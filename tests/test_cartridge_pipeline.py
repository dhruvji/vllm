# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test mid-generation cartridge KV overwrite (pipeline support).

Requires VLLM_TEST_ENABLE_CARTRIDGE=1 and a GPU.
"""

import asyncio
import os

import pytest

from tests.utils import RemoteOpenAIServer
from vllm.platforms import current_platform

os.environ["VLLM_PASSTHROUGH_ERRORS"] = "1"

pytest_plugins = ("pytest_asyncio",)
pytestmark = [
    pytest.mark.asyncio(loop_scope="session"),
    pytest.mark.benchmark,
]

BASE_MODEL = "Qwen/Qwen3-0.6B"
CHECKPOINT_PATH = os.environ.get(
    "VLLM_TEST_CARTRIDGE_PATH",
    "s3://engram-cartridges/weights-toy/addition/qwen-0.6b/model.pt",
)
UPDATED_CHECKPOINT_PATH = os.environ.get("VLLM_TEST_CARTRIDGE_PATH_NEW")


@pytest.fixture(scope="session")
def pipeline_server():
    """Start vLLM server for cartridge pipelining tests."""
    if current_platform.is_cpu():
        pytest.skip("GPU required for cartridge tests")
    if not os.environ.get("VLLM_TEST_ENABLE_CARTRIDGE"):
        pytest.skip(
            "Cartridge tests disabled (set VLLM_TEST_ENABLE_CARTRIDGE=1 to enable)"
        )

    args = [
        "--served-model-name",
        BASE_MODEL,
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--enable-lora",
        "--max-lora-rank",
        "256",
        "--max-loras",
        "1",
        "--max-model-len",
        "2048",
        "--enforce-eager",
    ]

    env = os.environ.copy()
    env["VLLM_PASSTHROUGH_ERRORS"] = "1"
    env["VLLM_USE_V1"] = "1"
    env["VLLM_CARTRIDGE_DEBUG_TIMING"] = "1"

    with RemoteOpenAIServer(
        BASE_MODEL, args, env_dict=env, max_wait_seconds=300
    ) as server:
        yield server


@pytest.fixture(scope="session")
def pipeline_client(pipeline_server):
    return pipeline_server.get_async_client()


async def test_cartridge_allow_pipelining_smoke(pipeline_client):
    """Smoke test: requests can include allow_pipelining without errors."""
    response = await pipeline_client.chat.completions.create(
        model=BASE_MODEL,
        messages=[{"role": "user", "content": "Problem inputs: 1 and 2"}],
        temperature=0.0,
        max_tokens=32,
        extra_body={
            "adapters": {
                "prefix": [
                    {
                        "id": CHECKPOINT_PATH,
                        "source": "s3",
                        "type": "prefix",
                        "allow_pipelining": True,
                    }
                ]
            },
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    text = response.choices[0].message.content.strip()
    assert len(text) > 0
    print("✓ allow_pipelining request succeeded")


async def test_cartridge_pipeline_update_during_generation(
    pipeline_server, pipeline_client
):
    """Test that a cartridge can be updated mid-generation via allow_pipelining."""
    if not UPDATED_CHECKPOINT_PATH:
        pytest.skip("Set VLLM_TEST_CARTRIDGE_PATH_NEW to test pipelining updates")
    # Start a streaming request that will take a while to generate
    stream = await pipeline_client.chat.completions.create(
        model=BASE_MODEL,
        messages=[{"role": "user", "content": "Problem inputs: 42 and 58"}],
        temperature=0.0,
        max_tokens=128,
        stream=True,
        extra_body={
            "adapters": {
                "prefix": [{"id": CHECKPOINT_PATH, "source": "s3", "type": "prefix"}]
            },
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )

    # Collect first few chunks to ensure generation has started
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
        if len(chunks) >= 3:
            break

    print(f"Generation started, collected {len(chunks)} chunks")

    # Trigger pipelining via a new request with a newer step path
    response = await pipeline_client.chat.completions.create(
        model=BASE_MODEL,
        messages=[{"role": "user", "content": "Problem inputs: 5 and 7"}],
        temperature=0.0,
        max_tokens=64,
        extra_body={
            "adapters": {
                "prefix": [
                    {
                        "id": UPDATED_CHECKPOINT_PATH,
                        "source": "s3",
                        "type": "prefix",
                        "allow_pipelining": True,
                    }
                ]
            },
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    assert response.choices[0].message.content.strip()

    # Continue collecting the rest of the stream
    async for chunk in stream:
        chunks.append(chunk)

    # Verify we got a complete response
    full_text = "".join(
        c.choices[0].delta.content or ""
        for c in chunks
        if c.choices and c.choices[0].delta.content
    )
    print(f"Full response ({len(chunks)} chunks): {full_text!r}")
    assert len(full_text) > 0, "Empty response after cartridge update"
    print("✓ Generation completed successfully after mid-generation update")


async def test_cartridge_pipeline_update_then_new_request(
    pipeline_server, pipeline_client
):
    """Test that after a pipelined update, requests still succeed."""
    if not UPDATED_CHECKPOINT_PATH:
        pytest.skip("Set VLLM_TEST_CARTRIDGE_PATH_NEW to test pipelining updates")
    # Step 1: Initial request to populate the cartridge cache
    response1 = await pipeline_client.chat.completions.create(
        model=BASE_MODEL,
        messages=[{"role": "user", "content": "Problem inputs: 10 and 20"}],
        temperature=0.0,
        max_tokens=64,
        extra_body={
            "adapters": {
                "prefix": [{"id": CHECKPOINT_PATH, "source": "s3", "type": "prefix"}]
            },
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    text1 = response1.choices[0].message.content.strip()
    print(f"Response 1: {text1!r}")
    assert len(text1) > 0

    # Step 2: Trigger pipelining with a newer cartridge path
    response_update = await pipeline_client.chat.completions.create(
        model=BASE_MODEL,
        messages=[{"role": "user", "content": "Problem inputs: 3 and 4"}],
        temperature=0.0,
        max_tokens=64,
        extra_body={
            "adapters": {
                "prefix": [
                    {
                        "id": UPDATED_CHECKPOINT_PATH,
                        "source": "s3",
                        "type": "prefix",
                        "allow_pipelining": True,
                    }
                ]
            },
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    assert response_update.choices[0].message.content.strip()

    # Step 3: New request after update — should work fine
    response2 = await pipeline_client.chat.completions.create(
        model=BASE_MODEL,
        messages=[{"role": "user", "content": "Problem inputs: 33 and 67"}],
        temperature=0.0,
        max_tokens=64,
        extra_body={
            "adapters": {
                "prefix": [{"id": CHECKPOINT_PATH, "source": "s3", "type": "prefix"}]
            },
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    text2 = response2.choices[0].message.content.strip()
    print(f"Response 2: {text2!r}")
    assert len(text2) > 0
    print("✓ New request after cartridge update succeeded")


async def test_cartridge_pipeline_concurrent_update_and_requests(
    pipeline_server, pipeline_client
):
    """Test updating a cartridge while multiple requests are in flight.

    Sends several concurrent requests, fires an update mid-flight,
    and verifies all requests complete without errors.
    """
    if not UPDATED_CHECKPOINT_PATH:
        pytest.skip("Set VLLM_TEST_CARTRIDGE_PATH_NEW to test pipelining updates")
    n_concurrent = 5

    async def make_request(i: int):
        return await pipeline_client.chat.completions.create(
            model=BASE_MODEL,
            messages=[{"role": "user", "content": f"Problem inputs: {i} and {i + 10}"}],
            temperature=0.0,
            max_tokens=64,
            extra_body={
                "adapters": {
                    "prefix": [
                        {
                            "id": CHECKPOINT_PATH,
                            "source": "s3",
                            "type": "prefix",
                        }
                    ]
                },
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

    async def send_update_after_delay():
        await asyncio.sleep(0.5)  # Let requests start generating
        return await pipeline_client.chat.completions.create(
            model=BASE_MODEL,
            messages=[{"role": "user", "content": "Problem inputs: 11 and 12"}],
            temperature=0.0,
            max_tokens=64,
            extra_body={
                "adapters": {
                    "prefix": [
                        {
                            "id": UPDATED_CHECKPOINT_PATH,
                            "source": "s3",
                            "type": "prefix",
                            "allow_pipelining": True,
                        }
                    ]
                },
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

    # Launch requests + update concurrently
    request_tasks = [make_request(i) for i in range(n_concurrent)]
    update_task = send_update_after_delay()
    all_tasks = request_tasks + [update_task]

    print(f"Sending {n_concurrent} requests + 1 update concurrently...")
    results = await asyncio.gather(*all_tasks, return_exceptions=True)

    # Check update succeeded
    update_result = results[-1]
    if isinstance(update_result, BaseException):
        raise update_result
    assert update_result.choices[0].message.content.strip()
    print("Update request succeeded")

    # Check all requests succeeded
    for i, result in enumerate(results[:-1]):
        if isinstance(result, BaseException):
            print(f"Request {i} failed: {result}")
            raise result
        text = result.choices[0].message.content.strip()
        print(f"  [{i}] response: {text[:80]}...")
        assert len(text) > 0, f"Empty response for request {i}"

    print(f"✓ All {n_concurrent} requests + update completed successfully")
