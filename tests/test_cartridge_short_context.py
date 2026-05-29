# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test vLLM cartridge error handling with short context length.

This test file runs in isolation to avoid GPU memory conflicts with
other cartridge tests that use a session-scoped vLLM server.
"""

import os

import openai
import pytest

from tests.utils import RemoteOpenAIServer
from vllm.platforms import current_platform

# Enable detailed error passthrough for testing
os.environ["VLLM_PASSTHROUGH_ERRORS"] = "1"

pytest_plugins = ("pytest_asyncio",)
pytestmark = [
    pytest.mark.asyncio(loop_scope="session"),
    pytest.mark.benchmark,
]

# Config
BASE_MODEL = "Qwen/Qwen3-0.6B"
CHECKPOINT_PATH = os.environ.get(
    "VLLM_TEST_CARTRIDGE_PATH",
    "s3://engram-cartridges/weights-toy/addition/qwen-0.6b/model.pt",
)


@pytest.fixture(scope="session")
def vllm_server_short_context():
    """Start vLLM server with very short max_model_len to test validation."""
    if current_platform.is_cpu():
        pytest.skip("GPU required for cartridge tests")

    if not os.environ.get("VLLM_TEST_ENABLE_CARTRIDGE"):
        pytest.skip("Cartridge tests disabled")

    # Use a very short max_model_len that will definitely be too small
    # when combined with cartridge + prompt + min output tokens
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
        "128",  # Very short to trigger the error
        "--enforce-eager",
    ]

    env = os.environ.copy()
    env["VLLM_PASSTHROUGH_ERRORS"] = "1"
    env["VLLM_USE_V1"] = "1"

    with RemoteOpenAIServer(
        BASE_MODEL, args, env_dict=env, max_wait_seconds=300
    ) as server:
        yield server


@pytest.fixture(scope="session")
def vllm_client_short_context(vllm_server_short_context):
    """Create OpenAI client for short context server."""
    return vllm_server_short_context.get_async_client()


@pytest.mark.benchmark
async def test_cartridge_length_exceeds_max_model_len_error(vllm_client_short_context):
    """Test that proper error is raised when cartridge + prompt exceeds max_model_len.

    This test verifies the fix for a bug where requests with
    max_model_len <= cartridge_length + request_length would spin forever
    instead of returning a clear error message.

    The test uses a too-short max_model_len (128) and verifies that:
    1. The request is rejected immediately (not stuck spinning)
    2. The error message clearly explains the issue with specific numbers
    """
    # Use a simple prompt - with max_model_len=128, even a short prompt
    # should fail when combined with the cartridge
    prompt = "Problem inputs: 1 and 2"

    print(
        "\nTesting error handling for cartridge + prompt exceeding max_model_len=128..."
    )

    try:
        response = await vllm_client_short_context.chat.completions.create(
            model=BASE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1,
            extra_body={
                "adapters": {
                    "prefix": [
                        {"id": CHECKPOINT_PATH, "source": "s3", "type": "prefix"}
                    ]
                },
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        # If we get here, the request succeeded when it should have failed
        print("ERROR: Request succeeded when it should have been rejected")
        print(f"Response: {response.choices[0].message.content[:100]}")
        pytest.fail(
            "Expected ValueError for cartridge + prompt exceeding max_model_len, "
            "but request succeeded"
        )
    except openai.BadRequestError as e:
        # This is what we expect - a validation error
        error_msg = str(e)
        print(f"✓ Got expected error: {error_msg}")

        # Verify the error message contains the expected information
        assert (
            "cartridge length" in error_msg.lower()
            or "cannot be scheduled" in error_msg.lower()
        ), (
            "Error message should mention cartridge length or scheduling issue, "
            f"but got: {error_msg}"
        )

        assert "max_model_len" in error_msg.lower(), (
            f"Error message should mention max_model_len, but got: {error_msg}"
        )

        # Verify it includes the actual numbers
        assert "128" in error_msg, (
            "Error message should include max_model_len value (128), but got: "
            f"{error_msg}"
        )

        print("✓ Error message correctly explains the issue with specific numbers")
    except Exception as e:
        # Some other error occurred
        print(f"Unexpected error type: {type(e).__name__}: {e}")
        raise
