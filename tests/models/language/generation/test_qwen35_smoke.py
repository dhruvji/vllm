# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

from vllm import SamplingParams
from vllm.outputs import RequestOutput
from vllm.v1.cartridge_loader import load_cartridge

QWEN35_MODEL = "Qwen/Qwen3.5-0.8B"
QWEN3_MODEL = "Qwen/Qwen3-0.6B"
ZEROED_CARTRIDGE = os.environ.get(
    "VLLM_TEST_ZEROED_CARTRIDGE_PATH",
    "s3://engram-cartridges/weights-toy/zeroed/qwen-0.6b/model.pt",
)


def test_qwen35_prefix_cache_smoke(vllm_runner) -> None:
    prompt = "The opposite of hot is cold. Repeat the phrase exactly once:"
    prompts = [prompt, prompt]

    with vllm_runner(
        QWEN35_MODEL,
        language_model_only=True,
        max_model_len=1024,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        enforce_eager=True,
        attention_backend="TRITON_ATTN",
    ) as llm:
        outputs = llm.generate_greedy(prompts, max_tokens=8)

    assert len(outputs) == len(prompts), "Expected one output per prompt"
    for _, output_text in outputs:
        assert output_text.strip(), "Expected non-empty generated text"


def test_qwen3_loads_zeroed_cartridge_and_generates(vllm_runner, monkeypatch) -> None:
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    prompt = "Write one short sentence about the ocean."
    cartridge = load_cartridge(ZEROED_CARTRIDGE)
    stacked_kv = cartridge.get_stacked_kv()

    with vllm_runner(
        QWEN3_MODEL,
        max_model_len=512,
        enforce_eager=True,
        attention_backend="TRITON_ATTN",
    ) as runner:
        llm = runner.get_llm()
        engine = llm.llm_engine
        params = SamplingParams(temperature=0.0, max_tokens=12)
        request = engine.input_processor.process_inputs(
            request_id="0",
            prompt=prompt,
            params=params,
            supported_tasks=engine.get_supported_tasks(),
            cartridge_kv=stacked_kv,
            cartridge_id=ZEROED_CARTRIDGE,
            cartridge_seq_len=cartridge.num_tokens,
        )
        engine.add_request("0", request, params, prompt_text=prompt)
        outputs = llm._run_engine(RequestOutput, use_tqdm=False)

    assert len(outputs) == 1, f"Expected one completed request, got {len(outputs)}"
    assert outputs[0].finished, "Expected request to finish successfully"
    assert len(outputs[0].outputs) == 1, "Expected one sample output"

    sample = outputs[0].outputs[0]
    assert sample.text.strip(), "Expected non-empty generated text"
    assert sample.token_ids, "Expected at least one generated token"
