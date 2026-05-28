#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run one text-only generation with a TorchTitan Qwen 3.5 prefix cartridge."""

import argparse

from vllm import LLM, SamplingParams
from vllm.v1.cartridge_loader import load_cartridge

DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_CARTRIDGE = (
    "s3://engram-cartridges/weights/torchtitan/2026-04-21/"
    "18-14-47-worktrial-singledoc-uniform-synced6-og-qwen3-5-9b-5c55ec98/"
    "step-0/adapter_model.safetensors"
)
DEFAULT_PROMPT = "Summarize the purpose of this adapter in one short sentence."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cartridge", default=DEFAULT_CARTRIDGE)
    parser.add_argument(
        "--no-cartridge",
        action="store_true",
        help="Run the same prompt without loading or attaching a cartridge.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--gdn-prefill-backend",
        choices=("auto", "flashinfer", "triton"),
        default="triton",
        help=(
            "Backend to use for Qwen 3.5 GDN prefill. "
            "Defaults to triton to avoid first-run FlashInfer JIT time."
        ),
    )
    args = parser.parse_args()

    cartridge = None
    stacked_kv = None
    if not args.no_cartridge:
        cartridge = load_cartridge(args.cartridge)
        stacked_kv = cartridge.get_stacked_kv()
        assert stacked_kv is not None

    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        dtype="bfloat16",
        language_model_only=True,
        max_model_len=6144,
        enforce_eager=True,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        gpu_memory_utilization=0.85,
        additional_config={"gdn_prefill_backend": args.gdn_prefill_backend},
    )

    params = SamplingParams(temperature=0.0, max_tokens=32)
    if args.no_cartridge:
        outputs = llm.generate(args.prompt, params, use_tqdm=False)
    else:
        assert cartridge is not None
        engine = llm.llm_engine
        request = engine.input_processor.process_inputs(
            request_id="0",
            prompt=args.prompt,
            params=params,
            supported_tasks=engine.get_supported_tasks(),
            cartridge_kv=stacked_kv,
            cartridge_id=args.cartridge,
            cartridge_seq_len=cartridge.num_tokens,
        )
        engine.add_request("0", request, params, prompt_text=args.prompt)
        outputs = llm._run_engine(use_tqdm=False)

    print(f"model={args.model}")
    print(f"cartridge={'<none>' if args.no_cartridge else args.cartridge}")
    print(f"prompt={args.prompt}")
    print(f"gdn_prefill_backend={args.gdn_prefill_backend}")
    print(f"generated_text={outputs[0].outputs[0].text}")


if __name__ == "__main__":
    main()
