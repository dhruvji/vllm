#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Run one isolated Engram upgrade inference-matrix cell."""

import argparse
import json
import sys
import time
from pathlib import Path

# Direct execution puts ``tools/`` first, where ``tools/vllm-*`` can be
# discovered as a namespace package instead of the repository's vllm package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--lora", required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--prefix-caching", action="store_true")
    parser.add_argument("--language-model-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        language_model_only=args.language_model_only,
        max_model_len=512,
        max_num_seqs=2,
        gpu_memory_utilization=0.2,
        enable_lora=True,
        enable_prefix_caching=args.prefix_caching,
        enforce_eager=not args.compile,
        attention_backend="TRITON_ATTN",
    )
    params = SamplingParams(temperature=0.0, max_tokens=8)
    prompt = "Answer with one word: what color is a clear daytime sky?"
    base = llm.generate([prompt, prompt], params, use_tqdm=False)
    adapted = llm.generate(
        [prompt, prompt],
        params,
        lora_request=LoRARequest("upgrade-smoke", 1, args.lora),
        use_tqdm=False,
    )
    base_text = [output.outputs[0].text for output in base]
    adapted_text = [output.outputs[0].text for output in adapted]
    assert all(text.strip() for text in base_text + adapted_text)
    print(
        json.dumps(
            {
                "model": args.model,
                "lora": args.lora,
                "compile": args.compile,
                "prefix_caching": args.prefix_caching,
                "base": base_text,
                "adapted": adapted_text,
                "elapsed_seconds": round(time.monotonic() - started, 2),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
