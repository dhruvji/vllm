# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors.torch import save_file

RANK = 32
ALPHA = 32
SEED = 0
NUM_LAYERS = 45

TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "f_a_proj",
    "f_b_proj",
    "b_proj",
    "g_a_proj",
    "g_b_proj",
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "o_proj",
)

KDA_PROJECTIONS = (
    ("q_proj", 4096, 8192),
    ("k_proj", 4096, 8192),
    ("v_proj", 4096, 8192),
    ("f_a_proj", 4096, 128),
    ("f_b_proj", 128, 8192),
    ("b_proj", 4096, 64),
    ("g_a_proj", 4096, 128),
    ("g_b_proj", 128, 8192),
    ("o_proj", 8192, 4096),
)

MLA_PROJECTIONS = (
    ("q_a_proj", 4096, 1536),
    ("q_b_proj", 1536, 16384),
    ("kv_a_proj_with_mqa", 4096, 512),
    ("kv_b_proj", 512, 32768),
    ("o_proj", 16384, 4096),
)


def projection_specs() -> list[tuple[int, str, int, int]]:
    specs = []
    for layer in range(NUM_LAYERS):
        projections = MLA_PROJECTIONS if layer % 4 == 3 else KDA_PROJECTIONS
        specs.extend(
            (layer, name, input_size, output_size)
            for name, input_size, output_size in projections
        )
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    tensors: dict[str, torch.Tensor] = {}
    for layer, module, input_size, output_size in projection_specs():
        prefix = f"base_model.model.model.layers.{layer}.self_attn.{module}"
        lora_a = torch.empty(
            RANK,
            input_size,
            dtype=torch.bfloat16,
            device=args.device,
        )
        torch.nn.init.kaiming_uniform_(lora_a, a=math.sqrt(5))
        tensors[f"{prefix}.lora_A.weight"] = lora_a.cpu()
        tensors[f"{prefix}.lora_B.weight"] = torch.zeros(
            output_size,
            RANK,
            dtype=torch.bfloat16,
        )

    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": "zai-org/GLM-5.3-Flash",
        "inference_mode": True,
        "init_lora_weights": True,
        "r": RANK,
        "lora_alpha": ALPHA,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": list(TARGET_MODULES),
    }
    (args.output / "adapter_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    save_file(tensors, args.output / "adapter_model.safetensors")


if __name__ == "__main__":
    main()
