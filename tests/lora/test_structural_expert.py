# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.lora.lora_model import LoRAModel
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.structural_expert import (
    extract_structural_experts,
    route_structural_expert,
)

pytestmark = pytest.mark.skip_global_cleanup


def structural_tensors() -> dict[str, torch.Tensor]:
    prefix = "model.layers.0.mlp"
    return {
        f"{prefix}.shared_expert.trainable_expert.gate_up_proj.weight": (
            torch.arange(48, dtype=torch.float32).reshape(8, 6)
        ),
        f"{prefix}.shared_expert.trainable_expert.down_proj.weight": (
            torch.arange(24, dtype=torch.float32).reshape(6, 4)
        ),
        f"{prefix}.gate.trainable_expert_router.weight": torch.ones(1, 6),
        f"{prefix}.shared_expert.expert_bias": torch.arange(3, dtype=torch.float32),
    }


def test_extract_structural_expert_preserves_portable_schema():
    ordinary, experts = extract_structural_experts(structural_tensors())

    assert ordinary == {}
    weights = experts["model.layers.0.mlp"]
    assert weights.is_routed
    assert weights.gate_up_proj.shape == (8, 6)
    assert weights.down_proj.shape == (6, 4)
    assert weights.router is not None and weights.router.shape == (1, 6)
    assert weights.expert_bias.shape == (3,)


def test_lora_model_accepts_structural_only_artifact():
    model = LoRAModel.from_lora_tensors(
        lora_model_id=1,
        tensors=structural_tensors(),
        peft_helper=PEFTHelper(r=1, lora_alpha=1, target_modules=[]),
        device="cpu",
        dtype=torch.bfloat16,
    )

    assert model.loras == {}
    weights = model.structural_experts["model.layers.0.mlp"]
    assert weights.gate_up_proj.dtype == torch.bfloat16
    assert weights.down_proj.dtype == torch.bfloat16
    assert weights.expert_bias.dtype == torch.float32


def test_extract_structural_expert_rejects_incomplete_layer():
    tensors = structural_tensors()
    tensors.pop("model.layers.0.mlp.shared_expert.trainable_expert.down_proj.weight")

    with pytest.raises(ValueError, match="missing tensors.*down_proj"):
        extract_structural_experts(tensors)


def test_route_structural_expert_replaces_added_slot_for_base_kernel():
    base_weights, base_ids, added_weights = route_structural_expert(
        gating_output=torch.tensor([[0.0, -2.0], [0.0, -2.0]]),
        extra_logits=torch.tensor([3.0, 3.0]),
        active=torch.tensor([True, False]),
        base_bias=torch.zeros(2, 2),
        topk=1,
        renormalize=True,
        scoring_func="sigmoid",
        routed_scaling_factor=2.5,
    )

    assert base_ids.tolist() == [[0], [0]]
    assert base_weights.tolist() == [[0.0], [2.5]]
    assert added_weights.tolist() == [2.5, 0.0]
