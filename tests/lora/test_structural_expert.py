# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from vllm.lora.lora_model import LoRAModel
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.structural_expert import (
    extract_structural_experts,
    route_structural_expert,
)
from vllm.models.glm5next.nvidia import model as glm5next_model
from vllm.models.glm5next.nvidia.model import Glm5NextMoE

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


def test_structural_only_checkpoint_does_not_require_adapter_config(tmp_path):
    save_file(
        structural_tensors(),
        tmp_path / "adapter_model.safetensors",
    )

    model = LoRAModel.from_local_checkpoint(
        str(tmp_path),
        expected_lora_modules=set(),
        peft_helper=PEFTHelper(r=1, lora_alpha=1, target_modules=[]),
        lora_model_id=1,
        device="cpu",
        dtype=torch.bfloat16,
    )

    assert not (tmp_path / "adapter_config.json").exists()
    assert set(model.structural_experts) == {"model.layers.0.mlp"}


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


def test_apply_structural_experts_only_runs_selected_slot_tokens(monkeypatch):
    moe = Glm5NextMoE.__new__(Glm5NextMoE)
    nn.Module.__init__(moe)
    moe.structural_present = torch.tensor([False, True])
    moe.structural_routed = torch.tensor([False, True])
    moe.structural_gate_up = torch.randn(2, 8, 6)
    moe.structural_down = torch.randn(2, 6, 4)
    moe.structural_act_fn = lambda gate_up: gate_up[:, :4]
    moe._structural_slot_indices = torch.tensor([1, -1, 1])
    moe._structural_route_weights = torch.tensor([0.5, 0.0, 0.0])
    moe.is_sequence_parallel = False
    moe.tp_size = 1
    hidden_states = torch.randn(3, 6)

    linear_batches = []
    linear = torch.nn.functional.linear

    def record_linear(inputs, weights):
        linear_batches.append(inputs.shape[0])
        return linear(inputs, weights)

    monkeypatch.setattr(torch.nn.functional, "linear", record_linear)

    output = moe._apply_structural_experts(hidden_states)

    assert linear_batches == [1, 1]
    assert output[1].eq(0).all()
    assert output[0].ne(0).any()
    assert output[2].eq(0).all()


def test_structural_slot_indices_follow_sequence_parallel_chunk():
    moe = Glm5NextMoE.__new__(Glm5NextMoE)
    nn.Module.__init__(moe)
    moe.experts = nn.Module()
    moe.experts.punica_wrapper = type(
        "Punica", (), {"token_lora_indices": torch.tensor([0, 0, 1])}
    )()
    moe.is_sequence_parallel = True
    moe.tp_rank = 1

    indices = moe._get_structural_slot_indices(torch.empty(2, 6))

    assert indices.tolist() == [1, -1]


def test_structural_routing_reads_base_gate_bias_after_lora_wrapping(monkeypatch):
    class GateWithLoRA(nn.Module):
        def __init__(self, correction_bias: torch.Tensor):
            super().__init__()
            self.base_layer = nn.Module()
            self.base_layer.e_score_correction_bias = nn.Parameter(correction_bias)

    moe = Glm5NextMoE.__new__(Glm5NextMoE)
    nn.Module.__init__(moe)
    moe.gate = GateWithLoRA(torch.tensor([0.25, -0.5]))
    moe.experts = nn.Module()
    moe.is_sequence_parallel = False
    moe.structural_present = torch.tensor([False])
    moe.structural_routed = torch.tensor([False])
    moe.structural_router = torch.zeros(1, 2)
    moe.structural_expert_bias = torch.zeros(1, 2)
    moe.scoring_func = "sigmoid"
    moe.routed_scaling_factor = 1.0
    captured = {}

    def capture_route(*args, **kwargs):
        captured["base_bias"] = args[3]
        return torch.ones(1, 1), torch.zeros(1, 1, dtype=torch.long), torch.zeros(1)

    monkeypatch.setattr(glm5next_model, "route_structural_expert", capture_route)

    moe._structural_expert_routing(
        hidden_states=torch.ones(1, 2),
        gating_output=torch.zeros(1, 2),
        topk=1,
        renormalize=True,
    )

    torch.testing.assert_close(captured["base_bias"], torch.tensor([[0.25, -0.5]]))
