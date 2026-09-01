# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.lora.utils import is_in_target_modules, is_supported_lora_module
from vllm.model_executor.models.interfaces import supports_lora


class TestIsSupportedLoraModule:
    """Tests for is_supported_lora_module (model-definition check)."""

    def test_suffix_match(self):
        assert is_supported_lora_module(
            "model.layers.0.self_attn.o_proj", ["o_proj", "q_proj"]
        )

    def test_no_match(self):
        assert not is_supported_lora_module(
            "model.layers.0.self_attn.o_proj", ["q_proj", "k_proj"]
        )

    def test_exact_match(self):
        assert is_supported_lora_module("o_proj", ["o_proj"])

    def test_regex_suffix_matching(self):
        """Regex anchors to end — partial suffix should not match."""
        assert not is_supported_lora_module("model.layers.0.self_attn.o_proj", ["proj"])

    def test_empty_supported_modules(self):
        assert not is_supported_lora_module("model.layers.0.self_attn.o_proj", [])

    def test_multiple_supported_modules(self):
        supported = ["q_proj", "k_proj", "v_proj", "o_proj"]
        assert is_supported_lora_module("model.layers.0.self_attn.v_proj", supported)
        assert not is_supported_lora_module("model.layers.0.mlp.gate_proj", supported)


class TestIsInTargetModules:
    """Tests for is_in_target_modules (deployment-time filter)."""

    def test_none_allows_all(self):
        assert is_in_target_modules("model.layers.0.self_attn.o_proj", None)

    def test_suffix_in_target(self):
        assert is_in_target_modules(
            "model.layers.0.self_attn.o_proj", ["o_proj", "q_proj"]
        )

    def test_suffix_not_in_target(self):
        assert not is_in_target_modules(
            "model.layers.0.self_attn.o_proj", ["q_proj", "k_proj"]
        )

    def test_empty_target_modules(self):
        assert not is_in_target_modules("model.layers.0.self_attn.o_proj", [])

    def test_exact_name_match(self):
        assert is_in_target_modules("dense1", ["dense1", "dense2"])

    def test_exact_name_no_match(self):
        assert not is_in_target_modules("dense3", ["dense1", "dense2"])

    def test_packed_parent_matches_child_target_modules(self):
        assert is_in_target_modules(
            "model.layers.0.mlp.gate_up_proj",
            ["gate_proj", "up_proj"],
            {"gate_up_proj": ["gate_proj", "up_proj"]},
        )

    def test_packed_child_matches_parent_target_modules(self):
        assert is_in_target_modules(
            "model.layers.0.mlp.gate_proj",
            ["gate_up_proj"],
            {"gate_up_proj": ["gate_proj", "up_proj"]},
        )

    def test_fused_parent_matches_child_target_modules(self):
        assert is_in_target_modules(
            "model.layers.0.self_attn.fused_qkv_a_proj",
            ["q_a_proj", "kv_a_proj_with_mqa"],
            {"fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"]},
        )


@pytest.mark.skip_global_cleanup
def test_glm5next_exposes_all_fused_lora_targets():
    from vllm.models.glm5next.nvidia.model import (
        Glm5NextForCausalLM,
        Glm5NextForConditionalGeneration,
    )

    expected = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
        "wk_weights_proj": ["wk", "weights_proj"],
        "in_proj_qkvbfg_a": [
            "q_proj",
            "k_proj",
            "v_proj",
            "b_proj",
            "f_a_proj",
            "g_a_proj",
        ],
    }

    assert supports_lora(Glm5NextForCausalLM)
    assert Glm5NextForCausalLM.packed_modules_mapping == expected
    assert Glm5NextForConditionalGeneration.packed_modules_mapping == {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        **expected,
    }


@pytest.mark.skip_global_cleanup
def test_glm5next_vlm_packed_lora_names_are_prefix_scoped():
    from vllm.lora.model_manager import LoRAModelManager
    from vllm.models.glm5next.nvidia.model import Glm5NextForConditionalGeneration

    manager = object.__new__(LoRAModelManager)
    manager.packed_modules_mapping = (
        Glm5NextForConditionalGeneration.packed_modules_mapping
    )
    manager.packed_modules = {}

    vision_parent = "visual.blocks.0.attn.qkv_proj"
    text_parent = "language_model.model.layers.0.mixer.in_proj_qkvbfg_a"
    manager._register_packed_modules(vision_parent)
    manager._register_packed_modules(text_parent)

    assert manager.packed_modules[vision_parent] == [
        "visual.blocks.0.attn.q_proj",
        "visual.blocks.0.attn.k_proj",
        "visual.blocks.0.attn.v_proj",
    ]
    assert manager.packed_modules[text_parent][:3] == [
        "language_model.model.layers.0.mixer.q_proj",
        "language_model.model.layers.0.mixer.k_proj",
        "language_model.model.layers.0.mixer.v_proj",
    ]


@pytest.mark.skip_global_cleanup
def test_glm5next_indexer_gate_adds_lora_delta_to_fp32_base():
    from vllm.models.glm5next.nvidia.attention import _compute_indexer_gate

    hidden_states = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    base_gate_weight = torch.tensor([[1.0, 0.5], [-0.25, 2.0]], dtype=torch.bfloat16)
    lora_delta = torch.tensor([[0.75, -1.25]], dtype=torch.bfloat16)
    fused_gate_output = (
        torch.nn.functional.linear(hidden_states, base_gate_weight) + lora_delta
    )
    base_gate_output = torch.nn.functional.linear(hidden_states, base_gate_weight)

    weights = _compute_indexer_gate(
        hidden_states,
        base_gate_weight.float().t().contiguous(),
        base_gate_output,
        fused_gate_output,
    )

    expected = torch.mm(hidden_states.float(), base_gate_weight.float().t())
    expected += lora_delta.float()
    torch.testing.assert_close(weights, expected)
