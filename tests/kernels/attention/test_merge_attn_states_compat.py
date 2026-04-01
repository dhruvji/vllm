# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.attention.ops import merge_attn_states as merge_attn_states_mod


def _make_inputs() -> tuple[torch.Tensor, ...]:
    output = torch.empty(2, 1, 8, dtype=torch.float16)
    prefix_output = torch.randn_like(output)
    suffix_output = torch.randn_like(output)
    prefix_lse = torch.randn(1, 2, dtype=torch.float32)
    suffix_lse = torch.randn(1, 2, dtype=torch.float32)
    return output, prefix_output, prefix_lse, suffix_output, suffix_lse


def test_merge_attn_states_uses_custom_kernel_without_prefill(monkeypatch):
    called = {}

    monkeypatch.setattr(merge_attn_states_mod.current_platform, "is_cuda", lambda: True)

    from vllm import _custom_ops
    from vllm.v1.attention.ops import triton_merge_attn_states

    monkeypatch.setattr(
        _custom_ops,
        "merge_attn_states_supports_prefill_tokens_with_context",
        lambda: False,
    )

    def fake_custom(*args):
        called["custom"] = args

    def fake_triton(*args):
        called["triton"] = len(args)

    monkeypatch.setattr(_custom_ops, "merge_attn_states", fake_custom)
    monkeypatch.setattr(triton_merge_attn_states, "merge_attn_states", fake_triton)

    merge_attn_states_mod.merge_attn_states(*_make_inputs())

    assert called["custom"][-1] is None


def test_merge_attn_states_falls_back_to_triton_with_prefill(monkeypatch):
    called = {}

    monkeypatch.setattr(merge_attn_states_mod.current_platform, "is_cuda", lambda: True)

    from vllm import _custom_ops
    from vllm.v1.attention.ops import triton_merge_attn_states

    monkeypatch.setattr(
        _custom_ops,
        "merge_attn_states_supports_prefill_tokens_with_context",
        lambda: False,
    )

    def fake_custom(*args):
        called["custom"] = args

    def fake_triton(*args):
        called["triton"] = args

    monkeypatch.setattr(_custom_ops, "merge_attn_states", fake_custom)
    monkeypatch.setattr(triton_merge_attn_states, "merge_attn_states", fake_triton)

    merge_attn_states_mod.merge_attn_states(*_make_inputs(), prefill_tokens_with_context=1)

    assert "custom" not in called
    assert called["triton"][-1] == 1
