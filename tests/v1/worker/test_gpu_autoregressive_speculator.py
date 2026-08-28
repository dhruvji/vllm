# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.autoregressive import speculator as spec_module
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.mtp import speculator as mtp_spec_module
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

pytestmark = pytest.mark.skip_global_cleanup


class _TestSpeculator(AutoRegressiveSpeculator):
    def load_draft_model(self, target_model, target_attn_layer_names):
        raise NotImplementedError


class _DraftModel(torch.nn.Module):
    def __init__(self, output: torch.Tensor | tuple[torch.Tensor, torch.Tensor]):
        super().__init__()
        self.output = output

    def forward(self, **kwargs):
        return self.output


def _make_speculator(
    monkeypatch,
    output: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> _TestSpeculator:
    monkeypatch.setattr(
        spec_module,
        "set_forward_context",
        lambda *args, **kwargs: nullcontext(),
    )

    speculator = object.__new__(_TestSpeculator)
    speculator.supports_mm_inputs = False
    speculator.vllm_config = None
    speculator.input_buffers = SimpleNamespace(
        input_ids=torch.arange(4),
        positions=torch.arange(4),
    )
    speculator.hidden_states = torch.zeros(4, 3)
    speculator.model = _DraftModel(output)
    return speculator


def test_run_model_unpacks_tuple_return_for_mtp(monkeypatch):
    logits_hidden = torch.full((4, 3), 1.0)
    feedback_hidden = torch.full((4, 3), 2.0)
    speculator = _make_speculator(monkeypatch, (logits_hidden, feedback_hidden))

    actual_logits_hidden, actual_feedback_hidden = speculator._run_model(
        4,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    assert actual_logits_hidden is logits_hidden
    assert actual_feedback_hidden is feedback_hidden


def test_run_model_reuses_tensor_return_for_mtp(monkeypatch):
    hidden = torch.full((4, 3), 1.0)
    speculator = _make_speculator(monkeypatch, hidden)

    actual_logits_hidden, actual_feedback_hidden = speculator._run_model(
        4,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    assert actual_logits_hidden is hidden
    assert actual_feedback_hidden is hidden


def test_mtp_reuses_compacted_topk_indices_between_draft_steps(monkeypatch):
    compacted: list[torch.Tensor] = []
    skip_topk: list[bool] = []
    draft_model = SimpleNamespace(
        model=SimpleNamespace(
            set_skip_topk=skip_topk.append,
            compact_topk_indices=lambda indices: compacted.append(indices.clone()),
        )
    )
    monkeypatch.setattr(
        mtp_spec_module,
        "load_eagle_model",
        lambda target_model, vllm_config: draft_model,
    )
    speculator = object.__new__(MTPSpeculator)
    speculator.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(index_share_for_mtp_iteration=True)
            )
        )
    )
    speculator.model = speculator.load_draft_model(None, set())
    speculator.num_speculative_steps = 5
    speculator.last_token_indices = torch.tensor([3, 7, 11])

    speculator.on_prefill_begin(2)
    speculator.on_prefill_end(2)
    speculator.on_multi_step_decode_begin(2)
    speculator.on_multi_step_decode_end(2)

    assert skip_topk == [False, True, False]
    assert len(compacted) == 1
    assert torch.equal(compacted[0], torch.tensor([3, 7]))
