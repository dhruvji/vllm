# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

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


class _StepAwareDraftModel(_DraftModel):
    def __init__(self, output: torch.Tensor):
        super().__init__(output)
        self.forward_step: int | None = None
        self.logits_step: int | None = None
        self.top_tokens_step: int | None = None

    def forward(self, spec_step_idx: int = 0, **kwargs):
        self.forward_step = spec_step_idx
        return self.output

    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor:
        self.logits_step = spec_step_idx
        return hidden_states

    def get_top_tokens(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor:
        self.top_tokens_step = spec_step_idx
        return hidden_states.argmax(dim=-1)


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


def test_run_model_forwards_spec_step_idx(monkeypatch):
    hidden = torch.full((4, 3), 1.0)
    speculator = _make_speculator(monkeypatch, hidden)
    model = _StepAwareDraftModel(hidden)
    speculator.model = model

    speculator._run_model(
        4,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        spec_step_idx=3,
    )

    assert model.forward_step == 3


@pytest.mark.parametrize("use_local_argmax_reduction", [False, True])
def test_greedy_sample_draft_forwards_spec_step_idx(use_local_argmax_reduction):
    hidden = torch.tensor([[0.0, 2.0, 1.0]])
    model = _StepAwareDraftModel(hidden)
    speculator = object.__new__(_TestSpeculator)
    speculator.model = model
    speculator.use_local_argmax_reduction = use_local_argmax_reduction

    tokens = speculator._greedy_sample_draft(hidden, spec_step_idx=4)

    assert torch.equal(tokens, torch.tensor([1]))
    if use_local_argmax_reduction:
        assert model.top_tokens_step == 4
        assert model.logits_step is None
    else:
        assert model.logits_step == 4
        assert model.top_tokens_step is None


def test_multi_step_decode_replays_layer_specific_full_graphs():
    speculator = object.__new__(_TestSpeculator)
    speculator.num_speculative_steps = 5
    speculator.current_draft_step = torch.tensor(0)
    speculator.input_buffers = SimpleNamespace(
        positions=torch.arange(2),
        query_start_loc=torch.arange(3),
    )
    speculator.idx_mapping = torch.arange(2)
    speculator.decode_cudagraph_managers = {
        step: SimpleNamespace(run_fullgraph=Mock()) for step in range(1, 5)
    }
    speculator._get_spec_step_idx = lambda step: step
    batch_desc = SimpleNamespace(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=2,
        num_reqs=2,
    )

    speculator._multi_step_decode(
        num_reqs=2,
        skip_attn=True,
        batch_desc=batch_desc,
        num_tokens_across_dp=None,
        seq_lens_cpu_upper_bound=None,
    )

    for manager in speculator.decode_cudagraph_managers.values():
        manager.run_fullgraph.assert_called_once_with(batch_desc)


def test_mtp_capture_uses_one_full_graph_per_predictor_layer():
    speculator = object.__new__(MTPSpeculator)
    speculator.num_speculative_steps = 5
    speculator.max_num_reqs = 2
    speculator.last_token_indices = torch.zeros(2, dtype=torch.int64)
    speculator.current_draft_step = torch.tensor(0)
    speculator.share_mtp_topk_indices = False
    speculator.prefill_cudagraph_manager = Mock(use_breakable_cg=False)
    speculator.decode_cudagraph_managers = {step: Mock() for step in range(1, 5)}
    speculator.decode_cudagraph_capture_steps = {step: step for step in range(1, 5)}
    speculator.model_state = Mock()
    speculator.target_input_buffers = Mock()
    speculator.input_buffers = Mock()
    speculator.block_tables = Mock()
    speculator.target_attn_groups = Mock()
    speculator.attn_groups = Mock()
    speculator.kv_cache_config = Mock()

    speculator.capture()

    for step, manager in speculator.decode_cudagraph_managers.items():
        capture_fn = manager.capture.call_args.args[0]
        assert capture_fn.keywords["spec_step_idx"] == step


def test_mtp_spec_step_idx_wraps_predictor_layers():
    speculator = object.__new__(MTPSpeculator)
    speculator.model = SimpleNamespace(
        supports_spec_step_idx=True,
        num_mtp_layers=5,
    )

    assert [speculator._get_spec_step_idx(step) for step in range(7)] == [
        0,
        1,
        2,
        3,
        4,
        0,
        1,
    ]

    speculator.model = SimpleNamespace()
    assert speculator._get_spec_step_idx(3) is None


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
