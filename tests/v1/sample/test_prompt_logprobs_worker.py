# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.v1.worker.gpu.sample.prompt_logprob as prompt_logprob_module
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.prompt_logprob import PromptLogprobsWorker


def test_prompt_logprobs_end_index_requires_prompt_logprobs():
    with pytest.raises(ValueError, match="require prompt_logprobs"):
        SamplingParams(prompt_logprobs_end_index=3)


def test_prompt_logprobs_end_index_must_not_precede_start_index():
    with pytest.raises(ValueError, match="greater than or equal"):
        SamplingParams(
            prompt_logprobs=2,
            prompt_logprobs_start_index=4,
            prompt_logprobs_end_index=3,
        )


def test_prompt_logprobs_worker_applies_bounded_range(monkeypatch):
    worker = PromptLogprobsWorker(max_num_reqs=1)
    sampling_params = SamplingParams(
        prompt_logprobs=2,
        prompt_logprobs_start_index=2,
        prompt_logprobs_end_index=4,
    )
    worker.add_request("req-1", 0, sampling_params)

    def fake_get_prompt_logprobs_token_ids(*args, **kwargs):
        return torch.arange(5)

    def fake_compute_prompt_logprobs_with_chunking(*args, **kwargs):
        return (
            torch.arange(5).view(5, 1),
            torch.arange(5, dtype=torch.float32).view(5, 1),
            torch.arange(5).view(5, 1),
        )

    monkeypatch.setattr(
        prompt_logprob_module,
        "get_prompt_logprobs_token_ids",
        fake_get_prompt_logprobs_token_ids,
    )
    monkeypatch.setattr(
        prompt_logprob_module,
        "compute_prompt_logprobs_with_chunking",
        fake_compute_prompt_logprobs_with_chunking,
    )

    input_batch = SimpleNamespace(
        idx_mapping_np=np.array([0], dtype=np.int32),
        idx_mapping=torch.tensor([0], dtype=torch.int32),
        num_tokens=5,
        query_start_loc=torch.tensor([0, 5], dtype=torch.int32),
        query_start_loc_np=np.array([0, 5], dtype=np.int32),
        num_scheduled_tokens=np.array([5], dtype=np.int32),
        req_ids=["req-1"],
    )

    result = worker.compute_prompt_logprobs(
        logits_fn=lambda x: x,
        hidden_states=torch.empty(5, 1),
        input_batch=input_batch,
        all_token_ids=torch.empty(1, 5, dtype=torch.int64),
        num_computed_tokens=torch.tensor([0], dtype=torch.int32),
        prompt_lens=np.array([5], dtype=np.int32),
        prefill_lens=np.array([5], dtype=np.int32),
        num_computed_prefill_tokens=np.array([0], dtype=np.int32),
    )

    logprobs = result["req-1"]
    assert logprobs.logprob_token_ids.tolist() == [[1], [2]]
    assert logprobs.logprobs.tolist() == [[1.0], [2.0]]
    assert logprobs.selected_token_ranks.tolist() == [[1], [2]]
