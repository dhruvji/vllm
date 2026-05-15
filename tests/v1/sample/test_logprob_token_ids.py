# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import MAX_LOGPROB_TOKEN_IDS, SamplingParams
from vllm.v1.worker.gpu.sample.logprob import (
    LogprobTokenIdsState,
    compute_topk_logprobs,
)


class DummyModelConfig:
    max_logprobs = 20

    def get_vocab_size(self) -> int:
        return 100


def test_logprob_token_ids_validation_rejects_mismatched_logprobs():
    params = SamplingParams(logprobs=1, logprob_token_ids=[3, 7])

    with pytest.raises(VLLMValidationError, match="logprobs must equal"):
        params.verify(DummyModelConfig(), None, None, None)


def test_logprob_token_ids_validation_rejects_too_many_tokens():
    params = SamplingParams(logprob_token_ids=list(range(MAX_LOGPROB_TOKEN_IDS + 1)))

    with pytest.raises(VLLMValidationError, match="greater than max allowed"):
        params.verify(DummyModelConfig(), None, None, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compute_logprobs_returns_requested_token_ids_outside_topk():
    device = torch.device("cuda")
    logits = torch.tensor([[0.0, 10.0, 9.0, -4.0]], device=device)
    sampled_token_ids = torch.tensor([1], device=device)
    state = LogprobTokenIdsState(max_num_reqs=1, device=device)
    params = SamplingParams(max_tokens=1, logprobs=2, logprob_token_ids=[3, 0])

    state.add_request(0, params)
    state.apply_staged_writes()

    output = compute_topk_logprobs(
        logits,
        num_logprobs=2,
        sampled_token_ids=sampled_token_ids,
        logprob_token_ids_state=state,
        expanded_idx_mapping=torch.tensor([0], device=device, dtype=torch.int32),
        max_per_req_token_ids=2,
    )

    assert output.logprob_token_ids.tolist() == [[1, 3, 0]]
    assert torch.isfinite(output.logprobs).all()
