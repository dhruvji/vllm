# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode import speculator as base_speculator
from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator


@pytest.mark.skip_global_cleanup
def test_build_draft_attn_metadata_advances_sequence_length_upper_bound():
    input_buffers = SimpleNamespace(
        query_start_loc=torch.zeros(5, dtype=torch.int32),
        seq_lens=torch.zeros(4, dtype=torch.int32),
    )
    block_tables = SimpleNamespace(
        input_block_tables=[torch.zeros(4, 4, dtype=torch.int32)],
        slot_mappings=torch.zeros(1, 8, dtype=torch.int64),
    )
    speculator = SimpleNamespace(
        arange=torch.arange(5, dtype=torch.int32),
        block_tables=block_tables,
        input_buffers=input_buffers,
        attn_groups=[],
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
        max_model_len=1024,
        draft_max_seq_len=1024,
    )
    captured = {}

    def capture_metadata(**kwargs):
        captured.update(kwargs)
        return {}

    with patch.object(base_speculator, "build_attn_metadata", capture_metadata):
        EagleSpeculator._build_draft_attn_metadata(
            speculator,
            num_reqs=3,
            num_reqs_padded=4,
            num_tokens_padded=4,
            seq_lens_cpu_upper_bound=torch.tensor(
                [100, 200, 1023, 0], dtype=torch.int32
            ),
            step=2,
        )

    torch.testing.assert_close(
        captured["seq_lens_cpu_upper_bound"],
        torch.tensor([102, 202, 1024, 0], dtype=torch.int32),
    )
