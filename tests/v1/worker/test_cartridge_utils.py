# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

from vllm.v1.worker.cartridge_utils import (
    strip_cartridge_offsets_from_seq_lens,
)


def test_strip_cartridge_offsets_from_seq_lens():
    seq_lens = np.array([4097, 4101, 3, 0], dtype=np.int32)
    req_ids = ["req-a", "req-b", "req-c"]
    offsets = {
        "req-a": 4096,
        "req-b": 4096,
    }

    adjusted = strip_cartridge_offsets_from_seq_lens(
        seq_lens,
        req_ids,
        offsets,
    )

    np.testing.assert_array_equal(adjusted, np.array([1, 5, 3, 0], dtype=np.int32))


def test_strip_cartridge_offsets_from_torch_seq_lens():
    seq_lens = torch.tensor([4097, 4101, 3, 0], dtype=torch.int32)
    req_ids = ["req-a", "req-b", "req-c"]
    offsets = {
        "req-a": 4096,
        "req-b": 4096,
    }

    adjusted = strip_cartridge_offsets_from_seq_lens(
        seq_lens,
        req_ids,
        offsets,
    )

    np.testing.assert_array_equal(adjusted, np.array([1, 5, 3, 0], dtype=np.int32))
