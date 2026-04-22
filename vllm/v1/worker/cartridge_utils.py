# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping, Sequence

import numpy as np


def strip_cartridge_offsets_from_seq_lens(
    seq_lens: np.ndarray,
    req_ids: Sequence[str],
    cartridge_position_offsets: Mapping[str, int],
) -> np.ndarray:
    """Remove attention-only cartridge offsets from sequence lengths."""
    adjusted = seq_lens.copy()
    for req_idx, req_id in enumerate(req_ids):
        adjusted[req_idx] -= cartridge_position_offsets.get(req_id, 0)
    return adjusted
