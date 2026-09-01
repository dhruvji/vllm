# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)

pytestmark = pytest.mark.skip_global_cleanup


def test_kda_state_shape_reserves_speculative_conv_tokens() -> None:
    conv_state, _ = MambaStateShapeCalculator.kda_state_shape(
        tp_world_size=2,
        num_heads=8,
        head_dim=128,
        conv_kernel_size=4,
        num_spec=5,
    )

    conv_dim = 3 * 8 * 128 // 2
    expected = (conv_dim, 8) if is_conv_state_dim_first() else (8, conv_dim)
    assert conv_state == expected
