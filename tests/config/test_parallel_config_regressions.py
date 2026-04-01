# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config import ParallelConfig


def test_parallel_config_restores_missing_elastic_ep_fields():
    config = ParallelConfig()

    assert config.enable_ep_weight_filter is False
    assert config.enable_elastic_ep is False
    assert config._coord_store_port == 0


def test_parallel_config_requires_eplb_for_elastic_ep():
    with pytest.raises(ValueError, match="enable_eplb=True"):
        ParallelConfig(enable_elastic_ep=True)
