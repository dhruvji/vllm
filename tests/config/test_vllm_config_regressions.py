# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.config import VllmConfig


def test_vllm_config_restores_reasoning_and_shutdown_defaults():
    config = VllmConfig()

    assert config.reasoning_config is None
    assert config.shutdown_timeout == 0
    assert config.num_speculative_tokens == 0


def test_vllm_config_num_speculative_tokens_forwards_to_speculative_config():
    config = VllmConfig()
    config.speculative_config = SimpleNamespace(num_speculative_tokens=7)

    assert config.num_speculative_tokens == 7
