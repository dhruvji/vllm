# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This test file includes some cases where it is inappropriate to
only get the `eos_token_id` from the tokenizer as defined by
`BaseRenderer.get_eos_token_id`.
"""

import pytest

from vllm.tokenizers import get_tokenizer
from vllm.transformers_utils.config import (
    try_get_generation_config,
)
from vllm.transformers_utils.configs.glm5_next import (
    Glm5NextConfig,
    Glm5NextTextConfig,
    Glm5NextVisionConfig,
)


@pytest.mark.skip_global_cleanup
def test_glm5_next_accepts_deepseek_sparse_attention_layers():
    layer_types = ["linear_attention", "deepseek_sparse_attention"]

    config = Glm5NextTextConfig(
        num_hidden_layers=len(layer_types), layer_types=layer_types
    )

    assert config.layer_types == layer_types
    assert config.layers_block_type == ["linear_attention", "attention"]


@pytest.mark.skip_global_cleanup
def test_glm5_next_accepts_prebuilt_subconfigs():
    text_config = Glm5NextTextConfig(hidden_size=1024)
    vision_config = Glm5NextVisionConfig(hidden_size=768)

    config = Glm5NextConfig(
        text_config=text_config,
        vision_config=vision_config,
    )

    assert config.text_config is text_config
    assert config.vision_config is vision_config


def test_get_llama3_eos_token():
    model_name = "meta-llama/Llama-3.2-1B-Instruct"

    tokenizer = get_tokenizer(model_name)
    assert tokenizer.eos_token_id == 128009

    generation_config = try_get_generation_config(model_name, trust_remote_code=False)
    assert generation_config is not None
    assert generation_config.eos_token_id == [128001, 128008, 128009]


def test_get_blip2_eos_token():
    model_name = "Salesforce/blip2-opt-2.7b"

    tokenizer = get_tokenizer(model_name)
    assert tokenizer.eos_token_id == 2

    generation_config = try_get_generation_config(model_name, trust_remote_code=False)
    assert generation_config is not None
    assert generation_config.eos_token_id == 50118
