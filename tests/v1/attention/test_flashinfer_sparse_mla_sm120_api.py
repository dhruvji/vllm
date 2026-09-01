# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavior checks for FlashInfer SM120 sparse MLA backend selection."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import set_current_vllm_config
from vllm.platforms.interface import DeviceCapability
from vllm.utils import flashinfer as fi_utils
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseSM120Backend,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum

pytestmark = pytest.mark.skip_global_cleanup


def _fake_vllm_config(model_type: str, index_kpool: int = 4) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                model_type=model_type,
                index_topk=2048,
                index_kpool=index_kpool,
            ),
        ),
    )


def test_sm120_backend_uses_dedicated_backend_name() -> None:
    assert FlashInferMLASparseSM120Backend.get_name() == "FLASHINFER_MLA_SPARSE_SM120"
    assert (
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120.get_class()
        is FlashInferMLASparseSM120Backend
    )


def test_v32_glm_sm120_backend_accepts_glm_block_size(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)

    with set_current_vllm_config(_fake_vllm_config("glm4_moe")):
        invalid_reasons = FlashInferMLASparseSM120Backend.validate_configuration(
            head_size=576,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8",
            block_size=256,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(12, 0),
            attn_type="decoder",
        )

    assert invalid_reasons == []


def test_sm120_compacts_kpool_tail_to_kernel_topk() -> None:
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import (
        _compact_topk_indices,
    )

    topk_indices = torch.tensor(
        [
            [10, 11, 12, 13, 20, 21, -1, -1],
            [30, 31, 32, 33, -1, -1, -1, -1],
            [40, 41, -1, -1, 50, 51, -1, -1],
            [60, -1, -1, -1, 70, -1, -1, -1],
            [-1, -1, -1, -1, 80, 81, -1, -1],
        ],
        dtype=torch.int32,
    )

    compacted = _compact_topk_indices(topk_indices, logical_topk=4)

    assert compacted.tolist() == [
        [10, 11, 20, 21],
        [30, 31, 32, 33],
        [40, 41, 50, 51],
        [60, 70, -1, -1],
        [80, 81, -1, -1],
    ]
