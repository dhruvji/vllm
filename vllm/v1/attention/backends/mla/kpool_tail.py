# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import torch

from vllm.config import VllmConfig
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import AttentionSpec


@dataclass
class KpoolTailMetadata:
    slot_mapping: torch.Tensor


class KpoolTailBackend(AttentionBackend):
    """Storage-only backend for the GLM-5.3-Flash kpool tail cache."""

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return num_blocks, num_kv_heads, block_size, head_size

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return 0, 1, 2, 3, 4
        return 0, 1, 2, 3

    @staticmethod
    def get_name() -> str:
        return "KPOOL_TAIL"

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @staticmethod
    def get_builder_cls() -> type["KpoolTailMetadataBuilder"]:
        return KpoolTailMetadataBuilder


def compute_kpool_tail_slot_mapping(
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    num_actual_tokens: int,
    num_reqs: int,
    kpool: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map every token to its request's one circular tail block."""
    if out is None:
        out = slot_mapping.clone()
    else:
        assert out.shape == slot_mapping.shape
        out.copy_(slot_mapping)
    if num_actual_tokens == 0:
        return out
    tokens = torch.arange(num_actual_tokens, device=slot_mapping.device)
    req = torch.searchsorted(query_start_loc, tokens, right=True) - 1
    req = req.clamp_(min=0, max=num_reqs - 1)
    own_block = block_table[:num_reqs, 0].index_select(0, req).to(torch.int64)
    pos = positions[:num_actual_tokens].to(torch.int64)
    out[:num_actual_tokens] = own_block * kpool + torch.remainder(pos, kpool)
    return out


class KpoolTailMetadataBuilder(AttentionMetadataBuilder):
    """Build only the circular slot mapping needed by the storage-only tail."""

    _cudagraph_support = AttentionCGSupport.ALWAYS
    supports_update_block_table = False
    reorder_batch_threshold = None

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.slot_mapping_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int64,
            device=device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> KpoolTailMetadata:
        slot_mapping = common_attn_metadata.slot_mapping
        positions = common_attn_metadata.positions
        if positions is not None:
            slot_mapping_buffer = self.slot_mapping_buffer[
                : slot_mapping.numel()
            ].view_as(slot_mapping)
            slot_mapping = compute_kpool_tail_slot_mapping(
                slot_mapping,
                common_attn_metadata.block_table_tensor,
                common_attn_metadata.query_start_loc,
                positions,
                common_attn_metadata.num_actual_tokens,
                common_attn_metadata.num_reqs,
                self.kv_cache_spec.block_size,
                out=slot_mapping_buffer,
            )
        return KpoolTailMetadata(slot_mapping=slot_mapping)
