# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backend import AttentionMetadataBuilder
from vllm.v1.attention.backends.mla.indexer import KpoolTailBackend
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KpoolTailSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    KVQuantMode,
    MambaSpec,
    MLAAttentionSpec,
)
from vllm.v1.worker.gpu.attn_utils import (
    _get_kv_cache_num_blocks,
    _reshape_attention_kv_cache,
    _reshape_kv_cache,
    _reshape_mamba_kv_cache,
)
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.utils import AttentionGroup


class FakeFlashAttentionBackend:
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        assert not include_num_layers_dimension
        return (0, 1, 2, 3, 4)


@pytest.mark.skip_global_cleanup
def test_reshape_packed_attention_cache_from_absolute_storage_offset():
    spec = FullAttentionSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.uint8,
    )
    raw_tensor = torch.zeros(13, dtype=torch.uint8)

    kv_cache = _reshape_attention_kv_cache(
        raw_tensor,
        spec,
        kv_cache_shape=(4, 2),
        kv_cache_stride_order=(0, 1),
        num_blocks=4,
        packing=(5, 4),
        blocks_per_page=2,
    )

    assert kv_cache.shape == (4, 2)
    assert kv_cache.stride() == (2, 1)
    assert kv_cache.storage_offset() == 5
    kv_cache[0].fill_(1)
    kv_cache[3].fill_(2)
    assert raw_tensor.tolist() == [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 2, 2]


@pytest.mark.skip_global_cleanup
def test_reshape_packed_mamba_cache_from_absolute_storage_offset():
    spec = MambaSpec(
        block_size=1,
        shapes=((2,),),
        dtypes=(torch.uint8,),
    )
    raw_tensor = torch.zeros(9, dtype=torch.uint8)
    packing = (5, 2)
    num_blocks = _get_kv_cache_num_blocks(raw_tensor, spec, packing, 2)

    state = _reshape_mamba_kv_cache(raw_tensor, spec, num_blocks, packing)[0]

    assert state.shape == (2, 2)
    assert state.stride() == (2, 1)
    assert state.storage_offset() == 5
    state[0].fill_(1)
    state[1].fill_(2)
    assert raw_tensor.tolist() == [0, 0, 0, 0, 0, 1, 1, 2, 2]


class FakeIndexerBackend:
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return num_blocks, block_size, head_size

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        assert not include_num_layers_dimension
        return 0, 1, 2

    @staticmethod
    def get_kv_cache_block_dim(*args, **kwargs) -> int:
        return 0


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize(
    ("metadata_block_size", "expected_shape"),
    [
        (128, (34, 32, 132)),
        (None, (68, 16, 132)),
    ],
)
def test_reshape_packed_compressed_indexer_uses_kernel_state_size(
    monkeypatch,
    metadata_block_size: int | None,
    expected_shape: tuple[int, ...],
):
    spec = MLAAttentionSpec(
        block_size=2176,
        num_kv_heads=1,
        head_size=132,
        dtype=torch.uint8,
        tokens_per_state=4,
        metadata_block_size=metadata_block_size,
    )
    num_blocks = 2
    offset = 5
    raw_tensor = torch.zeros(
        offset + num_blocks * spec.page_size_bytes,
        dtype=torch.uint8,
    )
    group = AttentionGroup(FakeIndexerBackend, ["indexer"], spec, 0)
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=raw_tensor.numel(),
                shared_by=["indexer"],
                offset=offset,
                block_stride=spec.page_size_bytes,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(["indexer"], spec)],
    )

    kv_cache = _reshape_kv_cache(
        [group],
        {"indexer": raw_tensor},
        "auto",
        [64],
        {},
        kv_cache_config,
    )["indexer"]

    assert kv_cache.shape == expected_shape
    assert kv_cache.stride() == (expected_shape[1] * 132, 132, 1)
    assert kv_cache.storage_offset() == offset

    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.kv_cache_config = kv_cache_config
    runner.attn_groups = [[group]]
    runner.runner_only_attn_layers = set()
    runner.cache_config = SimpleNamespace(cache_dtype="auto")  # type: ignore[assignment]
    monkeypatch.setattr(
        GPUModelRunner,
        "_has_mixed_attention_kv_layout",
        lambda *args: False,
    )

    runner_kv_cache = runner._reshape_kv_cache_tensors({"indexer": raw_tensor}, [64])[
        "indexer"
    ]

    assert runner_kv_cache.shape == kv_cache.shape
    assert runner_kv_cache.stride() == kv_cache.stride()
    assert runner_kv_cache.storage_offset() == offset


@pytest.mark.skip_global_cleanup
def test_reshape_packed_kpool_tail_uses_storage_backend_shape(monkeypatch):
    assert KpoolTailBackend.get_kv_cache_stride_order() == (0, 1, 2, 3)
    assert KpoolTailBackend.get_kv_cache_stride_order(True) == (0, 1, 2, 3, 4)
    assert KpoolTailBackend.indexes_kv_by_block_stride() is False

    spec = KpoolTailSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=3,
        head_size_v=0,
        dtype=torch.bfloat16,
        sliding_window=16,
    )
    num_blocks = 2
    offset = 8
    raw_tensor = torch.zeros(
        offset + num_blocks * spec.page_size_bytes,
        dtype=torch.uint8,
    )
    group = AttentionGroup(KpoolTailBackend, ["tail"], spec, 0)
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=raw_tensor.numel(),
                shared_by=["tail"],
                offset=offset,
                block_stride=spec.page_size_bytes,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(["tail"], spec)],
    )

    kv_cache = _reshape_kv_cache(
        [group],
        {"tail": raw_tensor},
        "auto",
        [spec.block_size],
        {},
        kv_cache_config,
    )["tail"]

    assert kv_cache.shape == (num_blocks, 2, 16, 3)
    assert kv_cache.stride() == (96, 48, 3, 1)
    assert kv_cache.storage_offset() == offset // 2

    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.kv_cache_config = kv_cache_config
    runner.attn_groups = [[group]]
    runner.runner_only_attn_layers = set()
    runner.cache_config = SimpleNamespace(cache_dtype="auto")  # type: ignore[assignment]
    monkeypatch.setattr(
        GPUModelRunner,
        "_has_mixed_attention_kv_layout",
        lambda *args: False,
    )

    runner_kv_cache = runner._reshape_kv_cache_tensors({"tail": raw_tensor}, [16])[
        "tail"
    ]

    assert runner_kv_cache.shape == kv_cache.shape
    assert runner_kv_cache.stride() == kv_cache.stride()
    assert runner_kv_cache.storage_offset() == offset // 2


class WidthAwareMetadataBuilder(AttentionMetadataBuilder):
    requires_block_table_width = True

    def __init__(
        self,
        kv_cache_spec,
        layer_names,
        vllm_config,
        device,
        *,
        block_table_width: int,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_table_width = block_table_width

    def build(self, *args, **kwargs):
        raise NotImplementedError


class WidthAwareBackend:
    @staticmethod
    def get_builder_cls():
        return WidthAwareMetadataBuilder


@pytest.mark.skip_global_cleanup
def test_metadata_builder_block_table_width_contract():
    assert AttentionMetadataBuilder.requires_block_table_width is False
    spec = FullAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
    )
    group = AttentionGroup(WidthAwareBackend, ["layer"], spec, 0)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=1024),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
    )

    group.create_metadata_builders(
        vllm_config,
        torch.device("cpu"),
        kernel_block_size=32,
    )

    assert group.metadata_builders[0].block_table_width == 32


@pytest.mark.skip_global_cleanup
def test_compressed_mla_metadata_uses_explicit_block_size():
    spec = MLAAttentionSpec(
        block_size=2176,
        num_kv_heads=1,
        head_size=132,
        dtype=torch.uint8,
        tokens_per_state=4,
        metadata_block_size=128,
    )
    group = AttentionGroup(WidthAwareBackend, ["indexer"], spec, 0)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=4096),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
    )

    group.create_metadata_builders(
        vllm_config,
        torch.device("cpu"),
        kernel_block_size=64,
    )

    builder = group.metadata_builders[0]
    assert builder.kv_cache_spec.block_size == 128
    assert builder.kv_cache_spec.storage_block_size == 32
    assert builder.kernel_block_size == 64


class FakeHNDFlashAttentionBackend(FakeFlashAttentionBackend):
    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        assert not include_num_layers_dimension
        return (0, 1, 3, 2, 4)


def test_reshape_padded_flash_attention_kv_cache_strides_by_page():
    num_blocks = 3
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
        page_size_padded=384,
    )
    assert spec.real_page_size_bytes == 256

    raw_tensors = {
        "layer": torch.zeros(spec.page_size_bytes * num_blocks, dtype=torch.int8)
    }
    attn_groups = [
        AttentionGroup(
            backend=FakeFlashAttentionBackend,
            layer_names=["layer"],
            kv_cache_spec=spec,
            kv_cache_group_id=0,
        )
    ]

    kv_cache = _reshape_kv_cache(
        attn_groups,
        raw_tensors,
        "auto",
        [spec.block_size],
        {},
    )["layer"]

    assert kv_cache.shape == (num_blocks, 2, 16, 1, 2)
    assert kv_cache.stride(0) == spec.page_size_bytes // 4
    assert kv_cache.stride(1) == spec.real_page_size_bytes // 2 // 4
    assert kv_cache[1, 0].storage_offset() == spec.page_size_bytes // 4
    assert (
        kv_cache[1, 1].storage_offset()
        == (spec.page_size_bytes + spec.real_page_size_bytes // 2) // 4
    )


def test_reshape_padded_hnd_flash_attention_kv_cache_strides_by_page():
    num_blocks = 3
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=3,
        head_size=2,
        dtype=torch.float32,
        page_size_padded=1024,
    )
    assert spec.real_page_size_bytes == 768

    raw_tensors = {
        "layer": torch.zeros(spec.page_size_bytes * num_blocks, dtype=torch.int8)
    }
    attn_groups = [
        AttentionGroup(
            backend=FakeHNDFlashAttentionBackend,
            layer_names=["layer"],
            kv_cache_spec=spec,
            kv_cache_group_id=0,
        )
    ]

    kv_cache = _reshape_kv_cache(
        attn_groups,
        raw_tensors,
        "auto",
        [spec.block_size],
        {},
    )["layer"]

    assert kv_cache.shape == (num_blocks, 2, 16, 3, 2)
    assert kv_cache.stride(0) == spec.page_size_bytes // 4
    assert kv_cache.stride(1) == spec.real_page_size_bytes // 2 // 4
    assert kv_cache.stride(2) == 2
    assert kv_cache.stride(3) == spec.block_size * spec.head_size
    assert kv_cache[1, 0].storage_offset() == spec.page_size_bytes // 4
    assert (
        kv_cache[1, 1].storage_offset()
        == (spec.page_size_bytes + spec.real_page_size_bytes // 2) // 4
    )
    assert (
        kv_cache[1, 1, 3, 2].storage_offset()
        == (
            spec.page_size_bytes
            + spec.real_page_size_bytes // 2
            + 3 * spec.head_size * 4
            + 2 * spec.block_size * spec.head_size * 4
        )
        // 4
    )


class FakeDiffKVBackend:
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, num_kv_heads, head_size * 2)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        assert not include_num_layers_dimension
        return (0, 1, 2, 3)


def test_reshape_padded_diff_kv_cache_does_not_infer_kv_dim():
    num_blocks = 3
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
        page_size_padded=384,
    )

    raw_tensors = {
        "layer": torch.zeros(spec.page_size_bytes * num_blocks, dtype=torch.int8)
    }
    attn_groups = [
        AttentionGroup(
            backend=FakeDiffKVBackend,
            layer_names=["layer"],
            kv_cache_spec=spec,
            kv_cache_group_id=0,
        )
    ]

    kv_cache = _reshape_kv_cache(
        attn_groups,
        raw_tensors,
        "auto",
        [spec.block_size],
        {},
    )["layer"]

    assert kv_cache.shape == (num_blocks, 16, 1, 4)
    assert kv_cache.stride(0) == spec.page_size_bytes // 4
    assert kv_cache.stride(1) == 4


class FakePerTokenScaleBackend:
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, 2, block_size, num_kv_heads, head_size + 4)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        assert not include_num_layers_dimension
        return (0, 1, 2, 3, 4)


def test_reshape_padded_quantized_kv_cache_preserves_scale_stride():
    num_blocks = 3
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.int8,
        kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
        page_size_padded=384,
    )
    assert spec.real_page_size_bytes == 128
    assert spec.page_size_bytes == 384

    raw_tensors = {
        "layer": torch.zeros(spec.page_size_bytes * num_blocks, dtype=torch.int8)
    }
    attn_groups = [
        AttentionGroup(
            backend=FakePerTokenScaleBackend,
            layer_names=["layer"],
            kv_cache_spec=spec,
            kv_cache_group_id=0,
        )
    ]

    kv_cache = _reshape_kv_cache(
        attn_groups,
        raw_tensors,
        "int8_per_token_head",
        [spec.block_size],
        {},
    )["layer"]

    assert kv_cache.shape == (num_blocks, 2, 16, 1, 8)
    assert kv_cache.stride(0) == spec.page_size_bytes
    assert kv_cache.stride(1) == 16 * 1 * 8
    assert kv_cache[1, 1].storage_offset() == spec.page_size_bytes + 16 * 1 * 8
