# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker.utils import AttentionGroup, KVBlockZeroer


class BlocksFirstBackend:
    @staticmethod
    def get_kv_cache_block_dim(*args, **kwargs) -> int:
        return 0


@pytest.mark.skip_global_cleanup
def test_non_uniform_page_sizes_have_per_segment_zeroing_metadata():
    device = torch.device("cpu")
    layers = {
        "narrow": torch.ones((4, 4), dtype=torch.int32),
        "wide": torch.ones((4, 6), dtype=torch.int32),
    }
    spec = FullAttentionSpec(
        block_size=1,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.int32,
    )

    zeroer = KVBlockZeroer(
        device,
        pin_memory=False,
        attn_groups_iter=[
            AttentionGroup(BlocksFirstBackend, [layer_name], spec, group_id)
            for group_id, layer_name in enumerate(layers)
        ],
        kernel_block_sizes=[1, 1],
        cache_dtype="auto",
        static_forward_context={
            layer_name: SimpleNamespace(kv_cache=cache)
            for layer_name, cache in layers.items()
        },
    )

    assert zeroer._meta is not None
    _, block_strides, page_sizes, max_chunks, block_size, num_segments = zeroer._meta
    assert block_strides.tolist() == [4, 6]
    assert page_sizes.tolist() == [4, 6]
    assert (max_chunks, block_size, num_segments) == (1, 8, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_block_ids_are_not_overwritten_while_copy_is_in_flight():
    device = torch.device("cuda")
    num_blocks = 4
    page_size_el = 4
    storage = torch.ones((num_blocks, page_size_el), dtype=torch.int32, device=device)

    # Build the minimal zeroer state directly so the test can focus on ID-buffer
    # lifetime without constructing model attention groups.
    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device
    zeroer.pin_memory = True
    zeroer.max_concurrency = 2
    zeroer._id_cap = 8
    zeroer._allocate_id_buffers()
    zeroer._meta = (
        torch.tensor([storage.data_ptr()], dtype=torch.uint64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        1,
        page_size_el,
        1,
    )

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # Keep the first nonblocking H2D copy pending while the host submits the
        # second call. A single shared pinned source would be overwritten here.
        torch.cuda._sleep(10_000_000)
        zeroer.zero_block_ids([1])
        zeroer.zero_block_ids([2])
    stream.synchronize()

    assert torch.all(storage[0] == 1)
    assert torch.all(storage[1] == 0)
    assert torch.all(storage[2] == 0)
    assert torch.all(storage[3] == 1)
