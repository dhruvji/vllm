# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)


def _custom_collective(name: str, x: torch.Tensor) -> torch.Tensor | None:
    communicator = get_tp_group().device_communicator
    if communicator is None:
        return None
    collective = getattr(communicator, name, None)
    return None if collective is None else collective(x)


def sp_all_gather(x: torch.Tensor) -> torch.Tensor:
    output = _custom_collective("custom_all_gather", x)
    return output if output is not None else tensor_model_parallel_all_gather(x, 0)


def sp_reduce_scatter(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2
    tp_size = get_tensor_model_parallel_world_size()
    padding = (-x.shape[0]) % tp_size
    if padding:
        x = torch.nn.functional.pad(x, (0, 0, 0, padding))
    output = _custom_collective("custom_reduce_scatter", x)
    return output if output is not None else tensor_model_parallel_reduce_scatter(x, 0)


def sp_shard(x: torch.Tensor) -> torch.Tensor:
    tp_size = get_tensor_model_parallel_world_size()
    padding = (-x.shape[0]) % tp_size
    if padding:
        x = torch.nn.functional.pad(x, (0, 0) * (x.ndim - 1) + (0, padding))
    chunk_size = x.shape[0] // tp_size
    tp_rank = get_tensor_model_parallel_rank()
    return x[tp_rank * chunk_size : (tp_rank + 1) * chunk_size]
