# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, replace
from typing import cast

from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import (
    KpoolTailSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)


@dataclass(frozen=True)
class Glm5NextKVCacheLayout:
    attn_group: KVCacheGroupSpec
    mamba_groups: tuple[KVCacheGroupSpec, ...]
    mla_names: tuple[str, ...]
    indexer_names: tuple[str, ...]
    mla_page_size: int
    indexer_page_size: int
    tail_group: KVCacheGroupSpec | None
    indexer_tail_pairs: tuple[tuple[str, str], ...]
    tail_page_size: int

    @property
    def bytes_per_block(self) -> int:
        return (
            len(self.mla_names) * self.mla_page_size
            + len(self.indexer_names) * self.indexer_page_size
        )


def _cache_owner(layer_name: str) -> str:
    return layer_name.rsplit(".", 1)[0]


def _pair_indexers_and_tails(
    indexer_names: list[str], tail_names: list[str]
) -> tuple[tuple[str, str], ...] | None:
    tails_by_owner = {_cache_owner(name): name for name in tail_names}
    if len(tails_by_owner) != len(tail_names):
        return None
    pairs = tuple(
        (indexer_name, tails_by_owner.get(_cache_owner(indexer_name), ""))
        for indexer_name in indexer_names
    )
    if len(pairs) != len(tail_names) or any(not tail for _, tail in pairs):
        return None
    return pairs


def _create_groups(
    kv_cache_spec: dict[str, KVCacheSpec], grouped_layer_names: list[list[str]]
) -> list[KVCacheGroupSpec]:
    groups = []
    for layer_names in grouped_layer_names:
        specs = [kv_cache_spec[layer_name] for layer_name in layer_names]
        groups.append(KVCacheGroupSpec(layer_names, specs[0].merge(specs)))
    return groups


def _pp_balanced_mamba_group_count(
    vllm_config: VllmConfig,
    mamba_layer_names: list[str],
    mla_layer_names: list[str],
) -> int | None:
    num_groups = cdiv(len(mamba_layer_names), len(mla_layer_names))
    pp_size = vllm_config.parallel_config.pipeline_parallel_size
    if pp_size == 1:
        return num_groups

    from vllm.distributed.utils import get_pp_indices
    from vllm.model_executor.models.utils import extract_layer_index

    total_layers = vllm_config.model_config.get_total_num_hidden_layers()
    mamba_indices = [extract_layer_index(name) for name in mamba_layer_names]
    mla_indices = [extract_layer_index(name) for name in mla_layer_names]
    for rank in range(pp_size):
        start, end = get_pp_indices(total_layers, rank, pp_size)
        num_mamba = sum(start <= index < end for index in mamba_indices)
        num_mla = sum(start <= index < end for index in mla_indices)
        if not num_mamba:
            continue
        if not num_mla:
            return None
        num_groups = max(num_groups, cdiv(num_mamba, num_mla))
    return num_groups


def get_glm5_next_kv_cache_groups(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec] | None:
    """Build GLM-5.3-Flash groups with Mamba/MLA and tail/indexer aliasing."""
    mamba_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if isinstance(spec, MambaSpec)
    }
    tail_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if isinstance(spec, KpoolTailSpec)
    }
    attn_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if not isinstance(spec, (MambaSpec, KpoolTailSpec))
    }
    if not mamba_specs or not all(
        type(spec) is MLAAttentionSpec for spec in attn_specs.values()
    ):
        return None

    mla_specs = cast(dict[str, MLAAttentionSpec], attn_specs)
    indexer_names = [
        name for name, spec in mla_specs.items() if spec.tokens_per_state > 1
    ]
    indexer_pages = {mla_specs[name].page_size_bytes for name in indexer_names}
    assert all(spec.page_size_padded is None for spec in mla_specs.values())
    assert len(indexer_pages) <= 1
    mla_names = [name for name, spec in mla_specs.items() if spec.tokens_per_state == 1]
    mla_pages = {mla_specs[name].page_size_bytes for name in mla_names}
    assert len(mla_pages) == 1
    mla_page = mla_pages.pop()
    uniform_spec = UniformTypeKVCacheSpecs.from_specs(attn_specs)
    assert uniform_spec is not None

    tail_group: KVCacheGroupSpec | None = None
    if tail_specs:
        pairs = _pair_indexers_and_tails(indexer_names, list(tail_specs))
        if not indexer_pages or pairs is None:
            raise ValueError("each kpool tail cache must have one sibling indexer")
        indexer_page = next(iter(indexer_pages))
        padded_tail_specs: dict[str, KVCacheSpec] = {
            name: replace(spec, page_size_padded=indexer_page)
            for name, spec in tail_specs.items()
        }
        tail_uniform = UniformTypeKVCacheSpecs.from_specs(padded_tail_specs)
        assert tail_uniform is not None
        tail_group = KVCacheGroupSpec(list(padded_tail_specs), tail_uniform)

    any_mamba = next(iter(mamba_specs.values()))
    assert all(spec == any_mamba for spec in mamba_specs.values())
    if any_mamba.real_page_size_bytes > mla_page:
        if not indexer_pages:
            return None
        raise ValueError(
            f"the mamba state page ({any_mamba.real_page_size_bytes} bytes) "
            f"does not fit the MLA page ({mla_page} bytes); increase tensor "
            "parallelism or use a wider KV cache dtype"
        )
    padded_specs: dict[str, KVCacheSpec] = {
        name: replace(any_mamba, page_size_padded=mla_page) for name in mamba_specs
    }
    num_groups = _pp_balanced_mamba_group_count(
        vllm_config, list(mamba_specs), mla_names
    )
    if num_groups is None:
        raise ValueError(
            "a pipeline stage has mamba layers but no MLA layer to share "
            "slots with; realign the stage boundaries (VLLM_PP_LAYER_PARTITION)"
        )
    mamba_grouped_names: list[list[str]] = [[] for _ in range(num_groups)]
    for index, name in enumerate(mamba_specs):
        mamba_grouped_names[index % num_groups].append(name)

    return (
        [KVCacheGroupSpec(list(attn_specs), uniform_spec)]
        + ([tail_group] if tail_group is not None else [])
        + _create_groups(padded_specs, mamba_grouped_names)
    )


def get_glm5_next_tensor_layout(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> Glm5NextKVCacheLayout | None:
    """Recognize the GLM-5.3-Flash grouping after optional PP projection."""
    uniform_groups = [
        group
        for group in kv_cache_groups
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
    ]
    mamba_groups = tuple(
        group for group in kv_cache_groups if isinstance(group.kv_cache_spec, MambaSpec)
    )
    attn_group: KVCacheGroupSpec | None = None
    tail_group: KVCacheGroupSpec | None = None
    for group in uniform_groups:
        inner = cast(UniformTypeKVCacheSpecs, group.kv_cache_spec).kv_cache_specs
        if all(type(spec) is MLAAttentionSpec for spec in inner.values()):
            attn_group = group
        elif all(isinstance(spec, KpoolTailSpec) for spec in inner.values()):
            tail_group = group
    if attn_group is None or not mamba_groups:
        return None
    if len(uniform_groups) + len(mamba_groups) != len(kv_cache_groups):
        return None

    attn_uniform = cast(UniformTypeKVCacheSpecs, attn_group.kv_cache_spec)
    mla_inner = cast(dict[str, MLAAttentionSpec], attn_uniform.kv_cache_specs)
    if not all(
        type(spec) is MLAAttentionSpec and spec.page_size_padded is None
        for spec in mla_inner.values()
    ):
        return None
    mla_names = tuple(
        name for name in attn_group.layer_names if mla_inner[name].tokens_per_state == 1
    )
    indexer_names = tuple(
        name for name in attn_group.layer_names if mla_inner[name].tokens_per_state > 1
    )
    mla_pages = {mla_inner[name].page_size_bytes for name in mla_names}
    indexer_pages = {mla_inner[name].page_size_bytes for name in indexer_names}
    if len(mla_pages) != 1 or len(indexer_pages) > 1:
        return None
    mla_page = mla_pages.pop()
    indexer_page = indexer_pages.pop() if indexer_pages else 0
    if any(group.kv_cache_spec.page_size_bytes != mla_page for group in mamba_groups):
        return None

    tail_page = 0
    pairs: tuple[tuple[str, str], ...] = ()
    if tail_group is not None:
        tail_inner = cast(
            UniformTypeKVCacheSpecs, tail_group.kv_cache_spec
        ).kv_cache_specs
        tail_pages = {
            cast(KpoolTailSpec, spec).unpadded_page_size_bytes
            for spec in tail_inner.values()
        }
        pairs = (
            _pair_indexers_and_tails(list(indexer_names), list(tail_group.layer_names))
            or ()
        )
        if len(tail_pages) != 1 or len(pairs) != len(indexer_names):
            return None
        tail_page = tail_pages.pop()
        if tail_page > indexer_page:
            return None

    return Glm5NextKVCacheLayout(
        attn_group=attn_group,
        mamba_groups=mamba_groups,
        mla_names=mla_names,
        indexer_names=indexer_names,
        mla_page_size=mla_page,
        indexer_page_size=indexer_page,
        tail_group=tail_group,
        indexer_tail_pairs=pairs,
        tail_page_size=tail_page,
    )
