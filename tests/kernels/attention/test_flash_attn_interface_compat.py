# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture
def fa_interface():
    module_names = (
        "vllm.vllm_flash_attn",
        "vllm.vllm_flash_attn.flash_attn_interface",
        "vllm.vllm_flash_attn._vllm_fa3_C",
    )
    previous_modules = {name: sys.modules.get(name) for name in module_names}
    for name in module_names:
        sys.modules.pop(name, None)
    sys.modules[module_names[2]] = ModuleType(module_names[2])
    try:
        yield importlib.import_module(module_names[1])
    finally:
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def _install_fake_fa3(
    monkeypatch,
    fa_interface,
    schema: str,
) -> list[tuple[object, ...]]:
    calls = []

    class FakeFwd:
        def __init__(self):
            self.default = SimpleNamespace(_schema=schema)

        def __call__(self, *args):
            calls.append(args)
            return args[0], None, None, None

    monkeypatch.setattr(
        fa_interface.torch.ops,
        "_vllm_fa3_C",
        SimpleNamespace(fwd=FakeFwd()),
        raising=False,
    )
    return calls


def _call_fa3(fa_interface, *, only_qv: bool):
    q = torch.empty(1, 1, 8)
    cu_seqlens = torch.tensor([0, 1], dtype=torch.int32)
    return fa_interface.flash_attn_varlen_func(
        q=q,
        k=q,
        v=q,
        max_seqlen_q=1,
        cu_seqlens_q=cu_seqlens,
        max_seqlen_k=1,
        cu_seqlens_k=cu_seqlens,
        fa_version=3,
        only_qv=only_qv,
    )


def test_fa3_uses_legacy_schema_when_only_qv_is_unavailable(monkeypatch, fa_interface):
    calls = _install_fake_fa3(
        monkeypatch,
        fa_interface,
        "fwd(..., bool? pack_gqa, int sm_margin, Tensor? s_aux) -> Tensor[]",
    )

    _call_fa3(fa_interface, only_qv=False)

    assert len(calls) == 1
    assert len(calls[0]) == 37
    assert calls[0][32] == 0


def test_fa3_rejects_only_qv_with_legacy_schema(monkeypatch, fa_interface):
    calls = _install_fake_fa3(
        monkeypatch,
        fa_interface,
        "fwd(..., bool? pack_gqa, int sm_margin, Tensor? s_aux) -> Tensor[]",
    )

    with pytest.raises(RuntimeError, match="only_qv.*newer.*extension"):
        _call_fa3(fa_interface, only_qv=True)

    assert calls == []


def test_fa3_passes_only_qv_to_new_schema(monkeypatch, fa_interface):
    calls = _install_fake_fa3(
        monkeypatch,
        fa_interface,
        "fwd(..., bool? pack_gqa, bool only_qv, int sm_margin) -> Tensor[]",
    )

    _call_fa3(fa_interface, only_qv=False)

    assert len(calls) == 1
    assert len(calls[0]) == 38
    assert calls[0][32] is False
    assert calls[0][33] == 0
