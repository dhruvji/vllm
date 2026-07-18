# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import threading
from collections import deque
from concurrent.futures import Future

import torch

from vllm.sampling_params import SamplingParams
from vllm.v1 import cartridge_loader
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine.core import EngineCore


def _reset_shm_state() -> None:
    cartridge_loader._active_shm_paths.clear()


def test_save_cartridge_to_shm_rewrites_invalid_existing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cartridge_loader, "SHM_CARTRIDGE_DIR", str(tmp_path / "cartridges")
    )
    _reset_shm_state()

    cartridge_id = "s3://bucket/path/to/cartridge.pt"
    shm_path = cartridge_loader._get_shm_path(cartridge_id)
    tmp_path.joinpath("cartridges").mkdir(parents=True, exist_ok=True)
    with open(shm_path, "wb") as f:
        f.write(b"not a torch archive")

    stacked_kv = [torch.randn(1, 1, 2, 3), torch.randn(1, 1, 2, 3)]

    saved_path = cartridge_loader.save_cartridge_to_shm(cartridge_id, stacked_kv)
    loaded = cartridge_loader.load_cartridge_from_shm(saved_path)

    assert saved_path == shm_path
    assert loaded is not None
    assert len(loaded) == 2
    assert torch.equal(loaded[0], stacked_kv[0])
    assert torch.equal(loaded[1], stacked_kv[1])


def test_save_cartridge_to_shm_overwrites_existing_file_when_requested(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        cartridge_loader, "SHM_CARTRIDGE_DIR", str(tmp_path / "cartridges")
    )
    _reset_shm_state()

    cartridge_id = "s3://bucket/path/to/cartridge.pt"
    stale_kv = [torch.zeros(1, 1, 2, 3), torch.zeros(1, 1, 2, 3)]
    fresh_kv = [torch.ones(1, 1, 2, 3), torch.ones(1, 1, 2, 3)]

    shm_path = cartridge_loader.save_cartridge_to_shm(cartridge_id, stale_kv)
    saved_path = cartridge_loader.save_cartridge_to_shm(
        cartridge_id, fresh_kv, overwrite=True
    )
    loaded = cartridge_loader.load_cartridge_from_shm(saved_path)

    assert saved_path == shm_path
    assert loaded is not None
    assert torch.equal(loaded[0], fresh_kv[0])
    assert torch.equal(loaded[1], fresh_kv[1])


def test_save_cartridge_to_shm_serializes_concurrent_writers(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cartridge_loader, "SHM_CARTRIDGE_DIR", str(tmp_path / "cartridges")
    )
    _reset_shm_state()

    cartridge_id = "s3://bucket/path/to/cartridge.pt"
    stacked_kv = [torch.randn(1, 1, 2, 3), torch.randn(1, 1, 2, 3)]

    original_save = cartridge_loader.torch.save
    save_started = threading.Event()
    allow_finish = threading.Event()
    save_calls = 0
    save_calls_lock = threading.Lock()

    def slow_save(obj, file_obj, *args, **kwargs):
        nonlocal save_calls
        with save_calls_lock:
            save_calls += 1
        save_started.set()
        allow_finish.wait(timeout=5)
        return original_save(obj, file_obj, *args, **kwargs)

    monkeypatch.setattr(cartridge_loader.torch, "save", slow_save)

    results: list[str] = []
    errors: list[Exception] = []

    def worker():
        try:
            results.append(
                cartridge_loader.save_cartridge_to_shm(cartridge_id, stacked_kv)
            )
        except Exception as exc:
            errors.append(exc)

    first = threading.Thread(target=worker)
    second = threading.Thread(target=worker)

    first.start()
    assert save_started.wait(timeout=5)
    second.start()
    allow_finish.set()

    first.join(timeout=5)
    second.join(timeout=5)

    assert not errors
    assert len(results) == 2
    assert results[0] == results[1]
    assert save_calls == 1

    loaded = cartridge_loader.load_cartridge_from_shm(results[0])
    assert loaded is not None
    assert torch.equal(loaded[0], stacked_kv[0])
    assert torch.equal(loaded[1], stacked_kv[1])


def test_load_cartridges_from_request_with_ids_uses_successful_source(monkeypatch):
    cartridge = cartridge_loader.CartridgeData(token_ids=[1, 2, 3])

    def fake_resolve_glob(cartridge_id):
        return {
            "missing": "missing",
            "fallback-*": "fallback-resolved",
        }[cartridge_id]

    def fake_load_cartridge(cartridge_id, force_redownload=False):
        if cartridge_id == "missing":
            raise FileNotFoundError(cartridge_id)
        assert force_redownload is True
        return cartridge

    monkeypatch.setattr(cartridge_loader, "_resolve_glob", fake_resolve_glob)
    monkeypatch.setattr(cartridge_loader, "load_cartridge", fake_load_cartridge)

    loaded = cartridge_loader.load_cartridges_from_request_with_ids(
        [
            {
                "sources": [{"id": "missing"}, {"id": "fallback-*"}],
                "force_redownload": True,
            }
        ]
    )

    assert loaded == [("fallback-resolved", cartridge)]


def test_real_model_layer_indices_stack_in_sorted_attention_order():
    kv_cache = {}
    expected_keys = []
    expected_values = []

    # Qwen 3.5 cartridges use real model layer indices for attention layers,
    # not dense 0..N-1 numbering. Insert them out of order to verify we sort by
    # the artifact's layer index before stacking.
    expected_layer_indices = (3, 7, 11)
    for layer_idx in (11, 3, 7):
        keys = torch.full((1, 2, 4, 5), float(layer_idx))
        values = torch.full((1, 2, 4, 5), float(layer_idx + 100))
        kv_cache[f"layers.{layer_idx}.attention.prefix.keys"] = keys
        kv_cache[f"layers.{layer_idx}.attention.prefix.values"] = values

    for layer_idx in expected_layer_indices:
        expected_keys.append(
            kv_cache[f"layers.{layer_idx}.attention.prefix.keys"]
            .squeeze(0)
            .permute(1, 0, 2)
        )
        expected_values.append(
            kv_cache[f"layers.{layer_idx}.attention.prefix.values"]
            .squeeze(0)
            .permute(1, 0, 2)
        )

    cartridge = cartridge_loader.CartridgeData(kv_cache=kv_cache, is_learned=True)
    stacked_keys, stacked_values = cartridge.get_stacked_kv()

    assert torch.equal(stacked_keys, torch.stack(expected_keys))
    assert torch.equal(stacked_values, torch.stack(expected_values))


def test_prepare_prefix_cartridge_request_builds_engine_fields(monkeypatch):
    kv_cache = {
        "layers.0.attention.prefix.keys": torch.zeros(1, 2, 4, 5),
        "layers.0.attention.prefix.values": torch.ones(1, 2, 4, 5),
    }
    cartridge = cartridge_loader.CartridgeData(kv_cache=kv_cache, is_learned=True)
    prefix_specs = [
        {
            "sources": [{"id": "missing"}, {"id": "fallback-*"}],
            "force_redownload": True,
            "allow_pipelining": True,
        }
    ]
    request = type(
        "Request",
        (),
        {"adapters": {"prefix": prefix_specs}, "truncate_prompt_tokens": None},
    )()
    saved = {}

    def fake_load_cartridges(cartridges_spec):
        assert cartridges_spec is prefix_specs
        return [("fallback-resolved", cartridge)]

    def fake_save_cartridge(cartridge_id, stacked_kv, overwrite=False):
        saved["cartridge_id"] = cartridge_id
        saved["stacked_kv"] = stacked_kv
        saved["overwrite"] = overwrite
        return "/tmp/fallback-resolved.pt"

    monkeypatch.setattr(
        cartridge_loader,
        "load_cartridges_from_request_with_ids",
        fake_load_cartridges,
    )
    monkeypatch.setattr(cartridge_loader, "save_cartridge_to_shm", fake_save_cartridge)

    params = SamplingParams(max_tokens=10)
    prepared = cartridge_loader.prepare_prefix_cartridge_request(
        request,
        prompt_len=5,
        max_model_len=8,
        expected_kv_heads=4,
        expected_head_size=5,
        sampling_params=params,
        truncate_prompt_tokens=None,
    )

    assert prepared is not None
    assert prepared.cartridge_id == "fallback-resolved"
    assert prepared.cartridge_seq_len == 2
    assert prepared.cartridge_shm_path == "/tmp/fallback-resolved.pt"
    assert prepared.cartridge_allow_pipelining is True
    assert prepared.sampling_params is not params
    assert prepared.sampling_params.max_tokens == 1
    assert saved["cartridge_id"] == "fallback-resolved"
    assert saved["overwrite"] is True
    assert saved["stacked_kv"] is cartridge.get_stacked_kv()


def test_prepare_prefix_cartridge_request_returns_none_without_prefix():
    request = type("Request", (), {"adapters": {"lora": []}})()

    prepared = cartridge_loader.prepare_prefix_cartridge_request(
        request,
        prompt_len=5,
        max_model_len=8,
        expected_kv_heads=4,
        expected_head_size=5,
        sampling_params=SamplingParams(max_tokens=1),
        truncate_prompt_tokens=None,
    )

    assert prepared is None


def test_engine_update_cartridge_tracks_shm_for_cleanup(monkeypatch):
    engine = EngineCore.__new__(EngineCore)
    engine._pending_cartridge_overwrites = {}
    engine._cartridge_kv_cache = {}
    engine._cartridge_shm_paths = {}
    engine._cartridge_shm_ids = set()
    stacked_kv = [torch.randn(1, 1, 2, 3), torch.randn(1, 1, 2, 3)]

    monkeypatch.setattr(
        "vllm.v1.engine.core.load_cartridge_from_shm",
        lambda path: stacked_kv,
    )

    result = engine.update_cartridge("cart-1", "/tmp/cart-1.pt")

    assert result == "queued"
    assert engine._pending_cartridge_overwrites["cart-1"] is stacked_kv
    assert engine._cartridge_kv_cache["cart-1"] is stacked_kv
    assert engine._cartridge_shm_paths["cart-1"] == "/tmp/cart-1.pt"
    assert "cart-1" in engine._cartridge_shm_ids


def test_engine_cleanup_covers_lru_evicted_shm_paths(monkeypatch):
    cleaned = []
    engine = EngineCore.__new__(EngineCore)
    engine._cartridge_shm_ids = {"cached-cart"}
    engine._cartridge_shm_paths = {
        "cached-cart": "/tmp/cached-cart.pt",
        "evicted-cart": "/tmp/evicted-cart.pt",
    }
    engine._cartridge_kv_cache = {"cached-cart": object()}

    monkeypatch.setattr(
        "vllm.v1.engine.core.cleanup_cartridge_shm",
        lambda cartridge_id: cleaned.append(cartridge_id),
    )

    engine._cleanup_cartridge_shm()

    assert set(cleaned) == {"cached-cart", "evicted-cart"}
    assert engine._cartridge_shm_ids == set()
    assert engine._cartridge_shm_paths == {}
    assert engine._cartridge_kv_cache == {}


def test_batch_queue_attaches_pending_cartridge_overwrites():
    scheduler_output = SchedulerOutput.make_empty()
    executed_outputs = []
    stacked_kv = [torch.randn(1, 1, 1, 1), torch.randn(1, 1, 1, 1)]
    engine = EngineCore.__new__(EngineCore)
    engine.batch_queue = deque(maxlen=2)
    engine.batch_queue_size = 2
    engine.is_ec_consumer = True
    engine.is_pooling_model = False
    engine.use_spec_decode = False
    engine._pending_cartridge_overwrites = {"cart-1": stacked_kv}

    class Scheduler:
        def has_requests(self):
            return True

        def schedule(self, defer_prefills=False):
            return scheduler_output

        def update_from_output(self, scheduler_output, model_output):
            return {}

    class ModelExecutor:
        def execute_model(self, scheduler_output, non_block=False):
            executed_outputs.append(scheduler_output)
            future: Future[object] = Future()
            future.set_result(object())
            return future

    engine.scheduler = Scheduler()
    engine.model_executor = ModelExecutor()
    engine.log_error_detail = lambda scheduler_output: contextlib.nullcontext()
    engine.log_iteration_details = lambda scheduler_output: contextlib.nullcontext()
    engine._process_aborts_queue = lambda: None

    outputs, model_executed = engine.step_with_batch_queue()

    assert outputs is None
    assert model_executed is False
    assert executed_outputs == [scheduler_output]
    assert scheduler_output.cartridge_overwrites == {"cart-1": stacked_kv}
    assert engine._pending_cartridge_overwrites == {}
