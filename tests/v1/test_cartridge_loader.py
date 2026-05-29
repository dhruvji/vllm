import threading

import torch

from vllm.v1 import cartridge_loader
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
