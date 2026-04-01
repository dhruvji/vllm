import threading

import torch

from vllm.v1 import cartridge_loader


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
