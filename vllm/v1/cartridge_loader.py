# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cartridge loader for loading pre-computed KV cache data.

Cartridges support two use cases:

1. **Pre-computed KV Cache** (requires token_ids):
   - KV cache computed from specific tokens (e.g., system prompt)
   - token_ids are prepended to prompt for prefix caching
   - The KV cache tensors are optional (vLLM will recompute from tokens)

2. **Learned KV Cache / Soft Prompts** (no token_ids):
   - KV cache is trained via gradient descent (like prompt tuning)
   - No corresponding tokens exist - the KV tensors ARE the representation
   - Requires direct KV injection into attention (not token prepending)
"""

import contextlib
import fcntl
import glob as globlib
import hashlib
import json
import os
import threading
import time
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import boto3
import regex as re
import torch

from vllm.logger import init_logger
from vllm.path import Path
from vllm.utils.adapter_manager import get_adapter_manager

logger = init_logger(__name__)


def _vllm_cache_root() -> str:
    cache_home = os.environ.get("XDG_CACHE_HOME")
    if cache_home:
        return os.path.join(cache_home, "vllm")
    return "/root/.cache/vllm"


# Directory for cartridge IPC files. Use the vLLM cache root instead of /dev/shm
# for Modal compatibility; Modal volumes are shared across processes, /dev/shm is
# not. BaseTen sets XDG_CACHE_HOME to a writable runtime cache.
SHM_CARTRIDGE_DIR = os.path.join(_vllm_cache_root(), "cartridges")

# Directory for checkpoint resolution cache. This is shared process-local state
# for caching latest.txt timestamps across containers.
CHECKPOINT_CACHE_DIR = os.path.join(_vllm_cache_root(), "checkpoint_cache")

# How often to check for new checkpoints (seconds)
CHECKPOINT_CHECK_INTERVAL = 30

# Track active shm paths for cleanup
_active_shm_paths: dict[str, str] = {}  # cartridge_id -> shm_path


def _get_shm_path(cartridge_id: str) -> str:
    """Get the shared memory path for a cartridge ID."""
    # Use SHA256 hash to avoid path issues with special characters
    id_hash = hashlib.sha256(cartridge_id.encode()).hexdigest()[:16]
    return os.path.join(SHM_CARTRIDGE_DIR, f"cart_{id_hash}.pt")


def _get_shm_lock_path(cartridge_id: str) -> str:
    """Get a process-shared lock path for cartridge shm writes."""
    id_hash = hashlib.sha256(cartridge_id.encode()).hexdigest()[:16]
    return os.path.join(SHM_CARTRIDGE_DIR, f"cart_{id_hash}.lock")


def save_cartridge_to_shm(
    cartridge_id: str,
    stacked_kv: list[torch.Tensor],
    overwrite: bool = False,
) -> str:
    """Save cartridge KV tensors to shared memory for zero-copy IPC.

    Args:
        cartridge_id: Unique identifier for the cartridge
        stacked_kv: List of [stacked_keys, stacked_values] tensors
        overwrite: If True, replace any existing shared-memory payload

    Returns:
        Path to the shared memory file
    """
    os.makedirs(SHM_CARTRIDGE_DIR, exist_ok=True)
    shm_path = _get_shm_path(cartridge_id)
    lock_path = _get_shm_lock_path(cartridge_id)
    tmp_path = f"{shm_path}.tmp.{os.getpid()}.{threading.get_ident()}"

    with open(lock_path, "a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            if os.path.exists(shm_path):
                if overwrite:
                    logger.info("Overwriting cartridge shm: %s", shm_path)
                    os.unlink(shm_path)
                elif zipfile.is_zipfile(shm_path):
                    # Another request may already have finished writing the
                    # cartridge. Because writes are lock-protected and completed
                    # via atomic rename, an existing target path is safe to
                    # reuse once we hold the lock.
                    logger.debug("Cartridge already in shm: %s", shm_path)
                    _active_shm_paths[cartridge_id] = shm_path
                    return shm_path
                else:
                    logger.warning(
                        "Found invalid cartridge shm file, rewriting: %s",
                        shm_path,
                    )
                    os.unlink(shm_path)

            # Write to a unique temp path and atomically publish once complete
            # so concurrent readers never observe a partial torch archive.
            with open(tmp_path, "wb") as tmp_file:
                torch.save(stacked_kv, tmp_file)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())

            os.replace(tmp_path, shm_path)
            _active_shm_paths[cartridge_id] = shm_path
            logger.info(
                "Saved cartridge to shm: %s (%.2f MB)",
                shm_path,
                os.path.getsize(shm_path) / (1024 * 1024),
            )
            return shm_path
        finally:
            if os.path.exists(tmp_path):
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def load_cartridge_from_shm(shm_path: str) -> list[torch.Tensor] | None:
    """Load cartridge KV tensors from shared memory.

    Args:
        shm_path: Path to the shared memory file

    Returns:
        List of [stacked_keys, stacked_values] tensors, or None if not found
    """
    if not os.path.exists(shm_path):
        logger.warning("Cartridge shm file not found: %s", shm_path)
        return None

    stacked_kv = torch.load(shm_path, map_location="cpu", weights_only=True)
    logger.info("Loaded cartridge from shm: %s", shm_path)
    return stacked_kv


def cleanup_cartridge_shm(cartridge_id: str) -> None:
    """Clean up shared memory file for a cartridge.

    Args:
        cartridge_id: Unique identifier for the cartridge
    """
    shm_path = _active_shm_paths.pop(cartridge_id, None)
    if shm_path is None:
        shm_path = _get_shm_path(cartridge_id)

    if os.path.exists(shm_path):
        try:
            os.unlink(shm_path)
            logger.info("Cleaned up cartridge shm: %s", shm_path)
        except OSError as e:
            logger.warning("Failed to cleanup cartridge shm %s: %s", shm_path, e)


def cleanup_all_cartridge_shm() -> None:
    """Clean up all shared memory cartridge files."""
    if os.path.exists(SHM_CARTRIDGE_DIR):
        for filename in os.listdir(SHM_CARTRIDGE_DIR):
            if filename.startswith("cart_") and filename.endswith(".pt"):
                filepath = os.path.join(SHM_CARTRIDGE_DIR, filename)
                with contextlib.suppress(OSError):
                    os.unlink(filepath)
        logger.info("Cleaned up all cartridge shm files")
    _active_shm_paths.clear()


# Compiled pattern for matching KV cache layer keys
_KV_LAYER_PATTERN = re.compile(r"layers\.(\d+)\.attention\.prefix\.(keys|values)")

# In-memory LRU cache for parsed CartridgeData objects
# Limited to 3 entries to prevent memory growth with frequent checkpoint updates
# Uses OrderedDict for LRU eviction (move to end on access, pop from front when full)
_CARTRIDGE_CACHE_MAX_SIZE = 20
_cartridge_data_cache: OrderedDict[str, "CartridgeData"] = OrderedDict()


def _parse_kv_cache_to_layers(
    kv_cache: dict[str, torch.Tensor],
    device: torch.device | str | None = None,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Parse KV cache dict into keys_by_layer and values_by_layer.

    Input: (1, seq_len, num_kv_heads, head_dim)
    Output: (num_kv_heads, seq_len, head_dim)
    """
    keys_by_layer: dict[int, torch.Tensor] = {}
    values_by_layer: dict[int, torch.Tensor] = {}

    for key_name, tensor in kv_cache.items():
        match = _KV_LAYER_PATTERN.match(key_name)
        if not match:
            continue

        layer_idx = int(match.group(1))
        kv_type = match.group(2)

        # Transform: (1, seq_len, num_kv_heads, head_dim) ->
        # (num_kv_heads, seq_len, head_dim)
        if tensor.dim() != 4:
            raise ValueError(
                f"Invalid cartridge format: {key_name} should be 4D "
                f"(1, seq_len, num_kv_heads, head_dim), got {tensor.shape}"
            )
        tensor = tensor.squeeze(0).permute(1, 0, 2)

        if device is not None:
            tensor = tensor.to(device)

        if kv_type == "keys":
            keys_by_layer[layer_idx] = tensor
        else:
            values_by_layer[layer_idx] = tensor

    return keys_by_layer, values_by_layer


def clear_cartridge_data_cache() -> None:
    """Clear the in-memory cartridge data cache."""
    _cartridge_data_cache.clear()
    logger.info("Cleared cartridge data cache")


class CartridgeData:
    """Container for loaded cartridge data.

    Supported formats:

    1. Standard format (pre-computed KV cache):
       {'token_ids': Tensor, 'kv_cache': dict, 'metadata': dict}

    2. Learned cartridge format (from cartridges library):
       {'layers.N.attention.prefix.keys': Tensor, ...}
       or {'trainable_keys': [...], 'trainable_values': [...], ...}
    """

    def __init__(
        self,
        kv_cache: dict[str, torch.Tensor] | None = None,
        token_ids: torch.Tensor | list[int] | None = None,
        metadata: dict[str, Any] | None = None,
        is_learned: bool = False,
    ):
        self.is_learned = is_learned
        self.kv_cache = kv_cache or {}
        self.metadata = metadata or {}
        self._stacked_kv: list[torch.Tensor] | None = None

        # Normalize token_ids to tensor
        if token_ids is None:
            self.token_ids = torch.tensor([], dtype=torch.long)
            self.num_tokens = self._infer_num_tokens_from_kv_cache()
        else:
            self.token_ids = (
                torch.tensor(token_ids, dtype=torch.long)
                if isinstance(token_ids, list)
                else token_ids
            )
            self.num_tokens = len(self.token_ids)

    def _infer_num_tokens_from_kv_cache(self) -> int:
        """Infer seq_len from KV cache.

        Expected shape: (1, seq_len, num_kv_heads, head_dim).
        """
        if not self.kv_cache:
            return 0
        tensor = next(iter(self.kv_cache.values()))
        if tensor.dim() != 4:
            raise ValueError(
                f"Invalid cartridge format: expected 4D tensor, "
                f"got {tensor.dim()}D: {tensor.shape}"
            )
        return tensor.shape[1]

    def has_valid_token_ids(self) -> bool:
        """Check if this cartridge has valid token IDs for prepending."""
        return len(self.token_ids) > 0 and not self.is_learned

    def get_stacked_kv(self) -> list[torch.Tensor] | None:
        """Get stacked KV tensors for IPC, computing and caching on first call.

        Output: [stacked_keys, stacked_values] with shape
            (num_attention_layers, num_kv_heads, seq_len, head_dim)
        """
        if not self.is_learned or not self.kv_cache:
            return None

        if self._stacked_kv is not None:
            return self._stacked_kv

        keys_by_layer, values_by_layer = _parse_kv_cache_to_layers(self.kv_cache)

        if keys_by_layer and values_by_layer:
            ordered_layers = sorted(set(keys_by_layer) & set(values_by_layer))
            if len(ordered_layers) != len(keys_by_layer) or len(ordered_layers) != len(
                values_by_layer
            ):
                missing_keys = sorted(set(values_by_layer) - set(keys_by_layer))
                missing_values = sorted(set(keys_by_layer) - set(values_by_layer))
                raise ValueError(
                    "Cartridge has mismatched sparse KV layers: "
                    f"missing_keys={missing_keys}, missing_values={missing_values}"
                )
            self._stacked_kv = [
                torch.stack([keys_by_layer[i] for i in ordered_layers]),
                torch.stack([values_by_layer[i] for i in ordered_layers]),
            ]
            logger.info(
                "Computed stacked cartridge KV: %d attention layers (%s)",
                len(ordered_layers),
                ordered_layers,
            )

        return self._stacked_kv

    @classmethod
    def from_dict(cls, data: dict) -> "CartridgeData":
        """Load CartridgeData from a dictionary."""
        has_flat_kv = any(_KV_LAYER_PATTERN.match(k) for k in data)
        has_trainable_kv = "trainable_keys" in data or "frozen_keys" in data
        is_learned = has_flat_kv or has_trainable_kv

        if not is_learned:
            # Standard format requires token_ids
            token_ids = data.get("token_ids")
            if token_ids is None:
                raise ValueError(
                    "Cartridge must contain 'token_ids' field. "
                    "For learned cartridges, use layer keys or trainable_keys format."
                )
            return cls(
                kv_cache=data.get("kv_cache"),
                token_ids=token_ids,
                metadata=data.get("metadata") or {},
                is_learned=False,
            )

        # Learned cartridge
        kv_cache = cls._extract_learned_kv_cache(data, has_trainable_kv)
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        metadata["_cartridge_type"] = "learned"

        logger.info("Loaded learned cartridge with %d KV cache tensors.", len(kv_cache))

        return cls(
            kv_cache=kv_cache,
            token_ids=data.get("token_ids"),
            metadata=metadata,
            is_learned=True,
        )

    @classmethod
    def _extract_learned_kv_cache(
        cls, data: dict, has_trainable_kv: bool
    ) -> dict[str, torch.Tensor]:
        """Extract KV cache from learned cartridge data.

        Expected format from torchtitan prefix adapters:
        - Keys: "layers.{N}.attention.prefix.keys/values"
        - Shape: (1, seq_len, num_kv_heads, head_dim)

        Raises ValueError if the format doesn't match expectations.
        """
        if not has_trainable_kv:
            # Flat layer format (from torchtitan prefix adapters)
            kv_cache = {k: v for k, v in data.items() if _KV_LAYER_PATTERN.match(k)}

            # Validate the format
            if kv_cache:
                cls._validate_kv_cache_format(kv_cache)
            else:
                # Log unexpected keys for debugging
                unexpected_keys = [k for k in data if not k.startswith("_")]
                if unexpected_keys:
                    logger.warning(
                        "Cartridge has no recognized KV cache keys. Found: %s",
                        unexpected_keys[:10],
                    )

            return kv_cache

        # TrainableCache format: convert to flat layer format
        all_keys = data.get("trainable_keys") or data.get("frozen_keys", [])
        all_values = data.get("trainable_values") or data.get("frozen_values", [])

        kv_cache = {}
        for layer_idx, (k, v) in enumerate(zip(all_keys, all_values)):
            if k is not None:
                kv_cache[f"layers.{layer_idx}.attention.prefix.keys"] = k
            if v is not None:
                kv_cache[f"layers.{layer_idx}.attention.prefix.values"] = v

        if kv_cache:
            cls._validate_kv_cache_format(kv_cache)

        return kv_cache

    @classmethod
    def _validate_kv_cache_format(cls, kv_cache: dict[str, torch.Tensor]) -> None:
        """Validate KV cache format from torchtitan prefix adapters.

        Expected shape: (1, seq_len, num_kv_heads, head_dim)
        All tensors must have identical shape (same seq_len, num_kv_heads, head_dim).
        """
        if not kv_cache:
            return

        first_key = next(iter(kv_cache))
        first_tensor = kv_cache[first_key]

        if first_tensor.dim() != 4:
            raise ValueError(
                f"Invalid cartridge format: expected 4D tensor "
                f"(1, seq_len, num_kv_heads, head_dim), "
                f"got {first_tensor.dim()}D: {first_tensor.shape}"
            )
        if first_tensor.shape[0] != 1:
            raise ValueError(
                f"Invalid cartridge format: expected batch dim = 1, "
                f"got {first_tensor.shape[0]}"
            )

        expected_shape = first_tensor.shape
        for key, tensor in kv_cache.items():
            if tensor.shape != expected_shape:
                raise ValueError(
                    f"Cartridge shape mismatch: {key} has {tensor.shape}, "
                    f"expected {expected_shape}"
                )

        seq_len, num_kv_heads, head_dim = (
            expected_shape[1],
            expected_shape[2],
            expected_shape[3],
        )
        logger.info(
            "Loaded cartridge: %d tensors, seq_len=%d, num_kv_heads=%d, head_dim=%d",
            len(kv_cache),
            seq_len,
            num_kv_heads,
            head_dim,
        )

    def __repr__(self) -> str:
        return (
            f"CartridgeData(type={'learned' if self.is_learned else 'precomputed'}, "
            f"num_tokens={self.num_tokens}, "
            f"has_kv_cache={bool(self.kv_cache)}, "
            f"kv_layers={len(self.kv_cache)})"
        )


def _get_cartridge_lock_path(cartridge_id: str) -> str:
    """Get the lock file path for a cartridge ID."""
    id_hash = hashlib.sha256(cartridge_id.encode()).hexdigest()[:16]
    lock_dir = os.path.join(SHM_CARTRIDGE_DIR, "locks")
    os.makedirs(lock_dir, exist_ok=True)
    return os.path.join(lock_dir, f"cart_{id_hash}.lock")


def load_cartridge(
    cartridge_id: str,
    force_redownload: bool = False,
) -> CartridgeData:
    """Load a KV cache cartridge.

    The cartridge_id can be either:
    - S3 URI: s3://bucket/path/to/cartridge.pt
    - Local path: /path/to/local/cartridge.pt

    The Path abstraction automatically handles both cases.

    Uses file-based locking to prevent thundering herd when multiple
    containers try to load the same cartridge simultaneously.

    Args:
        cartridge_id: The identifier/path of the cartridge (S3 URI or local path)
        force_redownload: If True, re-download even if cached

    Returns:
        CartridgeData containing the loaded cartridge
    """
    cache_key = cartridge_id

    # Fast path: check cache without lock
    if not force_redownload and cache_key in _cartridge_data_cache:
        logger.debug("Using cached CartridgeData for: %s", cartridge_id)
        _cartridge_data_cache.move_to_end(cache_key)
        return _cartridge_data_cache[cache_key]

    # Slow path: acquire file lock to prevent thundering herd across containers
    lock_path = _get_cartridge_lock_path(cartridge_id)

    with open(lock_path, "w") as lock_file:
        logger.debug("Acquiring lock for cartridge: %s", cartridge_id)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            # Double-check after acquiring lock; another process may have loaded it.
            if not force_redownload and cache_key in _cartridge_data_cache:
                logger.debug("Cache hit after lock for: %s", cartridge_id)
                _cartridge_data_cache.move_to_end(cache_key)
                return _cartridge_data_cache[cache_key]

            logger.info("Loading cartridge (holding lock): %s", cartridge_id)

            manager = get_adapter_manager()
            cartridge_tensor = manager.get_adapter(
                adapter_id=cartridge_id,
                force_redownload=force_redownload,
            )

            if not isinstance(cartridge_tensor, dict):
                raise ValueError(
                    "Invalid cartridge format. Expected dict, got "
                    f"{type(cartridge_tensor)}"
                )

            cartridge_data = CartridgeData.from_dict(cartridge_tensor)
            logger.info("Successfully loaded cartridge: %s", cartridge_data)

            # LRU eviction: remove oldest if at max size
            while len(_cartridge_data_cache) >= _CARTRIDGE_CACHE_MAX_SIZE:
                evicted_key, evicted_data = _cartridge_data_cache.popitem(last=False)
                logger.info("Evicted cartridge from cache: %s", evicted_key)
                cleanup_cartridge_shm(evicted_key)
                del evicted_data

            _cartridge_data_cache[cache_key] = cartridge_data
            return cartridge_data
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


class CheckpointCache:
    """Cache for checkpoint resolution with TTL and HEAD request optimization.

    Caches checkpoint paths on disk (with file locking) to share across containers.
    Uses HEAD requests to check if latest.txt has been updated.

    Process-safe: Uses fcntl.flock() for atomic reads/writes across containers.
    """

    def __init__(
        self,
        cache_dir: str = CHECKPOINT_CACHE_DIR,
        ttl: int = CHECKPOINT_CHECK_INTERVAL,
    ):
        self.cache_dir = cache_dir
        self.ttl = ttl
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_path(self, run_dir: str) -> str:
        """Get local cache file path for a checkpoint run directory."""
        run_hash = hashlib.sha256(run_dir.encode()).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"{run_hash}.json")

    def get(self, run_dir: str) -> dict | None:
        """Get cached checkpoint info if fresh (< TTL seconds old).

        Uses file locking for process-safety across containers.
        """
        cache_path = self._get_cache_path(run_dir)

        if not os.path.exists(cache_path):
            return None

        try:
            with open(cache_path) as f:
                # Acquire shared lock for reading
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    cache = json.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            # Check if cache is fresh
            if time.time() - cache.get("last_checked", 0) < self.ttl:
                logger.debug("Using fresh cache for %s", run_dir)
                return cache

            logger.debug("Cache stale for %s", run_dir)
            return None
        except Exception as e:
            logger.warning("Failed to read checkpoint cache: %s", e)
            return None

    def set(self, run_dir: str, checkpoint_path: str, latest_txt_mtime: float) -> None:
        """Write checkpoint info to disk cache.

        Uses exclusive file locking for process-safety across containers.
        """
        cache_path = self._get_cache_path(run_dir)

        cache = {
            "checkpoint_path": checkpoint_path,
            "latest_txt_mtime": latest_txt_mtime,
            "last_checked": time.time(),
        }

        try:
            # Use exclusive lock for writing (atomic write)
            with open(cache_path, "w") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump(cache, f)
                    f.flush()
                    os.fsync(f.fileno())  # Ensure written to disk
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            logger.debug("Wrote checkpoint cache for %s", run_dir)
        except Exception as e:
            logger.warning("Failed to write checkpoint cache: %s", e)

    def check_latest_txt(
        self, run_dir: str, cached_mtime: float | None = None
    ) -> tuple[str | None, float | None]:
        """Check latest.txt for newest checkpoint using HEAD request.

        Returns (checkpoint_path, mtime) if updated, else (None, None).
        """
        if not run_dir.startswith("s3://"):
            return None, None

        # Parse s3://bucket/path
        s3_path = run_dir[5:]
        bucket, _, prefix = s3_path.partition("/")
        latest_txt_key = f"{prefix}/latest.txt" if prefix else "latest.txt"

        try:
            s3 = boto3.client("s3")

            # HEAD request to check Last-Modified
            response = s3.head_object(Bucket=bucket, Key=latest_txt_key)
            latest_txt_mtime = response["LastModified"].timestamp()

            # Skip if unchanged
            if cached_mtime is not None and latest_txt_mtime <= cached_mtime:
                logger.debug("latest.txt unchanged for %s", run_dir)
                return None, None

            # Read latest.txt content
            logger.info("Detected new checkpoint in latest.txt for %s", run_dir)
            obj = s3.get_object(Bucket=bucket, Key=latest_txt_key)
            checkpoint_path = obj["Body"].read().decode("utf-8").strip()

            return checkpoint_path, latest_txt_mtime

        except s3.exceptions.NoSuchKey:
            logger.debug("latest.txt not found for %s", run_dir)
            return None, None
        except Exception as e:
            logger.warning("Failed to check latest.txt for %s: %s", run_dir, e)
            return None, None


# Global cache instance
_checkpoint_cache = CheckpointCache()


def _extract_run_dir_from_glob(s3_pattern: str) -> str | None:
    """Extract run directory from glob pattern.

    Example: s3://bucket/weights/torchtitan/2026-02-03/run-id/step-*/model.pt
          -> s3://bucket/weights/torchtitan/2026-02-03/run-id
    """
    if not s3_pattern.startswith("s3://") or "*" not in s3_pattern:
        return None

    # Find the part before the first glob
    parts = s3_pattern.split("/")
    run_dir_parts = []
    for part in parts:
        if "*" in part:
            break
        run_dir_parts.append(part)

    return "/".join(run_dir_parts) if len(run_dir_parts) > 3 else None


def _resolve_s3_glob(s3_pattern: str) -> str | None:
    """Resolve S3 glob to most recently modified matching file with smart caching.

    Uses latest.txt for dynamic checkpoint updates:
    1. Checks local cache (<30s old) for checkpoint path
    2. If stale, does HEAD request to latest.txt to check for updates
    3. Only reads latest.txt if Last-Modified changed
    4. Falls back to listing objects if latest.txt not found

    Args:
        s3_pattern: S3 URI with optional * glob (e.g., s3://bucket/path/step-*/file.pt)

    Returns:
        Resolved S3 URI, or original if no glob/matches
    """
    # Only resolve if glob present
    if "*" not in s3_pattern:
        return s3_pattern

    if not s3_pattern.startswith("s3://"):
        return s3_pattern

    # Extract run directory for caching (e.g., s3://bucket/run-id)
    run_dir = _extract_run_dir_from_glob(s3_pattern)

    # Try cache first (if < 30s old)
    if run_dir:
        cache = _checkpoint_cache.get(run_dir)
        if cache:
            logger.debug("Using cached checkpoint: %s", cache["checkpoint_path"])
            return cache["checkpoint_path"]

        # Cache miss or stale - check latest.txt
        cached_mtime = cache.get("latest_txt_mtime") if cache else None
        checkpoint_path, latest_txt_mtime = _checkpoint_cache.check_latest_txt(
            run_dir, cached_mtime
        )

        if checkpoint_path:
            # Found new checkpoint via latest.txt
            assert latest_txt_mtime is not None
            _checkpoint_cache.set(run_dir, checkpoint_path, latest_txt_mtime)
            logger.info(
                "Resolved %s -> %s (via latest.txt)", s3_pattern, checkpoint_path
            )
            return checkpoint_path

        # latest.txt exists but unchanged - use cached path
        if cache and latest_txt_mtime is None and cached_mtime is not None:
            # Update last_checked timestamp
            _checkpoint_cache.set(run_dir, cache["checkpoint_path"], cached_mtime)
            return cache["checkpoint_path"]

    # Fallback: No latest.txt or error - list objects (original logic)
    try:
        s3_path = s3_pattern[5:]
        bucket, _, path_pattern = s3_path.partition("/")
        prefix, _, suffix = path_pattern.partition("*")

        s3 = boto3.client("s3")
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

        if "Contents" not in response:
            return None

        # Find matches, sort by last modified (newest first)
        matches = [
            (obj["LastModified"], obj["Key"])
            for obj in response["Contents"]
            if obj["Key"].endswith(suffix)
        ]

        if not matches:
            return None

        matches.sort(reverse=True)
        latest_key = matches[0][1]
        resolved = f"s3://{bucket}/{latest_key}"

        logger.info("Resolved %s -> %s (via listing)", s3_pattern, resolved)
        return resolved

    except Exception as e:
        logger.error("Failed to resolve S3 glob %s: %s", s3_pattern, e)
        return None


def _resolve_modal_glob(modal_pattern: str) -> str | None:
    """Resolve Modal volume glob to most recent matching file.

    Uses the active path backend for modal:// paths and checks for latest.txt
    for dynamic checkpoint resolution.

    Args:
        modal_pattern: Modal URI with optional * glob
                       (e.g., modal://volume-name/step-*/model.pt)

    Returns:
        Resolved modal:// URI, or original if no glob/matches
    """
    # Only resolve if glob present
    if "*" not in modal_pattern:
        return modal_pattern

    if not modal_pattern.startswith("modal://"):
        return modal_pattern

    try:
        without_prefix = modal_pattern[8:]
        volume_name, _, remainder = without_prefix.partition("/")

        pattern_parts = remainder.split("/") if remainder else []
        base_parts: list[str] = []
        for part in pattern_parts:
            if "*" in part:
                break
            base_parts.append(part)

        base_prefix = f"modal://{volume_name}"
        if base_parts:
            base_prefix = f"{base_prefix}/{'/'.join(base_parts)}"

        latest_txt = Path(f"{base_prefix}/latest.txt")
        if latest_txt.isfile():
            try:
                latest_path = latest_txt.read_text().strip()
                if latest_path.startswith("modal://"):
                    if Path(latest_path).exists():
                        logger.info(
                            "Resolved %s -> %s (via latest.txt)",
                            modal_pattern,
                            latest_path,
                        )
                        return latest_path
                elif latest_path.startswith("/"):
                    latest_parts = latest_path.strip("/").split("/", 1)
                    rel_path = latest_parts[1] if len(latest_parts) > 1 else ""
                    candidate = (
                        f"modal://{volume_name}/{rel_path}"
                        if rel_path
                        else f"modal://{volume_name}"
                    )
                    if Path(candidate).exists():
                        logger.info(
                            "Resolved %s -> %s (via latest.txt)",
                            modal_pattern,
                            candidate,
                        )
                        return candidate
                elif latest_path:
                    candidate = f"modal://{volume_name}/{latest_path.lstrip('/')}"
                    if Path(candidate).exists():
                        logger.info(
                            "Resolved %s -> %s (via latest.txt)",
                            modal_pattern,
                            candidate,
                        )
                        return candidate
            except Exception as e:
                logger.debug("Could not read latest.txt: %s", e)

        glob_root = Path(base_prefix)
        relative_pattern = "/".join(pattern_parts[len(base_parts) :]) or "*"
        matches = sorted(str(path) for path in glob_root.glob(relative_pattern))
        if not matches:
            return None

        resolved = matches[-1]

        logger.info("Resolved %s -> %s (via glob)", modal_pattern, resolved)
        return resolved

    except Exception as e:
        logger.error("Failed to resolve Modal glob %s: %s", modal_pattern, e)
        return None


def _resolve_glob(pattern: str) -> str | None:
    """Resolve glob pattern for any supported backend.

    Dispatches to the appropriate resolver based on URI scheme:
    - s3:// -> _resolve_s3_glob
    - modal:// -> _resolve_modal_glob
    - local paths -> local glob

    Args:
        pattern: Path pattern with optional * glob

    Returns:
        Resolved path, or original if no glob/matches
    """
    if pattern.startswith("s3://"):
        return _resolve_s3_glob(pattern)
    elif pattern.startswith("modal://"):
        return _resolve_modal_glob(pattern)
    elif "*" in pattern:
        # Local path glob - first check for latest.txt
        parts = pattern.split("/")
        base_parts = []
        for part in parts:
            if "*" in part:
                break
            base_parts.append(part)
        base_dir = "/".join(base_parts) or "/"
        latest_txt = os.path.join(base_dir, "latest.txt")

        if os.path.exists(latest_txt):
            try:
                with open(latest_txt) as f:
                    latest_path = f.read().strip()
                if os.path.exists(latest_path):
                    logger.info(
                        "Resolved %s -> %s (via latest.txt)", pattern, latest_path
                    )
                    return latest_path
            except Exception as e:
                logger.debug("Could not read latest.txt: %s", e)

        # Fallback: use glob
        matches = globlib.glob(pattern)
        if not matches:
            return None
        # Sort by modification time (newest first)
        matches_with_mtime = [(os.path.getmtime(m), m) for m in matches]
        matches_with_mtime.sort(reverse=True)
        resolved = matches_with_mtime[0][1]
        logger.info("Resolved %s -> %s (via local glob)", pattern, resolved)
        return resolved
    else:
        return pattern


def _spec_sources(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the ranked list of ``{id, source}`` candidates for a spec.

    Accepts the new ``sources`` list and the legacy single ``id``/``source``
    pair (which is materialised as a single-entry list)."""
    sources = spec.get("sources")
    if sources:
        return list(sources)
    cartridge_id = spec.get("id")
    if not cartridge_id:
        return []
    logger.warning(
        "Adapter spec uses deprecated top-level `id`/`source` fields. "
        "Switch callers to `sources: [{id, source}, ...]` to get ranked "
        "fallback. Saw: id=%s source=%s",
        cartridge_id,
        spec.get("source"),
    )
    return [{"id": cartridge_id, "source": spec.get("source")}]


@dataclass(frozen=True)
class PreparedPrefixCartridgeRequest:
    cartridge_id: str
    cartridge_seq_len: int
    cartridge_shm_path: str
    cartridge_allow_pipelining: bool
    sampling_params: Any


def get_prefix_cartridge_specs(request: Any) -> list[dict[str, Any]] | None:
    adapters = getattr(request, "adapters", None)
    if not isinstance(adapters, dict) or "prefix" not in adapters:
        return None

    prefix_specs = adapters["prefix"]
    if not prefix_specs:
        return None
    if not isinstance(prefix_specs, list):
        raise ValueError("`adapters.prefix` must be a list.")
    if len(prefix_specs) != 1:
        raise ValueError("Exactly one prefix cartridge is currently supported.")
    if not isinstance(prefix_specs[0], dict):
        raise ValueError("Each prefix cartridge spec must be an object.")
    return prefix_specs


def prepare_prefix_cartridge_request(
    request: Any,
    *,
    prompt_len: int,
    max_model_len: int,
    expected_kv_heads: int,
    expected_head_size: int,
    sampling_params: Any,
    truncate_prompt_tokens: int | None,
) -> PreparedPrefixCartridgeRequest | None:
    prefix_specs = get_prefix_cartridge_specs(request)
    if prefix_specs is None:
        return None

    logger.info("Processing prefix cartridge adapter request.")
    loaded_cartridges = load_cartridges_from_request_with_ids(prefix_specs)
    if len(loaded_cartridges) != 1:
        raise ValueError("Failed to load prefix cartridge.")

    cartridge_id, cartridge = loaded_cartridges[0]
    stacked_kv = cartridge.get_stacked_kv()
    if stacked_kv is None:
        raise ValueError(
            "Only learned prefix cartridges with KV tensors are currently supported."
        )

    stacked_keys = stacked_kv[0]
    _, cart_kv_heads, cart_seq_len, cart_head_dim = stacked_keys.shape
    if cart_kv_heads != expected_kv_heads or cart_head_dim != expected_head_size:
        raise ValueError(
            "Prefix cartridge shape does not match model: "
            f"kv_heads={cart_kv_heads} expected={expected_kv_heads}, "
            f"head_dim={cart_head_dim} expected={expected_head_size}."
        )

    if truncate_prompt_tokens is not None:
        prompt_len = min(
            prompt_len,
            max_model_len if truncate_prompt_tokens == -1 else truncate_prompt_tokens,
        )
    effective_input_len = prompt_len + cartridge.num_tokens
    if effective_input_len > max_model_len:
        raise ValueError(
            "Input length plus prefix cartridge length exceeds model's "
            "maximum context length: "
            f"{prompt_len} + {cartridge.num_tokens} > {max_model_len}."
        )

    max_tokens_with_cartridge = max_model_len - effective_input_len
    if (
        sampling_params.max_tokens is None
        or sampling_params.max_tokens > max_tokens_with_cartridge
    ):
        sampling_params = sampling_params.clone()
        sampling_params.max_tokens = max_tokens_with_cartridge

    cartridge_spec = prefix_specs[0]
    cartridge_shm_path = save_cartridge_to_shm(
        cartridge_id,
        stacked_kv,
        overwrite=bool(cartridge_spec.get("force_redownload", False)),
    )
    logger.info(
        "Loaded prefix cartridge adapter: seq_len=%d kv_heads=%d head_dim=%d",
        cart_seq_len,
        cart_kv_heads,
        cart_head_dim,
    )

    return PreparedPrefixCartridgeRequest(
        cartridge_id=cartridge_id,
        cartridge_seq_len=cartridge.num_tokens,
        cartridge_shm_path=cartridge_shm_path,
        cartridge_allow_pipelining=bool(cartridge_spec.get("allow_pipelining", False)),
        sampling_params=sampling_params,
    )


def load_cartridges_from_request_with_ids(
    cartridges_spec: list[dict[str, Any]],
) -> list[tuple[str, CartridgeData]]:
    """Load cartridges and return the resolved ID used for each one."""
    loaded_cartridges = []

    for spec in cartridges_spec:
        sources = _spec_sources(spec)
        if not sources:
            logger.warning("Skipping cartridge with no id/sources: %s", spec)
            continue

        last_err: Exception | None = None
        loaded = False
        for source_ref in sources:
            cartridge_id = source_ref.get("id")
            if not cartridge_id:
                continue
            resolved_id = _resolve_glob(cartridge_id)
            if resolved_id is None:
                logger.warning("Could not resolve cartridge glob: %s", cartridge_id)
                continue
            if resolved_id != cartridge_id:
                logger.info("Using cartridge: %s -> %s", cartridge_id, resolved_id)
            try:
                cartridge_data = load_cartridge(
                    cartridge_id=resolved_id,
                    force_redownload=spec.get("force_redownload", False),
                )
                loaded_cartridges.append((resolved_id, cartridge_data))
                loaded = True
                break
            except FileNotFoundError as e:
                logger.warning(
                    "Cartridge source %s not found; trying next: %s",
                    resolved_id,
                    e,
                )
                last_err = e
                continue
            except Exception as e:
                logger.error("Failed to load cartridge %s: %s", resolved_id, e)
                raise RuntimeError(
                    f"Failed to load cartridge {resolved_id}: {e}"
                ) from e

        if not loaded:
            tried = [s.get("id") for s in sources]
            msg = f"Cartridge not found in any source: {tried}"
            logger.error("%s (%s)", msg, last_err)
            raise RuntimeError(msg) from last_err

    return loaded_cartridges


def load_cartridges_from_request(
    cartridges_spec: list[dict[str, Any]],
) -> list[CartridgeData]:
    """Load multiple cartridges from a request specification."""
    return [
        cartridge
        for _, cartridge in load_cartridges_from_request_with_ids(cartridges_spec)
    ]
