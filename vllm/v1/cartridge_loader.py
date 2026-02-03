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

import fcntl
import hashlib
import json
import os
import re
import time
from typing import Any

import boto3
import torch

from vllm.logger import init_logger
from vllm.utils.adapter_manager import get_adapter_manager

logger = init_logger(__name__)

# Directory for cartridge IPC files
# Use /root/.cache/vllm/cartridges instead of /dev/shm for Modal compatibility
# Modal volumes are shared across processes, /dev/shm is not
SHM_CARTRIDGE_DIR = "/root/.cache/vllm/cartridges"

# Directory for checkpoint resolution cache
# Shared volume for caching latest.txt timestamps across containers
CHECKPOINT_CACHE_DIR = "/root/.cache/vllm/checkpoint_cache"

# How often to check for new checkpoints (seconds)
CHECKPOINT_CHECK_INTERVAL = 30

# Track active shm paths for cleanup
_active_shm_paths: dict[str, str] = {}  # cartridge_id -> shm_path


def _get_shm_path(cartridge_id: str) -> str:
    """Get the shared memory path for a cartridge ID."""
    # Use SHA256 hash to avoid path issues with special characters
    id_hash = hashlib.sha256(cartridge_id.encode()).hexdigest()[:16]
    return os.path.join(SHM_CARTRIDGE_DIR, f"cart_{id_hash}.pt")


def save_cartridge_to_shm(
    cartridge_id: str,
    stacked_kv: list[torch.Tensor],
) -> str:
    """Save cartridge KV tensors to shared memory for zero-copy IPC.

    Args:
        cartridge_id: Unique identifier for the cartridge
        stacked_kv: List of [stacked_keys, stacked_values] tensors

    Returns:
        Path to the shared memory file
    """
    os.makedirs(SHM_CARTRIDGE_DIR, exist_ok=True)
    shm_path = _get_shm_path(cartridge_id)

    # Skip if already saved (another request may have saved it)
    if os.path.exists(shm_path):
        logger.debug("Cartridge already in shm: %s", shm_path)
        _active_shm_paths[cartridge_id] = shm_path
        return shm_path

    # Save tensors to shm
    torch.save(stacked_kv, shm_path)
    _active_shm_paths[cartridge_id] = shm_path
    logger.info(
        "Saved cartridge to shm: %s (%.2f MB)",
        shm_path,
        os.path.getsize(shm_path) / (1024 * 1024),
    )
    return shm_path


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
                try:
                    os.unlink(filepath)
                except OSError:
                    pass
        logger.info("Cleaned up all cartridge shm files")
    _active_shm_paths.clear()

# Compiled pattern for matching KV cache layer keys
_KV_LAYER_PATTERN = re.compile(r"layers\.(\d+)\.attention\.prefix\.(keys|values)")

# In-memory cache for parsed CartridgeData objects
# Maps (cartridge_id, source) -> CartridgeData
_cartridge_data_cache: dict[tuple[str, str], "CartridgeData"] = {}


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
            (num_layers, num_kv_heads, seq_len, head_dim)
        """
        if not self.is_learned or not self.kv_cache:
            return None

        if self._stacked_kv is not None:
            return self._stacked_kv

        keys_by_layer, values_by_layer = _parse_kv_cache_to_layers(self.kv_cache)

        if keys_by_layer and values_by_layer:
            num_layers = max(keys_by_layer.keys()) + 1
            self._stacked_kv = [
                torch.stack([keys_by_layer[i] for i in range(num_layers)]),
                torch.stack([values_by_layer[i] for i in range(num_layers)]),
            ]
            logger.info("Computed stacked cartridge KV: %d layers", num_layers)

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

        logger.info(
            "Loaded learned cartridge with %d KV cache tensors.", len(kv_cache)
        )

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


def load_cartridge(
    cartridge_id: str,
    force_redownload: bool = False,
) -> CartridgeData:
    """Load a KV cache cartridge.

    The cartridge_id can be either:
    - S3 URI: s3://bucket/path/to/cartridge.pt
    - Local path: /path/to/local/cartridge.pt

    The Path abstraction automatically handles both cases.

    Args:
        cartridge_id: The identifier/path of the cartridge (S3 URI or local path)
        force_redownload: If True, re-download even if cached

    Returns:
        CartridgeData containing the loaded cartridge
    """
    cache_key = cartridge_id

    if not force_redownload and cache_key in _cartridge_data_cache:
        logger.debug("Using cached CartridgeData for: %s", cartridge_id)
        return _cartridge_data_cache[cache_key]

    logger.info("Loading cartridge: %s", cartridge_id)

    manager = get_adapter_manager()
    cartridge_tensor = manager.get_adapter(
        adapter_id=cartridge_id,
        force_redownload=force_redownload,
    )

    if not isinstance(cartridge_tensor, dict):
        raise ValueError(
            f"Invalid cartridge format. Expected dict, got {type(cartridge_tensor)}"
        )

    cartridge_data = CartridgeData.from_dict(cartridge_tensor)
    logger.info("Successfully loaded cartridge: %s", cartridge_data)

    _cartridge_data_cache[cache_key] = cartridge_data
    return cartridge_data


class CheckpointCache:
    """Cache for checkpoint resolution with TTL and HEAD request optimization.

    Caches checkpoint paths on disk (with file locking) to share across containers.
    Uses HEAD requests to check if latest.txt has been updated.

    Process-safe: Uses fcntl.flock() for atomic reads/writes across containers.
    """

    def __init__(self, cache_dir: str = CHECKPOINT_CACHE_DIR, ttl: int = CHECKPOINT_CHECK_INTERVAL):
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
            with open(cache_path, "r") as f:
                # Acquire shared lock for reading
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    cache = json.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            # Check if cache is fresh
            if time.time() - cache.get("last_checked", 0) < self.ttl:
                logger.debug(f"Using fresh cache for {run_dir}")
                return cache

            logger.debug(f"Cache stale for {run_dir}")
            return None
        except Exception as e:
            logger.warning(f"Failed to read checkpoint cache: {e}")
            return None

    def set(self, run_dir: str, checkpoint_path: str, latest_txt_mtime: float) -> None:
        """Write checkpoint info to disk cache.

        Uses exclusive file locking for process-safety across containers.
        """
        cache_path = self._get_cache_path(run_dir)

        cache = {
            "checkpoint_path": checkpoint_path,
            "latest_txt_mtime": latest_txt_mtime,
            "last_checked": time.time()
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
            logger.debug(f"Wrote checkpoint cache for {run_dir}")
        except Exception as e:
            logger.warning(f"Failed to write checkpoint cache: {e}")

    def check_latest_txt(self, run_dir: str, cached_mtime: float | None = None) -> tuple[str | None, float | None]:
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
                logger.debug(f"latest.txt unchanged for {run_dir}")
                return None, None

            # Read latest.txt content
            logger.info(f"Detected new checkpoint in latest.txt for {run_dir}")
            obj = s3.get_object(Bucket=bucket, Key=latest_txt_key)
            checkpoint_path = obj["Body"].read().decode("utf-8").strip()

            return checkpoint_path, latest_txt_mtime

        except s3.exceptions.NoSuchKey:
            logger.debug(f"latest.txt not found for {run_dir}")
            return None, None
        except Exception as e:
            logger.warning(f"Failed to check latest.txt for {run_dir}: {e}")
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
            logger.debug(f"Using cached checkpoint: {cache['checkpoint_path']}")
            return cache["checkpoint_path"]

        # Cache miss or stale - check latest.txt
        cached_mtime = cache.get("latest_txt_mtime") if cache else None
        checkpoint_path, latest_txt_mtime = _checkpoint_cache.check_latest_txt(run_dir, cached_mtime)

        if checkpoint_path:
            # Found new checkpoint via latest.txt
            _checkpoint_cache.set(run_dir, checkpoint_path, latest_txt_mtime)
            logger.info(f"Resolved {s3_pattern} -> {checkpoint_path} (via latest.txt)")
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

        logger.info(f"Resolved {s3_pattern} -> {resolved} (via listing)")
        return resolved

    except Exception as e:
        logger.error(f"Failed to resolve S3 glob {s3_pattern}: {e}")
        return None


def load_cartridges_from_request(
    cartridges_spec: list[dict[str, Any]],
) -> list[CartridgeData]:
    """Load multiple cartridges from a request specification."""
    loaded_cartridges = []

    for spec in cartridges_spec:
        cartridge_id = spec.get("id")
        if not cartridge_id:
            logger.warning("Skipping cartridge with missing id: %s", spec)
            continue

        # Resolve glob patterns in cartridge ID
        resolved_id = _resolve_s3_glob(cartridge_id)
        if resolved_id is None:
            logger.warning("Could not resolve cartridge glob: %s", cartridge_id)
            continue

        try:
            cartridge_data = load_cartridge(
                cartridge_id=resolved_id,
                force_redownload=spec.get("force_redownload", False),
            )
            loaded_cartridges.append(cartridge_data)
        except Exception as e:
            logger.error("Failed to load cartridge %s: %s", resolved_id, e)
            raise RuntimeError(f"Failed to load cartridge {resolved_id}: {e}") from e

    return loaded_cartridges
