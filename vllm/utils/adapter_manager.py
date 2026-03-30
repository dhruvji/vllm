# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapter manager for downloading and caching adapter artifacts.

Handles downloading and caching of:
- KV cache cartridges (prefix adapters)
- LoRA adapter weights
- Other trained artifacts

Supports multiple storage backends:
- S3 (s3://bucket/path) - downloaded and cached locally
- Modal Volumes (modal://volume-name/path) - loaded directly from mount
- Local paths (/path/to/file) - loaded directly

This class is storage-agnostic and doesn't handle adapter-specific logic.
"""

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path as StdPath

import filelock
import torch
from safetensors.torch import load as safetensors_load

from vllm.path import Path


def _resolve_adapter_file(path_str: str) -> str:
    """If path is a directory, resolve to the best adapter file inside it.

    Prefers adapter_model.safetensors, then model.pt.
    Returns path unchanged if it already points to a file.
    """
    # Quick check: if it looks like a file path already, return as-is.
    if path_str.endswith((".safetensors", ".pt", ".bin")):
        logger.info("[resolve] path is already a file: %s", path_str)
        return path_str

    candidates = (
        "adapter_model.safetensors",
        "model.safetensors",
        "adapter_model.bin",
        "adapter_model.pt",
        "model.pt",
    )

    path = Path(path_str)

    if path.isdir():
        for candidate in candidates:
            candidate_path = path / candidate
            if candidate_path.isfile():
                size = candidate_path.size
                logger.info("[resolve] found local %s: %d bytes", candidate_path, size)
                return str(candidate_path)
        raise FileNotFoundError(f"No adapter file found in directory: {path_str}")

    return path_str


def _looks_like_adapter_file(path_str: str) -> bool:
    return path_str.endswith((".safetensors", ".pt", ".bin"))


def _resolve_adapter_path(path: str | StdPath) -> str:
    """Resolve adapter path, handling directories transparently.

    If *path* points to a directory, looks for ``model.pt`` or
    ``adapter_model.safetensors`` inside it.  Otherwise returns
    the path as-is.  This lets callers pass either a directory
    (e.g. an S3 prefix / checkpoint dir) or a direct file path.
    """
    p = StdPath(path)
    if p.is_dir():
        for candidate in ("model.pt", "adapter_model.safetensors"):
            child = p / candidate
            if child.exists():
                return str(child)
        raise FileNotFoundError(
            f"Adapter directory {path} does not contain model.pt "
            f"or adapter_model.safetensors"
        )
    return str(path)


def _load_adapter(path: str | StdPath, **kwargs) -> dict:
    """Load adapter from .pt (torch) or .safetensors format.

    If path is a directory, resolves to the best file inside it.
    For safetensors we read the entire file into memory first and
    deserialize from bytes.
    """
    path_str = _resolve_adapter_file(str(path))
    path_obj = Path(path_str)
    logger.info("[load_adapter] resolved path: %s", path_str)
    if path_str.endswith(".safetensors"):
        data = _read_file_bytes(path_str)
        logger.info(
            "[load_adapter] safetensors bytes: %d, first16=%s, last16=%s",
            len(data),
            data[:16].hex(),
            data[-16:].hex() if len(data) >= 16 else data.hex(),
        )
        result = safetensors_load(data)
        logger.info(
            "[load_adapter] safetensors loaded: %d keys, keys=%s",
            len(result),
            list(result.keys())[:5],
        )
        return result
    with path_obj.open("rb") as f:
        return torch.load(f, map_location="cpu", **kwargs)


def _read_file_bytes(path: str) -> bytes:
    """Read file bytes via the configured path backend."""
    data = Path(path).read_bytes()
    logger.info("[read_file_bytes] %s: %d bytes", path, len(data))
    return data

logger = logging.getLogger(__name__)

MODAL_LOAD_RETRY_DELAY_SECONDS = 5.0


def _normalize_adapter_id(adapter_id: str) -> str:
    """Trim accidental surrounding whitespace without changing the path body."""
    normalized = adapter_id.strip()
    if normalized != adapter_id:
        logger.warning(
            "Normalized adapter path by trimming surrounding whitespace: %r -> %r",
            adapter_id,
            normalized,
        )
    return normalized


class AdapterManager:
    """
    Manager for downloading and caching adapter artifacts from multiple backends.

    This class handles:
    - Downloading adapters from S3 URIs (s3://bucket/path/to/adapter.pt)
    - Loading adapters from Modal Volumes (modal://volume-name/path/to/adapter.pt)
    - Loading adapters from local file paths
    - Caching downloaded files to avoid re-downloading (S3 only)
    - Thread-safe file locking during downloads

    Supports multiple adapter types:
    - KV cache cartridges (prefix adapters)
    - LoRA adapter weights
    - Any other trained model artifacts
    """

    def __init__(self, cache_dir: str | None = None):
        """
        Initialize the AdapterManager.

        Args:
            cache_dir: Directory to cache downloaded adapters.
                      Defaults to ~/.cache/vllm/adapters
        """
        if cache_dir is None:
            cache_dir = os.path.join(
                os.path.expanduser("~"), ".cache", "vllm", "adapters"
            )

        self.cache_dir = StdPath(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "AdapterManager initialized with cache_dir: %s", self.cache_dir
        )

    def _get_cache_path(self, uri: str) -> StdPath:
        """
        Get the local cache path for a given URI.

        Args:
            uri: The S3 URI or identifier of the adapter

        Returns:
            Path to the cached file
        """
        # Use hash of URI as filename, preserving the original extension so
        # _load_adapter can pick the right deserializer (safetensors vs torch).
        uri_hash = hashlib.sha256(uri.encode()).hexdigest()
        for ext in (".safetensors", ".pt", ".bin"):
            if uri.endswith(ext):
                return self.cache_dir / f"{uri_hash}{ext}"
        return self.cache_dir / f"{uri_hash}.pt"

    def get_adapter(
        self,
        adapter_id: str,
        force_redownload: bool = False,
    ) -> torch.Tensor:
        """
        Get an adapter, downloading from S3 or loading from local/volume path.

        The adapter_id can be either:
        - S3 URI: s3://bucket/path/to/adapter.pt (downloaded and cached)
        - Modal Volume: modal://volume-name/path/to/adapter.pt (loaded directly)
        - Local path: /path/to/local/adapter.pt or ./adapter.pt (loaded directly)

        The Path abstraction automatically handles all cases.

        Args:
            adapter_id: The identifier/path of the adapter
            force_redownload: If True, re-download even if cached
                (only applies to S3 paths)

        Returns:
            Loaded adapter tensor data

        Raises:
            FileNotFoundError: If file doesn't exist
            RuntimeError: If download or loading fails
        """
        adapter_id = _normalize_adapter_id(adapter_id)
        source_path = Path(adapter_id)

        # For S3 paths, use caching (need to download)
        if adapter_id.startswith("s3://"):
            cache_path = self._get_cache_path(adapter_id)
            lock_path = cache_path.with_suffix(".lock")

            def _copy_with_progress(*, reason: str) -> None:
                """Copy adapter from source_path to cache_path with progress logging.

                NOTE: Path.copy doesn't expose callbacks, so we poll
                the destination size. boto3's download_file writes to a
                temp file (*.XXXXXX) then renames atomically.
                """
                import glob as globlib

                start_t = time.time()
                stop_evt = threading.Event()

                def _get_download_size() -> int:
                    """Get size of download in progress.

                    Checks both final and temp files.
                    """
                    # Check final destination first
                    if cache_path.exists():
                        return cache_path.stat().st_size
                    # Check for boto3 temp files (pattern: cache_path.XXXXXX)
                    temp_pattern = str(cache_path) + ".*"
                    temp_files = globlib.glob(temp_pattern)
                    if temp_files:
                        # Return size of largest temp file (in case of stale ones)
                        return max(StdPath(f).stat().st_size for f in temp_files)
                    return 0

                def _log_progress() -> None:
                    last_size = -1
                    last_change_t = time.time()
                    while not stop_evt.wait(10.0):
                        try:
                            size = _get_download_size()
                            now = time.time()
                            if size != last_size:
                                last_change_t = now
                                delta_mb = (
                                    (size - last_size) / (1024 * 1024)
                                    if last_size >= 0
                                    else 0.0
                                )
                                logger.info(
                                    "Adapter download progress (%s): "
                                    "%.2f MB (Δ%.2f MB) -> %s",
                                    reason,
                                    size / (1024 * 1024),
                                    delta_mb,
                                    cache_path,
                                )
                                last_size = size
                            elif now - last_change_t > 120:
                                # If nothing changes for a while,
                                # still emit a heartbeat so it doesn't look like a hang.
                                logger.info(
                                    "Adapter download still in progress (%s): "
                                    "%.2f MB written to %s (no change for %.0fs)",
                                    reason,
                                    size / (1024 * 1024),
                                    cache_path,
                                    now - last_change_t,
                                )
                                last_change_t = now
                        except Exception:
                            # Never let progress logging impact the download path.
                            pass

                t = threading.Thread(target=_log_progress, daemon=True)
                t.start()
                try:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    if _looks_like_adapter_file(adapter_id):
                        with source_path.open("rb") as src, open(cache_path, "wb") as dst:
                            while True:
                                chunk = src.read(8 * 1024 * 1024)
                                if not chunk:
                                    break
                                dst.write(chunk)
                    else:
                        source_path.copy(str(cache_path))
                finally:
                    stop_evt.set()
                final_size_mb = cache_path.stat().st_size / (1024 * 1024)
                logger.info(
                    "Finished adapter download (%s): %s (%.2f MB) in %.1fs",
                    reason,
                    cache_path,
                    final_size_mb,
                    time.time() - start_t,
                )

            def _cleanup_temp_files() -> None:
                """Remove stale boto3 temp files from previous interrupted downloads."""
                import glob as globlib

                temp_pattern = str(cache_path) + ".*"
                for temp_file in globlib.glob(temp_pattern):
                    try:
                        StdPath(temp_file).unlink()
                        logger.info("Cleaned up stale temp file: %s", temp_file)
                    except Exception:
                        pass

            # Use file lock to prevent concurrent downloads
            with filelock.FileLock(lock_path, timeout=300):
                # Check if we need to download
                should_download = force_redownload or not cache_path.exists()

                if should_download:
                    # Clean up any stale temp files from previous interrupted downloads
                    _cleanup_temp_files()
                    logger.info("Downloading adapter from %s", adapter_id)
                    try:
                        _copy_with_progress(reason="initial")
                    except Exception as e:
                        # Clean up partial download and temp files
                        _cleanup_temp_files()
                        if cache_path.exists():
                            cache_path.unlink()
                        raise RuntimeError(
                            f"Failed to download adapter from {adapter_id}: {e}"
                        ) from e
                else:
                    logger.info("Using cached adapter: %s", cache_path)

                # Load the adapter
                try:
                    data = _load_adapter(cache_path)
                    return data
                except Exception as e:
                    # If loading failed, try re-downloading once
                    if not should_download:
                        logger.info("Retrying with fresh download...")
                        try:
                            # Clean up corrupted cache and any stale temp files
                            _cleanup_temp_files()
                            if cache_path.exists():
                                cache_path.unlink()
                            _copy_with_progress(reason="retry_after_load_failure")
                            data = _load_adapter(cache_path)
                            return data
                        except Exception as retry_e:
                            _cleanup_temp_files()
                            raise RuntimeError(
                                f"Failed to load adapter even after retry: {retry_e}"
                            ) from retry_e
                    raise RuntimeError(f"Failed to load adapter: {e}") from e

        # For Modal Volume paths, load directly via the path abstraction.
        elif adapter_id.startswith("modal://"):
            for attempt in range(2):
                logger.info("Loading adapter from Modal volume: %s", adapter_id)
                try:
                    data = _load_adapter(adapter_id)
                    return data
                except FileNotFoundError as e:
                    if attempt == 0:
                        logger.warning(
                            "Modal adapter load failed for %s; waiting %.1fs "
                            "before retrying: %s",
                            adapter_id,
                            MODAL_LOAD_RETRY_DELAY_SECONDS,
                            e,
                        )
                        time.sleep(MODAL_LOAD_RETRY_DELAY_SECONDS)
                        continue
                    logger.error(
                        "Failed to load adapter from Modal volume %s: %s",
                        adapter_id,
                        e,
                    )
                    raise
                except Exception as e:
                    logger.error(
                        "Failed to load adapter from Modal volume %s: %s",
                        adapter_id,
                        e,
                    )
                    raise

        # For local paths, load directly.  _load_adapter uses the Modal
        # Volume API for hot-checkpoint paths, bypassing FUSE entirely,
        # so the file doesn't need to be visible on the local mount.
        else:
            adapter_str = str(source_path)
            logger.info("Loading adapter from local path: %s", adapter_str)
            try:
                data = _load_adapter(source_path)
                return data
            except Exception as e:
                logger.error("Failed to load local adapter: %s", e)
                raise

    def download_json(
        self,
        json_id: str,
        target_path: StdPath | None = None,
    ) -> dict:
        """
        Download and parse a JSON file from S3 or local path.

        The json_id can be either:
        - S3 URI: s3://bucket/path/to/file.json
        - Local path: /path/to/local/file.json

        The Path abstraction automatically handles both cases.

        Args:
            json_id: The identifier/path of the JSON file (S3 URI or local path)
            target_path: Optional local path to save the downloaded JSON.
                        If not provided, returns parsed dict without saving.

        Returns:
            Parsed JSON content as a dictionary

        Raises:
            FileNotFoundError: If file doesn't exist
            RuntimeError: If download or parsing fails
        """
        source_path = Path(json_id)

        if not source_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_id}")

        # Read and parse JSON using the unified Path API
        try:
            content = source_path.read_text()
            data = json.loads(content)
        except Exception as e:
            raise RuntimeError(
                f"Failed to read or parse JSON from {json_id}: {e}"
            ) from e

        # Save to target path if provided
        if target_path is not None:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_path, "w") as f:
                json.dump(data, f, indent=2)

        return data

    def clear_cache(self) -> None:
        """Clear all cached adapters."""
        logger.info("Clearing adapter cache...")
        for file in self.cache_dir.glob("*.pt"):
            file.unlink()
        for file in self.cache_dir.glob("*.lock"):
            file.unlink()
        logger.info("Cache cleared")


# Global adapter manager instance
_global_adapter_manager: AdapterManager | None = None


def get_adapter_manager() -> AdapterManager:
    """Get or create the global adapter manager instance."""
    global _global_adapter_manager
    if _global_adapter_manager is None:
        _global_adapter_manager = AdapterManager()
    return _global_adapter_manager
