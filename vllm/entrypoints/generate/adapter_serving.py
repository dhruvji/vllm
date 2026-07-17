# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import hashlib
import os
import shutil
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from safetensors.torch import save_file as safetensors_save_file

from vllm import envs
from vllm.entrypoints.serve.engine.typing import AnyRequest
from vllm.exceptions import VLLMNotFoundError
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.utils.adapter_manager import get_adapter_manager
from vllm.v1.cartridge_loader import _resolve_glob, _spec_sources

logger = init_logger(__name__)


def _normalize_lora_runtime_path(lora_id: str) -> str:
    if lora_id.startswith("file://"):
        return lora_id.removeprefix("file://")
    return os.path.expanduser(lora_id)


def _requires_staged_lora_dir(lora_id: str) -> bool:
    if lora_id.startswith(("s3://", "gs://", "modal://")):
        return True
    path = Path(_normalize_lora_runtime_path(lora_id))
    return path.is_file() or path.suffix in {".safetensors", ".pt", ".bin"}


def _lora_config_source_ids(lora_id: str) -> list[str]:
    source_id = _normalize_lora_runtime_path(lora_id)
    source_path = Path(source_id)
    if source_path.suffix in {".safetensors", ".pt", ".bin"}:
        source_id = os.path.dirname(source_id) or "."
    return [
        os.path.join(source_id, "adapter_config.json"),
        os.path.join(source_id, "lora_config.json"),
    ]


class EngramAdapterServingMixin:
    """Request-body LoRA loading shared by upstream generation entrypoints."""

    def _init_engram_adapter_serving(self) -> None:
        self._prepared_lora_dirs: OrderedDict[str, str] = OrderedDict()
        self._prepared_lora_dirs_max_size = 128
        self._lora_staging_locks: dict[str, threading.Lock] = {}
        self._lora_staging_locks_meta = threading.Lock()

    def _prepare_staged_lora_dir(
        self,
        lora_id: str,
        force_redownload: bool,
    ) -> str:
        lora_hash = hashlib.sha256(lora_id.encode()).hexdigest()[:16]
        lora_cache_dir = envs.VLLM_LORA_RESOLVER_CACHE_DIR or tempfile.gettempdir()
        local_lora_dir = Path(lora_cache_dir) / lora_hash
        adapter_model_path = local_lora_dir / "adapter_model.safetensors"
        adapter_config_path = local_lora_dir / "adapter_config.json"

        def can_reuse() -> bool:
            cached = self._prepared_lora_dirs.get(lora_id)
            return (
                not force_redownload
                and cached == str(local_lora_dir)
                and adapter_model_path.exists()
                and adapter_config_path.exists()
            )

        if can_reuse():
            self._prepared_lora_dirs.move_to_end(lora_id)
            return str(local_lora_dir)

        with self._lora_staging_locks_meta:
            lock = self._lora_staging_locks.setdefault(lora_id, threading.Lock())

        with lock:
            if can_reuse():
                self._prepared_lora_dirs.move_to_end(lora_id)
                return str(local_lora_dir)

            self._prepared_lora_dirs.pop(lora_id, None)
            manager = get_adapter_manager()
            lora_data = manager.get_adapter(
                adapter_id=lora_id,
                force_redownload=force_redownload,
            )

            if local_lora_dir.exists():
                shutil.rmtree(local_lora_dir)
            local_lora_dir.mkdir(parents=True, exist_ok=True)
            safetensors_save_file(lora_data, str(adapter_model_path))

            config_ids = _lora_config_source_ids(lora_id)
            last_error: Exception | None = None
            for config_id in config_ids:
                try:
                    manager.download_json(
                        json_id=config_id,
                        target_path=adapter_config_path,
                    )
                    break
                except Exception as exc:
                    last_error = exc
            else:
                raise RuntimeError(
                    "Failed to download LoRA config from any known path "
                    f"{config_ids}: {last_error}"
                ) from last_error

            self._prepared_lora_dirs[lora_id] = str(local_lora_dir)
            self._prepared_lora_dirs.move_to_end(lora_id)
            while len(self._prepared_lora_dirs) > self._prepared_lora_dirs_max_size:
                self._prepared_lora_dirs.popitem(last=False)
            return str(local_lora_dir)

    async def _process_lora(
        self,
        lora_specs: list[dict[str, Any]] | None,
    ) -> LoRARequest | None:
        if not lora_specs:
            return None
        if not isinstance(lora_specs, list):
            raise ValueError("`adapters.lora` must be a list.")
        if len(lora_specs) > 1:
            logger.warning(
                "Multiple LoRA adapters detected (%d); using the first.",
                len(lora_specs),
            )

        lora_spec = lora_specs[0]
        if not isinstance(lora_spec, dict):
            raise ValueError("Each LoRA adapter spec must be an object.")
        sources = _spec_sources(lora_spec)
        if not sources:
            raise ValueError("LoRA adapter spec missing `id`/`sources`.")

        force_redownload = bool(lora_spec.get("force_redownload", False))
        resolved_id: str | None = None
        runtime_lora_path: str | None = None
        last_error: Exception | None = None
        for source_ref in sources:
            lora_id = source_ref.get("id")
            if not lora_id:
                continue
            try:
                candidate_id = await asyncio.to_thread(_resolve_glob, lora_id)
                if candidate_id is None:
                    continue
                if _requires_staged_lora_dir(candidate_id):
                    candidate_path = await asyncio.to_thread(
                        self._prepare_staged_lora_dir,
                        candidate_id,
                        force_redownload,
                    )
                else:
                    candidate_path = _normalize_lora_runtime_path(candidate_id)
                resolved_id = candidate_id
                runtime_lora_path = candidate_path
                break
            except FileNotFoundError as exc:
                logger.warning(
                    "LoRA source %s not found; trying next: %s", lora_id, exc
                )
                last_error = exc

        if resolved_id is None or runtime_lora_path is None:
            tried = [source.get("id") for source in sources]
            raise RuntimeError(f"LoRA not found in any source: {tried}") from last_error

        async with self.models.lora_resolver_lock[resolved_id]:
            if resolved_id in self.models.lora_requests and not force_redownload:
                return self.models.lora_requests[resolved_id]

            lora_int_id = (
                self.models.lora_requests[resolved_id].lora_int_id
                if resolved_id in self.models.lora_requests
                else self.models.lora_id_counter.inc(1)
            )
            lora_request = LoRARequest(
                lora_name=resolved_id,
                lora_int_id=lora_int_id,
                lora_path=runtime_lora_path,
                load_inplace=force_redownload,
            )
            if self.models.is_base_model(self.model_config.model):
                lora_request.base_model_name = self.model_config.model

            await self.engine_client.add_lora(lora_request)
            self.models.lora_requests[resolved_id] = lora_request
            return lora_request

    async def _maybe_get_adapters(
        self,
        request: AnyRequest,
        supports_default_mm_loras: bool = False,
    ) -> LoRARequest | None:
        if request.model in self.models.lora_requests:
            return self.models.lora_requests[request.model]

        adapters = getattr(request, "adapters", None)
        if isinstance(adapters, dict):
            lora_request = await self._process_lora(adapters.get("lora"))
            if lora_request is not None:
                return lora_request

        if supports_default_mm_loras:
            default_mm_lora = self._get_active_default_mm_loras(request)
            if default_mm_lora is not None:
                return default_mm_lora

        if self._is_model_supported(request.model):
            return None
        raise VLLMNotFoundError(f"The model `{request.model}` does not exist.")
