# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import hashlib
import os
import shutil
import threading
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from safetensors.torch import save_file as safetensors_save_file

from vllm import envs
from vllm.entrypoints.pooling.typing import AnyPoolingRequest
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

    # Supplied by GenerateBaseServing. Keeping these structural declarations
    # here lets the mixin remain independent of the concrete serving class.
    models: Any
    model_config: Any
    engine_client: Any

    if TYPE_CHECKING:

        def _get_active_default_mm_loras(
            self, request: AnyRequest | AnyPoolingRequest
        ) -> LoRARequest | None: ...

        def _is_model_supported(self, model_name: str | None) -> bool: ...

    def _init_engram_adapter_serving(self) -> None:
        self._prepared_lora_dirs: OrderedDict[tuple[str, str], str] = OrderedDict()
        self._prepared_lora_dirs_max_size = 128
        self._lora_staging_locks: dict[tuple[str, str], threading.Lock] = {}
        self._lora_staging_locks_meta = threading.Lock()
        self._structural_expert_requests: dict[str, LoRARequest] = {}

    def _staged_adapter_dir(self, kind: str, adapter_id: str) -> Path:
        cache_root = Path(
            envs.VLLM_LORA_RESOLVER_CACHE_DIR
            or Path(envs.VLLM_CACHE_ROOT) / "adapters" / "staged"
        )
        adapter_hash = hashlib.sha256(f"{kind}:{adapter_id}".encode()).hexdigest()[:16]
        return cache_root / adapter_hash

    def _remember_staged_dir(
        self, cache_key: tuple[str, str], staged_dir: Path
    ) -> None:
        self._prepared_lora_dirs[cache_key] = str(staged_dir)
        self._prepared_lora_dirs.move_to_end(cache_key)
        while len(self._prepared_lora_dirs) > self._prepared_lora_dirs_max_size:
            _, evicted_dir = self._prepared_lora_dirs.popitem(last=False)
            shutil.rmtree(evicted_dir, ignore_errors=True)

    def _prepare_staged_lora_dir(
        self,
        lora_id: str,
        force_redownload: bool,
    ) -> str:
        cache_key = ("lora", lora_id)
        local_lora_dir = self._staged_adapter_dir(*cache_key)
        adapter_model_path = local_lora_dir / "adapter_model.safetensors"
        adapter_config_path = local_lora_dir / "adapter_config.json"

        def can_reuse() -> bool:
            cached = self._prepared_lora_dirs.get(cache_key)
            return (
                not force_redownload
                and cached == str(local_lora_dir)
                and adapter_model_path.exists()
                and adapter_config_path.exists()
            )

        if can_reuse():
            self._prepared_lora_dirs.move_to_end(cache_key)
            return str(local_lora_dir)

        with self._lora_staging_locks_meta:
            lock = self._lora_staging_locks.setdefault(cache_key, threading.Lock())

        with lock:
            if can_reuse():
                self._prepared_lora_dirs.move_to_end(cache_key)
                return str(local_lora_dir)

            self._prepared_lora_dirs.pop(cache_key, None)
            manager = get_adapter_manager()
            lora_data = manager.get_adapter(
                adapter_id=lora_id,
                force_redownload=force_redownload,
            )

            try:
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
            except Exception:
                shutil.rmtree(local_lora_dir, ignore_errors=True)
                raise

            self._remember_staged_dir(cache_key, local_lora_dir)
            return str(local_lora_dir)

    def _prepare_staged_expert_dir(
        self,
        expert_id: str,
        force_redownload: bool,
    ) -> str:
        cache_key = ("expert", expert_id)
        local_dir = self._staged_adapter_dir(*cache_key)
        model_path = local_dir / "adapter_model.safetensors"

        cached = self._prepared_lora_dirs.get(cache_key)
        if not force_redownload and cached == str(local_dir) and model_path.exists():
            self._prepared_lora_dirs.move_to_end(cache_key)
            return str(local_dir)

        with self._lora_staging_locks_meta:
            lock = self._lora_staging_locks.setdefault(cache_key, threading.Lock())
        with lock:
            cached = self._prepared_lora_dirs.get(cache_key)
            if (
                not force_redownload
                and cached == str(local_dir)
                and model_path.exists()
            ):
                self._prepared_lora_dirs.move_to_end(cache_key)
                return str(local_dir)

            expert_data = get_adapter_manager().get_adapter(
                adapter_id=expert_id,
                force_redownload=force_redownload,
            )
            try:
                if local_dir.exists():
                    shutil.rmtree(local_dir)
                local_dir.mkdir(parents=True, exist_ok=True)
                safetensors_save_file(expert_data, str(model_path))
            except Exception:
                shutil.rmtree(local_dir, ignore_errors=True)
                raise
            self._remember_staged_dir(cache_key, local_dir)
            return str(local_dir)

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

    async def _process_expert(
        self,
        expert_specs: list[dict[str, Any]] | None,
    ) -> LoRARequest | None:
        if not expert_specs:
            return None
        if not isinstance(expert_specs, list) or len(expert_specs) != 1:
            raise ValueError("`adapters.expert` must contain exactly one adapter.")
        expert_spec = expert_specs[0]
        if not isinstance(expert_spec, dict):
            raise ValueError("Each structural expert spec must be an object.")
        sources = _spec_sources(expert_spec)
        if not sources:
            raise ValueError("Structural expert spec missing `id`/`sources`.")

        force_redownload = bool(expert_spec.get("force_redownload", False))
        resolved_id: str | None = None
        runtime_path: str | None = None
        last_error: Exception | None = None
        for source_ref in sources:
            expert_id = source_ref.get("id")
            if not expert_id:
                continue
            try:
                candidate_id = await asyncio.to_thread(_resolve_glob, expert_id)
                if candidate_id is None:
                    continue
                if _requires_staged_lora_dir(candidate_id):
                    candidate_path = await asyncio.to_thread(
                        self._prepare_staged_expert_dir,
                        candidate_id,
                        force_redownload,
                    )
                else:
                    candidate_path = _normalize_lora_runtime_path(candidate_id)
                resolved_id = candidate_id
                runtime_path = candidate_path
                break
            except FileNotFoundError as exc:
                last_error = exc

        if resolved_id is None or runtime_path is None:
            tried = [source.get("id") for source in sources]
            raise RuntimeError(
                f"Structural expert not found in any source: {tried}"
            ) from last_error

        lock_key = f"structural-expert:{resolved_id}"
        async with self.models.lora_resolver_lock[lock_key]:
            if resolved_id in self._structural_expert_requests and not force_redownload:
                return self._structural_expert_requests[resolved_id]
            lora_int_id = (
                self._structural_expert_requests[resolved_id].lora_int_id
                if resolved_id in self._structural_expert_requests
                else self.models.lora_id_counter.inc(1)
            )
            request = LoRARequest(
                lora_name=resolved_id,
                lora_int_id=lora_int_id,
                lora_path=runtime_path,
                load_inplace=force_redownload,
                is_structural_expert=True,
            )
            if self.models.is_base_model(self.model_config.model):
                request.base_model_name = self.model_config.model
            await self.engine_client.add_lora(request)
            self._structural_expert_requests[resolved_id] = request
            return request

    async def _maybe_get_engram_adapters(
        self,
        request: AnyRequest,
        supports_default_mm_loras: bool = False,
    ) -> LoRARequest | None:
        if request.model in self.models.lora_requests:
            return self.models.lora_requests[request.model]

        adapters = getattr(request, "adapters", None)
        if isinstance(adapters, dict):
            if adapters.get("lora") and adapters.get("expert"):
                raise ValueError(
                    "A request may select one LoRA or one structural expert, not both."
                )
            expert_request = await self._process_expert(adapters.get("expert"))
            if expert_request is not None:
                return expert_request
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
