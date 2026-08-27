# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import filelock
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
        self._structural_expert_requests: dict[str, LoRARequest] = {}

    def _staged_adapter_dir(self, kind: str, adapter_id: str) -> Path:
        cache_root = Path(
            envs.VLLM_LORA_RESOLVER_CACHE_DIR
            or Path(envs.VLLM_CACHE_ROOT) / "adapters" / "staged"
        )
        adapter_hash = hashlib.sha256(f"{kind}:{adapter_id}".encode()).hexdigest()[:16]
        return cache_root / adapter_hash

    @staticmethod
    def _staged_adapter_lock(staged_dir: Path) -> filelock.FileLock:
        staged_dir.parent.mkdir(parents=True, exist_ok=True)
        return filelock.FileLock(f"{staged_dir}.lock", timeout=300)

    @staticmethod
    def _current_staged_adapter_dir(
        staging_root: Path,
        required_files: tuple[str, ...],
    ) -> Path | None:
        try:
            generation = (staging_root / "current").read_text().strip()
        except FileNotFoundError:
            return None
        if (
            not generation.startswith("generation-")
            or Path(generation).name != generation
        ):
            return None
        published_dir = staging_root / generation
        if not (published_dir / ".ready").is_file():
            return None
        if not all((published_dir / filename).is_file() for filename in required_files):
            return None
        return published_dir

    @staticmethod
    def _publish_staged_adapter_dir(
        staging_root: Path,
        temp_dir: Path,
    ) -> Path:
        (temp_dir / ".ready").touch()
        published_dir = staging_root / temp_dir.name.removeprefix(".")
        os.rename(temp_dir, published_dir)

        file_descriptor, current_temp_name = tempfile.mkstemp(
            dir=staging_root,
            prefix=".current.",
        )
        current_temp_path = Path(current_temp_name)
        try:
            with os.fdopen(file_descriptor, "w") as current_file:
                current_file.write(published_dir.name)
            os.replace(current_temp_path, staging_root / "current")
        finally:
            current_temp_path.unlink(missing_ok=True)
        return published_dir

    def _prepare_staged_lora_dir(
        self,
        lora_id: str,
        force_redownload: bool,
    ) -> str:
        staging_root = self._staged_adapter_dir("lora", lora_id)
        required_files = ("adapter_model.safetensors", "adapter_config.json")

        with self._staged_adapter_lock(staging_root):
            current_dir = self._current_staged_adapter_dir(staging_root, required_files)
            if not force_redownload and current_dir is not None:
                return str(current_dir)

            manager = get_adapter_manager()
            lora_data = manager.get_adapter(
                adapter_id=lora_id,
                force_redownload=force_redownload,
            )

            staging_root.mkdir(parents=True, exist_ok=True)
            temp_dir = Path(
                tempfile.mkdtemp(
                    dir=staging_root,
                    prefix=".generation-",
                )
            )
            temp_model_path = temp_dir / required_files[0]
            temp_config_path = temp_dir / required_files[1]
            try:
                safetensors_save_file(lora_data, str(temp_model_path))

                config_ids = _lora_config_source_ids(lora_id)
                last_error: Exception | None = None
                for config_id in config_ids:
                    try:
                        manager.download_json(
                            json_id=config_id,
                            target_path=temp_config_path,
                        )
                        break
                    except Exception as exc:
                        last_error = exc
                else:
                    raise RuntimeError(
                        "Failed to download LoRA config from any known path "
                        f"{config_ids}: {last_error}"
                    ) from last_error

                published_dir = self._publish_staged_adapter_dir(staging_root, temp_dir)
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

            return str(published_dir)

    def _prepare_staged_expert_dir(
        self,
        expert_id: str,
        force_redownload: bool,
    ) -> str:
        staging_root = self._staged_adapter_dir("expert", expert_id)
        required_files = ("adapter_model.safetensors",)

        with self._staged_adapter_lock(staging_root):
            current_dir = self._current_staged_adapter_dir(staging_root, required_files)
            if not force_redownload and current_dir is not None:
                return str(current_dir)

            expert_data = get_adapter_manager().get_adapter(
                adapter_id=expert_id,
                force_redownload=force_redownload,
            )
            staging_root.mkdir(parents=True, exist_ok=True)
            temp_dir = Path(
                tempfile.mkdtemp(
                    dir=staging_root,
                    prefix=".generation-",
                )
            )
            temp_model_path = temp_dir / required_files[0]
            try:
                safetensors_save_file(expert_data, str(temp_model_path))
                published_dir = self._publish_staged_adapter_dir(staging_root, temp_dir)
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
            return str(published_dir)

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
        lora_config = self.engine_client.vllm_config.lora_config
        if lora_config is None or not lora_config.enable_structural_experts:
            raise ValueError(
                "Structural expert adapters require `--enable-structural-experts`."
            )
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
