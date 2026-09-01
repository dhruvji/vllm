# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm.config.multimodal import MultiModalConfig
from vllm.entrypoints.generate.adapter_serving import (
    EngramAdapterServingMixin,
    _lora_config_source_ids,
)
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.lora.request import LoRARequest
from vllm.lora.resolver import LoRAResolver, LoRAResolverRegistry
from vllm.renderers.hf import HfRenderer
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.tokenizers.registry import cached_tokenizer_from_config
from vllm.v1.engine.async_llm import AsyncLLM

MODEL_NAME = "openai-community/gpt2"
BASE_MODEL_PATHS = [BaseModelPath(name=MODEL_NAME, model_path=MODEL_NAME)]

MOCK_RESOLVER_NAME = "mock_test_resolver"


@pytest.mark.parametrize(
    ("lora_id", "expected_config_ids"),
    [
        (
            "s3://bucket/path/adapter_model.safetensors",
            [
                "s3://bucket/path/adapter_config.json",
                "s3://bucket/path/lora_config.json",
            ],
        ),
        (
            "s3://bucket/path/adapter.v1.safetensors",
            [
                "s3://bucket/path/adapter_config.json",
                "s3://bucket/path/lora_config.json",
            ],
        ),
        (
            "modal://volume/run/adapter_model.pt",
            [
                "modal://volume/run/adapter_config.json",
                "modal://volume/run/lora_config.json",
            ],
        ),
        (
            "file:///tmp/run/adapter_model.bin",
            ["/tmp/run/adapter_config.json", "/tmp/run/lora_config.json"],
        ),
        (
            "/tmp/run/lora-dir",
            [
                "/tmp/run/lora-dir/adapter_config.json",
                "/tmp/run/lora-dir/lora_config.json",
            ],
        ),
    ],
)
def test_request_body_lora_config_sources_use_file_parent(lora_id, expected_config_ids):
    assert _lora_config_source_ids(lora_id) == expected_config_ids


@dataclass
class MockHFConfig:
    model_type: str = "any"


@dataclass
class MockModelConfig:
    """Minimal mock ModelConfig for testing."""

    model: str = MODEL_NAME
    runner_type = "generate"
    tokenizer: str = MODEL_NAME
    trust_remote_code: bool = False
    tokenizer_mode: str = "auto"
    max_model_len: int = 100
    tokenizer_revision: str | None = None
    multimodal_config: MultiModalConfig = field(default_factory=MultiModalConfig)
    hf_config: MockHFConfig = field(default_factory=MockHFConfig)
    logits_processors: list[str] | None = None
    diff_sampling_param: dict | None = None
    allowed_local_media_path: str = ""
    allowed_media_domains: list[str] | None = None
    encoder_config = None
    generation_config: str = "auto"
    skip_tokenizer_init: bool = False
    is_encoder_decoder: bool = False
    is_multimodal_model: bool = False
    renderer_num_workers: int = 1

    def get_diff_sampling_param(self):
        return self.diff_sampling_param or {}


@dataclass
class MockParallelConfig:
    _api_process_rank: int = 0


@dataclass
class MockVllmConfig:
    model_config: MockModelConfig
    parallel_config: MockParallelConfig


class MockLoRAResolver(LoRAResolver):
    async def resolve_lora(
        self, base_model_name: str, lora_name: str
    ) -> LoRARequest | None:
        if lora_name == "test-lora":
            return LoRARequest(
                lora_name="test-lora",
                lora_int_id=1,
                lora_path="/fake/path/test-lora",
            )
        elif lora_name == "invalid-lora":
            return LoRARequest(
                lora_name="invalid-lora",
                lora_int_id=2,
                lora_path="/fake/path/invalid-lora",
            )
        return None


@pytest.fixture(autouse=True)
def register_mock_resolver():
    """Fixture to register and unregister the mock LoRA resolver."""
    resolver = MockLoRAResolver()
    LoRAResolverRegistry.register_resolver(MOCK_RESOLVER_NAME, resolver)
    yield
    # Cleanup: remove the resolver after the test runs
    if MOCK_RESOLVER_NAME in LoRAResolverRegistry.resolvers:
        del LoRAResolverRegistry.resolvers[MOCK_RESOLVER_NAME]


def _build_renderer(model_config: MockModelConfig):
    return HfRenderer(
        MockVllmConfig(model_config, parallel_config=MockParallelConfig()),
        cached_tokenizer_from_config(model_config),
    )


@pytest.fixture
def mock_serving_setup():
    """Provides a mocked engine and serving completion instance."""
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False

    async def mock_add_lora_side_effect(lora_request: LoRARequest):
        """Simulate engine behavior when adding LoRAs."""
        if lora_request.lora_name == "test-lora":
            # Simulate successful addition
            return True
        if lora_request.lora_name == "invalid-lora":
            # Simulate failure during addition (e.g. invalid format)
            raise ValueError(f"Simulated failure adding LoRA: {lora_request.lora_name}")
        return True

    mock_engine.add_lora = AsyncMock(side_effect=mock_add_lora_side_effect)

    async def mock_generate(*args, **kwargs):
        for _ in []:
            yield _

    mock_engine.generate = MagicMock(spec=AsyncLLM.generate, side_effect=mock_generate)

    mock_engine.generate.reset_mock()
    mock_engine.add_lora.reset_mock()

    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.renderer = _build_renderer(mock_engine.model_config)

    models = OpenAIServingModels(
        engine_client=mock_engine,
        base_model_paths=BASE_MODEL_PATHS,
    )

    online_renderer = OnlineRenderer(
        model_config=mock_engine.model_config,
        renderer=mock_engine.renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )
    serving_completion = OpenAIServingCompletion(
        mock_engine, models, online_renderer=online_renderer, request_logger=None
    )

    return mock_engine, serving_completion


@pytest.mark.asyncio
async def test_serving_completion_with_lora_resolver(mock_serving_setup, monkeypatch):
    monkeypatch.setenv("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "true")

    mock_engine, serving_completion = mock_serving_setup

    lora_model_name = "test-lora"
    req_found = CompletionRequest(
        model=lora_model_name,
        prompt="Generate with LoRA",
    )

    # Suppress potential errors during the mocked generate call,
    # as we are primarily checking for add_lora and generate calls
    with suppress(Exception):
        await serving_completion.create_completion(req_found)

    mock_engine.add_lora.assert_awaited_once()
    called_lora_request = mock_engine.add_lora.call_args[0][0]
    assert isinstance(called_lora_request, LoRARequest)
    assert called_lora_request.lora_name == lora_model_name

    mock_engine.generate.assert_called_once()
    called_lora_request = mock_engine.generate.call_args[1]["lora_request"]
    assert isinstance(called_lora_request, LoRARequest)
    assert called_lora_request.lora_name == lora_model_name


@pytest.mark.asyncio
async def test_serving_completion_with_request_body_lora(mock_serving_setup):
    mock_engine, serving_completion = mock_serving_setup

    req = CompletionRequest(
        model=MODEL_NAME,
        prompt="Generate with request-body LoRA",
        adapters={
            "lora": [{"id": "/fake/path/request-body-lora"}],
        },
    )

    with suppress(Exception):
        await serving_completion.create_completion(req)

    mock_engine.add_lora.assert_awaited_once()
    added_lora_request = mock_engine.add_lora.call_args[0][0]
    assert isinstance(added_lora_request, LoRARequest)
    assert added_lora_request.lora_name == "/fake/path/request-body-lora"
    assert added_lora_request.lora_path == "/fake/path/request-body-lora"

    mock_engine.generate.assert_called_once()
    generated_lora_request = mock_engine.generate.call_args[1]["lora_request"]
    assert isinstance(generated_lora_request, LoRARequest)
    assert generated_lora_request.lora_name == "/fake/path/request-body-lora"
    assert generated_lora_request.lora_path == "/fake/path/request-body-lora"


@pytest.mark.asyncio
@pytest.mark.skip_global_cleanup
async def test_request_body_structural_expert_requires_feature_flag(monkeypatch):
    serving = EngramAdapterServingMixin()
    serving._init_engram_adapter_serving()
    adapter_path = "/fake/path/structural-expert"
    counter = MagicMock()
    counter.inc.return_value = 7
    serving.models = SimpleNamespace(
        lora_resolver_lock=defaultdict(asyncio.Lock),
        lora_requests={},
        lora_id_counter=counter,
        is_base_model=lambda _model: True,
    )
    serving.model_config = SimpleNamespace(model=MODEL_NAME)
    serving.engine_client = SimpleNamespace(
        add_lora=AsyncMock(),
        vllm_config=SimpleNamespace(
            lora_config=SimpleNamespace(enable_structural_experts=False)
        ),
    )
    monkeypatch.setattr(
        "vllm.entrypoints.generate.adapter_serving._resolve_glob",
        lambda expert_id: expert_id,
    )

    async def run_inline(function, *args):
        return function(*args)

    monkeypatch.setattr(
        "vllm.entrypoints.generate.adapter_serving.asyncio.to_thread",
        run_inline,
    )

    with pytest.raises(ValueError, match="--enable-structural-experts"):
        await serving._process_expert([{"id": adapter_path}])

    serving.engine_client.add_lora.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.skip_global_cleanup
async def test_request_body_structural_expert_uses_lora_scheduler_slot(monkeypatch):
    serving = EngramAdapterServingMixin()
    serving._init_engram_adapter_serving()
    counter = MagicMock()
    counter.inc.return_value = 7
    adapter_path = "/fake/path/structural-expert"
    ordinary_lora = LoRARequest(
        lora_name=adapter_path,
        lora_int_id=3,
        lora_path=adapter_path,
    )
    serving.models = SimpleNamespace(
        lora_resolver_lock=defaultdict(asyncio.Lock),
        lora_requests={adapter_path: ordinary_lora},
        lora_id_counter=counter,
        is_base_model=lambda _model: True,
    )
    serving.model_config = SimpleNamespace(model=MODEL_NAME)
    serving.engine_client = SimpleNamespace(
        add_lora=AsyncMock(),
        vllm_config=SimpleNamespace(
            lora_config=SimpleNamespace(enable_structural_experts=True)
        ),
    )
    monkeypatch.setattr(
        "vllm.entrypoints.generate.adapter_serving._resolve_glob",
        lambda expert_id: expert_id,
    )

    async def run_inline(function, *args):
        return function(*args)

    monkeypatch.setattr(
        "vllm.entrypoints.generate.adapter_serving.asyncio.to_thread",
        run_inline,
    )

    request = await serving._process_expert([{"id": adapter_path}])

    assert request is not None
    assert request.is_structural_expert
    assert request.lora_int_id == 7
    assert request.lora_path == adapter_path
    assert request != ordinary_lora
    assert len({request, ordinary_lora}) == 2
    assert serving.models.lora_requests[adapter_path] is ordinary_lora
    serving.engine_client.add_lora.assert_awaited_once_with(request)


@pytest.mark.skip_global_cleanup
def test_staged_adapter_cache_reuses_durable_disk_path(monkeypatch, tmp_path):
    monkeypatch.delenv("VLLM_LORA_RESOLVER_CACHE_DIR", raising=False)
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    serving = EngramAdapterServingMixin()
    serving._init_engram_adapter_serving()

    staging_root = serving._staged_adapter_dir("expert", "first")
    assert staging_root.parent == tmp_path / "adapters" / "staged"
    staged_dir = staging_root / "generation-existing"
    staged_dir.mkdir(parents=True)
    model_path = staged_dir / "adapter_model.safetensors"
    model_path.touch()
    (staged_dir / ".ready").touch()
    (staging_root / "current").write_text(staged_dir.name)
    manager = MagicMock()
    monkeypatch.setattr(
        "vllm.entrypoints.generate.adapter_serving.get_adapter_manager", manager
    )

    resolved = serving._prepare_staged_expert_dir("first", force_redownload=False)

    assert resolved == str(staged_dir)
    assert model_path.exists()
    manager.get_adapter.assert_not_called()


@pytest.mark.skip_global_cleanup
def test_staged_lora_publishes_complete_directory_before_reuse(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    serving = EngramAdapterServingMixin()
    serving._init_engram_adapter_serving()
    manager = MagicMock()
    manager.get_adapter.return_value = {}

    def download_json(*, json_id, target_path):
        Path(target_path).write_text(json_id)

    manager.download_json = MagicMock(side_effect=download_json)
    monkeypatch.setattr(
        "vllm.entrypoints.generate.adapter_serving.get_adapter_manager",
        lambda: manager,
    )
    monkeypatch.setattr(
        "vllm.entrypoints.generate.adapter_serving.safetensors_save_file",
        lambda _data, path: Path(path).write_bytes(b"model"),
    )

    staged = Path(serving._prepare_staged_lora_dir("s3://bucket/adapter.bin", False))

    assert (staged / "adapter_model.safetensors").read_bytes() == b"model"
    assert (staged / "adapter_config.json").is_file()
    assert (staged / ".ready").is_file()

    manager.reset_mock()
    assert serving._prepare_staged_lora_dir("s3://bucket/adapter.bin", False) == str(
        staged
    )
    manager.get_adapter.assert_not_called()
    manager.download_json.assert_not_called()


@pytest.mark.skip_global_cleanup
def test_staged_adapter_lock_serializes_shared_disk_writes(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    first_save_started = threading.Event()
    release_first_save = threading.Event()
    second_save_started = threading.Event()
    save_call_lock = threading.Lock()
    save_calls = 0

    def save_adapter(_data, path):
        nonlocal save_calls
        with save_call_lock:
            save_calls += 1
            call = save_calls
        if call == 1:
            first_save_started.set()
            assert release_first_save.wait(timeout=5)
        else:
            second_save_started.set()
        Path(path).touch()

    manager = MagicMock()
    manager.get_adapter.return_value = {}
    monkeypatch.setattr(
        "vllm.entrypoints.generate.adapter_serving.get_adapter_manager",
        lambda: manager,
    )
    monkeypatch.setattr(
        "vllm.entrypoints.generate.adapter_serving.safetensors_save_file",
        save_adapter,
    )

    first = EngramAdapterServingMixin()
    first._init_engram_adapter_serving()
    second = EngramAdapterServingMixin()
    second._init_engram_adapter_serving()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_result = executor.submit(first._prepare_staged_expert_dir, "shared", True)
        assert first_save_started.wait(timeout=5)
        second_result = executor.submit(
            second._prepare_staged_expert_dir, "shared", True
        )
        writes_overlapped = second_save_started.wait(timeout=0.5)
        release_first_save.set()
        first_result.result(timeout=5)
        second_result.result(timeout=5)

    assert not writes_overlapped


@pytest.mark.skip_global_cleanup
def test_force_redownload_publishes_immutable_adapter_generation(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    serving = EngramAdapterServingMixin()
    serving._init_engram_adapter_serving()
    manager = MagicMock()
    manager.get_adapter.side_effect = [b"first", b"second"]
    monkeypatch.setattr(
        "vllm.entrypoints.generate.adapter_serving.get_adapter_manager",
        lambda: manager,
    )
    monkeypatch.setattr(
        "vllm.entrypoints.generate.adapter_serving.safetensors_save_file",
        lambda data, path: Path(path).write_bytes(data),
    )

    first_dir = Path(serving._prepare_staged_expert_dir("shared", False))
    second_dir = Path(serving._prepare_staged_expert_dir("shared", True))

    assert second_dir != first_dir
    assert (first_dir / "adapter_model.safetensors").read_bytes() == b"first"
    assert (second_dir / "adapter_model.safetensors").read_bytes() == b"second"
    assert serving._prepare_staged_expert_dir("shared", False) == str(second_dir)


@pytest.mark.skip_global_cleanup
def test_failed_force_redownload_preserves_published_adapter(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    serving = EngramAdapterServingMixin()
    serving._init_engram_adapter_serving()
    manager = MagicMock()
    manager.get_adapter.return_value = {}
    monkeypatch.setattr(
        "vllm.entrypoints.generate.adapter_serving.get_adapter_manager",
        lambda: manager,
    )
    monkeypatch.setattr(
        "vllm.entrypoints.generate.adapter_serving.safetensors_save_file",
        lambda _data, path: Path(path).write_bytes(b"published"),
    )
    staged_dir = Path(
        serving._prepare_staged_expert_dir("shared", force_redownload=False)
    )
    model_path = staged_dir / "adapter_model.safetensors"

    def fail_save(_data, _path):
        raise RuntimeError("write failed")

    monkeypatch.setattr(
        "vllm.entrypoints.generate.adapter_serving.safetensors_save_file",
        fail_save,
    )

    with pytest.raises(RuntimeError, match="write failed"):
        serving._prepare_staged_expert_dir("shared", force_redownload=True)

    assert model_path.read_bytes() == b"published"
    assert (staged_dir / ".ready").exists()
    assert serving._prepare_staged_expert_dir("shared", False) == str(staged_dir)


@pytest.mark.asyncio
async def test_serving_completion_resolver_not_found(mock_serving_setup, monkeypatch):
    monkeypatch.setenv("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "true")

    mock_engine, serving_completion = mock_serving_setup

    non_existent_model = "non-existent-lora-adapter"
    req = CompletionRequest(
        model=non_existent_model,
        prompt="what is 1+1?",
    )

    response = await serving_completion.create_completion(req)

    mock_engine.add_lora.assert_not_awaited()
    mock_engine.generate.assert_not_called()

    assert isinstance(response, ErrorResponse)
    assert response.error.code == HTTPStatus.NOT_FOUND.value
    assert non_existent_model in response.error.message


@pytest.mark.asyncio
async def test_serving_completion_resolver_add_lora_fails(
    mock_serving_setup, monkeypatch
):
    monkeypatch.setenv("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "true")

    mock_engine, serving_completion = mock_serving_setup

    invalid_model = "invalid-lora"
    req = CompletionRequest(
        model=invalid_model,
        prompt="what is 1+1?",
    )

    response = await serving_completion.create_completion(req)

    # Assert add_lora was called before the failure
    mock_engine.add_lora.assert_awaited_once()
    called_lora_request = mock_engine.add_lora.call_args[0][0]
    assert isinstance(called_lora_request, LoRARequest)
    assert called_lora_request.lora_name == invalid_model

    # Assert generate was *not* called due to the failure
    mock_engine.generate.assert_not_called()

    # Assert the correct error response
    assert isinstance(response, ErrorResponse)
    assert response.error.code == HTTPStatus.BAD_REQUEST.value
    assert invalid_model in response.error.message


@pytest.mark.asyncio
async def test_serving_completion_flag_not_set(mock_serving_setup):
    mock_engine, serving_completion = mock_serving_setup

    lora_model_name = "test-lora"
    req_found = CompletionRequest(
        model=lora_model_name,
        prompt="Generate with LoRA",
    )

    await serving_completion.create_completion(req_found)

    mock_engine.add_lora.assert_not_called()
    mock_engine.generate.assert_not_called()
