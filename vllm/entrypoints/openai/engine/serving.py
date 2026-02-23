# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import json
import sys
import time
import traceback
from collections.abc import AsyncGenerator, Callable, Mapping
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any, ClassVar, Generic, Protocol, TypeAlias, TypeVar

import numpy as np
import torch
from fastapi import Request
from openai.types.responses import (
    ToolChoiceFunction,
)
from pydantic import ConfigDict, TypeAdapter
from starlette.datastructures import Headers

import vllm.envs as envs
from vllm.beam_search import BeamSearchSequence, create_sort_beams_key_function
from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ChatTemplateContentFormatOption,
    ConversationMessage,
)
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
    CompletionResponse,
)
from vllm.entrypoints.openai.engine.protocol import (
    ErrorInfo,
    ErrorResponse,
    FunctionCall,
    FunctionDefinition,
)
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.openai.responses.context import (
    ConversationContext,
    HarmonyContext,
    ParsableContext,
    StreamingHarmonyContext,
)
from vllm.entrypoints.openai.responses.protocol import (
    ResponseInputOutputItem,
    ResponsesRequest,
)
from vllm.entrypoints.openai.responses.utils import (
    construct_input_messages,
)
from vllm.entrypoints.openai.speech_to_text.protocol import (
    TranscriptionRequest,
    TranscriptionResponse,
    TranslationRequest,
)
from vllm.entrypoints.pooling.classify.protocol import (
    ClassificationChatRequest,
    ClassificationCompletionRequest,
    ClassificationResponse,
)
from vllm.entrypoints.pooling.embed.protocol import (
    EmbeddingBytesResponse,
    EmbeddingChatRequest,
    EmbeddingCompletionRequest,
    EmbeddingResponse,
)
from vllm.entrypoints.pooling.pooling.protocol import (
    IOProcessorRequest,
    PoolingChatRequest,
    PoolingCompletionRequest,
    PoolingResponse,
)
from vllm.entrypoints.pooling.score.protocol import (
    RerankRequest,
    ScoreDataRequest,
    ScoreQueriesDocumentsRequest,
    ScoreRequest,
    ScoreResponse,
    ScoreTextRequest,
)
from vllm.entrypoints.serve.disagg.protocol import GenerateRequest, GenerateResponse
from vllm.entrypoints.serve.tokenize.protocol import (
    DetokenizeRequest,
    TokenizeChatRequest,
    TokenizeCompletionRequest,
    TokenizeResponse,
)
from vllm.entrypoints.utils import get_max_tokens, sanitize_message
from vllm.exceptions import VLLMValidationError
from vllm.inputs.data import PromptType, SingletonPrompt, TokensPrompt
from vllm.logger import init_logger
from vllm.logprobs import Logprob, PromptLogprobs
from vllm.lora.request import LoRARequest
from vllm.multimodal import MultiModalDataDict
from vllm.outputs import CompletionOutput, PoolingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.renderers import ChatParams, TokenizeParams, merge_kwargs
from vllm.renderers.inputs import TokPrompt
from vllm.renderers.inputs.preprocess import (
    extract_prompt_components,
    extract_prompt_len,
    parse_model_prompt,
    prompt_to_seq,
)
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers import ToolParser
from vllm.tracing import (
    contains_trace_headers,
    extract_trace_headers,
    log_tracing_disabled_warning,
)
from vllm.utils import random_uuid
from vllm.utils.async_utils import (
    collect_from_async_generator,
    merge_async_iterators,
)


class GenerationError(Exception):
    """raised when finish_reason indicates internal server error (500)"""

    def __init__(self, message: str = "Internal server error"):
        super().__init__(message)
        self.status_code = HTTPStatus.INTERNAL_SERVER_ERROR


logger = init_logger(__name__)


class RendererRequest(Protocol):
    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams:
        raise NotImplementedError


class RendererChatRequest(RendererRequest, Protocol):
    def build_chat_params(
        self,
        default_template: str | None,
        default_template_content_format: ChatTemplateContentFormatOption,
    ) -> ChatParams:
        raise NotImplementedError


CompletionLikeRequest: TypeAlias = (
    CompletionRequest
    | TokenizeCompletionRequest
    | DetokenizeRequest
    | EmbeddingCompletionRequest
    | ClassificationCompletionRequest
    | RerankRequest
    | ScoreRequest
    | PoolingCompletionRequest
)

ChatLikeRequest: TypeAlias = (
    ChatCompletionRequest
    | TokenizeChatRequest
    | EmbeddingChatRequest
    | ClassificationChatRequest
    | PoolingChatRequest
)

SpeechToTextRequest: TypeAlias = TranscriptionRequest | TranslationRequest

AnyRequest: TypeAlias = (
    CompletionLikeRequest
    | ChatLikeRequest
    | SpeechToTextRequest
    | ResponsesRequest
    | IOProcessorRequest
    | GenerateRequest
)

AnyResponse: TypeAlias = (
    CompletionResponse
    | ChatCompletionResponse
    | EmbeddingResponse
    | EmbeddingBytesResponse
    | TranscriptionResponse
    | TokenizeResponse
    | PoolingResponse
    | ClassificationResponse
    | ScoreResponse
    | GenerateResponse
)


RequestT = TypeVar("RequestT", bound=AnyRequest)


@dataclass(kw_only=True)
class ServeContext(Generic[RequestT]):
    request: RequestT
    raw_request: Request | None = None
    model_name: str
    request_id: str
    created_time: int = field(default_factory=lambda: int(time.time()))
    lora_request: LoRARequest | None = None
    engine_prompts: list[TokPrompt] | None = None

    result_generator: AsyncGenerator[tuple[int, PoolingRequestOutput], None] | None = (
        None
    )
    final_res_batch: list[PoolingRequestOutput] = field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class OpenAIServing:
    request_id_prefix: ClassVar[str] = """
    A short string prepended to every request’s ID (e.g. "embd", "classify")
    so you can easily tell “this ID came from Embedding vs Classification.”
    """

    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        return_tokens_as_token_ids: bool = False,
        log_error_stack: bool = False,
    ):
        super().__init__()

        self.engine_client = engine_client

        self.models = models

        self.request_logger = request_logger
        self.return_tokens_as_token_ids = return_tokens_as_token_ids

        self.log_error_stack = log_error_stack

        self.input_processor = self.models.input_processor
        self.io_processor = self.models.io_processor
        self.renderer = self.models.renderer
        self.model_config = self.models.model_config
        self.max_model_len = self.model_config.max_model_len
        # Track cartridges that have already sent KV to engine core.
        # Bounded to prevent unbounded memory growth with frequent checkpoint updates.
        from collections import OrderedDict
        self._sent_cartridge_ids: OrderedDict[str, str | None] = OrderedDict()
        self._sent_cartridge_ids_max_size: int = 8

    async def _process_adapters(
        self,
        adapters_config: dict[str, Any] | None,
        prompt_token_ids: list[int],
        request_id: str | None = None,
    ) -> tuple[
        list[int],
        list["torch.Tensor"] | None,
        str | None,
        int | None,
        str | None,
        bool | None,
        LoRARequest | None,
    ]:
        """
        Process all adapters (prefix, lora) from the request.

        Args:
            adapters_config: New adapters config with 'prefix' and 'lora' keys
            prompt_token_ids: Original prompt token IDs
            request_id: Request ID to associate with adapters

        Returns:
            Tuple of:
            - Combined token IDs with prefix tokens prepended (for pre-computed),
              or original prompt_token_ids (for learned prefix)
            - Stacked prefix KV tensors [keys, values] for learned prefix,
              or None if using shm
            - Prefix ID for deduplication, or None if no learned prefix
            - Prefix sequence length, or None if no learned prefix
            - Path to prefix KV in shared memory, or None if not saved
            - Whether this request allows cartridge pipelining
            - LoRARequest object if LoRA adapters specified, or None
        """
        if not adapters_config:
            return prompt_token_ids, None, None, None, None, None, None

        # Process prefix adapters (cartridges)
        prefix_specs = adapters_config.get("prefix")
        (
            prompt_token_ids,
            prefix_kv,
            prefix_id,
            prefix_seq_len,
            prefix_shm_path,
            allow_pipelining,
        ) = await self._process_cartridges(
            cartridges=prefix_specs,
            prompt_token_ids=prompt_token_ids,
            request_id=request_id,
        )

        # Process LoRA adapters
        lora_specs = adapters_config.get("lora")
        lora_request = self._process_lora(
            lora_specs=lora_specs,
            request_id=request_id,
        )

        return (
            prompt_token_ids,
            prefix_kv,
            prefix_id,
            prefix_seq_len,
            prefix_shm_path,
            allow_pipelining,
            lora_request,
        )

    async def _process_cartridges(
        self,
        cartridges: list[dict[str, Any]] | None,
        prompt_token_ids: list[int],
        request_id: str | None = None,
    ) -> tuple[
        list[int],
        list["torch.Tensor"] | None,
        str | None,
        int | None,
        str | None,
        bool | None,
    ]:
        """
        Load cartridges and process them for the request.

        For pre-computed cartridges (with token_ids): prepend token IDs to prompt.
        For learned cartridges (soft prompts): saves KV to shared memory for fast IPC.

        Args:
            cartridges: List of cartridge specifications from the request
            prompt_token_ids: Original prompt token IDs
            request_id: Request ID to associate with learned cartridges

        Returns:
            Tuple of:
            - Combined token IDs with cartridge tokens prepended (for pre-computed),
              or original prompt_token_ids (for learned cartridges)
            - Stacked cartridge KV tensors [keys, values] for learned cartridges,
              or None (use shm_path instead for zero-copy)
            - Cartridge ID for deduplication, or None if no learned cartridge
            - Cartridge sequence length, or None if no learned cartridge
            - Path to cartridge KV in shared memory, or None if not saved
            - Whether this request allows cartridge pipelining
        """
        import asyncio

        import torch

        if not cartridges:
            return prompt_token_ids, None, None, None, None, None

        from vllm.logger import init_logger
        from vllm.v1.cartridge_loader import (
            load_cartridges_from_request,
            save_cartridge_to_shm,
        )

        logger = init_logger(__name__)

        try:
            # Load cartridges in a thread to avoid blocking the event loop
            # during volume reloads, file locks, and torch.load().
            loaded_cartridges = await asyncio.to_thread(
                load_cartridges_from_request, cartridges
            )

            # Separate learned and pre-computed cartridges
            learned_cartridges = []
            learned_cartridge_ids = []
            learned_cartridge_allow_pipelining = []
            precomputed_cartridges = []

            for i, cartridge_data in enumerate(loaded_cartridges):
                if cartridge_data.is_learned:
                    learned_cartridges.append(cartridge_data)
                    # Get cartridge ID from original spec
                    learned_cartridge_ids.append(
                        cartridges[i].get("id", f"unknown_{i}")
                    )
                    learned_cartridge_allow_pipelining.append(
                        bool(cartridges[i].get("allow_pipelining", False))
                    )
                elif cartridge_data.has_valid_token_ids():
                    precomputed_cartridges.append(cartridge_data)
                else:
                    logger.warning(
                        "Skipping cartridge with no valid token_ids "
                        "and not marked as learned"
                    )

            # Handle learned cartridges - save to shm for zero-copy IPC
            stacked_cartridge_kv: list[torch.Tensor] | None = None
            cartridge_id: str | None = None
            cartridge_seq_len: int | None = None
            cartridge_shm_path: str | None = None
            allow_pipelining: bool | None = None
            if learned_cartridges:
                # Currently only supports one learned cartridge per request
                if len(learned_cartridges) > 1:
                    logger.warning(
                        "Multiple learned cartridges detected. "
                        "Only the first one will be used."
                    )

                cartridge_data = learned_cartridges[0]
                cartridge_id = learned_cartridge_ids[0]
                cartridge_seq_len = cartridge_data.num_tokens
                allow_pipelining = learned_cartridge_allow_pipelining[0]
                assert cartridge_id is not None  # Guaranteed by learned_cartridge_ids

                # Get stacked KV tensors and save to shared storage for zero-copy IPC.
                # Use volume-backed storage instead of /dev/shm for Modal compatibility.
                # Only save once per cartridge to avoid repeated disk writes.
                if cartridge_id in self._sent_cartridge_ids:
                    self._sent_cartridge_ids.move_to_end(cartridge_id)  # LRU touch
                    # Already saved to shared storage — re-send the shm path
                    # so engine core can reload if it evicted from its LRU cache.
                    stacked_cartridge_kv = None
                    cartridge_shm_path = self._sent_cartridge_ids[cartridge_id]
                else:
                    stacked_cartridge_kv = cartridge_data.get_stacked_kv()
                    if stacked_cartridge_kv is not None:
                        # Validate cartridge is compatible with this model
                        stacked_keys = stacked_cartridge_kv[0]
                        _, cart_kv_heads, cart_seq_len, cart_head_dim = stacked_keys.shape
                        expected_kv_heads = self.model_config.get_total_num_kv_heads()
                        expected_head_size = self.model_config.get_head_size()
                        if cart_kv_heads != expected_kv_heads:
                            raise ValueError(
                                f"Cartridge incompatible with model "
                                f"{self.model_config.model}: cartridge has "
                                f"{cart_kv_heads} KV heads but model expects "
                                f"{expected_kv_heads}. The cartridge may have "
                                f"been trained on a different model."
                            )
                        if cart_head_dim != expected_head_size:
                            raise ValueError(
                                f"Cartridge incompatible with model "
                                f"{self.model_config.model}: cartridge has "
                                f"head_dim={cart_head_dim} but model expects "
                                f"{expected_head_size}. The cartridge may have "
                                f"been trained on a different model."
                            )

                        # Save to shared storage (Modal volume) for zero-copy IPC
                        cartridge_shm_path = save_cartridge_to_shm(
                            cartridge_id, stacked_cartridge_kv
                        )
                        # Don't send tensors via IPC, engine core will load from shared storage
                        stacked_cartridge_kv = None
                        self._sent_cartridge_ids[cartridge_id] = cartridge_shm_path
                        # LRU eviction
                        while len(self._sent_cartridge_ids) > self._sent_cartridge_ids_max_size:
                            self._sent_cartridge_ids.popitem(last=False)
                    else:
                        # No KV data available
                        cartridge_shm_path = None

            # Handle pre-computed cartridges (token prepending for prefix caching)
            cartridge_token_ids = []
            for cartridge_data in precomputed_cartridges:
                cartridge_token_ids.extend(cartridge_data.token_ids.tolist())

            if cartridge_token_ids:
                logger.info(
                    "Prepending %d tokens from %d pre-computed cartridge(s) to prompt",
                    len(cartridge_token_ids),
                    len(precomputed_cartridges),
                )
                return (
                    cartridge_token_ids + prompt_token_ids,
                    stacked_cartridge_kv,
                    cartridge_id,
                    cartridge_seq_len,
                    cartridge_shm_path,
                    allow_pipelining,
                )
            else:
                return (
                    prompt_token_ids,
                    stacked_cartridge_kv,
                    cartridge_id,
                    cartridge_seq_len,
                    cartridge_shm_path,
                    allow_pipelining,
                )

        except Exception as e:
            logger.error("Failed to process cartridges: %s", e)
            raise RuntimeError(f"Failed to process cartridges: {e}") from e

    def _process_lora(
        self,
        lora_specs: list[dict[str, Any]] | None,
        request_id: str | None = None,
    ) -> LoRARequest | None:
        """
        Process LoRA adapter specifications and create a LoRARequest.

        Args:
            lora_specs: List of LoRA adapter specifications from the request
            request_id: Request ID to associate with the LoRA adapter

        Returns:
            LoRARequest object if LoRA adapters are specified, None otherwise
        """
        if not lora_specs:
            return None

        from vllm.logger import init_logger
        from vllm.utils.adapter_manager import get_adapter_manager

        logger = init_logger(__name__)

        try:
            # Only support one LoRA adapter per request
            if len(lora_specs) > 1:
                logger.warning(
                    "Multiple LoRA adapters detected (%d). "
                    "Only the first one will be used.",
                    len(lora_specs),
                )

            lora_spec = lora_specs[0]
            lora_id = lora_spec.get("id")
            # lora_source is available but not yet used for different backends
            _ = lora_spec.get("source", "s3")
            force_redownload = lora_spec.get("force_redownload", False)

            if not lora_id:
                logger.error("LoRA adapter spec missing 'id' field")
                return None

            # Download/load the LoRA adapter directory
            # (contains adapter_model.pt + adapter_config.json)
            manager = get_adapter_manager()

            import hashlib
            import os
            import tempfile
            from pathlib import Path

            import torch

            import vllm.envs as envs

            # Create a unique local path for this LoRA adapter
            # Use persistent cache directory if configured, otherwise fall back to /tmp
            lora_hash = hashlib.sha256(lora_id.encode()).hexdigest()[:16]
            lora_cache_dir = envs.VLLM_LORA_RESOLVER_CACHE_DIR or tempfile.gettempdir()
            local_lora_dir = Path(lora_cache_dir) / lora_hash
            local_lora_dir.mkdir(parents=True, exist_ok=True)
            logger.info("LoRA adapter storage directory: %s", local_lora_dir)

            # Download model.pt and lora_config.json from S3,
            # save as adapter_model.pt and adapter_config.json (vLLM format)
            adapter_model_path = local_lora_dir / "adapter_model.pt"
            adapter_config_path = local_lora_dir / "adapter_config.json"

            if not adapter_model_path.exists() or force_redownload:
                # lora_id should be a directory path (e.g., s3://bucket/path/to/checkpoint)
                # We need to download model.pt from that directory (S3 naming)

                # Download model.pt from S3, save as adapter_model.pt locally
                adapter_model_id = (
                    os.path.join(lora_id, "model.pt")
                    if not lora_id.endswith(".pt")
                    else lora_id
                )

                lora_data = manager.get_adapter(
                    adapter_id=adapter_model_id,
                    force_redownload=force_redownload,
                )

                torch.save(lora_data, adapter_model_path)

                # Download lora_config.json from S3,
                # save as adapter_config.json locally (vLLM format)
                # If lora_id ends with .pt, use its directory;
                # otherwise use lora_id as-is
                lora_dir = (
                    os.path.dirname(lora_id) if lora_id.endswith(".pt") else lora_id
                )
                adapter_config_id = os.path.join(lora_dir, "lora_config.json")

                try:
                    manager.download_json(
                        json_id=adapter_config_id,
                        target_path=adapter_config_path,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to download lora_config.json: %s. "
                        "vLLM may fail to load LoRA.",
                        e,
                    )

            # Verify files exist before creating LoRARequest
            if not adapter_model_path.exists():
                raise FileNotFoundError(
                    f"adapter_model.pt not found at {adapter_model_path}"
                )
            if not adapter_config_path.exists():
                logger.warning(
                    "adapter_config.json not found at %s, vLLM may fail to load LoRA",
                    adapter_config_path,
                )

            # Create LoRARequest
            # Generate a stable int ID based on lora_id only (not request_id)
            # This ensures the same LoRA gets the same ID across all requests,
            # allowing vLLM to reuse the loaded LoRA instead of reloading it
            lora_int_id = hash(lora_id) & 0x7FFFFFFF  # Positive int

            lora_request = LoRARequest(
                lora_name=lora_id,  # Use full lora_id for better traceability
                lora_int_id=lora_int_id,
                lora_path=f"{str(local_lora_dir)}/",  # vLLM expects directory path
            )

            return lora_request

        except Exception as e:
            logger.error("Failed to process LoRA adapters: %s", e)
            raise RuntimeError(f"Failed to process LoRA adapters: {e}") from e

    async def beam_search(
        self,
        prompt: TokPrompt,
        request_id: str,
        params: BeamSearchParams,
        lora_request: LoRARequest | None = None,
        trace_headers: Mapping[str, str] | None = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        beam_width = params.beam_width
        max_tokens = params.max_tokens
        ignore_eos = params.ignore_eos
        temperature = params.temperature
        length_penalty = params.length_penalty
        include_stop_str_in_output = params.include_stop_str_in_output

        input_processor = self.input_processor
        tokenizer = input_processor.tokenizer
        if tokenizer is None:
            raise VLLMValidationError(
                "You cannot use beam search when `skip_tokenizer_init=True`",
                parameter="skip_tokenizer_init",
                value=True,
            )

        eos_token_id: int = tokenizer.eos_token_id  # type: ignore

        if isinstance(prompt, dict) and "encoder_prompt" in prompt:
            raise NotImplementedError("Encoder-decoder prompt not supported")

        prompt_text: str | None = prompt.get("prompt")  # type: ignore
        prompt_token_ids: list[int] = prompt.get("prompt_token_ids", [])  # type: ignore
        multi_modal_data: MultiModalDataDict | None = prompt.get("multi_modal_data")  # type: ignore

        mm_processor_kwargs: dict[str, Any] | None = None

        # This is a workaround to fix multimodal beam search; this is a
        # bandaid fix for 2 small problems:
        # 1. Multi_modal_data on the processed_inputs currently resolves to
        #    `None`.
        # 2. preprocessing above expands the multimodal placeholders. However,
        #    this happens again in generation, so the double expansion causes
        #    a mismatch.
        # TODO - would be ideal to handle this more gracefully.

        tokenized_length = len(prompt_token_ids)

        sort_beams_key = create_sort_beams_key_function(eos_token_id, length_penalty)

        logprobs_num = 2 * beam_width
        beam_search_params = SamplingParams(
            logprobs=logprobs_num,
            max_tokens=1,
            temperature=temperature,
        )
        all_beams = [
            BeamSearchSequence(
                tokens=prompt_token_ids,
                cum_logprob=0,
                logprobs=[],
                multi_modal_data=multi_modal_data,
                mm_processor_kwargs=mm_processor_kwargs,
                lora_request=lora_request,
            )
        ]
        completed = []

        for _ in range(max_tokens):
            prompts_batch, lora_req_batch = zip(
                *[
                    (
                        TokensPrompt(
                            prompt_token_ids=beam.tokens,
                            multi_modal_data=beam.multi_modal_data,
                            mm_processor_kwargs=beam.mm_processor_kwargs,
                        ),
                        beam.lora_request,
                    )
                    for beam in all_beams
                ]
            )

            tasks = []
            request_id_batch = f"{request_id}-{random_uuid()}"

            for i, (individual_prompt, lora_req) in enumerate(
                zip(prompts_batch, lora_req_batch)
            ):
                request_id_item = f"{request_id_batch}-beam-{i}"
                task = asyncio.create_task(
                    collect_from_async_generator(
                        self.engine_client.generate(
                            individual_prompt,
                            beam_search_params,
                            request_id_item,
                            lora_request=lora_req,
                            trace_headers=trace_headers,
                        )
                    )
                )
                tasks.append(task)

            output = [x[0] for x in await asyncio.gather(*tasks)]

            new_beams = []
            # Store all new tokens generated by beam
            all_beams_token_id = []
            # Store the cumulative probability of all tokens
            # generated by beam search
            all_beams_logprob = []
            # Iterate through all beam inference results
            for i, result in enumerate(output):
                current_beam = all_beams[i]

                # check for error finish reason and abort beam search
                if result.outputs[0].finish_reason == "error":
                    # yield error output and terminate beam search
                    yield RequestOutput(
                        request_id=request_id,
                        prompt=prompt_text,
                        outputs=[
                            CompletionOutput(
                                index=0,
                                text="",
                                token_ids=[],
                                cumulative_logprob=None,
                                logprobs=None,
                                finish_reason="error",
                            )
                        ],
                        finished=True,
                        prompt_token_ids=prompt_token_ids,
                        prompt_logprobs=None,
                    )
                    return

                if result.outputs[0].logprobs is not None:
                    logprobs = result.outputs[0].logprobs[0]
                    all_beams_token_id.extend(list(logprobs.keys()))
                    all_beams_logprob.extend(
                        [
                            current_beam.cum_logprob + obj.logprob
                            for obj in logprobs.values()
                        ]
                    )

            # Handle the token for the end of sentence (EOS)
            all_beams_token_id = np.array(all_beams_token_id)
            all_beams_logprob = np.array(all_beams_logprob)

            if not ignore_eos:
                # Get the index position of eos token in all generated results
                eos_idx = np.where(all_beams_token_id == eos_token_id)[0]
                for idx in eos_idx:
                    current_beam = all_beams[idx // logprobs_num]
                    result = output[idx // logprobs_num]
                    assert result.outputs[0].logprobs is not None
                    logprobs_entry = result.outputs[0].logprobs[0]
                    completed.append(
                        BeamSearchSequence(
                            tokens=current_beam.tokens + [eos_token_id]
                            if include_stop_str_in_output
                            else current_beam.tokens,
                            logprobs=current_beam.logprobs + [logprobs_entry],
                            cum_logprob=float(all_beams_logprob[idx]),
                            finish_reason="stop",
                            stop_reason=eos_token_id,
                        )
                    )
                # After processing, set the log probability of the eos condition
                # to negative infinity.
                all_beams_logprob[eos_idx] = -np.inf

            # Processing non-EOS tokens
            # Get indices of the top beam_width probabilities
            topn_idx = np.argpartition(np.negative(all_beams_logprob), beam_width)[
                :beam_width
            ]

            for idx in topn_idx:
                current_beam = all_beams[idx // logprobs_num]
                result = output[idx // logprobs_num]
                token_id = int(all_beams_token_id[idx])
                assert result.outputs[0].logprobs is not None
                logprobs_entry = result.outputs[0].logprobs[0]
                new_beams.append(
                    BeamSearchSequence(
                        tokens=current_beam.tokens + [token_id],
                        logprobs=current_beam.logprobs + [logprobs_entry],
                        lora_request=current_beam.lora_request,
                        cum_logprob=float(all_beams_logprob[idx]),
                        multi_modal_data=current_beam.multi_modal_data,
                        mm_processor_kwargs=current_beam.mm_processor_kwargs,
                    )
                )

            all_beams = new_beams

        completed.extend(all_beams)
        sorted_completed = sorted(completed, key=sort_beams_key, reverse=True)
        best_beams = sorted_completed[:beam_width]

        for beam in best_beams:
            if beam.tokens[-1] == eos_token_id and not ignore_eos:
                # Skip the eos token in the text.
                tokens = beam.tokens[tokenized_length:-1]
            else:
                tokens = beam.tokens[tokenized_length:]
            beam.text = tokenizer.decode(tokens)

        yield RequestOutput(
            request_id=request_id,
            prompt=prompt_text,
            outputs=[
                CompletionOutput(
                    text=beam.text,  # type: ignore
                    cumulative_logprob=beam.cum_logprob,
                    token_ids=beam.tokens[tokenized_length:],
                    index=i,
                    logprobs=beam.logprobs,
                    finish_reason=beam.finish_reason
                    if beam.finish_reason is not None
                    else "length",
                    stop_reason=beam.stop_reason,
                )
                for (i, beam) in enumerate(best_beams)
            ],
            finished=True,
            prompt_token_ids=prompt_token_ids,
            prompt_logprobs=None,
        )

    async def _preprocess(
        self,
        ctx: ServeContext,
    ) -> ErrorResponse | None:
        """
        Default preprocessing hook. Subclasses may override
        to prepare `ctx` (classification, embedding, etc.).
        """
        return None

    def _build_response(
        self,
        ctx: ServeContext,
    ) -> AnyResponse | ErrorResponse:
        """
        Default response builder. Subclass may override this method
        to return the appropriate response object.
        """
        return self.create_error_response("unimplemented endpoint")

    async def handle(
        self,
        ctx: ServeContext,
    ) -> AnyResponse | ErrorResponse:
        async for response in self._pipeline(ctx):
            return response

        return self.create_error_response("No response yielded from pipeline")

    async def _pipeline(
        self,
        ctx: ServeContext,
    ) -> AsyncGenerator[AnyResponse | ErrorResponse, None]:
        """Execute the request processing pipeline yielding responses."""
        if error := await self._check_model(ctx.request):
            yield error
        if error := self._validate_request(ctx):
            yield error

        preprocess_ret = await self._preprocess(ctx)
        if isinstance(preprocess_ret, ErrorResponse):
            yield preprocess_ret

        generators_ret = await self._prepare_generators(ctx)
        if isinstance(generators_ret, ErrorResponse):
            yield generators_ret

        collect_ret = await self._collect_batch(ctx)
        if isinstance(collect_ret, ErrorResponse):
            yield collect_ret

        yield self._build_response(ctx)

    def _validate_request(self, ctx: ServeContext) -> ErrorResponse | None:
        truncate_prompt_tokens = getattr(ctx.request, "truncate_prompt_tokens", None)

        if (
            truncate_prompt_tokens is not None
            and truncate_prompt_tokens > self.max_model_len
        ):
            return self.create_error_response(
                "truncate_prompt_tokens value is "
                "greater than max_model_len."
                " Please, select a smaller truncation size."
            )
        return None

    def _create_pooling_params(
        self,
        ctx: ServeContext,
    ) -> PoolingParams | ErrorResponse:
        if not hasattr(ctx.request, "to_pooling_params"):
            return self.create_error_response(
                "Request type does not support pooling parameters"
            )

        return ctx.request.to_pooling_params()

    async def _prepare_generators(
        self,
        ctx: ServeContext,
    ) -> ErrorResponse | None:
        """Schedule the request and get the result generator."""
        generators: list[AsyncGenerator[PoolingRequestOutput, None]] = []

        try:
            trace_headers = (
                None
                if ctx.raw_request is None
                else await self._get_trace_headers(ctx.raw_request.headers)
            )

            pooling_params = self._create_pooling_params(ctx)
            if isinstance(pooling_params, ErrorResponse):
                return pooling_params

            if ctx.engine_prompts is None:
                return self.create_error_response("Engine prompts not available")

            for i, engine_prompt in enumerate(ctx.engine_prompts):
                request_id_item = f"{ctx.request_id}-{i}"

                self._log_inputs(
                    request_id_item,
                    engine_prompt,
                    params=pooling_params,
                    lora_request=ctx.lora_request,
                )

                generator = self.engine_client.encode(
                    engine_prompt,
                    pooling_params,
                    request_id_item,
                    lora_request=ctx.lora_request,
                    trace_headers=trace_headers,
                    priority=getattr(ctx.request, "priority", 0),
                )

                generators.append(generator)

            ctx.result_generator = merge_async_iterators(*generators)

            return None

        except Exception as e:
            return self.create_error_response(e)

    async def _collect_batch(
        self,
        ctx: ServeContext,
    ) -> ErrorResponse | None:
        """Collect batch results from the result generator."""
        try:
            if ctx.engine_prompts is None:
                return self.create_error_response("Engine prompts not available")

            num_prompts = len(ctx.engine_prompts)
            final_res_batch: list[PoolingRequestOutput | None]
            final_res_batch = [None] * num_prompts

            if ctx.result_generator is None:
                return self.create_error_response("Result generator not available")

            async for i, res in ctx.result_generator:
                final_res_batch[i] = res

            if None in final_res_batch:
                return self.create_error_response(
                    "Failed to generate results for all prompts"
                )

            ctx.final_res_batch = [res for res in final_res_batch if res is not None]

            return None

        except Exception as e:
            return self.create_error_response(e)

    def create_error_response(
        self,
        message: str | Exception,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
        param: str | None = None,
    ) -> ErrorResponse:
        exc: Exception | None = None

        if isinstance(message, Exception):
            exc = message

            from vllm.exceptions import VLLMValidationError

            if isinstance(exc, VLLMValidationError):
                err_type = "BadRequestError"
                status_code = HTTPStatus.BAD_REQUEST
                param = exc.parameter
            elif isinstance(exc, (ValueError, TypeError, RuntimeError, OverflowError)):
                # Common validation errors from user input
                err_type = "BadRequestError"
                status_code = HTTPStatus.BAD_REQUEST
                param = None
            elif isinstance(exc, NotImplementedError):
                err_type = "NotImplementedError"
                status_code = HTTPStatus.NOT_IMPLEMENTED
                param = None
            elif exc.__class__.__name__ == "TemplateError":
                # jinja2.TemplateError (avoid importing jinja2)
                err_type = "BadRequestError"
                status_code = HTTPStatus.BAD_REQUEST
                param = None
            else:
                err_type = "InternalServerError"
                status_code = HTTPStatus.INTERNAL_SERVER_ERROR
                param = None

            message = str(exc)

        if self.log_error_stack:
            exc_type, _, _ = sys.exc_info()
            if exc_type is not None:
                traceback.print_exc()
            else:
                traceback.print_stack()

        return ErrorResponse(
            error=ErrorInfo(
                message=sanitize_message(message),
                type=err_type,
                code=status_code.value,
                param=param,
            )
        )

    def create_streaming_error_response(
        self,
        message: str | Exception,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
        param: str | None = None,
    ) -> str:
        json_str = json.dumps(
            self.create_error_response(
                message=message,
                err_type=err_type,
                status_code=status_code,
                param=param,
            ).model_dump()
        )
        return json_str

    def _raise_if_error(self, finish_reason: str | None, request_id: str) -> None:
        """Raise GenerationError if finish_reason indicates an error."""
        if finish_reason == "error":
            logger.error(
                "Request %s failed with an internal error during generation",
                request_id,
            )
            raise GenerationError("Internal server error")

    def _convert_generation_error_to_response(
        self, e: GenerationError
    ) -> ErrorResponse:
        """Convert GenerationError to ErrorResponse."""
        return self.create_error_response(
            str(e),
            err_type="InternalServerError",
            status_code=e.status_code,
        )

    def _convert_generation_error_to_streaming_response(
        self, e: GenerationError
    ) -> str:
        """Convert GenerationError to streaming error response."""
        return self.create_streaming_error_response(
            str(e),
            err_type="InternalServerError",
            status_code=e.status_code,
        )

    async def _check_model(
        self,
        request: AnyRequest,
    ) -> ErrorResponse | None:
        error_response = None

        if self._is_model_supported(request.model):
            return None
        if request.model in self.models.lora_requests:
            return None
        if (
            envs.VLLM_ALLOW_RUNTIME_LORA_UPDATING
            and request.model
            and (load_result := await self.models.resolve_lora(request.model))
        ):
            if isinstance(load_result, LoRARequest):
                return None
            if (
                isinstance(load_result, ErrorResponse)
                and load_result.error.code == HTTPStatus.BAD_REQUEST.value
            ):
                error_response = load_result

        return error_response or self.create_error_response(
            message=f"The model `{request.model}` does not exist.",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
            param="model",
        )

    def _get_active_default_mm_loras(self, request: AnyRequest) -> LoRARequest | None:
        """Determine if there are any active default multimodal loras."""
        # TODO: Currently this is only enabled for chat completions
        # to be better aligned with only being enabled for .generate
        # when run offline. It would be nice to support additional
        # tasks types in the future.
        message_types = self._get_message_types(request)
        default_mm_loras = set()

        for lora in self.models.lora_requests.values():
            # Best effort match for default multimodal lora adapters;
            # There is probably a better way to do this, but currently
            # this matches against the set of 'types' in any content lists
            # up until '_', e.g., to match audio_url -> audio
            if lora.lora_name in message_types:
                default_mm_loras.add(lora)

        # Currently only support default modality specific loras if
        # we have exactly one lora matched on the request.
        if len(default_mm_loras) == 1:
            return default_mm_loras.pop()
        return None

    def _maybe_get_adapters(
        self,
        request: AnyRequest,
        supports_default_mm_loras: bool = False,
    ) -> LoRARequest | None:
        if request.model in self.models.lora_requests:
            return self.models.lora_requests[request.model]

        # Currently only support default modality specific loras
        # if we have exactly one lora matched on the request.
        if supports_default_mm_loras:
            default_mm_lora = self._get_active_default_mm_loras(request)
            if default_mm_lora is not None:
                return default_mm_lora

        if self._is_model_supported(request.model):
            return None

        # if _check_model has been called earlier, this will be unreachable
        raise ValueError(f"The model `{request.model}` does not exist.")

    def _get_message_types(self, request: AnyRequest) -> set[str]:
        """Retrieve the set of types from message content dicts up
        until `_`; we use this to match potential multimodal data
        with default per modality loras.
        """
        message_types: set[str] = set()

        if not hasattr(request, "messages"):
            return message_types

        messages = request.messages
        if messages is None or isinstance(messages, (str, bytes)):
            return message_types

        for message in messages:
            if (
                isinstance(message, dict)
                and "content" in message
                and isinstance(message["content"], list)
            ):
                for content_dict in message["content"]:
                    if "type" in content_dict:
                        message_types.add(content_dict["type"].split("_")[0])
        return message_types

    def _validate_input(
        self,
        request: object,
        input_ids: list[int],
        input_text: str,
    ) -> TokensPrompt:
        token_num = len(input_ids)

        # Note: EmbeddingRequest, ClassificationRequest,
        # and ScoreRequest doesn't have max_tokens
        if isinstance(
            request,
            (
                EmbeddingChatRequest,
                EmbeddingCompletionRequest,
                ScoreDataRequest,
                ScoreTextRequest,
                ScoreQueriesDocumentsRequest,
                RerankRequest,
                ClassificationCompletionRequest,
                ClassificationChatRequest,
            ),
        ):
            # Note: input length can be up to the entire model context length
            # since these requests don't generate tokens.
            if token_num > self.max_model_len:
                operations: dict[type[AnyRequest], str] = {
                    ScoreDataRequest: "score",
                    ScoreTextRequest: "score",
                    ScoreQueriesDocumentsRequest: "score",
                    ClassificationCompletionRequest: "classification",
                    ClassificationChatRequest: "classification",
                }
                operation = operations.get(type(request), "embedding generation")
                raise VLLMValidationError(
                    f"This model's maximum context length is "
                    f"{self.max_model_len} tokens. However, you requested "
                    f"{token_num} tokens in the input for {operation}. "
                    f"Please reduce the length of the input.",
                    parameter="input_tokens",
                    value=token_num,
                )
            return TokensPrompt(prompt=input_text, prompt_token_ids=input_ids)

        # Note: TokenizeRequest and DetokenizeRequest doesn't have max_tokens
        # and does not require model context length validation
        if isinstance(
            request,
            (TokenizeCompletionRequest, TokenizeChatRequest, DetokenizeRequest),
        ):
            return TokensPrompt(prompt=input_text, prompt_token_ids=input_ids)

        # chat completion endpoint supports max_completion_tokens
        if isinstance(request, ChatCompletionRequest):
            # TODO(#9845): remove max_tokens when field dropped from OpenAI API
            max_tokens = request.max_completion_tokens or request.max_tokens
        else:
            max_tokens = getattr(request, "max_tokens", None)

        # Note: input length can be up to model context length - 1 for
        # completion-like requests.
        if token_num >= self.max_model_len:
            raise VLLMValidationError(
                f"This model's maximum context length is "
                f"{self.max_model_len} tokens. However, your request has "
                f"{token_num} input tokens. Please reduce the length of "
                "the input messages.",
                parameter="input_tokens",
                value=token_num,
            )

        if max_tokens is not None and token_num + max_tokens > self.max_model_len:
            raise VLLMValidationError(
                "'max_tokens' or 'max_completion_tokens' is too large: "
                f"{max_tokens}. This model's maximum context length is "
                f"{self.max_model_len} tokens and your request has "
                f"{token_num} input tokens ({max_tokens} > {self.max_model_len}"
                f" - {token_num}).",
                parameter="max_tokens",
                value=max_tokens,
            )

        return TokensPrompt(prompt=input_text, prompt_token_ids=input_ids)

    def _validate_chat_template(
        self,
        request_chat_template: str | None,
        chat_template_kwargs: dict[str, Any] | None,
        trust_request_chat_template: bool,
    ) -> ErrorResponse | None:
        if not trust_request_chat_template and (
            request_chat_template is not None
            or (
                chat_template_kwargs
                and chat_template_kwargs.get("chat_template") is not None
            )
        ):
            return self.create_error_response(
                "Chat template is passed with request, but "
                "--trust-request-chat-template is not set. "
                "Refused request with untrusted chat template."
            )
        return None

    @staticmethod
    def _prepare_extra_chat_template_kwargs(
        request_chat_template_kwargs: dict[str, Any] | None = None,
        default_chat_template_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Helper to merge server-default and request-specific chat template kwargs."""
        request_chat_template_kwargs = request_chat_template_kwargs or {}
        if default_chat_template_kwargs is None:
            return request_chat_template_kwargs
        # Apply server defaults first, then request kwargs override.
        return default_chat_template_kwargs | request_chat_template_kwargs

    async def _preprocess_completion(
        self,
        request: RendererRequest,
        prompt_input: str | list[str] | list[int] | list[list[int]] | None,
        prompt_embeds: bytes | list[bytes] | None,
    ) -> list[TokPrompt]:
        renderer = self.renderer
        model_config = self.model_config

        prompts = list[SingletonPrompt | bytes]()
        if prompt_embeds is not None:  # embeds take higher priority
            prompts.extend(prompt_to_seq(prompt_embeds))
        if prompt_input is not None:
            prompts.extend(prompt_to_seq(prompt_input))

        parsed_prompts = [
            (
                prompt
                if isinstance(prompt, bytes)
                else parse_model_prompt(model_config, prompt)
            )
            for prompt in prompts
        ]
        tok_params = request.build_tok_params(model_config)

        return await renderer.render_cmpl_async(
            parsed_prompts,
            tok_params,
            prompt_extras={
                k: v
                for k in ("mm_processor_kwargs", "cache_salt")
                if (v := getattr(request, k, None)) is not None
            },
        )

    async def _preprocess_chat(
        self,
        request: RendererChatRequest,
        messages: list[ChatCompletionMessageParam],
        default_template: str | None,
        default_template_content_format: ChatTemplateContentFormatOption,
        default_template_kwargs: dict[str, Any] | None,
        tool_dicts: list[dict[str, Any]] | None = None,
        tool_parser: Callable[[TokenizerLike], ToolParser] | None = None,
    ) -> tuple[list[ConversationMessage], list[TokPrompt]]:
        from vllm.tokenizers.mistral import MistralTokenizer

        renderer = self.renderer

        default_template_kwargs = merge_kwargs(
            default_template_kwargs,
            dict(
                tools=tool_dicts,
                tokenize=isinstance(renderer.tokenizer, MistralTokenizer),
            ),
        )

        tok_params = request.build_tok_params(self.model_config)
        chat_params = request.build_chat_params(
            default_template, default_template_content_format
        ).with_defaults(default_template_kwargs)

        (conversation,), (engine_prompt,) = await renderer.render_chat_async(
            [messages],
            chat_params,
            tok_params,
            prompt_extras={
                k: v
                for k in ("mm_processor_kwargs", "cache_salt")
                if (v := getattr(request, k, None)) is not None
            },
        )

        # tool parsing is done only if a tool_parser has been set and if
        # tool_choice is not "none" (if tool_choice is "none" but a tool_parser
        # is set, we want to prevent parsing a tool_call hallucinated by the LLM
        if tool_parser is not None:
            tool_choice = getattr(request, "tool_choice", "none")
            if tool_choice != "none":
                if not isinstance(request, ChatCompletionRequest | ResponsesRequest):
                    msg = (
                        "Tool usage is only supported for Chat Completions API "
                        "or Responses API requests."
                    )
                    raise NotImplementedError(msg)

                # TODO: Update adjust_request to accept ResponsesRequest
                tokenizer = renderer.get_tokenizer()
                request = tool_parser(tokenizer).adjust_request(request=request)  # type: ignore[arg-type]

        return conversation, [engine_prompt]

    def _extract_prompt_components(self, prompt: object):
        return extract_prompt_components(self.model_config, prompt)

    def _extract_prompt_text(self, prompt: object):
        return self._extract_prompt_components(prompt).text

    def _extract_prompt_len(self, prompt: object):
        return extract_prompt_len(self.model_config, prompt)

    async def _render_next_turn(
        self,
        request: ResponsesRequest,
        messages: list[ResponseInputOutputItem],
        tool_dicts: list[dict[str, Any]] | None,
        tool_parser: Callable[[TokenizerLike], ToolParser] | None,
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
    ):
        new_messages = construct_input_messages(
            request_input=messages,
        )

        _, engine_prompts = await self._preprocess_chat(
            request,
            new_messages,
            default_template=chat_template,
            default_template_content_format=chat_template_content_format,
            default_template_kwargs=None,
            tool_dicts=tool_dicts,
            tool_parser=tool_parser,
        )
        return engine_prompts

    async def _generate_with_builtin_tools(
        self,
        request_id: str,
        engine_prompt: TokPrompt,
        sampling_params: SamplingParams,
        tok_params: TokenizeParams,
        context: ConversationContext,
        lora_request: LoRARequest | None = None,
        priority: int = 0,
        trace_headers: Mapping[str, str] | None = None,
    ):
        prompt_text = self._extract_prompt_text(engine_prompt)

        orig_priority = priority
        sub_request = 0
        while True:
            # Ensure that each sub-request has a unique request id.
            sub_request_id = f"{request_id}_{sub_request}"

            self._log_inputs(
                sub_request_id,
                engine_prompt,
                params=sampling_params,
                lora_request=lora_request,
            )

            tokenization_kwargs = tok_params.get_encode_kwargs()
            engine_request = self.input_processor.process_inputs(
                sub_request_id,
                engine_prompt,
                sampling_params,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                priority=priority,
            )

            generator = self.engine_client.generate(
                engine_request,
                sampling_params,
                sub_request_id,
                lora_request=lora_request,
                trace_headers=trace_headers,
                priority=priority,
                prompt_text=prompt_text,
                tokenization_kwargs=tokenization_kwargs,
            )

            async for res in generator:
                context.append_output(res)
                # NOTE(woosuk): The stop condition is handled by the engine.
                yield context

            if not context.need_builtin_tool_call():
                # The model did not ask for a tool call, so we're done.
                break

            # Call the tool and update the context with the result.
            tool_output = await context.call_tool()
            context.append_tool_output(tool_output)

            # TODO: uncomment this and enable tool output streaming
            # yield context

            # Create inputs for the next turn.
            # Render the next prompt token ids and update sampling_params.
            if isinstance(context, (HarmonyContext, StreamingHarmonyContext)):
                token_ids = context.render_for_completion()
                engine_prompt = TokensPrompt(prompt_token_ids=token_ids)

                sampling_params.max_tokens = self.max_model_len - len(token_ids)
            elif isinstance(context, ParsableContext):
                engine_prompts = await self._render_next_turn(
                    context.request,
                    context.parser.response_messages,
                    context.tool_dicts,
                    context.tool_parser_cls,
                    context.chat_template,
                    context.chat_template_content_format,
                )
                engine_prompt = engine_prompts[0]
                prompt_text = self._extract_prompt_text(engine_prompt)

                sampling_params.max_tokens = get_max_tokens(
                    self.max_model_len,
                    context.request.max_output_tokens,
                    self._extract_prompt_len(engine_prompt),
                    self.default_sampling_params,  # type: ignore
                )

            # OPTIMIZATION
            priority = orig_priority - 1
            sub_request += 1

    def _log_inputs(
        self,
        request_id: str,
        inputs: PromptType | TokPrompt,
        params: SamplingParams | PoolingParams | BeamSearchParams | None,
        lora_request: LoRARequest | None,
    ) -> None:
        if self.request_logger is None:
            return

        components = self._extract_prompt_components(inputs)

        self.request_logger.log_inputs(
            request_id,
            components.text,
            components.token_ids,
            components.embeds,
            params=params,
            lora_request=lora_request,
        )

    async def _get_trace_headers(
        self,
        headers: Headers,
    ) -> Mapping[str, str] | None:
        is_tracing_enabled = await self.engine_client.is_tracing_enabled()

        if is_tracing_enabled:
            return extract_trace_headers(headers)

        if contains_trace_headers(headers):
            log_tracing_disabled_warning()

        return None

    @staticmethod
    def _base_request_id(
        raw_request: Request | None, default: str | None = None
    ) -> str | None:
        """Pulls the request id to use from a header, if provided"""
        if raw_request is not None and (
            (req_id := raw_request.headers.get("X-Request-Id")) is not None
        ):
            return req_id

        return random_uuid() if default is None else default

    @staticmethod
    def _get_data_parallel_rank(raw_request: Request | None) -> int | None:
        """Pulls the data parallel rank from a header, if provided"""
        if raw_request is None:
            return None

        rank_str = raw_request.headers.get("X-data-parallel-rank")
        if rank_str is None:
            return None

        try:
            return int(rank_str)
        except ValueError:
            return None

    @staticmethod
    def _parse_tool_calls_from_content(
        request: ResponsesRequest | ChatCompletionRequest,
        tokenizer: TokenizerLike | None,
        enable_auto_tools: bool,
        tool_parser_cls: Callable[[TokenizerLike], ToolParser] | None,
        content: str | None = None,
    ) -> tuple[list[FunctionCall] | None, str | None]:
        function_calls = list[FunctionCall]()
        if request.tool_choice and isinstance(request.tool_choice, ToolChoiceFunction):
            assert content is not None
            # Forced Function Call
            function_calls.append(
                FunctionCall(name=request.tool_choice.name, arguments=content)
            )
            content = None  # Clear content since tool is called.
        elif request.tool_choice and isinstance(
            request.tool_choice, ChatCompletionNamedToolChoiceParam
        ):
            assert content is not None
            # Forced Function Call
            function_calls.append(
                FunctionCall(name=request.tool_choice.function.name, arguments=content)
            )
            content = None  # Clear content since tool is called.
        elif request.tool_choice == "required":
            assert content is not None
            tool_calls = TypeAdapter(list[FunctionDefinition]).validate_json(content)
            function_calls.extend(
                [
                    FunctionCall(
                        name=tool_call.name,
                        arguments=json.dumps(tool_call.parameters, ensure_ascii=False),
                    )
                    for tool_call in tool_calls
                ]
            )
            content = None  # Clear content since tool is called.
        elif (
            tool_parser_cls
            and enable_auto_tools
            and (request.tool_choice == "auto" or request.tool_choice is None)
        ):
            if tokenizer is None:
                raise ValueError(
                    "Tokenizer not available when `skip_tokenizer_init=True`"
                )

            # Automatic Tool Call Parsing
            try:
                tool_parser = tool_parser_cls(tokenizer)
            except RuntimeError as e:
                logger.exception("Error in tool parser creation.")
                raise e
            tool_call_info = tool_parser.extract_tool_calls(
                content if content is not None else "",
                request=request,  # type: ignore
            )
            if tool_call_info is not None and tool_call_info.tools_called:
                # extract_tool_calls() returns a list of tool calls.
                function_calls.extend(
                    FunctionCall(
                        id=tool_call.id,
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments,
                    )
                    for tool_call in tool_call_info.tool_calls
                )
                content = tool_call_info.content
                if content and content.strip() == "":
                    content = None
            else:
                # No tool calls.
                return None, content

        return function_calls, content

    @staticmethod
    def _get_decoded_token(
        logprob: Logprob,
        token_id: int,
        tokenizer: TokenizerLike | None,
        return_as_token_id: bool = False,
    ) -> str:
        if return_as_token_id:
            return f"token_id:{token_id}"

        if logprob.decoded_token is not None:
            return logprob.decoded_token

        if tokenizer is None:
            raise ValueError(
                "Unable to get tokenizer because `skip_tokenizer_init=True`"
            )

        return tokenizer.decode([token_id])

    def _is_model_supported(self, model_name: str | None) -> bool:
        if not model_name:
            return True
        return self.models.is_base_model(model_name)


def clamp_prompt_logprobs(
    prompt_logprobs: PromptLogprobs | None,
) -> PromptLogprobs | None:
    if prompt_logprobs is None:
        return prompt_logprobs

    for logprob_dict in prompt_logprobs:
        if logprob_dict is None:
            continue
        for logprob_values in logprob_dict.values():
            if logprob_values.logprob == float("-inf"):
                logprob_values.logprob = -9999.0
    return prompt_logprobs
