# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    try_get_attention_backend,
)
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.config.load import LoadConfig
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KpoolTailSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.spec_decode import multi_group as multi_group_module
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.multi_group import MultiGroupEagleProposer

mimo_7b_dir = "XiaomiMiMo/MiMo-7B-Base"
DEVICE_TYPE = current_platform.device_type


@pytest.mark.skip_global_cleanup
def test_glm5next_mtp_uses_multi_group_proposer():
    speculative_config = SimpleNamespace(
        method="mtp",
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="glm5_next_mtp")
        ),
    )

    assert SpeculativeConfig.use_glm5next_mtp(speculative_config)


@pytest.mark.skip_global_cleanup
def test_multi_group_proposer_initializes_each_draft_cache_group(monkeypatch):
    class MetadataBuilder:
        requires_block_table_width = False

        def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
            self.kv_cache_spec = kv_cache_spec

    class Backend:
        @classmethod
        def full_cls_name(cls):
            return cls.__module__, cls.__name__

        @staticmethod
        def get_builder_cls():
            return MetadataBuilder

    layers = {
        "draft.main": SimpleNamespace(get_attn_backend=lambda: Backend),
        "draft.tail": SimpleNamespace(get_attn_backend=lambda: Backend),
    }
    monkeypatch.setattr(
        multi_group_module,
        "get_layers_from_vllm_config",
        lambda *args: layers,
    )
    proposer = MultiGroupEagleProposer.__new__(MultiGroupEagleProposer)
    proposer.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=1024),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
    )
    proposer.device = torch.device("cpu")
    proposer._draft_attn_layer_names = set(layers)
    main_spec = FullAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
    )
    tail_spec = KpoolTailSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=2,
        dtype=torch.float32,
        sliding_window=16,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["draft.main"], kv_cache_spec=main_spec),
            KVCacheGroupSpec(layer_names=["draft.tail"], kv_cache_spec=tail_spec),
        ],
    )

    proposer.validate_same_kv_cache_group(kv_cache_config)
    proposer.initialize_attn_backend(kv_cache_config, [32, 16])

    assert [group.kv_cache_group_id for group in proposer.draft_attn_groups] == [
        0,
        1,
    ]
    assert [
        group.get_metadata_builder().kv_cache_spec.block_size
        for group in proposer.draft_attn_groups
    ] == [32, 16]
    assert proposer.kv_cache_gid == 0
    assert proposer.block_size == 32


def _create_mtp_proposer(num_speculative_tokens: int) -> EagleProposer:
    """Create an MTP proposer with unified model configuration."""
    model_config = ModelConfig(
        model=mimo_7b_dir, runner="generate", max_model_len=100, trust_remote_code=True
    )

    speculative_config = SpeculativeConfig(
        target_model_config=model_config,
        target_parallel_config=ParallelConfig(),
        model=mimo_7b_dir,
        method="mtp",
        num_speculative_tokens=num_speculative_tokens,
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(),
        speculative_config=speculative_config,
        device_config=DeviceConfig(device=DEVICE_TYPE),
        parallel_config=ParallelConfig(),
        load_config=LoadConfig(),
        scheduler_config=SchedulerConfig(
            max_model_len=model_config.max_model_len,
            is_encoder_decoder=model_config.is_encoder_decoder,
        ),
    )

    return EagleProposer(vllm_config=vllm_config, device=DEVICE_TYPE)


@mock.patch(
    "vllm.v1.spec_decode.llm_base_proposer.supports_multimodal",
    return_value=True,
)
@mock.patch(
    "vllm.v1.spec_decode.llm_base_proposer.current_platform."
    "update_block_size_for_backend"
)
@mock.patch("vllm.v1.spec_decode.llm_base_proposer.get_pp_group")
@mock.patch("vllm.v1.spec_decode.llm_base_proposer.get_layers_from_vllm_config")
@mock.patch("vllm.v1.spec_decode.llm_base_proposer.get_model")
def test_mtp_load_model_unified(
    mock_get_model,
    mock_get_layers,
    mock_get_pp_group,
    mock_update_block_size,
    _mock_supports_multimodal,
):
    """Test MTP-specific model loading with unified model approach."""

    # Setup mocks
    mock_model = mock.MagicMock()
    mock_model.model.embed_tokens.weight.shape = (131072, 4096)
    mock_get_model.return_value = mock_model
    # MTP does not have its own embed_tokens or lm_head
    # so it should share them with the target model
    mock_model.has_own_embed_tokens = False
    mock_model.has_own_lm_head = False

    target_attn_layers = {"target_attn_1": mock.MagicMock()}
    draft_attn_layer = mock.MagicMock()
    all_attn_layers = {**target_attn_layers, "draft_attn_1": draft_attn_layer}
    target_indexer_layers: dict = {}
    all_indexer_layers: dict = {}

    mock_get_layers.side_effect = [
        target_attn_layers,
        target_indexer_layers,
        all_attn_layers,
        all_indexer_layers,
    ]

    mock_pp_group = mock.MagicMock()
    mock_pp_group.world_size = 1
    mock_get_pp_group.return_value = mock_pp_group

    def resolve_block_size(vllm_config):
        vllm_config.cache_config.block_size = 128

    def get_draft_kv_cache_spec(vllm_config):
        assert vllm_config.cache_config.block_size == 128
        return mock.MagicMock()

    mock_update_block_size.side_effect = resolve_block_size
    draft_attn_layer.get_kv_cache_spec.side_effect = get_draft_kv_cache_spec

    # Create target model
    class _TargetModelStub(LlamaForCausalLM):
        model: mock.MagicMock
        lm_head: mock.MagicMock

    target_model = mock.create_autospec(_TargetModelStub, instance=True)
    target_model.model = mock.MagicMock()
    target_model.model.embed_tokens.weight.shape = (131072, 4096)
    target_model.lm_head = mock.MagicMock()
    target_model.config = mock.Mock(spec_set=["image_token_id"], image_token_id=151363)
    target_model.get_language_model = mock.MagicMock(return_value=target_model)

    # Create MTP proposer
    proposer = _create_mtp_proposer(num_speculative_tokens=4)
    proposer.get_model_name = mock.MagicMock(
        return_value="Glm5NextForConditionalGeneration"
    )
    proposer.load_model(target_model)

    # Verify MTP-specific behavior:
    # Model is loaded
    mock_get_model.assert_called_once()
    mock_update_block_size.assert_called_once_with(proposer.vllm_config)
    # MTP shares lm_head with target model
    assert proposer.model.lm_head == target_model.lm_head
    # MTP shares embed_tokens with target model
    assert proposer.model.model.embed_tokens == target_model.model.embed_tokens
    assert proposer.model.config.image_token_index == 151363


@pytest.mark.parametrize("num_speculative_tokens", [1])
def test_mtp_propose(num_speculative_tokens, monkeypatch):
    """Test that MTP's forward method returns hidden states directly"""

    device = torch.device(DEVICE_TYPE)
    batch_size = 2
    seq_lens = [5, 3]
    total_tokens = sum(seq_lens)
    vocab_size = 100

    proposer = _create_mtp_proposer(num_speculative_tokens)
    hidden_size = proposer.hidden_size

    # Mock the MTP model to verify it returns hidden states directly
    model_mock = mock.MagicMock()

    # MTP returns hidden states directly
    if num_speculative_tokens == 1:
        model_mock.return_value = torch.zeros(total_tokens, hidden_size, device=device)
    else:
        # Multiple forward passes for multi-token speculation
        forward_returns = []
        for i in range(num_speculative_tokens):
            if i == 0:
                h_states = torch.zeros(total_tokens, hidden_size, device=device)
            else:
                h_states = torch.zeros(batch_size, hidden_size, device=device)
            forward_returns.append(h_states)
        model_mock.side_effect = forward_returns

    # Mock compute_logits
    def create_deterministic_logits(batch_size, vocab_size, token_offset):
        logits = torch.full((batch_size, vocab_size), -100.0, device=device)
        logits[:, token_offset] = 100.0
        return logits

    if num_speculative_tokens == 1:
        model_mock.compute_logits.return_value = create_deterministic_logits(
            batch_size, vocab_size, 42
        )
    else:
        logits_returns = [
            create_deterministic_logits(batch_size, vocab_size, 42 + i)
            for i in range(num_speculative_tokens)
        ]
        model_mock.compute_logits.side_effect = logits_returns

    proposer.model = model_mock
    proposer._draft_attn_layer_names = {"layer.0"}

    # Prepare inputs
    batch_spec = BatchSpec(seq_lens=seq_lens, query_lens=seq_lens)
    common_attn_metadata = create_common_attn_metadata(
        batch_spec, block_size=16, device=device
    )

    target_token_ids = torch.randint(0, vocab_size, (total_tokens,), device=device)
    target_positions = torch.cat(
        [
            torch.arange(seq_lens[0], device=device),
            torch.arange(seq_lens[1], device=device),
        ]
    )
    target_hidden_states = torch.randn(total_tokens, hidden_size, device=device)
    next_token_ids = torch.randint(
        0, vocab_size, (batch_size,), dtype=torch.int32, device=device
    )
    sampling_metadata = mock.MagicMock()

    # Setup attention metadata
    attn_metadata_builder_cls, _ = try_get_attention_backend(
        AttentionBackendEnum.FLASH_ATTN
    )

    attn_metadata_builder = attn_metadata_builder_cls(
        kv_cache_spec=create_standard_kv_cache_spec(proposer.vllm_config),
        layer_names=list(proposer._draft_attn_layer_names),
        vllm_config=proposer.vllm_config,
        device=device,
    )

    proposer.runner = mock.MagicMock()
    mock_attn_group = mock.MagicMock()
    mock_attn_group.get_metadata_builder.return_value = attn_metadata_builder
    mock_attn_group.layer_names = list(proposer._draft_attn_layer_names)
    mock_attn_group.kv_cache_spec = attn_metadata_builder.kv_cache_spec
    proposer.draft_attn_groups = [mock_attn_group]

    # Run propose
    result = proposer.propose(
        num_speculative_tokens=num_speculative_tokens,
        target_token_ids=target_token_ids,
        target_positions=target_positions,
        target_hidden_states=target_hidden_states,
        next_token_ids=next_token_ids,
        token_indices_to_sample=None,
        common_attn_metadata=common_attn_metadata,
        sampling_metadata=sampling_metadata,
    )

    # Verify the model was called correctly
    assert model_mock.called
    # Verify output shape
    assert result.shape == (batch_size, num_speculative_tokens)
