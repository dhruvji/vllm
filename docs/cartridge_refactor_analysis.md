# Cartridge Implementation Analysis: Post-Mortem


This document analyzes the vLLM cartridge implementation across **three distinct phases**:

1. **In-Attention Two-Pass** (Dec 2025 - Jan 14, 2026): Original approach, complex but worked
2. **Pre-Populate (Broken)** (Jan 14-17, 2026): Refactor attempt, right idea but incomplete
3. **Pre-Populate (Working)** (Jan 17+): Current implementation, simple and correct

---

## Timeline

### Phase 1: In-Attention Two-Pass (Working but Complex)

| Date | Commit | What Happened |
|------|--------|---------------|
| **2025-11-26** | `f7396eb42` | **Initial implementation**: S3 KV cache loading, in-attention two-pass with `merge_attn_states` |
| **2025-12-04** | `e635240d2` | Fixed tensor contiguity bug, LSE transpose for merge |
| **2025-12-05** | `ff2b66d56` | Optimized single-sequence case (avoid replication) |
| **2025-12-09** | `de486de2d` | Fixed flash_attn shape handling, LSE format |
| **2026-01-08** | `1df05a67e` | Added tensor parallelism (TP head sharding) |
| **2026-01-10** | `5667957b1` | Added `cartridge_id` to BatchDescriptor for CUDA graph differentiation |
| **2026-01-11** | `0dc3dc585` | Added `_forward_with_cartridge_inject` (alternative in-attention approach) |

### Phase 2: Pre-Populate Refactor (Broken)

| Date | Commit | What Happened |
|------|--------|---------------|
| **2026-01-14** | `66138457a` | **❌ BROKEN REFACTOR**: Added `_prepopulate_cartridge_to_cache`, removed in-attention code (`cartridge_kv` from ForwardContext, 316 lines from flash_attn.py). Missing: GPU caching, prefix cache hit detection, API params |
| **2026-01-14** | `956d4779e` | Fix: check `num_computed_tokens < cart_seq_len` before pre-populating |
| **2026-01-14** | `2e5687d5c` | Fix: add `gpu_cartridge_cache` to avoid repeated CPU→GPU transfers |
| **2026-01-15** | `4507398c5` | Fix: track `cartridge_cache_hit`, skip resending for shared prefixes |
| **2026-01-16** | `6efa2583c` | Fix: prevent duplicate IPC sends |

### Phase 3: Pre-Populate Fixed (Current Working Implementation)

| Date | Commit | What Happened |
|------|--------|---------------|
| **2026-01-17** | `a0ded9652` | **✅ MAJOR FIX** (+3114 lines): Added `cartridge_loader.py`, `adapter_manager.py`, scheduler integration, GPU caching, prefix cache detection, IPC via shared memory. Commit message says "temporary fix before refactor" but this made pre-populate work. |
| **2026-01-19** | `c1d99b93c` | Cleanup: removed dead code from old in-attention implementation |

**Note**: `a0ded9652` is NOT a revert to the two-pass approach. It's a forward-fix that added the infrastructure the refactor was missing. The current code still uses pre-populate (no in-attention code).

**Summary**: 
- **Phase 1** (in-attention): Worked but was complex - required per-cartridge CUDA graphs, two-pass attention, `merge_attn_states`
- **Phase 2** (pre-populate broken): Right architecture but incomplete - removed old code without adding GPU caching, prefix cache detection, or API plumbing
- **Phase 3** (pre-populate working): Current implementation - simple, no attention-layer changes needed

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Phase 3: Pre-Populate Working (Current)](#phase-3-pre-populate-working-current-implementation)
4. [Phase 1: In-Attention Two-Pass (Original)](#phase-1-in-attention-two-pass-original-implementation)
5. [Phase 2: Pre-Populate Broken (Refactor)](#phase-2-pre-populate-broken-refactor)
6. [Specific Bugs in Phase 1](#specific-bugs-in-phase-1-in-attention-approach)
7. [Recommendations](#recommendations-for-future-simplification)
8. [Gotchas from Commit History](#appendix-additional-gotchas-from-commit-history)

---

## Executive Summary

| Aspect | Phase 3: Pre-Populate (Working) | Phase 2: Pre-Populate (Broken) | Phase 1: In-Attention (Original) |
|--------|--------------------------------|-------------------------------|----------------------------------|
| **Commits** | `a0ded9652`+ | `66138457a` - `6efa2583c` | `f7396eb42` - `0dc3dc585` |
| **Approach** | Pre-populate KV cache | Pre-populate (incomplete) | Two-pass attention + merge |
| **flash_attn.py changes** | None | None | +316 lines for cartridge handling |
| **ForwardContext.cartridge_kv** | No | No | Yes |
| **GPU caching** | Yes (`gpu_cartridge_cache`) | No (removed, not replaced) | Yes (`_gpu_cartridge_cache`) |
| **Prefix cache detection** | Yes (`num_computed_tokens` check) | No | N/A (different mechanism) |
| **CUDA graphs** | Standard graphs work | Standard graphs work | Per-cartridge graphs needed |
| **Status** | ✅ Working | ❌ Broken | ✅ Worked (but complex) |

**Key Insight**: The correct approach writes cartridge KV to the cache *before* the forward pass, so attention "just works" with no special handling. The refactor had the right idea but was incomplete.

---

## Architecture Overview

### What is a Cartridge?

A "cartridge" is a pre-computed or learned KV cache that gets prepended to attention:

1. **Pre-computed KV Cache**: KV cache computed from specific tokens (e.g., system prompt)
2. **Learned Soft Prompts**: KV cache trained via gradient descent (no corresponding tokens)

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          REQUEST PROCESSING                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. Request arrives with cartridge_id                                   │
│                    ↓                                                     │
│  2. Cartridge loaded (S3/local) → CartridgeData                         │
│                    ↓                                                     │
│  3. Stacked KV tensors transferred to GPU                               │
│                    ↓                                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  ✅ CORRECT APPROACH (Pre-populate)                               │   │
│  │  ──────────────────────────────────                               │   │
│  │  4a. Write cartridge KV to cache slots 0..cart_seq_len-1         │   │
│  │  4b. Offset input positions by cart_seq_len                       │   │
│  │  4c. Offset slot_mapping by cart_seq_len                          │   │
│  │  4d. Adjust seq_lens to include cartridge                         │   │
│  │  4e. Normal attention runs (no special handling needed)           │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  ❌ BROKEN APPROACH (In-attention injection)                      │   │
│  │  ───────────────────────────────────────────                      │   │
│  │  4a. Pass cartridge_kv through ForwardContext                     │   │
│  │  4b. In flash_attn.forward(), detect cartridge_kv                 │   │
│  │  4c. Run two-pass attention: cartridge + input                    │   │
│  │  4d. Merge results with log-sum-exp                               │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Three Phases**:
- **Phase 1** (in-attention): Original implementation that worked but was complex
- **Phase 2** (pre-populate broken): Refactor with right idea but incomplete - broke everything
- **Phase 3** (pre-populate working): Current implementation after fixes

---

## Phase 3: Pre-Populate Working (Current Implementation)

### Key Files

| File | Purpose |
|------|---------|
| `vllm/entrypoints/openai/engine/serving.py` | API-level cartridge processing from request |
| `vllm/v1/cartridge_loader.py` | Load cartridges from S3/local, IPC via shared memory |
| `vllm/v1/worker/gpu_model_runner.py` | Pre-populate cache, manage offsets, GPU caching |
| `vllm/v1/core/sched/scheduler.py` | Cartridge-aware scheduling, prefix cache hit detection |
| `vllm/v1/core/kv_cache_utils.py` | Block hashing with cartridge_id for prefix sharing |
| `vllm/forward_context.py` | Forward context (NO cartridge_kv field - not needed) |

---

### 1. State Management

**WHY CORRECT**: Tracking offsets per-request allows each request to have its own cartridge length; GPU cache avoids repeated CPU→GPU transfers; reference counting prevents memory leaks while allowing sharing.

```python
# vllm/v1/worker/gpu_model_runner.py - __init__()

# Position/slot offset for cartridge requests.
# Cartridge KV is pre-populated into cache slots 0..cart_seq_len-1,
# so input tokens need their positions and slot_mapping offset by cart_seq_len.
# Maps request_id -> cartridge_seq_len
self.cartridge_position_offsets: dict[str, int] = {}

# GPU cache for cartridge KV tensors to avoid repeated CPU->GPU transfers.
# Maps cartridge_id -> list of (layer_keys, layer_values) tuples,
# already transposed and ready for reshape_and_cache_flash.
self.gpu_cartridge_cache: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}

# Reference counting for cartridge GPU cache cleanup.
# Tracks how many active requests use each cartridge.
self._cartridge_ref_counts: dict[str, int] = {}
self._request_to_cartridge: dict[str, str] = {}
```

---

### 2. Pre-population Function (_prepopulate_cartridge_to_cache)

**WHY CORRECT**: Writing KV to cache before the forward pass means attention sees cartridge tokens as already-computed prefix tokens - no special attention code needed. Using `reshape_and_cache_flash` ensures proper format for FlashAttention.

```python
# vllm/v1/worker/gpu_model_runner.py

def _prepopulate_cartridge_to_cache(
    self,
    block_ids: tuple[list[int], ...],
    layer_kv_list: list[tuple[torch.Tensor, torch.Tensor]],
    cart_seq_len: int,
) -> None:
    """Pre-populate cartridge KV into the KV cache.

    This writes the cartridge KV to cache slots 0..cart_seq_len-1 so that
    subsequent tokens can attend to it without special handling in attention.

    Args:
        block_ids: The allocated block IDs for this request (tuple of lists,
                   one per KV cache group).
        layer_kv_list: List of (layer_keys, layer_values) tuples, one per layer.
                      Each tensor has shape (seq_len, num_kv_heads, head_dim).
        cart_seq_len: Length of the cartridge sequence.
    """
    block_size = self.cache_config.block_size

    # Use first KV cache group's block IDs (most common case)
    block_ids_list = block_ids[0]

    # Compute slot mapping for all cartridge positions
    positions = torch.arange(cart_seq_len, device=self.device, dtype=torch.long)
    block_indices = positions // block_size
    block_offsets = positions % block_size

    # Map virtual block indices to physical block IDs
    block_ids_tensor = torch.tensor(
        block_ids_list, device=self.device, dtype=torch.long
    )
    physical_blocks = block_ids_tensor[block_indices]

    # Compute slot indices: slot = physical_block * block_size + offset
    slot_mapping = physical_blocks * block_size + block_offsets

    # Get cache dtype string for reshape_and_cache_flash
    kv_cache_dtype_str = self.cache_config.cache_dtype

    # Use scale=1.0 for cartridge (skip quantization)
    k_scale = torch.ones(1, device=self.device, dtype=torch.float32)
    v_scale = torch.ones(1, device=self.device, dtype=torch.float32)

    # Write cartridge KV to each layer's cache
    num_layers = len(layer_kv_list)
    for layer_idx in range(num_layers):
        if layer_idx >= len(self.kv_caches):
            break

        kv_cache = self.kv_caches[layer_idx]
        key_cache, value_cache = kv_cache.unbind(0)

        # Get precomputed layer tensors (already transposed and on GPU)
        layer_keys, layer_values = layer_kv_list[layer_idx]

        # Write to cache
        reshape_and_cache_flash(
            layer_keys,
            layer_values,
            key_cache,
            value_cache,
            slot_mapping,
            kv_cache_dtype_str,
            k_scale,
            v_scale,
        )
```

---

### 3. Full New Request Handling

**WHY CORRECT**: Validating dimensions catches incompatible cartridges early with clear error messages. TP sharding ensures each GPU only gets its assigned KV heads. Caching avoids repeated work. Checking `cartridge_cache_hit` skips pre-population when prefix cache already has the data.

```python
# vllm/v1/worker/gpu_model_runner.py - _update_states()

# Pre-populate cartridge KV into cache if present.
# The cartridge occupies cache slots 0..cart_seq_len-1, and input tokens
# will be offset to start at slot cart_seq_len.
#
# When cartridge_cache_hit is True, cartridge_kv may be None (not sent
# via IPC to avoid serializing large tensors). In that case, we use the
# cached tensors from gpu_cartridge_cache.
if new_req_data.cartridge_id is not None:
    cartridge_id = new_req_data.cartridge_id
    prepopulate_ms = 0.0
    prepare_ms = 0.0
    gpu_cache_hit = False
    cartridge_cache_hit = getattr(
        new_req_data, "cartridge_cache_hit", False
    )
    cart_seq_len = None
    layer_kv_list = None

    # Check GPU cache first to avoid repeated
    # CPU->GPU transfers and tensor ops
    if cartridge_id in self.gpu_cartridge_cache:
        # Use cached layer tensors (already transposed and on GPU)
        layer_kv_list = self.gpu_cartridge_cache[cartridge_id]
        cart_seq_len = layer_kv_list[0][0].shape[0]  # (seq_len, num_kv_heads, head_dim)
        gpu_cache_hit = True
        
    elif new_req_data.cartridge_kv is not None:
        # First time seeing this cartridge:
        # validate, shard, transpose, and cache
        stacked_keys, stacked_values = new_req_data.cartridge_kv
        prepare_start = time.perf_counter()

        # Validate cartridge dimensions match model architecture
        total_kv_heads = self.model_config.get_total_num_kv_heads()
        expected_head_size = self.model_config.get_head_size()
        expected_num_layers = self.model_config.get_num_layers(
            self.parallel_config
        )
        cart_num_layers, cart_kv_heads, cart_seq_len, cart_head_dim = (
            stacked_keys.shape
        )
        if cart_num_layers != expected_num_layers:
            raise ValueError(
                f"Cartridge incompatible with model: "
                f"num_layers={cart_num_layers} but model expects "
                f"{expected_num_layers}. Cartridge may have been "
                f"trained on a different model."
            )
        if cart_kv_heads != total_kv_heads:
            raise ValueError(
                f"Cartridge incompatible with model: "
                f"num_kv_heads={cart_kv_heads} but model expects "
                f"{total_kv_heads}. Cartridge may have been "
                f"trained on a different model."
            )
        if cart_head_dim != expected_head_size:
            raise ValueError(
                f"Cartridge incompatible with model: "
                f"head_dim={cart_head_dim} but model expects "
                f"{expected_head_size}. Cartridge may have been "
                f"trained on a different model."
            )

        # Shard cartridge KV for tensor parallelism
        tp_size = self.parallel_config.tensor_parallel_size
        if tp_size > 1 and total_kv_heads >= tp_size:
            tp_rank = get_tp_group().rank_in_group
            heads_per_gpu = total_kv_heads // tp_size
            start_head = tp_rank * heads_per_gpu
            end_head = start_head + heads_per_gpu
            stacked_keys = stacked_keys[
                :, start_head:end_head, :, :
            ].contiguous()
            stacked_values = stacked_values[
                :, start_head:end_head, :, :
            ].contiguous()

        # Precompute transposed layer tensors and move to GPU once.
        # This avoids repeated transpose().contiguous().to() calls.
        layer_kv_list = []
        for layer_idx in range(cart_num_layers):
            # (num_kv_heads, seq_len, head_dim) ->
            # (seq_len, num_kv_heads, head_dim)
            layer_keys = (
                stacked_keys[layer_idx].transpose(0, 1).contiguous()
            )
            layer_values = (
                stacked_values[layer_idx].transpose(0, 1).contiguous()
            )
            # Transfer to GPU and convert dtype
            layer_keys = layer_keys.to(
                self.device, dtype=self.kv_cache_dtype
            )
            layer_values = layer_values.to(
                self.device, dtype=self.kv_cache_dtype
            )
            layer_kv_list.append((layer_keys, layer_values))

        # Cache the precomputed layer tensors
        self.gpu_cartridge_cache[cartridge_id] = layer_kv_list

    else:
        # No GPU cache and no KV payload; fall back to provided seq_len.
        cart_seq_len = getattr(new_req_data, "cartridge_seq_len", 0)

    if not cartridge_cache_hit:
        if layer_kv_list is None:
            # We need KV tensors to prepopulate on cache miss.
            raise RuntimeError(
                f"Cartridge {cartridge_id} KV not available for "
                f"prepopulate (cache_hit={cartridge_cache_hit})."
            )
        self._prepopulate_cartridge_to_cache(
            new_req_data.block_ids, layer_kv_list, cart_seq_len
        )

    # Store position offset:
    # input tokens need positions/slots offset by cart_seq_len
    self.cartridge_position_offsets[req_id] = cart_seq_len

    # Track reference count for GPU cache cleanup.
    self._request_to_cartridge[req_id] = cartridge_id
    self._cartridge_ref_counts[cartridge_id] = (
        self._cartridge_ref_counts.get(cartridge_id, 0) + 1
    )
```

---

### 4. Position Offset Application

**WHY CORRECT**: Offsetting positions BEFORE `compute_slot_mapping` ensures input tokens go to cache slots AFTER the cartridge. This also gives correct RoPE positions (input token 0 gets position cart_seq_len). Doing this once, before the forward pass, avoids inconsistent state.

```python
# vllm/v1/worker/gpu_model_runner.py - _prepare_inputs()

# Apply cartridge position offsets BEFORE computing slot_mapping.
# This ensures input tokens are stored in cache slots AFTER the cartridge.
# The offset also affects RoPE positions (applied later when copying to GPU).
if self.cartridge_position_offsets:
    token_idx = 0
    for req_idx in range(num_reqs):
        req_id = self.input_batch.req_ids[req_idx]
        n_tokens = num_scheduled_tokens[req_idx]
        if req_id in self.cartridge_position_offsets:
            offset = self.cartridge_position_offsets[req_id]
            positions_np[token_idx:token_idx + n_tokens] += offset
        token_idx += n_tokens

# Then compute slot_mapping with the offset positions
self.input_batch.block_table.compute_slot_mapping(req_indices, positions_np)
self.input_batch.block_table.commit_slot_mapping(total_num_scheduled_tokens)
```

---

### 5. seq_lens Adjustment

**WHY CORRECT**: Adding cartridge length to seq_lens tells attention how many total tokens to attend over (cartridge + input). This happens once before copying to GPU, so all layers see consistent values.

```python
# vllm/v1/worker/gpu_model_runner.py - _prepare_inputs()

self.seq_lens.np[:num_reqs] = (
    self.input_batch.num_computed_tokens_cpu[:num_reqs] + num_scheduled_tokens
)
# Add cartridge lengths to seq_lens so attention covers cartridge + input tokens.
if self.cartridge_position_offsets:
    for req_idx in range(num_reqs):
        req_id = self.input_batch.req_ids[req_idx]
        if req_id in self.cartridge_position_offsets:
            self.seq_lens.np[req_idx] += self.cartridge_position_offsets[req_id]
# Fill unused with 0 for full cuda graph mode.
self.seq_lens.np[num_reqs:].fill(0)
self.seq_lens.copy_to_gpu()
```

---

### 6. Cleanup on Request Finish

**WHY CORRECT**: Reference counting ensures GPU memory is freed only when no requests use the cartridge, preventing both leaks (never freeing) and use-after-free (freeing too early).

```python
# vllm/v1/worker/gpu_model_runner.py - _update_states()

# Clear cartridge state for finished requests.
# Decrement ref counts and free GPU cache when no longer used.
for req_id in scheduler_output.finished_req_ids:
    self.cartridge_position_offsets.pop(req_id, None)
    cartridge_id = self._request_to_cartridge.pop(req_id, None)
    if cartridge_id is not None:
        self._cartridge_ref_counts[cartridge_id] -= 1
        if self._cartridge_ref_counts[cartridge_id] <= 0:
            # No more requests using this cartridge - free GPU memory
            del self._cartridge_ref_counts[cartridge_id]
            if cartridge_id in self.gpu_cartridge_cache:
                del self.gpu_cartridge_cache[cartridge_id]
```

---

### 7. Scheduler Budget Accounting

**WHY CORRECT**: Counting cartridge tokens in budget only on cache miss prevents starvation. Cache hits share existing memory and don't need to process cartridge tokens, so they shouldn't consume extra budget.

```python
# vllm/v1/core/sched/scheduler.py - schedule()

# Check if cartridge prefix is already cached
cart_seq_len = getattr(request, "cartridge_seq_len", 0)
if cart_seq_len > 0:
    num_cart_blocks = (cart_seq_len + self.block_size - 1) // self.block_size
    cartridge_cache_hit = all(
        len(group) >= num_cart_blocks
        for group in new_computed_blocks.blocks
    )
    req_to_cartridge_cache_hit[request.request_id] = cartridge_cache_hit

# For cartridge cache MISS, count cart_seq_len in budget
# (need to load cartridge). For cache HIT, don't count
# (cartridge already in memory, shared via prefix cache).
cart_seq_len = getattr(request, "cartridge_seq_len", 0)
cartridge_cache_hit = req_to_cartridge_cache_hit.get(
    request.request_id, False
)
if cart_seq_len > 0 and not cartridge_cache_hit:
    budget_cost = num_new_tokens + cart_seq_len
else:
    budget_cost = num_new_tokens

if budget_cost > token_budget:
    # Not enough budget for this request
    break
```

---

### 8. Prefix Cache Block Hashing

**WHY CORRECT**: Including cartridge_id in ALL block hashes ensures requests with the same cartridge share cache (same RoPE positions), while different cartridges don't collide (different RoPE positions).

```python
# vllm/v1/core/kv_cache_utils.py

def generate_block_hash_extra_keys(request, start_token_idx, end_token_idx):
    # Include cartridge_id in ALL blocks (not just first) because cartridge
    # shifts RoPE positions for all tokens. Requests with the same cartridge
    # can share prefix cache; different cartridges will have different hashes.
    cartridge_keys: list[str] = [request.cartridge_id] if request.cartridge_id else []
    
    extra_keys: list[Any] = (
        lora_extra_keys
        + mm_extra_keys
        + cache_salt_keys
        + cartridge_keys  # <-- Added for cartridge support
        + prompt_embeds_keys
    )
    return extra_keys

def _generate_cartridge_virtual_tokens(
    cartridge_id: str,
    seq_len: int,
    hash_fn,
) -> list[int]:
    """Generate deterministic virtual token IDs for cartridge prefix sharing."""
    # Use negative IDs to avoid collision with real token IDs
    base_hash = int.from_bytes(hash_fn(cartridge_id)[:4], "big")
    return [-(base_hash + i) for i in range(seq_len)]
```

---

## Phase 1: In-Attention Two-Pass (Original Implementation)

### Original Implementation (Dec 2025 - Jan 2026)

This was the **original** approach used from `e635240d2` through `0dc3dc585`. The refactor commit `66138457a` removed this code, but the code snippets below show what the broken approach looked like.

**WHY BROKEN**: Handling cartridge KV *inside* the attention layer requires modifying metadata (slot_mapping, seq_lens) during the forward pass, which causes inconsistent state across layers and breaks CUDA graphs.

---

### 1. Added cartridge_kv to ForwardContext

**WHY BROKEN**: Passing tensors through context makes them part of the forward signature, requiring different CUDA graphs for different cartridges. With many cartridges, this causes memory explosion.

```python
# vllm/forward_context.py (BROKEN - these fields were removed)

@dataclass
class ForwardContext:
    no_compile_layers: dict[str, Any]
    attn_metadata: dict[str, "AttentionMetadata"] | list[dict[str, "AttentionMetadata"]]
    virtual_engine: int
    dp_metadata: DPMetadata | None = None
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE
    batch_descriptor: BatchDescriptor | None = None
    ubatch_slices: UBatchSlices | None = None
    
    # ❌ REMOVED: Learned cartridge KV cache to prepend to attention
    # Maps layer index -> (key_tensor, value_tensor)
    # Key/value shape: (num_kv_heads, seq_len, head_size)
    cartridge_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None

class BatchDescriptor(NamedTuple):
    num_tokens: int
    uniform_decode: bool = False
    has_lora: bool = False
    # ❌ REMOVED: Different cartridges need different CUDA graphs!
    cartridge_id: str | None = None
```

---

### 2. Layer Index Extraction via Regex

**WHY BROKEN**: Parsing layer names with regex is fragile and breaks if model architecture changes naming conventions. It also adds overhead to every attention call.

```python
# vllm/v1/attention/backends/flash_attn.py (BROKEN - this method was removed)

def _get_cartridge_kv_for_layer(
    self, layer: torch.nn.Module
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Get cartridge KV for this layer if available.
    
    Returns:
        Tuple of (key, value) tensors with shape 
        (num_kv_heads, seq_len, head_size) or None if no cartridge.
    """
    from vllm.forward_context import get_forward_context
    
    forward_context = get_forward_context()
    if forward_context.cartridge_kv is None:
        return None
    
    # ❌ FRAGILE: Extract layer index from layer name 
    # (e.g., "model.layers.5.self_attn" -> 5)
    layer_name = getattr(layer, 'layer_name', '')
    import re
    match = re.search(r'layers\.(\d+)', layer_name)
    if match is None:
        logger.warning(f"Could not extract layer index from layer_name='{layer_name}'")
        return None
    
    layer_idx = int(match.group(1))
    return forward_context.cartridge_kv.get(layer_idx)
```

---

### 3. Cartridge Injection During Attention (_forward_with_cartridge_inject)

**WHY BROKEN**: Computing slot_mapping inside attention uses fragile heuristics (inferring block_size from first slot). Modifying slot_mapping and seq_lens affects other code that reads these values. The offset computation assumes single-sequence prefill.

```python
# vllm/v1/attention/backends/flash_attn.py (BROKEN - this method was removed)

def _forward_with_cartridge_inject(
    self,
    layer: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: FlashAttentionMetadata,
    output: torch.Tensor,
    cartridge_kv: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Inject cartridge KV into cache during first prefill, then use normal attention.

    This method writes the cartridge KV into the first cartridge_seq_len cache positions,
    then writes the input KV into subsequent positions, and runs normal single-pass attention.
    """
    cartridge_k, cartridge_v = cartridge_kv
    num_actual_tokens = attn_metadata.num_actual_tokens

    # Cartridge shape: (num_kv_heads_per_gpu, seq_len, head_size)
    # Reshape to: (seq_len, num_kv_heads, head_size) and convert dtype
    cart_seq_len = cartridge_k.shape[1]
    cartridge_k_formatted = cartridge_k.transpose(0, 1).contiguous().to(query.dtype)
    cartridge_v_formatted = cartridge_v.transpose(0, 1).contiguous().to(query.dtype)

    # Get cache
    key_cache, value_cache = kv_cache.unbind(0)

    # ❌ FRAGILE: Infer block_size from the slot_mapping and block_table
    # This fails when first_block_num is 0 (division by zero)
    first_input_slot = attn_metadata.slot_mapping[0].item()
    first_block_num = attn_metadata.block_table[0, 0].item()
    block_size = first_input_slot // first_block_num if first_block_num > 0 else 16

    positions = torch.arange(cart_seq_len, device=cartridge_k_formatted.device, dtype=torch.long)
    virtual_blocks = positions // block_size
    block_offsets = positions % block_size

    # ❌ ASSUMES single sequence during prefill
    physical_blocks = attn_metadata.block_table[0, virtual_blocks]
    cartridge_slots = physical_blocks * block_size + block_offsets

    # Write cartridge KV to cache
    reshape_and_cache_flash(
        cartridge_k_formatted, cartridge_v_formatted,
        key_cache, value_cache,
        cartridge_slots, self.kv_cache_dtype,
        layer._k_scale, layer._v_scale,
    )

    # ❌ PROBLEM: Modifying slot_mapping affects other layers/code
    if self.kv_sharing_target_layer_name is None and key is not None and value is not None:
        offset_slot_mapping = attn_metadata.slot_mapping + cart_seq_len
        reshape_and_cache_flash(
            key, value, key_cache, value_cache,
            offset_slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale, layer._v_scale,
        )

    # ❌ PROBLEM: Creating new tensors for seqused_k but max_seqlen_k is modified
    seqused_k = attn_metadata.seq_lens + cart_seq_len  # Add cartridge length
    max_seqlen_q = attn_metadata.max_query_len
    max_seqlen_k = attn_metadata.max_seq_len + cart_seq_len  # Add cartridge length

    flash_attn_varlen_func(
        q=query[:num_actual_tokens],
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=attn_metadata.query_start_loc,
        max_seqlen_q=max_seqlen_q,
        seqused_k=seqused_k,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=self.scale,
        causal=attn_metadata.causal,
        block_table=attn_metadata.block_table,
        ...
    )
    return output
```

---

### 4. Two-Pass Attention with Merge (_forward_with_cartridge)

**WHY BROKEN**: Running attention twice doubles compute cost. Non-causal attention on cartridge ignores RoPE positioning. Replicating cartridge per sequence wastes memory. merge_attn_states has numerical edge cases.

```python
# vllm/v1/attention/backends/flash_attn.py (BROKEN - this method was removed)

def _forward_with_cartridge(
    self, layer, query, key, value, kv_cache, attn_metadata, output, cartridge_kv
) -> torch.Tensor:
    """Two-pass attention: cartridge + regular cache, then merge.
    
    ⚠️ THIS APPROACH IS FUNDAMENTALLY BROKEN - DO NOT USE ⚠️
    """
    cartridge_k, cartridge_v = cartridge_kv
    cartridge_seq_len = cartridge_k.shape[1]  # (num_kv_heads, seq_len, head_size)
    
    # Transpose: (num_kv_heads, seq_len, head_size) -> (seq_len, num_kv_heads, head_size)
    cartridge_k_t = cartridge_k.transpose(0, 1).contiguous().to(query.dtype)
    cartridge_v_t = cartridge_v.transpose(0, 1).contiguous().to(query.dtype)
    
    # ❌ MEMORY WASTE: Replicating cartridge for each sequence in batch
    num_seqs = attn_metadata.query_start_loc.shape[0] - 1
    cu_seqlens_k_cartridge = torch.arange(
        0, (num_seqs + 1) * cartridge_seq_len, cartridge_seq_len,
        device=query.device, dtype=torch.int32
    )
    cartridge_k_batch = cartridge_k_t.repeat(num_seqs, 1, 1)
    cartridge_v_batch = cartridge_v_t.repeat(num_seqs, 1, 1)
    
    # ══════════════════════════════════════════════════════════════════════
    # PASS 1: Attention over cartridge KV (non-causal)
    # ❌ NO ROPE APPLIED TO CARTRIDGE POSITIONS!
    # ══════════════════════════════════════════════════════════════════════
    cartridge_output, cartridge_lse = flash_attn_varlen_func(
        q=query[:num_actual_tokens],
        k=cartridge_k_batch,
        v=cartridge_v_batch,
        cu_seqlens_q=attn_metadata.query_start_loc,
        cu_seqlens_k=cu_seqlens_k_cartridge,
        max_seqlen_q=attn_metadata.max_query_len,
        max_seqlen_k=cartridge_seq_len,
        softmax_scale=self.scale,
        causal=False,  # ❌ Non-causal ignores RoPE positioning!
        return_softmax_lse=True,
        fa_version=self.vllm_flash_attn_version,
    )
    
    # Write input KV to cache (same as normal path)
    key_cache, value_cache = kv_cache.unbind(0)
    reshape_and_cache_flash(
        key, value, key_cache, value_cache,
        attn_metadata.slot_mapping, self.kv_cache_dtype,
        layer._k_scale, layer._v_scale,
    )
    
    # ══════════════════════════════════════════════════════════════════════
    # PASS 2: Regular attention with KV cache
    # ══════════════════════════════════════════════════════════════════════
    suffix_output, suffix_lse = flash_attn_varlen_func(
        q=query[:num_actual_tokens],
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=attn_metadata.query_start_loc,
        max_seqlen_q=attn_metadata.max_query_len,
        seqused_k=attn_metadata.seq_lens,
        max_seqlen_k=attn_metadata.max_seq_len,
        softmax_scale=self.scale,
        causal=attn_metadata.causal,
        block_table=attn_metadata.block_table,
        return_softmax_lse=True,
        ...
    )
    
    # ══════════════════════════════════════════════════════════════════════
    # Merge outputs using log-sum-exp
    # ❌ NUMERICAL ISSUES when one LSE is much larger than the other
    # ══════════════════════════════════════════════════════════════════════
    merge_attn_states(
        output[:num_actual_tokens],
        cartridge_output,
        cartridge_lse,
        suffix_output,
        suffix_lse,
    )
    
    return output
```

---

### 5. CUDA Graph Differentiation by Cartridge ID

**WHY BROKEN**: Each unique cartridge needs its own CUDA graph because tensor addresses differ. With many cartridges, this causes memory explosion and slow graph warmup.

```python
# vllm/v1/worker/gpu_model_runner.py (BROKEN - this method was removed)

def _get_cartridge_id_for_batch(
    self, scheduler_output: "SchedulerOutput"
) -> str | None:
    """Get the cartridge_id for the current batch if any requests have cartridges.

    This is used to differentiate CUDA graphs - batches with different cartridges
    need separate graphs since cartridge KV tensors have different memory addresses.
    
    ⚠️ PROBLEM: This causes memory explosion with many cartridges!

    Returns:
        The cartridge_id of the first request with a cartridge, or None if no
        requests have cartridges. If multiple cartridges exist, only the first
        is returned (matching _get_cartridge_kv_for_batch behavior).
    """
    scheduled_req_ids = list(scheduler_output.num_scheduled_tokens.keys())
    for req_id in scheduled_req_ids:
        cartridge_ref = self.cartridge_kv_refs.get(req_id)
        if cartridge_ref is not None:
            return cartridge_ref[2]  # (keys, values, cartridge_id)
    return None
```

---

### 6. Mixed Cartridge Batch Handling

**WHY BROKEN**: Only using the first cartridge in a batch means other requests attend to the wrong cartridge. This silently produces incorrect results without error.

```python
# vllm/v1/worker/gpu_model_runner.py (BROKEN - this method was removed)

def _get_cartridge_kv_for_batch(
    self, scheduler_output: "SchedulerOutput"
) -> dict[int, tuple[torch.Tensor, torch.Tensor]] | None:
    """Get cartridge KV for the current batch if any requests have active cartridges.
    
    Uses a shared GPU cache keyed by cartridge_id to avoid re-transferring 
    the same cartridge. Multiple requests using the same cartridge will share
    the GPU tensors even after IPC (unlike data_ptr which changes).
    
    ⚠️ LIMITATION: If requests have different cartridges, only the first is used.
    This is logged once to avoid spam.
    """
    scheduled_req_ids = list(scheduler_output.num_scheduled_tokens.keys())
    
    first_ref: tuple[torch.Tensor, torch.Tensor, str] | None = None
    first_cartridge_id: str | None = None
    has_conflict = False
    
    for req_id in scheduled_req_ids:
        cartridge_ref = self.cartridge_kv_refs.get(req_id)
        if cartridge_ref is not None:
            cartridge_id = cartridge_ref[2]  # (keys, values, cartridge_id)
            if first_ref is None:
                first_ref = cartridge_ref
                first_cartridge_id = cartridge_id
            elif cartridge_id != first_cartridge_id:
                has_conflict = True  # ❌ Different cartridges in same batch!
    
    if first_ref is None:
        return None
    
    # ❌ Only warns, uses first cartridge for ALL requests - silent corruption!
    if has_conflict:
        if not hasattr(self, '_mixed_cartridge_warned'):
            self._mixed_cartridge_warned = True
            logger.warning(
                "Batch has requests with different cartridges. "
                "Only the first cartridge will be used for all requests in batch. "
                "For correct results, use same cartridge for concurrent requests "
                "or reduce MAX_NUM_SEQS to 1."
            )
    
    stacked_keys, stacked_values, cartridge_id = first_ref
    
    # Check GPU cache first - keyed by cartridge_id (survives IPC)
    if cartridge_id in self._gpu_cartridge_cache:
        return self._gpu_cartridge_cache[cartridge_id]
    
    # Transfer to GPU and cache
    num_layers = stacked_keys.shape[0]
    layer_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer_idx in range(num_layers):
        layer_kv[layer_idx] = (
            stacked_keys[layer_idx].to(self.device, non_blocking=True),
            stacked_values[layer_idx].to(self.device, non_blocking=True),
        )
    
    self._gpu_cartridge_cache[cartridge_id] = layer_kv
    return layer_kv
```

---

### 7. execute_model() with Broken Context

**WHY BROKEN**: Passing cartridge_kv through set_forward_context and including cartridge_id in BatchDescriptor ties the forward pass to specific cartridge tensors, breaking CUDA graph reuse.

```python
# vllm/v1/worker/gpu_model_runner.py (BROKEN - these lines were removed)

# In execute_model():

# ❌ REMOVED: BatchDescriptor included cartridge_id for CUDA graph dispatch
batch_descriptor = BatchDescriptor(
    num_tokens=num_input_tokens,
    uniform_decode=uniform_decode,
    has_lora=len(self.input_batch.lora_id_to_lora_request) > 0,
    cartridge_id=self._get_cartridge_id_for_batch(scheduler_output),  # ❌ REMOVED
)

# ❌ REMOVED: Cartridge KV passed through forward context to attention layer
cartridge_kv = self._get_cartridge_kv_for_batch(scheduler_output)

with set_forward_context(
    attn_metadata,
    self.vllm_config,
    cudagraph_runtime_mode=cudagraph_runtime_mode,
    batch_descriptor=batch_descriptor,
    ubatch_slices=ubatch_slices,
    cartridge_kv=cartridge_kv,  # ❌ REMOVED - no longer needed
):
    hidden_states = self.model(...)
```

---

## Phase 2: Pre-Populate Broken (Refactor)

### Commit: `66138457a` - "refactor: simplify cartridge kv caching"

The refactor had the **right architecture** (pre-populate) but was **incomplete**. It removed the in-attention machinery without adding the necessary infrastructure.

### What the Refactor Added (Correct)

```python
# ✅ ADDED: Pre-populate cartridge KV into cache BEFORE forward pass
def _prepopulate_cartridge_to_cache(self, block_ids, stacked_keys, stacked_values):
    """Write cartridge KV to cache slots 0..cart_seq_len-1."""
    # Compute slot mapping for cartridge positions
    positions = torch.arange(cart_seq_len, device=self.device)
    slot_mapping = physical_blocks * block_size + block_offsets
    
    # Write to each layer's cache
    for layer_idx in range(num_layers):
        reshape_and_cache_flash(layer_keys, layer_values, key_cache, value_cache, slot_mapping, ...)

# ✅ ADDED: Offset positions/slots so input tokens come AFTER cartridge
if req_id in self.cartridge_position_offsets:
    positions_np[token_idx:token_idx + n_tokens] += offset

# ✅ ADDED: Include cartridge in seq_lens for attention
self.seq_lens.np[req_idx] += self.cartridge_position_offsets[req_id]
```

### What the Refactor Removed (Correct Direction)

```python
# ❌ REMOVED: No longer pass cartridge_kv to attention layer
- cartridge_kv = self._get_cartridge_kv_for_batch(scheduler_output)
- set_forward_context(..., cartridge_kv=cartridge_kv)

# ❌ REMOVED: No per-cartridge CUDA graph differentiation needed
- cartridge_id=self._get_cartridge_id_for_batch(scheduler_output)

# ❌ REMOVED: 316 lines of in-attention cartridge handling from flash_attn.py
- def _forward_with_cartridge(...)
- def _get_cartridge_kv_for_layer(...)
```

### What the Refactor FORGOT (Caused Breakage)

**1. Removed GPU cache without replacement:**
```python
# ❌ REMOVED but not replaced:
- self._gpu_cartridge_cache: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}
- def _get_cartridge_kv_for_batch()  # Had GPU caching logic

# Result: Every request re-transferred cartridge CPU→GPU (~50-100ms each)
```

**2. No prefix cache hit detection:**
```python
# ❌ MISSING: Always pre-populated, even when blocks already filled
self._prepopulate_cartridge_to_cache(block_ids, stacked_keys, stacked_values)
# Should check: if new_req_data.num_computed_tokens < cart_seq_len
```

**3. API layer not updated:**
```python
# ❌ MISSING: cartridge_seq_len and cartridge_shm_path not passed through API
# IPC was still trying to serialize full cartridge tensors
```

**4. Reference counting not added:**
```python
# ❌ MISSING: No tracking of which requests use which cartridges
# GPU cache would grow unbounded without cleanup
```

### Subsequent Fixes (Jan 14-17)

| Commit | Fix |
|--------|-----|
| `956d4779e` | Check `num_computed_tokens < cart_seq_len` before pre-populating |
| `2e5687d5c` | Add `gpu_cartridge_cache` dict |
| `7c774494e` | Implement prefix sharing logic |
| `b49ee440e` | Fix prefix caching, fix scheduling, add logging |
| `4507398c5` | Track `cartridge_cache_hit`, skip resending for shared prefixes |
| `6efa2583c` | Prevent duplicate IPC sends |
| `a0ded9652` | **MAJOR** (+3114 lines): `cartridge_loader.py`, `adapter_manager.py`, GPU caching with ref counts, scheduler integration, IPC via shared memory |
| `6dca3074c` | Use shared memory for IPC (avoiding serialization) |

---

## Specific Bugs in Phase 1 (In-Attention Approach)

These bugs existed in the original in-attention implementation and were part of why the pre-populate approach is better.

### Bug 1: Block Size Inference Was Fragile

**WHY BROKEN**: Dividing first slot by first block number fails when first block is 0 (division by zero or incorrect block_size of 16).

```python
# ❌ BROKEN:
first_input_slot = attn_metadata.slot_mapping[0].item()
first_block_num = attn_metadata.block_table[0, 0].item()
block_size = first_input_slot // first_block_num if first_block_num > 0 else 16

# ✅ CORRECT: Use config directly
block_size = self.cache_config.block_size
```

### Bug 2: slot_mapping Modification Affected Other Code

**WHY BROKEN**: `attn_metadata` is shared across layers. Creating `offset_slot_mapping = slot_mapping + cart_seq_len` doesn't modify the original, but the offset isn't applied consistently.

```python
# ❌ BROKEN: Creates new tensor in attention, inconsistent with other code
offset_slot_mapping = attn_metadata.slot_mapping + cart_seq_len

# ✅ CORRECT: Modify positions BEFORE slot_mapping is computed (in _prepare_inputs)
positions_np[token_idx:token_idx + n_tokens] += offset
self.input_batch.block_table.compute_slot_mapping(req_indices, positions_np)
```

### Bug 3: seq_lens Modified at Wrong Time

**WHY BROKEN**: Creating `seqused_k = seq_lens + cart_seq_len` during attention means each layer creates a new tensor. The original seq_lens is unchanged, but attention sees inconsistent values if this logic has bugs.

```python
# ❌ BROKEN: New tensor created during forward pass, potential inconsistency
seqused_k = attn_metadata.seq_lens + cart_seq_len

# ✅ CORRECT: Modified once before forward pass, all layers see same value
self.seq_lens.np[req_idx] += self.cartridge_position_offsets[req_id]
self.seq_lens.copy_to_gpu()
```

### Bug 4: RoPE Positions Not Adjusted Consistently

**WHY BROKEN**: Cartridge KV was computed with positions 0..N-1, but input tokens also used 0..M-1 instead of N..N+M-1. The two-pass approach with non-causal attention ignores RoPE entirely for cartridge.

```python
# ❌ BROKEN: Non-causal attention ignores positional information
cartridge_output, cartridge_lse = flash_attn_varlen_func(
    q=query, k=cartridge_k, v=cartridge_v,
    causal=False,  # No RoPE effect!
    ...
)

# ✅ CORRECT: Input positions offset so RoPE is correct
positions_np[token_idx:token_idx + n_tokens] += cart_seq_len
# Now input token 0 has position cart_seq_len, giving correct relative positions
```

### Bug 5: Two-Pass Merge Had Numerical Issues

**WHY BROKEN**: `merge_attn_states` using log-sum-exp has numerical issues when one LSE is much larger than the other, potentially causing NaN or incorrect outputs.

### Bug 6: CUDA Graph Incompatibility

**WHY BROKEN**: Each cartridge has different tensor addresses, so passing through ForwardContext requires separate CUDA graphs. With N cartridges, you need N graphs per batch size.

```python
# ❌ BROKEN: CUDA graph per cartridge
batch_descriptor = BatchDescriptor(
    cartridge_id=self._get_cartridge_id_for_batch(scheduler_output),
)

# ✅ CORRECT: No cartridge in batch descriptor, standard graphs work
batch_descriptor = BatchDescriptor(
    num_tokens=num_input_tokens,
    uniform_decode=uniform_decode,
    has_lora=has_lora,
    # No cartridge_id - cache is pre-populated, attention is unchanged
)
```

---

## Recommendations for Future Simplification

### 1. Keep the Pre-Population Approach

**WHY**: It's fundamentally simpler - attention sees cartridge as regular cached tokens. No special code in attention layer.

### 2. Consider Lazy Pre-Population

**WHY**: Currently pre-populates on first schedule. Could defer to first forward pass if that simplifies state management.

### 3. Add Comprehensive Tests

**WHY**: The broken implementation failed because there were no tests catching edge cases:
- Multi-sequence batches with cartridges
- Mixed cartridge/non-cartridge batches
- CUDA graph capture with cartridges
- Tensor parallelism with cartridges
- Prefix cache with cartridges

---

## Appendix: Additional Gotchas from Commit History

### Gotcha 1: IPC Serialization of Large Tensors

**Commit**: `6dca3074c` - "use shm"

**Problem**: Sending full KV tensors through multiprocessing queues is slow (~100s of MB per cartridge).

```python
# ❌ BAD: Serialize full tensor via IPC
return prompt_token_ids, prefix_kv, prefix_id, prefix_seq_len, lora_request
```

```python
# ✅ GOOD: Pass shared memory path instead of tensor
return prompt_token_ids, prefix_kv, prefix_id, prefix_seq_len, prefix_shm_path, lora_request
```

**Why**: File I/O via shared memory is faster than pickle serialization for large tensors.

### Gotcha 2: CUDA Graphs Need Per-Cartridge Differentiation (in Phase 1)

**Commit**: `5667957b1` - "fix: make cartridges compatible with cudagraphs"

**Problem**: In-attention approach needed different CUDA graphs for different cartridges because tensor addresses differ.

```python
# ❌ PHASE 1 NEEDED THIS: Different cartridges = different graphs
class BatchDescriptor(NamedTuple):
    num_tokens: int
    uniform_decode: bool = False
    has_lora: bool = False
    cartridge_id: str | None = None  # Different cartridges need different graphs!
```

```python
# ✅ PHASE 3: No cartridge_id needed - pre-populate means standard graphs work
class BatchDescriptor(NamedTuple):
    num_tokens: int
    uniform_decode: bool = False
    has_lora: bool = False
    # No cartridge_id - cache is pre-populated, attention is unchanged
```

**Why**: Pre-populated cache is just regular KV cache from attention's perspective.

### Gotcha 3: Tensor Parallelism Requires KV Head Sharding

**Commit**: `1df05a67e` - "shard cartridge for TP"

**Problem**: Cartridge has ALL KV heads, but each GPU only has `num_kv_heads / tp_size` heads.

```python
# ❌ BAD: Use full cartridge on each GPU
stacked_keys, stacked_values = new_req_data.cartridge_kv
# crash: shape mismatch with model's per-GPU KV heads!
```

```python
# ✅ GOOD: Shard by TP rank before using
tp_size = self.parallel_config.tensor_parallel_size
if tp_size > 1 and total_kv_heads >= tp_size:
    tp_rank = get_tp_group().rank_in_group
    heads_per_gpu = total_kv_heads // tp_size
    start_head = tp_rank * heads_per_gpu
    end_head = start_head + heads_per_gpu
    # Slice: (num_layers, total_kv_heads, seq_len, head_dim) -> (num_layers, heads_per_gpu, ...)
    stacked_keys = stacked_keys[:, start_head:end_head, :, :].contiguous()
    stacked_values = stacked_values[:, start_head:end_head, :, :].contiguous()
```

**Why**: FlashAttention expects head count to match the GPU's portion.

### Gotcha 4: Tensor Shape Format Ambiguity

**Problem**: Cartridges from different sources have different shapes.
**Solution**: Detect format and normalize to `(num_layers, num_kv_heads, seq_len, head_dim)`.
**Why**: Consistent shape prevents silent corruption from misinterpreted dimensions.

### Gotcha 5: GPU Cache Memory Leaks

**Commit**: `a0ded9652`

**Problem**: GPU tensors not freed when cartridge no longer used.

```python
# ❌ BAD: Just delete offset, GPU tensors leak
for req_id in scheduler_output.finished_req_ids:
    self.cartridge_position_offsets.pop(req_id, None)
```

```python
# ✅ GOOD: Reference counting - free when ref count hits zero
# Track references
self._cartridge_ref_counts: dict[str, int] = {}
self._request_to_cartridge: dict[str, str] = {}

# On request finish:
for req_id in scheduler_output.finished_req_ids:
    self.cartridge_position_offsets.pop(req_id, None)
    cartridge_id = self._request_to_cartridge.pop(req_id, None)
    if cartridge_id is not None:
        self._cartridge_ref_counts[cartridge_id] -= 1
        if self._cartridge_ref_counts[cartridge_id] <= 0:
            # No more requests using this cartridge - free GPU memory
            del self._cartridge_ref_counts[cartridge_id]
            if cartridge_id in self.gpu_cartridge_cache:
                del self.gpu_cartridge_cache[cartridge_id]
```

**Why**: Multiple requests can share same cartridge; only free when all done.

### Gotcha 6: Cascade Attention Prefix Length Calculation

**Commit**: `b49ee440e` - "fix prefix caching"

**Problem**: Cascade attention uses `num_computed_tokens` for prefix, but cartridge tokens aren't "computed" in normal sense.

```python
# ❌ BAD: Ignore cartridge length
max_cache_hit_length = request.num_tokens - 1
```

```python
# ✅ GOOD: Include cartridge tokens so those blocks can be matched
cart_seq_len = getattr(request, 'cartridge_seq_len', 0)
max_cache_hit_length = request.num_tokens + cart_seq_len - 1
```

**Why**: Cascade attention needs to know full prefix length including cartridge.

### Gotcha 7: Scheduler Budget Accounting for Prefix Cache Hits

**Problem**: Counting cartridge in budget even on cache hit starves other requests.

```python
# ❌ BAD: Always count cartridge in budget
budget_cost = num_new_tokens + cart_seq_len
```

```python
# ✅ GOOD: Only count cartridge on cache miss
if not cartridge_cache_hit:
    budget_cost = num_new_tokens + cart_seq_len
else:
    budget_cost = num_new_tokens  # Cartridge blocks already in cache
```

**Why**: Cache hits share existing memory; no additional tokens need processing.

### Gotcha 8: Only Populate Cache Once

**Problem**: Re-scheduling a request would re-populate, overwriting valid cache.

```python
# ❌ BAD: Always populate
self._prepopulate_cartridge_to_cache(block_ids, layer_kv_list, cart_seq_len)
```

```python
# ✅ GOOD: Check cartridge_cache_hit before populating
if not cartridge_cache_hit:
    self._prepopulate_cartridge_to_cache(block_ids, layer_kv_list, cart_seq_len)
```

**Why**: Cartridge slots are written once; input tokens append after.

### Gotcha 9: Corrupted Downloads Need Retry Logic

**Problem**: Partial S3 downloads produce corrupted files that fail silently.
**Solution**: Validate after download, delete and retry on failure.
**Why**: `torch.load` on corrupted file can give wrong data without error.

### Gotcha 10: Don't Resend Cartridge for Prefix-Shared Requests

**Commit**: `4507398c5` - "dont resend cartridge to shared prefixes"

**Problem**: Multiple requests with same prefix sent cartridge KV repeatedly via IPC.

```python
# ❌ BAD: Always send cartridge_kv via IPC
return cls(
    ...
    cartridge_kv=request.cartridge_kv,  # ~100s of MB serialized every time!
)
```

```python
# ✅ GOOD: Skip sending when cache already has it
# When cartridge_cache_hit is True, don't send cartridge_kv via IPC.
# The worker already has the tensors cached in gpu_cartridge_cache,
# and the KV block cache already has the cartridge blocks populated.
cartridge_kv = None if cartridge_cache_hit else request.cartridge_kv
return cls(
    ...
    cartridge_kv=cartridge_kv,
)
```

**Why**: Prefix cache means KV is already in blocks; no need to transfer again.

### Gotcha 11: Prefix Cache Block Hashing Must Include cartridge_id

**Commit**: `b49ee440e` - "fix prefix caching"

**Problem**: Without cartridge_id in hash, different cartridges could share cache incorrectly.

```python
# ❌ BAD: Hash only request tokens
hash_token_ids = list(request.all_token_ids)
```

```python
# ✅ GOOD: Prepend virtual tokens derived from cartridge_id
def _generate_cartridge_virtual_tokens(cartridge_id: str, seq_len: int, hash_fn):
    """Generate deterministic virtual token IDs for cartridge prefix sharing."""
    base_hash = int.from_bytes(hash_fn(cartridge_id)[:4], 'big')
    return [-(base_hash + i) for i in range(seq_len)]

if cart_seq_len > 0 and cart_id:
    virtual_tokens = _generate_cartridge_virtual_tokens(cart_id, cart_seq_len, caching_hash_fn)
    hash_token_ids = virtual_tokens + list(request.all_token_ids)
else:
    hash_token_ids = list(request.all_token_ids)
```

**Why**: Different cartridges have different KV content and RoPE offsets; must not share blocks.

### Gotcha 12: Adjust Token Indices for MM/LoRA After Cartridge Portion

**Commit**: `b49ee440e` - "fix prefix caching"

**Problem**: Token indices for multimodal/LoRA hash key computation were off by cart_seq_len.

```python
# ❌ BAD: Use raw indices (includes cartridge)
extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
    request, start_token_idx, end_token_idx, curr_mm_idx
)
```

```python
# ✅ GOOD: Adjust indices for actual request tokens (after cartridge portion)
req_start = max(0, start_token_idx - cart_seq_len)
req_end = max(0, end_token_idx - cart_seq_len)
extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
    request, req_start, req_end, curr_mm_idx
)
```

**Why**: Hash key computation expects indices into request tokens, not including virtual cartridge tokens.

### Gotcha 13: TrainableCache vs Legacy Tensor Shape Format

**Commit**: `1549dac2b` - "fix: handle TrainableCache tensor shape correctly"

**Problem**: The cartridges library saves trainable KV cache as `(1, num_kv_heads, seq_len, head_dim)`, but the loader was assuming `(1, seq_len, num_kv_heads, head_dim)`. This caused garbled output because attention was computing with incorrectly shaped tensors.

**Solution**: Added heuristic detection to handle both formats:

```python
# After squeeze(0), we have shape (A, B, C)
# Need to determine if this is:
# - TrainableCache: (num_kv_heads, seq_len, head_dim) - already correct
# - Legacy: (seq_len, num_kv_heads, head_dim) - needs permute

tensor = tensor.squeeze(0)
dim0, dim1, dim2 = tensor.shape

# Heuristic: num_kv_heads is typically small (power of 2, <= 128)
# and head_dim is typically 64, 128, 256
is_trainable_cache_format = (
    dim0 <= 128 and
    (dim0 & (dim0 - 1)) == 0 and  # power of 2
    dim2 in (64, 80, 96, 128, 256)  # common head_dim values
)

if not is_trainable_cache_format:
    # Legacy format needs permute
    tensor = tensor.permute(1, 0, 2)
```

**Why**: Different cartridge sources use different axis orderings; detecting and handling both prevents silent data corruption.

### Gotcha 14: Early Validation with Clear Error Messages

**Commit**: `8e94b5d46` - "fix: propagate shape errors to client"

**Problem**: Shape mismatches between cartridge and model were caught deep in the worker, giving cryptic errors. Users couldn't tell why their cartridge wasn't working.

**Solution**: Validate cartridge compatibility at the API level (before sending to workers) with clear error messages:

```python
# In serving_engine.py, before sending to engine:
if stacked_cartridge_kv is not None:
    stacked_keys = stacked_cartridge_kv[0]
    _, cart_kv_heads, cart_seq_len, cart_head_dim = stacked_keys.shape
    expected_kv_heads = self.model_config.get_total_num_kv_heads()
    expected_head_size = self.model_config.get_head_size()
    
    if cart_kv_heads != expected_kv_heads:
        raise ValueError(
            f"Cartridge incompatible with model {self.model_config.model}: "
            f"cartridge has {cart_kv_heads} KV heads but model expects {expected_kv_heads}. "
            f"The cartridge may have been trained on a different model."
        )
```

**Why**: Early validation with descriptive errors helps users understand and fix problems quickly.

### Gotcha 15: GPU Cartridge Cache for Performance

**Commit**: `2e5687d5c` - "Add GPU cartridge cache to avoid repeated CPU->GPU transfers"

**Problem**: Every request with a cartridge was transferring the same large tensors from CPU to GPU, even if the cartridge was already on GPU from a previous request.

**Solution**: Cache cartridge tensors on GPU by cartridge_id:

```python
# Check GPU cache first to avoid repeated CPU->GPU transfers
if cartridge_id in self.gpu_cartridge_cache:
    # Use cached GPU tensors (already sharded for TP)
    stacked_keys, stacked_values = self.gpu_cartridge_cache[cartridge_id]
    cart_seq_len = stacked_keys.shape[2]
else:
    # First time seeing this cartridge - validate, shard, and cache
    stacked_keys, stacked_values = new_req_data.cartridge_kv
    # ... validation and sharding ...
    self.gpu_cartridge_cache[cartridge_id] = (stacked_keys, stacked_values)
```

**Why**: CPU→GPU transfer for a 4096-token cartridge takes ~50-100ms; caching eliminates this for repeated use.

### Gotcha 16: Skip Pre-population on Prefix Cache Hit

**Commit**: `956d4779e` - "only populate cache with cartridge once"

**Problem**: When prefix cache hit, the cartridge blocks were already filled, but we were still calling `_prepopulate_cartridge_to_cache`, wasting GPU cycles.

**Solution**: Check `num_computed_tokens` before pre-populating:

```python
# Only pre-populate if prefix cache didn't already cover it.
# When num_computed_tokens >= cart_seq_len, the cartridge blocks are
# already filled from prefix cache hit - skip the expensive kernel calls.
if new_req_data.num_computed_tokens < cart_seq_len:
    self._prepopulate_cartridge_to_cache(
        new_req_data.block_ids, stacked_keys, stacked_values
    )
# Always store position offset - needed for slot_mapping and RoPE
self.cartridge_position_offsets[req_id] = cart_seq_len
```

**Why**: Redundant `reshape_and_cache_flash` calls waste GPU cycles and can cause correctness issues if blocks are partially overwritten.

### Gotcha 17: Download Progress and Cleanup for Large Cartridges

**Commit**: `a676c8c33` - "fix: overwrite corrupted downloads, throw error properly"

**Problem**: Large cartridge downloads from S3 (1-2GB) could fail partway, leaving corrupted files. Retry would use the corrupted cached file. Also, no progress logging made it look like the server was hung.

**Solution**: 
1. Clean up stale temp files before download
2. Add progress logging during download
3. Validate downloaded files before use

```python
def _copy_with_progress(*, reason: str) -> None:
    """Copy with progress logging (polls destination size every 10s)."""
    start_t = time.time()
    stop_evt = threading.Event()
    
    def _log_progress():
        while not stop_evt.wait(10.0):
            size = _get_download_size()
            logger.info(f"Adapter download progress: {size / 1024 / 1024:.2f} MB")
    
    t = threading.Thread(target=_log_progress, daemon=True)
    t.start()
    try:
        source_path.copy(str(cache_path))
    finally:
        stop_evt.set()

def _cleanup_temp_files():
    """Remove stale boto3 temp files from interrupted downloads."""
    temp_pattern = str(cache_path) + ".*"
    for temp_file in glob.glob(temp_pattern):
        Path(temp_file).unlink()
```

**Why**: Large file downloads need progress feedback and proper cleanup to avoid corrupt cached files.

---

## Summary Table

| Aspect | Broken Implementation | Correct Implementation | Winner |
|--------|----------------------|------------------------|--------|
| Simplicity | Complex (2-pass attention, merge) | Simple (pre-populate, offset) | ✅ Correct |
| Correctness | Multiple bugs | Works correctly | ✅ Correct |
| Performance | 2x attention calls | Single attention pass | ✅ Correct |
| CUDA Graphs | Per-cartridge graphs needed | Standard graphs work | ✅ Correct |
| Memory | Replicates cartridge per seq | Shares via cache | ✅ Correct |
| Maintainability | Special code in attention | No special attention code | ✅ Correct |

**Conclusion**: Pre-populate the KV cache before the forward pass. **Never handle cartridges inside the attention layer.**
