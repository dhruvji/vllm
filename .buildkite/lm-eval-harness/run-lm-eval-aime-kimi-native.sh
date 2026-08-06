#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

model_args="pretrained=moonshotai/Kimi-K2.6,revision=100231dd6b04cec5fff2b2f649754f6b760e9476,tensor_parallel_size=16,enable_expert_parallel=true,dtype=bfloat16,language_model_only=true,trust_remote_code=true,max_model_len=65536,max_num_batched_tokens=64,enable_chunked_prefill=true"

run_eval() {
  local profile=$1
  local lora_args=$2
  shift 2

  printf '\nRunning native Kimi AIME 2024 %s\n' "$profile"
  lm_eval \
    --model vllm \
    --model_args "${model_args}${lora_args}" \
    --tasks aime24 \
    --apply_chat_template \
    --batch_size 1 \
    "$@"
}

run_eval "without LoRA" "" "$@"
run_eval "with one empty rank-16 LoRA slot" \
  ",enable_lora=true,max_lora_rank=16,max_loras=1" "$@"
