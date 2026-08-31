#!/bin/bash
set -euo pipefail

endpoint=${1:?endpoint is required}
artifact_file=${2:?artifact file is required}
lengths=${BENCH_LENGTHS:-128 256 512 1024 2048 4096 8192}
repeats=${BENCH_REPEATS:-2}
warmup_length=${BENCH_WARMUP_LENGTH:-1024}
readonly concurrency=64
base_model=glm-5.3-flash
lora_model=step0-lora-r32
smoke_file=${artifact_file%.jsonl}-smoke.json

install -d -m 0755 "$(dirname "$artifact_file")"
test ! -e "$artifact_file"
test ! -e "$smoke_file"
curl -fsS "$endpoint/health" >/dev/null
models=$(curl -fsS "$endpoint/v1/models")
jq -e --arg model "$base_model" '.data | any(.id == $model)' <<<"$models" >/dev/null
jq -e --arg model "$lora_model" '.data | any(.id == $model)' <<<"$models" >/dev/null

request_body() {
    local model=$1
    local sequence_length=$2
    jq -nc --arg model "$model" --argjson max_tokens "$sequence_length" '{
        model: $model,
        messages: [{
            role: "user",
            content: "Count upward from 1, writing one integer per line."
        }],
        temperature: 0,
        ignore_eos: true,
        max_tokens: $max_tokens
    }'
}

smoke_dir=$(mktemp -d /var/tmp/glm-step0-lora-smoke.XXXXXX)
trap 'rm -rf -- "$smoke_dir"' EXIT
for model in "$base_model" "$lora_model"; do
    body=$(request_body "$model" 64)
    curl -fsS --max-time 1200 "$endpoint/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "X-data-parallel-rank: 0" \
        --data-binary "$body" \
        -o "$smoke_dir/$model.json"
done
for model in "$base_model" "$lora_model"; do
    tokens=$(jq -r '.usage.completion_tokens' "$smoke_dir/$model.json")
    test "$tokens" -eq 64
done
jq -n \
    --arg base_model "$base_model" \
    --arg lora_model "$lora_model" \
    '{base_model:$base_model,lora_model:$lora_model,dp_rank:0,
      completion_tokens_per_response:64,status:"passed"}' \
    >"$smoke_file"
rm -rf -- "$smoke_dir"
trap - EXIT

run_once() {
    local model=$1
    local sequence_length=$2
    local repeat=$3
    local label=$4
    local run_dir
    run_dir=$(mktemp -d "/var/tmp/glm-step0-lora-${label}.XXXXXX")
    local body
    body=$(request_body "$model" "$sequence_length")
    export body endpoint run_dir

    local start_ns end_ns
    start_ns=$(date +%s%N)
    # shellcheck disable=SC2016
    seq 1 "$concurrency" | xargs -P"$concurrency" -I{} bash -c '
        curl -sS --max-time 1200 "$endpoint/v1/chat/completions" \
            -H "Content-Type: application/json" \
            --data-binary "$body" \
            -o "$run_dir/{}.json" \
            -w "%{http_code}\n" >"$run_dir/{}.code"
    '
    end_ns=$(date +%s%N)

    local total_tokens=0
    local errors=0
    local index code tokens
    for index in $(seq 1 "$concurrency"); do
        code=$(<"$run_dir/$index.code")
        tokens=$(jq -r '.usage.completion_tokens // 0' "$run_dir/$index.json")
        if test "$code" != 200 || test "$tokens" != "$sequence_length"; then
            errors=$((errors + 1))
        fi
        total_tokens=$((total_tokens + tokens))
    done

    jq -nc \
        --arg request_model "$model" \
        --arg label "$label" \
        --argjson sequence_length "$sequence_length" \
        --argjson repeat "$repeat" \
        --argjson concurrency "$concurrency" \
        --argjson total_completion_tokens "$total_tokens" \
        --argjson errors "$errors" \
        --argjson elapsed_ns "$((end_ns - start_ns))" \
        '{request_model:$request_model,label:$label,
          sequence_length:$sequence_length,repeat:$repeat,
          concurrency:$concurrency,total_completion_tokens:$total_completion_tokens,
          errors:$errors,wall_seconds:($elapsed_ns / 1000000000),
          completion_tokens_per_second:
              ($total_completion_tokens / ($elapsed_ns / 1000000000))}' \
        | tee -a "$artifact_file"
    rm -rf -- "$run_dir"
    test "$errors" -eq 0
    test "$total_tokens" -eq $((concurrency * sequence_length))
}

run_once "$base_model" "$warmup_length" 0 warmup-base
run_once "$lora_model" "$warmup_length" 0 warmup-step0-lora
for sequence_length in $lengths; do
    for repeat in $(seq 1 "$repeats"); do
        if test "$((repeat % 2))" -eq 1; then
            run_once "$base_model" "$sequence_length" "$repeat" measured-base
            run_once "$lora_model" "$sequence_length" "$repeat" measured-step0-lora
        else
            run_once "$lora_model" "$sequence_length" "$repeat" measured-step0-lora
            run_once "$base_model" "$sequence_length" "$repeat" measured-base
        fi
    done
done

curl -fsS "$endpoint/health" >/dev/null
sha256sum "$artifact_file" "$smoke_file"
