#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/pyproject.toml" || -f "$SCRIPT_DIR/setup.py" ]]; then
  REPO_ROOT="$SCRIPT_DIR"
elif [[ -f "$SCRIPT_DIR/../pyproject.toml" || -f "$SCRIPT_DIR/../setup.py" ]]; then
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
else
  echo "error: could not determine repo root from $SCRIPT_DIR" >&2
  exit 1
fi

cd "$REPO_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required but not installed" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3.13}"
VENV_DIR="${VENV_DIR:-.venv}"
VENV_PY="$REPO_ROOT/$VENV_DIR/bin/python"

# Mirror the current engrams Modal vLLM image defaults unless overridden.
VLLM_PRECOMPILED_WHEEL_COMMIT="${VLLM_PRECOMPILED_WHEEL_COMMIT:-a97954b6a8fa41a162ebf58f80a1460a98e0baf0}"
VLLM_MAIN_CUDA_VERSION="${VLLM_MAIN_CUDA_VERSION:-12.8}"
VLLM_CUDA_VARIANT="${VLLM_CUDA_VARIANT:-cu129}"
TORCH_VERSION="${TORCH_VERSION:-2.10.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.25.0}"
FLASHINFER_VERSION="${FLASHINFER_VERSION:-0.6.4}"
FLASHINFER_CUDA_INDEX="${FLASHINFER_CUDA_INDEX:-https://flashinfer.ai/whl/cu128}"
VLLM_NIGHTLY_INDEX="${VLLM_NIGHTLY_INDEX:-https://wheels.vllm.ai/nightly}"

echo "==> repo: $REPO_ROOT"
echo "==> python: $PYTHON_BIN"
echo "==> venv: $VENV_DIR"
echo "==> wheel commit: $VLLM_PRECOMPILED_WHEEL_COMMIT"
echo "==> main CUDA version: $VLLM_MAIN_CUDA_VERSION"
echo "==> torch variant: $VLLM_CUDA_VARIANT"

# Avoid local shell config accidentally forcing uv to load a missing .env file.
unset UV_ENV_FILE
export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_COMMIT
export VLLM_MAIN_CUDA_VERSION

uv venv --python "$PYTHON_BIN" "$VENV_DIR"

find "$REPO_ROOT" -name '*.txt' -o -name 'pyproject.toml' | \
  xargs -r sed -i 's/flashinfer-python==0\.[0-9]*\.[0-9]*/flashinfer-python/g'

uv pip install --python "$VENV_PY" \
  --index-url "https://download.pytorch.org/whl/${VLLM_CUDA_VARIANT}" \
  "torch==${TORCH_VERSION}+${VLLM_CUDA_VARIANT}" \
  "torchvision==${TORCHVISION_VERSION}+${VLLM_CUDA_VARIANT}"

uv pip install --python "$VENV_PY" \
  --extra-index-url "$VLLM_NIGHTLY_INDEX" \
  --extra-index-url "$FLASHINFER_CUDA_INDEX" \
  "vllm @ file://$REPO_ROOT" \
  "huggingface-hub[hf_transfer]" \
  "fastsafetensors" \
  "boto3" \
  "pytest" \
  "pytest-asyncio" \
  "httpx" \
  "openai" \
  "tblib" \
  "flashinfer-python==${FLASHINFER_VERSION}" \
  "flashinfer-cubin==${FLASHINFER_VERSION}" \
  "flashinfer-jit-cache==${FLASHINFER_VERSION}"

echo
echo "Install complete."
echo "Activate with: source \"$VENV_DIR/bin/activate\""
echo "Test with:    python -m pytest tests/test_cartridge_addition_eval.py -v -x -m benchmark"
