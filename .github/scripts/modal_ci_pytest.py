#!/usr/bin/env python3
"""Run selected vLLM pytest targets on Modal GPU CI."""

import os
import shlex
import subprocess
import sys

import modal

CUDA_IMAGE = "nvidia/cuda:13.0.2-devel-ubuntu22.04"
PYTHON_VERSION = "3.12"
REPO_URL = "https://github.com/engram-lab/vllm.git"
REPO_DIR = "/tmp/vllm"
SECONDS_PER_MINUTE = 60
CUDA_VARIANT = "cu130"
FLASHINFER_CUDA_INDEX = "https://flashinfer.ai/whl/cu130"
PYTORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu130"
FLASHINFER_VERSION = "0.6.8.post1"

BRANCH = os.environ.get("BRANCH", "main")
COMMIT_HASH = os.environ.get("COMMIT_HASH") or os.environ.get("GITHUB_SHA")
GPU_TYPE = os.environ.get("GPU_TYPE", "A100")
GPU_COUNT = int(os.environ.get("GPU_COUNT", 1))
SECRETS = os.environ.get("SECRETS", "dev-api-keys")
USE_LOCAL_SOURCE = os.environ.get("VLLM_CI_USE_LOCAL_SOURCE") == "1"
LOCAL_SOURCE_DIR = os.environ.get("VLLM_CI_LOCAL_SOURCE_DIR", ".")
PRECOMPILED_WHEEL_COMMIT = os.environ.get(
    "VLLM_PRECOMPILED_WHEEL_COMMIT",
    "9da56fd18b95ff1b11360e8400f7c41b126d190b",
)

BASE_ENV = {
    "VLLM_USE_PRECOMPILED": "1",
    "VLLM_PRECOMPILED_WHEEL_COMMIT": PRECOMPILED_WHEEL_COMMIT,
    "VLLM_PRECOMPILED_WHEEL_VARIANT": CUDA_VARIANT,
    "VLLM_MAIN_CUDA_VERSION": "13.0",
    "VLLM_ATTENTION_BACKEND": "FLASHINFER",
    "VLLM_USE_FLASHINFER_SAMPLER": "1",
}
RUNTIME_ENV = {
    "HF_XET_HIGH_PERFORMANCE": "1",
    "XFORMERS_ENABLE_TRITON": "1",
    "VLLM_SERVER_DEV_MODE": "1",
    "TORCHINDUCTOR_COMPILE_THREADS": "1",
    "VLLM_HTTP_TIMEOUT_KEEP_ALIVE": "300",
    "VLLM_PASSTHROUGH_ERRORS": "1",
}
TORCH_PACKAGES = (
    f"torch==2.11.0+{CUDA_VARIANT}",
    f"torchvision==0.26.0+{CUDA_VARIANT}",
)
FLASHINFER_PACKAGES = (
    f"flashinfer-python=={FLASHINFER_VERSION}",
    f"flashinfer-cubin=={FLASHINFER_VERSION}",
    f"flashinfer-jit-cache=={FLASHINFER_VERSION}+{CUDA_VARIANT}",
)
LOCAL_SOURCE_IGNORE = modal.FilePatternMatcher(
    ".git/**",
    ".env",
    ".cursor/**",
    ".claude/**",
    ".codex/**",
    ".venv/**",
    "**/__pycache__/**",
    ".pytest_cache/**",
    ".ruff_cache/**",
    ".mypy_cache/**",
    "build/**",
    "dist/**",
    "downloads/**",
    "wheels/**",
    "**/*.egg-info/**",
)


def _clone_commands(branch: str, commit_hash: str | None) -> list[str]:
    askpass_setup = (
        "printf '%s\\n' '#!/bin/sh' 'case \"$1\" in' "
        "'*Username*) echo x-access-token ;;' "
        "'*Password*) echo \"$GITHUB_TOKEN\" ;;' 'esac' > /tmp/git-askpass && "
        "python3 -c 'import os; os.chmod(\"/tmp/git-askpass\", 0o700)'"
    )
    auth_env = "GIT_TERMINAL_PROMPT=0 GIT_ASKPASS=/tmp/git-askpass"
    branch_arg = shlex.quote(branch)
    clone_cmd = f"{auth_env} git clone --depth 1 -b {branch_arg} {REPO_URL} {REPO_DIR}"
    if not commit_hash:
        return [askpass_setup, clone_cmd]

    commit_arg = shlex.quote(commit_hash)
    checkout_cmd = (
        f"cd {REPO_DIR} && "
        f"{auth_env} git fetch --depth 1 origin {commit_arg} && "
        f"git checkout {commit_arg} && "
        f"echo 'vllm@{commit_hash}'"
    )
    return [askpass_setup, clone_cmd, checkout_cmd]


def _local_version_override(commit_hash: str | None) -> str:
    if not commit_hash:
        return "0.1.dev0+local"

    suffix = "".join(
        char.lower() if char.isalnum() else "." for char in commit_hash
    ).strip(".")
    return f"0.1.dev0+g{suffix[:12] or 'local'}"


def _add_repo_source(base_image):
    if USE_LOCAL_SOURCE:
        return (
            base_image.add_local_dir(
                local_path=LOCAL_SOURCE_DIR,
                remote_path=REPO_DIR,
                copy=True,
                ignore=LOCAL_SOURCE_IGNORE,
            )
            .env({"VLLM_VERSION_OVERRIDE": _local_version_override(COMMIT_HASH)})
            .run_commands(f"echo 'vllm local source: {REPO_DIR}'")
        )

    return base_image.run_commands(
        *_clone_commands(BRANCH, COMMIT_HASH), secrets=secrets
    )


secrets = [modal.Secret.from_name(SECRETS)]

image = _add_repo_source(
    modal.Image.from_registry(CUDA_IMAGE, add_python=PYTHON_VERSION)
    .entrypoint([])
    .apt_install("git", "rsync")
    .env(BASE_ENV)
    .pip_install("packaging", "setuptools>=77,<82", "wheel", "setuptools-scm")
    .uv_pip_install(
        *TORCH_PACKAGES,
        index_url=PYTORCH_CUDA_INDEX,
    )
    .uv_pip_install(
        *FLASHINFER_PACKAGES,
        extra_index_url=FLASHINFER_CUDA_INDEX,
    )
    .uv_pip_install(
        "huggingface-hub[hf_transfer]==0.36.0",
        "boto3",
        "pytest",
        "pytest-asyncio",
        "httpx",
        "openai",
        "tblib",
    )
)

image = image.run_commands(
    "/.uv/uv pip install --system --no-build-isolation "
    "'vllm @ file:///tmp/vllm' "
    "huggingface-hub[hf_transfer] fastsafetensors boto3 'fastapi<0.137' "
    f"torch==2.11.0+{CUDA_VARIANT} "
    f"torchvision==0.26.0+{CUDA_VARIANT} "
    "prometheus-fastapi-instrumentator==7.1.0 "
    f"flashinfer-python=={FLASHINFER_VERSION} "
    f"flashinfer-cubin=={FLASHINFER_VERSION} "
    f"flashinfer-jit-cache=={FLASHINFER_VERSION}+{CUDA_VARIANT} "
    f"--extra-index-url {FLASHINFER_CUDA_INDEX} "
    f"--torch-backend {CUDA_VARIANT}"
).env({**RUNTIME_ENV, "HF_HUB_ENABLE_HF_TRANSFER": "1"})

# Overlay local tests at runtime so workflow changes are visible in clone mode.
image = image.add_local_dir(
    local_path="./tests",
    remote_path="/mnt/local_tests",
    copy=False,
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
flashinfer_cache_vol = modal.Volume.from_name(
    "flashinfer-cache", create_if_missing=True
)

app = modal.App("vllm-ci-pytest")


@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    timeout=45 * SECONDS_PER_MINUTE,
    secrets=secrets,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
        "/root/.cache/flashinfer": flashinfer_cache_vol,
    },
    enable_memory_snapshot=False,
)
def run_pytest(test_path: str, pytest_args: str = "", env_vars: str = ""):
    print("=" * 80)
    print("vLLM CI PYTEST")
    print("=" * 80)
    print(f"Branch: {BRANCH}")
    print(f"Commit: {COMMIT_HASH}")
    print(f"Use local source: {USE_LOCAL_SOURCE}")
    print(f"Test path: {test_path}")
    print(f"Pytest args: {pytest_args}")
    print(f"GPU allocation: {GPU_COUNT}x {GPU_TYPE}")
    print("=" * 80)

    env = os.environ.copy()
    if env_vars:
        for pair in env_vars.split(","):
            if "=" in pair:
                key, value = pair.split("=", 1)
                env[key.strip()] = value.strip()
                print(f"Set env: {key.strip()}={value.strip()}")

    if (
        env.get("VLLM_TEST_ENABLE_CARTRIDGE")
        and "VLLM_ADAPTER_FORCE_REDOWNLOAD" not in env
    ):
        env["VLLM_ADAPTER_FORCE_REDOWNLOAD"] = "1"
        print("Set env: VLLM_ADAPTER_FORCE_REDOWNLOAD=1")

    print("\nOverlaying local test files onto /tmp/vllm/tests/...")
    subprocess.run(
        ["rsync", "-av", "/mnt/local_tests/", f"{REPO_DIR}/tests/"],
        check=True,
    )

    os.chdir(REPO_DIR)
    cmd = ["python", "-m", "pytest", test_path]
    if pytest_args:
        cmd.extend(pytest_args.split())

    print(f"\nRunning: {' '.join(cmd)}")
    print("=" * 80)
    result = subprocess.run(cmd, env=env, capture_output=False)

    print("=" * 80)
    if result.returncode == 0:
        print("PYTEST PASSED")
        return {"status": "passed", "returncode": result.returncode}

    print(f"PYTEST FAILED (exit code: {result.returncode})")
    return {"status": "failed", "returncode": result.returncode}


@app.local_entrypoint()
def main(
    test_path: str,
    pytest_args: str = "-v",
    env_vars: str = "",
):
    result = run_pytest.remote(
        test_path=test_path,
        pytest_args=pytest_args,
        env_vars=env_vars,
    )

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Status: {result.get('status')}")
    print(f"Return code: {result.get('returncode')}")

    sys.exit(0 if result.get("status") == "passed" else 1)
