#!/usr/bin/env python3
"""Shared Modal image helpers for vLLM CI workloads."""

import os
from typing import Iterable

import modal

CUDA_IMAGE = "nvidia/cuda:12.8.0-devel-ubuntu22.04"
PYTHON_VERSION = "3.12"
REPO_URL = "https://$GITHUB_TOKEN@github.com/engram-lab/vllm.git"
REPO_DIR = "/tmp/vllm"

PRECOMPILED_WHEEL_COMMIT = os.environ.get(
    "VLLM_PRECOMPILED_WHEEL_COMMIT",
    "2b465570e6dd327e8422ef9c87e9b2b1454ceaed",
)

BASE_ENV = {
    "VLLM_USE_PRECOMPILED": "1",
    "VLLM_PRECOMPILED_WHEEL_COMMIT": PRECOMPILED_WHEEL_COMMIT,
}

RUNTIME_ENV = {
    "HF_XET_HIGH_PERFORMANCE": "1",
    "XFORMERS_ENABLE_TRITON": "1",
    "VLLM_SERVER_DEV_MODE": "1",
    "TORCHINDUCTOR_COMPILE_THREADS": "1",
    "VLLM_HTTP_TIMEOUT_KEEP_ALIVE": "300",
    "VLLM_PASSTHROUGH_ERRORS": "1",
}

TORCH_PACKAGES = ("torch==2.10.0", "torchvision==0.25.0")
FLASHINFER_PACKAGES = (
    "flashinfer-python==0.6.7",
    "flashinfer-cubin==0.6.7",
    "flashinfer-jit-cache==0.6.7",
)


def get_commit_hash() -> str | None:
    return os.environ.get("COMMIT_HASH") or os.environ.get("GITHUB_SHA")


def _clone_commands(branch: str, commit_hash: str | None) -> list[str]:
    clone_cmd = f"git clone --depth 1 -b {branch} {REPO_URL} {REPO_DIR}"
    if not commit_hash:
        return [clone_cmd]

    return [
        clone_cmd,
        "cd /tmp/vllm && "
        f"git fetch --depth 1 origin {commit_hash} && "
        f"git checkout {commit_hash} && "
        f"echo 'vllm@{commit_hash}'",
    ]


def _as_tuple(packages: Iterable[str] | None) -> tuple[str, ...]:
    if packages is None:
        return ()
    return tuple(packages)


def build_ci_image(
    *,
    branch: str,
    commit_hash: str | None,
    secrets: list[modal.Secret],
    apt_packages: Iterable[str] = (),
    pre_repo_packages: Iterable[str] = (),
    post_repo_packages: Iterable[str] = (),
    additional_env: dict[str, str] | None = None,
) -> modal.Image:
    apt_pkgs = ("git",) + _as_tuple(apt_packages)
    image = (
        modal.Image.from_registry(CUDA_IMAGE, add_python=PYTHON_VERSION)
        .entrypoint([])
        .apt_install(*apt_pkgs)
        .env(BASE_ENV)
        .uv_pip_install(
            *TORCH_PACKAGES,
            extra_index_url="https://download.pytorch.org/whl/cu128",
        )
        .uv_pip_install(
            *FLASHINFER_PACKAGES,
            extra_index_url="https://flashinfer.ai/whl/cu128",
        )
    )

    if pre_repo_packages:
        image = image.uv_pip_install(*_as_tuple(pre_repo_packages))

    image = image.run_commands(
        *_clone_commands(branch, commit_hash),
        secrets=secrets,
    ).uv_pip_install("vllm @ file:///tmp/vllm")

    if post_repo_packages:
        image = image.uv_pip_install(*_as_tuple(post_repo_packages))

    return image.env({**RUNTIME_ENV, **(additional_env or {})})
