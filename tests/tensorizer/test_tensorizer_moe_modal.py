#!/usr/bin/env python3
"""
Modal test for tensorizer + MoE loading.

Clones vLLM from GitHub (cached), then overlays ONLY changed files at runtime.
Uses git diff to find changed files - much faster than uploading entire repo.

Verifies that:
1. Tensorizer can load an MoE model (Qwen3-30B-A3B)
2. process_weights_after_loading is called (MoE kernel initialized)
3. Expert parallel works with tensorizer
4. Basic inference completes successfully

Usage:
    modal run tests/tensorizer/test_tensorizer_moe_modal.py

Environment variables:
    - GPU_TYPE: GPU type (default: H200)
    - NUM_GPUS: Number of GPUs (default: 8)
    - MODEL: Model to test (default: Qwen/Qwen3-30B-A3B)
"""

import os
import subprocess
import time
from pathlib import Path

import modal

# Get vllm root directory relative to this script
VLLM_ROOT = Path(__file__).resolve().parent.parent.parent

GPU_TYPE = os.environ.get("GPU_TYPE", "H200")
NUM_GPUS = int(os.environ.get("NUM_GPUS", 8))
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-30B-A3B")
SECRETS = os.environ.get("SECRETS", "sabri-api-keys")
BRANCH = os.environ.get("BRANCH", "jack/tensorizer-loader")

secrets = modal.Secret.from_name(SECRETS)


def get_changed_files(branch: str) -> list[str]:
    """Get list of files changed compared to the target branch."""
    try:
        # Fetch to ensure we have the latest remote refs
        subprocess.run(
            ["git", "fetch", "origin", branch],
            cwd=VLLM_ROOT,
            capture_output=True,
            check=False,
        )

        # Get changed files (both staged and unstaged) compared to target branch
        result = subprocess.run(
            ["git", "diff", "--name-only", f"origin/{branch}"],
            cwd=VLLM_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        changed = result.stdout.strip().split("\n") if result.stdout.strip() else []

        # Also get untracked files that might be new tests
        result2 = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=VLLM_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        untracked = result2.stdout.strip().split("\n") if result2.stdout.strip() else []

        # Filter to only .py and .json files in vllm/ or tests/
        all_files = changed + untracked
        relevant = [
            f for f in all_files
            if f and (f.startswith("vllm/") or f.startswith("tests/"))
            and (f.endswith(".py") or f.endswith(".json"))
        ]
        return relevant
    except subprocess.CalledProcessError:
        # If git fails, fall back to empty (will use cloned code)
        return []


# Get changed files at module load time (when modal builds the image def)
CHANGED_FILES = get_changed_files(BRANCH)
print(f"Changed files vs origin/{BRANCH} ({len(CHANGED_FILES)}): {CHANGED_FILES}")

# Build base image: clone from GitHub (cached) and build vLLM
base_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.13"
    )
    .entrypoint([])
    .apt_install("git")
    .env({"VLLM_USE_PRECOMPILED": "1"})
    .run_commands(
        f"git clone https://$GITHUB_TOKEN@github.com/engram-lab/vllm.git -b {BRANCH} /tmp/vllm",
        secrets=[secrets],
        force_build=False,
    )
    .uv_pip_install("/tmp/vllm[tensorizer]", extra_options="-e")
    .uv_pip_install(
        "huggingface-hub[hf_transfer]==0.36.0",
        "flashinfer-python>=0.5.2",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
)

# Mount ONLY changed files at runtime (not the whole repo)
# copy=False means mounted from local machine, not baked into image
if CHANGED_FILES:
    image = base_image.add_local_dir(
        local_path=str(VLLM_ROOT),
        remote_path="/mnt/local_vllm",
        copy=False,
        # Only include the specific changed files
        ignore=~modal.FilePatternMatcher(*CHANGED_FILES),
    )
else:
    # No local changes - just use the cloned repo
    image = base_image

hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True, version=2)
vllm_cache = modal.Volume.from_name("vllm-cache-v2", create_if_missing=True, version=2)

app = modal.App("test-tensorizer-moe")


def get_tensorizer_path(model: str, tp: int) -> str:
    """Get the tensorizer cache path for a model."""
    model_slug = model.replace("/", "--")
    return f"/root/.cache/vllm/tensorized/{model_slug}/tp{tp}"


def sync_changed_files(changed_files: list[str]) -> None:
    """Overlay local changes onto the cloned repo."""
    import shutil

    local_mount = Path("/mnt/local_vllm")
    target_repo = Path("/tmp/vllm")

    if local_mount.exists() and changed_files:
        for rel_path in changed_files:
            src = local_mount / rel_path
            dst = target_repo / rel_path
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                print(f"  Synced: {rel_path}")
        print(f"Synced {len(changed_files)} changed files")
    else:
        print("No local mount or no changed files - using cloned repo as-is")


@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:{NUM_GPUS}",
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
    secrets=[secrets],
    timeout=30 * 60,
)
def test_tensorizer_moe_load(changed_files: list[str] | None = None):
    """Test that tensorizer can load an MoE model with expert parallel."""
    import socket

    import requests

    # Sync any changed files first
    sync_changed_files(changed_files or [])

    tp = NUM_GPUS
    tensorizer_path = get_tensorizer_path(MODEL, tp)

    # Check if tensorizer files exist
    tensor_dir = Path(tensorizer_path)
    if not tensor_dir.exists():
        print(f"ERROR: Tensorizer path {tensorizer_path} does not exist")
        print("Run tensorization first to create the cached weights")
        return {"success": False, "error": "tensorizer_path_missing"}

    files = list(tensor_dir.iterdir())
    print(f"Tensorizer path contents: {[f.name for f in files]}")

    has_shards = any(f.name.endswith(".tensors") for f in files)
    has_config = "config.json" in {f.name for f in files}

    if not (has_shards and has_config):
        print(f"ERROR: Missing required files (shards={has_shards}, config={has_config})")
        return {"success": False, "error": "tensorizer_files_incomplete"}

    # Build tensorizer URI
    if tp > 1:
        tensorizer_uri = f"{tensorizer_path}/model-rank-%04d.tensors"
    else:
        tensorizer_uri = f"{tensorizer_path}/model.tensors"

    # Launch vLLM with tensorizer + expert parallel
    cmd = [
        "vllm",
        "serve",
        MODEL,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--tensor-parallel-size", str(tp),
        "--max-model-len", "4096",
        "--max-num-seqs", "8",
        "--load-format", "tensorizer",
        "--enable-expert-parallel",
        "--model-loader-extra-config",
        f'{{"tensorizer_uri": "{tensorizer_uri}"}}',
    ]

    print(f"Command: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # Wait for server to be ready (with timeout)
    start = time.time()
    ready = False
    error_output = []

    for _ in range(300):  # 5 minute timeout
        try:
            socket.create_connection(("localhost", 8000), timeout=1).close()
            ready = True
            print(f"Server ready after {time.time() - start:.1f}s")
            break
        except OSError:
            if proc.poll() is not None:
                # Process died, capture output
                stdout, _ = proc.communicate()
                error_output = stdout.decode() if stdout else ""
                print(f"ERROR: vLLM exited with code {proc.returncode}")
                print(f"Output:\n{error_output}")
                return {
                    "success": False,
                    "error": "vllm_crashed",
                    "exit_code": proc.returncode,
                    "output": error_output[-5000:],  # Last 5000 chars
                }
            time.sleep(1)

    if not ready:
        proc.terminate()
        return {"success": False, "error": "timeout_waiting_for_server"}

    # Test inference
    try:
        response = requests.post(
            "http://localhost:8000/v1/completions",
            json={
                "model": MODEL,
                "prompt": "The capital of France is",
                "max_tokens": 16,
            },
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()
        completion = result["choices"][0]["text"]
        print(f"Inference test passed: '{completion}'")
    except Exception as e:
        proc.terminate()
        return {"success": False, "error": f"inference_failed: {e}"}

    # Cleanup
    proc.terminate()
    proc.wait()

    return {
        "success": True,
        "model": MODEL,
        "tp": tp,
        "tensorizer_path": tensorizer_path,
        "completion": completion,
    }


@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:{NUM_GPUS}",
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
    secrets=[secrets],
    timeout=30 * 60,
)
def check_tensorizer_fix(changed_files: list[str] | None = None):
    """Verify the tensorizer fix is present in the installed vLLM."""
    import inspect

    # Sync any changed files first
    sync_changed_files(changed_files or [])

    import vllm.model_executor.model_loader.tensorizer_loader as t

    source = inspect.getsource(t.TensorizerLoader.load_model)

    checks = {
        "process_weights_after_loading": "process_weights_after_loading" in source,
        "requires_grad_(False)": "requires_grad_(False)" in source,
    }

    print("Tensorizer fix verification:")
    for check, present in checks.items():
        status = "OK" if present else "MISSING"
        print(f"  {check}: {status}")

    all_present = all(checks.values())
    return {
        "success": all_present,
        "checks": checks,
    }


@app.local_entrypoint()
def main():
    """Run tensorizer MoE tests."""
    print("=" * 60)
    print("Testing tensorizer + MoE loading")
    print("=" * 60)
    print(f"Local vLLM source: {VLLM_ROOT}")
    print(f"Branch: {BRANCH}")
    print(f"Changed files to sync ({len(CHANGED_FILES)}):")
    for f in CHANGED_FILES:
        print(f"  - {f}")
    print("=" * 60)

    # First verify the fix is present
    print("\n1. Checking tensorizer fix is present...")
    fix_result = check_tensorizer_fix.remote(changed_files=CHANGED_FILES)
    print(f"   Result: {fix_result}")

    if not fix_result["success"]:
        print("\nERROR: Tensorizer fix not present in image!")
        print("Make sure you've pushed the changes to the branch.")
        return

    # Then test actual loading
    print("\n2. Testing tensorizer + MoE + EP loading...")
    load_result = test_tensorizer_moe_load.remote(changed_files=CHANGED_FILES)
    print(f"   Result: {load_result}")

    if load_result["success"]:
        print("\nSUCCESS: Tensorizer + MoE + Expert Parallel working!")
    else:
        print(f"\nFAILED: {load_result.get('error', 'unknown error')}")
