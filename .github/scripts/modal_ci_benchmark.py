#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""
Modal CI script for vLLM benchmarking (latency and throughput).

Runs benchmarks and asserts thresholds based on config:
- Latency mode: asserts TPOT (Time Per Output Token) < threshold
- Throughput mode: asserts output throughput > threshold

Exits with status code 0 if all passed, 1 if any failed.

Usage:
    # Single config
    modal run .github/scripts/modal_ci_benchmark.py \
        --config-paths .github/benchmark_configs/base_latency.json

    # Multiple configs
    modal run .github/scripts/modal_ci_benchmark.py \
        --config-paths .github/benchmark_configs/base_latency.json \
        --config-paths .github/benchmark_configs/base_throughput.json
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time

import modal

try:
    from modal_ci_image import build_ci_image, get_commit_hash
except ModuleNotFoundError:
    CUDA_IMAGE = "nvidia/cuda:13.0.2-devel-ubuntu22.04"
    PYTHON_VERSION = "3.12"
    REPO_URL = "https://$GITHUB_TOKEN@github.com/engram-lab/vllm.git"
    REPO_DIR = "/tmp/vllm"
    USE_LOCAL_SOURCE = os.environ.get("VLLM_CI_USE_LOCAL_SOURCE") == "1"
    LOCAL_SOURCE_DIR = os.environ.get("VLLM_CI_LOCAL_SOURCE_DIR", ".")
    LOCAL_SOURCE_IGNORE = modal.FilePatternMatcher(
        ".git/**",
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
    PRECOMPILED_WHEEL_COMMIT = os.environ.get(
        "VLLM_PRECOMPILED_WHEEL_COMMIT",
        "877dae9c684cd8cb2f66eed33f1c3261c6792c10",
    )
    BASE_ENV = {
        "VLLM_USE_PRECOMPILED": "1",
        "VLLM_PRECOMPILED_WHEEL_COMMIT": PRECOMPILED_WHEEL_COMMIT,
        "VLLM_PRECOMPILED_WHEEL_VARIANT": "cu130",
        "VLLM_MAIN_CUDA_VERSION": "13.0",
    }
    RUNTIME_ENV = {
        "HF_XET_HIGH_PERFORMANCE": "1",
        "XFORMERS_ENABLE_TRITON": "1",
        "VLLM_SERVER_DEV_MODE": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "VLLM_HTTP_TIMEOUT_KEEP_ALIVE": "300",
        "VLLM_PASSTHROUGH_ERRORS": "1",
    }
    TORCH_PACKAGES = ("torch==2.11.0+cu130", "torchvision==0.26.0+cu130")
    FLASHINFER_PACKAGES = (
        "flashinfer-python==0.6.14",
        "flashinfer-cubin==0.6.14",
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

    def _as_tuple(packages):
        if packages is None:
            return ()
        return tuple(packages)

    def build_ci_image(
        *,
        branch: str,
        commit_hash: str | None,
        secrets: list[modal.Secret],
        apt_packages=(),
        pre_repo_packages=(),
        post_repo_packages=(),
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
                index_url="https://download.pytorch.org/whl/cu130",
            )
            .uv_pip_install(
                *FLASHINFER_PACKAGES,
                extra_index_url="https://flashinfer.ai/whl/",
            )
        )

        if pre_repo_packages:
            image = image.uv_pip_install(*_as_tuple(pre_repo_packages))

        if USE_LOCAL_SOURCE:
            image = image.add_local_dir(
                local_path=LOCAL_SOURCE_DIR,
                remote_path=REPO_DIR,
                copy=True,
                ignore=LOCAL_SOURCE_IGNORE,
            ).env({"VLLM_VERSION_OVERRIDE": "0.1.dev0+ci"})
        else:
            image = image.run_commands(
                *_clone_commands(branch, commit_hash),
                secrets=secrets,
            )

        image = image.run_commands(
            "/.uv/uv pip install --system --no-build-isolation "
            "'vllm @ file:///tmp/vllm' --torch-backend cu130"
        )

        if post_repo_packages:
            image = image.uv_pip_install(*_as_tuple(post_repo_packages))

        return image.env({**RUNTIME_ENV, **(additional_env or {})})


# --- Configuration ---
PORT = 8000
BRANCH = os.environ.get("BRANCH", "main")
COMMIT_HASH = get_commit_hash()
GPU_TYPE = os.environ.get("GPU_TYPE", "H200")
GPU_COUNT = int(os.environ.get("GPU_COUNT", 1))
SECRETS = os.environ.get("SECRETS", "dev-api-keys")

# vLLM optimization args
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "128"))
MAX_NUM_BATCHED_TOKENS = int(os.environ.get("MAX_NUM_BATCHED_TOKENS", "32768"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "16384"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.90"))

# Benchmark thresholds (can be overridden by config)
TPOT_THRESHOLD_MS = float(os.environ.get("TPOT_THRESHOLD_MS", "10.0"))
THROUGHPUT_THRESHOLD_TOK_S = float(
    os.environ.get("THROUGHPUT_THRESHOLD_TOK_S", "2500.0")
)

MINUTES = 60  # seconds

secrets = modal.Secret.from_name(SECRETS)
image = build_ci_image(
    branch=BRANCH,
    commit_hash=COMMIT_HASH,
    secrets=[secrets],
    pre_repo_packages=(
        "packaging",
        "setuptools>=77,<82",
        "setuptools-rust>=1.9.0",
        "setuptools-scm",
        "wheel",
        "huggingface-hub[hf_transfer]==0.36.0",
        "boto3",
    ),
    additional_env={"HF_HUB_ENABLE_HF_TRANSFER": "0"},
)

hf_cache_vol = modal.Volume.from_name(
    "huggingface-cache", create_if_missing=True, version=2
)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True, version=2)
flashinfer_cache_vol = modal.Volume.from_name(
    "flashinfer-cache", create_if_missing=True, version=2
)

app = modal.App("vllm-ci-benchmark")


@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    timeout=30 * MINUTES,
    secrets=[secrets],
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
        "/root/.cache/flashinfer": flashinfer_cache_vol,
    },
    enable_memory_snapshot=False,
)
def run_benchmarks_with_server(
    model: str,
    tensor_parallel_size: int,
    config_jsons: list[str],
    enforce_eager: bool,
):
    """
    Start vLLM server, run all benchmarks, then stop server.

    Args:
        model: Model name to serve
        tensor_parallel_size: Tensor parallel size
        config_jsons: List of JSON strings containing benchmark configurations

    Returns:
        List of dictionaries with benchmark results and pass/fail status
    """
    import requests

    print("=" * 80)
    print("STARTING vLLM SERVER")
    print("=" * 80)
    print(f"Model: {model}")
    print(f"GPU allocation: {GPU_COUNT}x {GPU_TYPE}")
    print(f"Tensor Parallel Size: {tensor_parallel_size}")
    print(f"Enforce eager: {enforce_eager}")
    print("=" * 80)
    print()

    # Build server command
    cmd = [
        "vllm",
        "serve",
        model,
        "--served-model-name",
        model,
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
        "--max-num-batched-tokens",
        str(MAX_NUM_BATCHED_TOKENS),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),
        "--uvicorn-log-level",
        "warning",
        "--disable-log-requests",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
    ]
    if enforce_eager:
        cmd.append("--enforce-eager")

    print(f"Server command: {' '.join(cmd)}")
    print()

    def ping() -> bool:
        try:
            response = requests.get(f"http://localhost:{PORT}/health", timeout=1.0)
            return response.status_code == 200
        except requests.RequestException:
            return False

    # Start server
    server_start_time = time.time()
    server_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Stream server logs in a separate thread
    def stream_logs():
        print("=" * 80)
        print("vLLM SERVER LOGS")
        print("=" * 80)
        if server_process.stdout:
            for line in iter(server_process.stdout.readline, ""):
                if line:
                    print(f"[SERVER] {line.rstrip()}")
        print("=" * 80)
        print("vLLM SERVER STOPPED")
        print("=" * 80)

    log_thread = threading.Thread(target=stream_logs, daemon=True)
    log_thread.start()

    # Wait for server to be ready
    print("Waiting for vLLM server to start...")
    ready = False
    while time.time() - server_start_time < 600:
        if ping():
            server_ready_time = time.time()
            server_startup_duration = server_ready_time - server_start_time
            print(
                f"✓ vLLM server is ready! (startup time: {server_startup_duration:.2f}s)"
            )
            print()
            ready = True
            break
        if server_process.poll() is not None:
            print("❌ Server process exited unexpectedly!")
            return [{"status": "failed", "error": "Server failed to start"}]
        time.sleep(2.0)

    if not ready:
        server_process.kill()
        print("❌ Server startup timeout!")
        return [{"status": "failed", "error": "Server startup timeout"}]

    base_url = f"http://localhost:{PORT}"

    # Run all benchmarks
    all_results = []

    try:
        for i, config_json in enumerate(config_jsons, 1):
            config = json.loads(config_json)
            extra_body = config.get("extra_body")
            assert_type = config.get("assert_type", "latency")

            print("\n" + "=" * 80)
            print(f"BENCHMARK {i}/{len(config_jsons)}: {assert_type.upper()}")
            print("=" * 80)
            print(f"Config: {config.get('name', 'custom')}")
            print(f"Description: {config.get('description', 'N/A')}")
            print(f"Model: {model}")
            print(f"Server URL: {base_url}")
            if assert_type == "throughput":
                threshold = config.get(
                    "throughput_threshold_tok_s", THROUGHPUT_THRESHOLD_TOK_S
                )
                print(f"Throughput Threshold: {threshold:.2f} tok/s")
            else:
                threshold = config.get("tpot_threshold_ms", TPOT_THRESHOLD_MS)
                print(f"TPOT Threshold: {threshold:.2f}ms")
            print("=" * 80)
            print()

            try:
                # Build the benchmark command
                bench_cmd = [
                    "vllm",
                    "bench",
                    "serve",
                    "--backend",
                    "openai",
                    "--base-url",
                    base_url,
                    "--model",
                    model,
                    "--dataset-name",
                    config["dataset_name"],
                    "--num-prompts",
                    str(config["num_prompts"]),
                    "--seed",
                    str(config.get("seed", 42)),
                    "--num-warmups",
                    str(config.get("num_warmups", 5)),
                    "--save-result",
                ]

                # Add dataset-specific arguments
                if config["dataset_name"] == "random":
                    bench_cmd.extend(
                        [
                            "--random-input-len",
                            str(config.get("prompt_len", 512)),
                            "--random-output-len",
                            str(config.get("output_len", 128)),
                        ]
                    )

                # Add request rate
                if config.get("request_rate"):
                    bench_cmd.extend(["--request-rate", str(config["request_rate"])])

                # Add optional arguments
                if config.get("max_concurrency"):
                    bench_cmd.extend(
                        ["--max-concurrency", str(config["max_concurrency"])]
                    )

                # Add optional sampling arguments so CI can pin deterministic
                # decode settings across upstream benchmark behavior changes.
                for arg_name in (
                    "temperature",
                    "top_p",
                    "top_k",
                    "min_p",
                    "frequency_penalty",
                    "presence_penalty",
                    "repetition_penalty",
                ):
                    value = config.get(arg_name)
                    if value is not None:
                        bench_cmd.extend(
                            [f"--{arg_name.replace('_', '-')}", str(value)]
                        )

                # Add extra body parameters
                if extra_body:
                    bench_cmd.extend(["--extra-body", json.dumps(extra_body)])

                result_filename = f"ci_result_{i}.json"
                bench_cmd.extend(["--result-filename", result_filename])

                print("=" * 80)
                print("Running benchmark:")
                print(" ".join(bench_cmd))
                print("=" * 80)
                print()

                # Run the benchmark
                result = subprocess.run(
                    bench_cmd,
                    capture_output=True,
                    text=True,
                )

                print(result.stdout)
                if result.stderr:
                    print("STDERR:")
                    print(result.stderr)

                if result.returncode != 0:
                    all_results.append(
                        {
                            "status": "failed",
                            "error": f"Benchmark failed with return code {result.returncode}",
                        }
                    )
                    continue

                # Read benchmark results from saved JSON file
                benchmark_result = None
                try:
                    with open(result_filename) as f:
                        benchmark_result = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    all_results.append(
                        {
                            "status": "failed",
                            "error": f"Failed to read benchmark results: {e}",
                        }
                    )
                    continue

                print("\n" + "=" * 80)

                if assert_type == "throughput":
                    # Throughput mode: check output_throughput
                    print("THROUGHPUT CHECK RESULTS")
                    print("=" * 80)

                    output_throughput = benchmark_result.get("output_throughput")
                    request_throughput = benchmark_result.get("request_throughput")
                    threshold = config.get(
                        "throughput_threshold_tok_s", THROUGHPUT_THRESHOLD_TOK_S
                    )

                    if output_throughput is None:
                        all_results.append(
                            {
                                "status": "failed",
                                "error": "Output throughput metric not found in results",
                            }
                        )
                        continue

                    print(f"Output throughput: {output_throughput:.2f} tok/s")
                    print(f"Request throughput: {request_throughput:.2f} req/s")
                    print(f"Threshold: {threshold:.2f} tok/s")
                    print()

                    passed = output_throughput > threshold

                    if passed:
                        print(
                            f"✓ PASSED: Output throughput ({output_throughput:.2f} tok/s) > {threshold:.2f} tok/s"
                        )
                    else:
                        print(
                            f"✗ FAILED: Output throughput ({output_throughput:.2f} tok/s) <= {threshold:.2f} tok/s"
                        )
                    print("=" * 80)

                    all_results.append(
                        {
                            "status": "passed" if passed else "failed",
                            "output_throughput": output_throughput,
                            "request_throughput": request_throughput,
                            "threshold_tok_s": threshold,
                            "passed": passed,
                            "full_results": benchmark_result,
                        }
                    )
                else:
                    # Latency mode: check TPOT
                    print("LATENCY CHECK RESULTS")
                    print("=" * 80)

                    mean_tpot_ms = benchmark_result.get("mean_tpot_ms")
                    median_tpot_ms = benchmark_result.get("median_tpot_ms")
                    threshold = config.get("tpot_threshold_ms", TPOT_THRESHOLD_MS)

                    if mean_tpot_ms is None:
                        all_results.append(
                            {
                                "status": "failed",
                                "error": "TPOT metric not found in results",
                            }
                        )
                        continue

                    print(f"Mean TPOT: {mean_tpot_ms:.2f}ms")
                    print(f"Median TPOT: {median_tpot_ms:.2f}ms")
                    print(f"Threshold: {threshold:.2f}ms")
                    print()

                    passed = mean_tpot_ms < threshold

                    if passed:
                        print(
                            f"✓ PASSED: Mean TPOT ({mean_tpot_ms:.2f}ms) < {threshold:.2f}ms"
                        )
                    else:
                        print(
                            f"✗ FAILED: Mean TPOT ({mean_tpot_ms:.2f}ms) >= {threshold:.2f}ms"
                        )
                    print("=" * 80)

                    all_results.append(
                        {
                            "status": "passed" if passed else "failed",
                            "mean_tpot_ms": mean_tpot_ms,
                            "median_tpot_ms": median_tpot_ms,
                            "threshold_ms": threshold,
                            "passed": passed,
                            "full_results": benchmark_result,
                        }
                    )

            except Exception as e:
                all_results.append(
                    {
                        "status": "failed",
                        "error": f"Benchmark execution error: {str(e)}",
                    }
                )

    finally:
        # Stop the server
        print("\nShutting down vLLM server...")
        try:
            server_process.send_signal(signal.SIGTERM)
            server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print("Server didn't stop gracefully, killing...")
            server_process.kill()
        print("✓ Server stopped")

    return all_results


@app.local_entrypoint()
def main(*args):
    """
    Local entrypoint to run CI benchmark (latency or throughput).

    Args:
        args: Command line arguments (parsed using argparse)

    Examples:
        # Single config
        modal run .github/scripts/modal_ci_benchmark.py \\
            --config-paths .github/benchmark_configs/base_latency.json

        # Multiple configs (using shared vLLM server)
        modal run .github/scripts/modal_ci_benchmark.py \\
            --config-paths .github/benchmark_configs/base_latency.json \\
            --config-paths .github/benchmark_configs/base_throughput.json

    Note: All configs must use the same model, tensor_parallel_size, and enforce_eager
    to share a server.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run vLLM CI benchmarks")
    parser.add_argument(
        "--config-paths",
        action="append",
        required=True,
        help="Path to benchmark config JSON file (can be specified multiple times)",
    )
    parsed_args = parser.parse_args(args)
    config_paths = parsed_args.config_paths

    # Read and validate all configs
    configs = []
    config_jsons = []
    for config_path in config_paths:
        with open(config_path) as f:
            config_json = f.read()
        config = json.loads(config_json)
        configs.append(config)
        config_jsons.append(config_json)

    # Validate all configs use the same model and tensor_parallel_size
    if len(configs) > 1:
        first_model = configs[0]["model_name"]
        first_tp_size = configs[0].get("tensor_parallel_size", GPU_COUNT)
        first_enforce_eager = configs[0].get("enforce_eager", False)

        for i, config in enumerate(configs[1:], 2):
            model = config["model_name"]
            tp_size = config.get("tensor_parallel_size", GPU_COUNT)

            if model != first_model:
                print(
                    f"ERROR: Config {i} uses different model ({model}) than config 1 ({first_model})"
                )
                print("All configs must use the same model to share a server.")
                sys.exit(1)

            if tp_size != first_tp_size:
                print(
                    f"ERROR: Config {i} uses different tensor_parallel_size ({tp_size}) than config 1 ({first_tp_size})"
                )
                print(
                    "All configs must use the same tensor_parallel_size to share a server."
                )
                sys.exit(1)
            enforce_eager = config.get("enforce_eager", False)
            if enforce_eager != first_enforce_eager:
                print(
                    f"ERROR: Config {i} uses different enforce_eager ({enforce_eager}) "
                    f"than config 1 ({first_enforce_eager})"
                )
                print("All configs must use the same enforce_eager to share a server.")
                sys.exit(1)

    # Get model and tensor_parallel_size from first config
    model = configs[0]["model_name"]
    tensor_parallel_size = configs[0].get("tensor_parallel_size", GPU_COUNT)
    enforce_eager = configs[0].get("enforce_eager", False)

    print("\n" + "=" * 80)
    print(f"STARTING SHARED vLLM SERVER FOR {len(configs)} BENCHMARK(S)")
    print("=" * 80)
    print(f"Model: {model}")
    print(f"Tensor Parallel Size: {tensor_parallel_size}")
    print(f"GPU allocation: {GPU_COUNT}x {GPU_TYPE}")
    print(f"Enforce eager: {enforce_eager}")
    print("=" * 80)
    print()

    # Run all benchmarks with a single server
    all_results = run_benchmarks_with_server.remote(
        model,
        tensor_parallel_size,
        config_jsons,
        enforce_eager,
    )

    # Print results for each config
    for i, (config_path, result) in enumerate(zip(config_paths, all_results), 1):
        print("\n" + "=" * 80)
        print(f"RESULTS FOR {config_path}")
        print("=" * 80)
        print(json.dumps(result, indent=2))
        print("=" * 80)

    # Print summary of all results
    print("\n\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    all_passed = True
    for config_path, result in zip(config_paths, all_results):
        status = result.get("status")

        if status == "passed":
            print(f"✓ PASSED: {config_path}")
        else:
            print(f"✗ FAILED: {config_path}")
            all_passed = False

    print("=" * 80)

    # Exit with appropriate status code
    if all_passed:
        print(f"\n✓ ALL {len(config_paths)} BENCHMARK(S) PASSED")
        sys.exit(0)
    else:
        print("\n✗ SOME BENCHMARKS FAILED")
        sys.exit(1)
