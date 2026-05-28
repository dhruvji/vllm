#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Smoke-test loading a TorchTitan Qwen 3.5 prefix cartridge from S3.

Before the sparse-layer fix in ``CartridgeData.get_stacked_kv()``, this script
failed with ``KeyError: 0`` for TorchTitan Qwen 3.5 cartridges because they
store KV only for attention layers, not every hidden layer.
"""

import argparse

from vllm.v1.cartridge_loader import load_cartridge

DEFAULT_CARTRIDGE = (
    "s3://engram-cartridges/weights/torchtitan/2026-04-21/"
    "18-14-47-worktrial-singledoc-uniform-synced6-og-qwen3-5-9b-5c55ec98/"
    "step-0/adapter_model.safetensors"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cartridge", nargs="?", default=DEFAULT_CARTRIDGE)
    args = parser.parse_args()

    cartridge = load_cartridge(args.cartridge)
    stacked_keys, stacked_values = cartridge.get_stacked_kv()

    print(f"cartridge={args.cartridge}")
    print(f"loaded={cartridge}")
    print(f"stacked_keys_shape={tuple(stacked_keys.shape)}")
    print(f"stacked_values_shape={tuple(stacked_values.shape)}")


if __name__ == "__main__":
    main()
