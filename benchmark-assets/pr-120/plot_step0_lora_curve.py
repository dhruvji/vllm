# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

LENGTHS = [128, 256, 512, 1024, 2048, 4096, 8192]
CONCURRENCY = 64
ROOT = Path(__file__).parent


def read_series(input_path: Path, model: str, mode: str) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in input_path.read_text().splitlines() if line]
    rows = [
        row
        for row in rows
        if row["label"].startswith("measured-") and row["request_model"] == model
    ]

    expected = {(length, repeat) for length in LENGTHS for repeat in (1, 2)}
    actual = {(row["sequence_length"], row["repeat"]) for row in rows}
    if (
        actual != expected
        or any(row["concurrency"] != CONCURRENCY for row in rows)
        or any(row["errors"] for row in rows)
        or any(
            row["total_completion_tokens"] != CONCURRENCY * row["sequence_length"]
            for row in rows
        )
    ):
        raise ValueError(f"invalid benchmark artifact: {input_path}")

    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["sequence_length"]].append(row["completion_tokens_per_second"])

    return [
        {
            "request_model": model,
            "mode": mode,
            "sequence_length": length,
            "median_output_tokens_per_second": round(
                statistics.median(grouped[length]), 2
            ),
            "min_output_tokens_per_second": round(min(grouped[length]), 2),
            "max_output_tokens_per_second": round(max(grouped[length]), 2),
        }
        for length in LENGTHS
    ]


def main() -> None:
    summary = [
        *read_series(
            ROOT / "glm-3f712-ep8-mtp5-base.jsonl",
            "glm-5.3-flash",
            "MTP5",
        ),
        *read_series(
            ROOT / "glm-3f712-ep8-mtp5-step0-lora.jsonl",
            "step0-lora-r32",
            "MTP5",
        ),
        *read_series(
            ROOT / "glm-3f712-ep8-no-mtp-base.jsonl",
            "glm-5.3-flash",
            "no MTP",
        ),
        *read_series(
            ROOT / "glm-3f712-ep8-no-mtp-step0-lora.jsonl",
            "step0-lora-r32",
            "no MTP",
        ),
    ]

    with (ROOT / "glm-step0-lora-summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=summary[0].keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summary)

    plt.figure(figsize=(9, 5.5))
    for model, mode, label, color, linestyle, marker in (
        (
            "glm-5.3-flash",
            "MTP5",
            "Normal model, MTP5",
            "tab:blue",
            "-",
            "o",
        ),
        (
            "glm-5.3-flash",
            "no MTP",
            "Normal model, no MTP",
            "tab:blue",
            "--",
            "o",
        ),
        (
            "step0-lora-r32",
            "MTP5",
            "Step-0 attention LoRA, MTP5",
            "tab:red",
            "-",
            "s",
        ),
        (
            "step0-lora-r32",
            "no MTP",
            "Step-0 attention LoRA, no MTP",
            "tab:red",
            "--",
            "s",
        ),
    ):
        model_rows = [
            row
            for row in summary
            if row["request_model"] == model and row["mode"] == mode
        ]
        plt.plot(
            [row["sequence_length"] for row in model_rows],
            [row["median_output_tokens_per_second"] for row in model_rows],
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=2,
            label=label,
        )

    plt.xscale("log", base=2)
    plt.xticks(LENGTHS, [f"{length:,}" for length in LENGTHS])
    plt.xlabel("Generated sequence length (tokens per request)")
    plt.ylabel("Output throughput (tokens/s)")
    plt.title("GLM-5.3-Flash on 8x B200, 64 concurrent requests")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "glm-step0-lora-throughput.png", dpi=160)


if __name__ == "__main__":
    main()
