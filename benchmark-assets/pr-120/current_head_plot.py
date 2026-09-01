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


def read_glm(path: Path, mode: str) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    rows = [row for row in rows if row["label"] == "measured"]
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
        raise ValueError(f"invalid benchmark artifact: {path}")

    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["sequence_length"]].append(row["completion_tokens_per_second"])
    return [
        {
            "model": "GLM-5.3-Flash",
            "mode": mode,
            "source": "f9448f10c7",
            "sequence_length": length,
            "median_output_tokens_per_second": round(
                statistics.median(grouped[length]), 2
            ),
            "min_output_tokens_per_second": round(min(grouped[length]), 2),
            "max_output_tokens_per_second": round(max(grouped[length]), 2),
        }
        for length in LENGTHS
    ]


def read_qwen(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    return [
        {
            "model": "Qwen3.8-27B",
            "mode": row["mode"],
            "source": "historical TP8 control",
            "sequence_length": int(row["sequence_length"]),
            "median_output_tokens_per_second": float(
                row["median_output_tokens_per_second"]
            ),
            "min_output_tokens_per_second": float(row["min_output_tokens_per_second"]),
            "max_output_tokens_per_second": float(row["max_output_tokens_per_second"]),
        }
        for row in rows
    ]


def main() -> None:
    summary = [
        *read_glm(ROOT / "glm-f944-ep8-mtp5.jsonl", "MTP5"),
        *read_glm(ROOT / "glm-f944-ep8-no-mtp.jsonl", "no MTP"),
        *read_qwen(ROOT / "qwen38-tp8-summary.csv"),
    ]
    output_csv = ROOT / "current-head-glm-qwen-summary.csv"
    with output_csv.open("w", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=summary[0].keys(),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(summary)

    series = (
        ("GLM-5.3-Flash", "MTP5", "tab:green", "-", "D"),
        ("GLM-5.3-Flash", "no MTP", "tab:green", "--", "D"),
        ("Qwen3.8-27B", "mtp1", "tab:orange", "-", "s"),
        ("Qwen3.8-27B", "no_mtp", "tab:orange", "--", "s"),
    )
    labels = {
        ("GLM-5.3-Flash", "MTP5"): "GLM current head, EP8, MTP5",
        ("GLM-5.3-Flash", "no MTP"): "GLM current head, EP8, no MTP",
        ("Qwen3.8-27B", "mtp1"): "Qwen3.8-27B, TP8, MTP1",
        ("Qwen3.8-27B", "no_mtp"): "Qwen3.8-27B, TP8, no MTP",
    }

    plt.figure(figsize=(10, 6))
    for model, mode, color, linestyle, marker in series:
        rows = [row for row in summary if row["model"] == model and row["mode"] == mode]
        plt.plot(
            [row["sequence_length"] for row in rows],
            [row["median_output_tokens_per_second"] for row in rows],
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=2,
            label=labels[(model, mode)],
        )

    glm_mtp_8192 = next(
        row
        for row in summary
        if row["model"] == "GLM-5.3-Flash"
        and row["mode"] == "MTP5"
        and row["sequence_length"] == 8192
    )
    glm_no_mtp_8192 = next(
        row
        for row in summary
        if row["model"] == "GLM-5.3-Flash"
        and row["mode"] == "no MTP"
        and row["sequence_length"] == 8192
    )
    speedup = (
        glm_mtp_8192["median_output_tokens_per_second"]
        / glm_no_mtp_8192["median_output_tokens_per_second"]
    )
    plt.annotate(
        f"GLM MTP5: {speedup:.2f}x at 8,192 tokens",
        xy=(8192, glm_mtp_8192["median_output_tokens_per_second"]),
        xytext=(900, 9000),
        arrowprops={"arrowstyle": "->", "color": "tab:green"},
        color="tab:green",
    )
    plt.xscale("log", base=2)
    plt.xticks(LENGTHS, [f"{length:,}" for length in LENGTHS])
    plt.xlabel("Generated sequence length (tokens per request)")
    plt.ylabel("Output throughput (tokens/s)")
    plt.title("8x B200 throughput, 64 concurrent requests")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "current-head-glm-qwen-throughput.png", dpi=160)


if __name__ == "__main__":
    main()
