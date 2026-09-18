#!/usr/bin/env python3
"""Summarize one or more raw planner benchmark CSV files as a Markdown table."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


METRICS = (
    "ttft_s",
    "total_s",
    "decode_tokens_per_second",
    "generated_tokens",
    "distributed_overhead_s",
    "remote_compute_s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, help="Also write the Markdown table here")
    return parser.parse_args()


def percentile_nearest_rank(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * percentile) - 1))
    return ordered[index]


def main() -> int:
    args = parse_args()
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    failed: dict[str, int] = defaultdict(int)
    layouts: dict[str, set[str]] = defaultdict(set)

    try:
        for path in args.csv:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                required = {"algorithm", "status", *METRICS}
                missing = required.difference(reader.fieldnames or [])
                if missing:
                    raise ValueError(f"{path}: missing columns {sorted(missing)}")
                for row in reader:
                    algorithm = row["algorithm"]
                    if row["status"] != "ok":
                        failed[algorithm] += 1
                        continue
                    grouped[algorithm].append(row)
                    if row.get("plan_segments"):
                        layouts[algorithm].add(row["plan_segments"])
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if not grouped:
        print("error: no successful benchmark rows", file=sys.stderr)
        return 1

    header = [
        "algorithm",
        "n",
        "failed",
        "TTFT median/p95 (s)",
        "total median/p95 (s)",
        "decode TPS median/p95",
        "generated tokens min/max",
        "dist. overhead median (s)",
        "remote compute median (s)",
        "plan",
        "risk weight",
        "planned devices",
        "pred. TTFT mean/tail (ms)",
        "pred. TPOT mean/tail (ms)",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for algorithm in sorted(grouped):
        rows = grouped[algorithm]

        def values(metric: str) -> list[float]:
            return [float(row[metric]) for row in rows if row.get(metric)]

        def median_p95(metric: str) -> str:
            data = values(metric)
            if not data:
                return "n/a"
            return f"{statistics.median(data):.4f} / {percentile_nearest_rank(data, 0.95):.4f}"

        def median_only(metric: str) -> str:
            data = values(metric)
            return "n/a" if not data else f"{statistics.median(data):.4f}"

        generated = values("generated_tokens")
        generated_range = (
            "n/a" if not generated else f"{int(min(generated))} / {int(max(generated))}"
        )

        layout_values = layouts[algorithm]
        if not layout_values:
            layout = "n/a"
        elif len(layout_values) == 1:
            layout = next(iter(layout_values))
        else:
            layout = f"{len(layout_values)} layouts"

        def plan_pair(mean_field: str, tail_field: str) -> str:
            mean_values = [float(row[mean_field]) for row in rows if row.get(mean_field)]
            tail_values = [float(row[tail_field]) for row in rows if row.get(tail_field)]
            if not mean_values or not tail_values:
                return "n/a"
            return f"{statistics.median(mean_values):.2f} / {statistics.median(tail_values):.2f}"

        risk_values = [float(row["plan_risk_weight"]) for row in rows if row.get("plan_risk_weight")]
        risk = "n/a" if not risk_values else f"{statistics.median(risk_values):.2f}"
        device_values = [
            float(row["plan_selected_device_count"])
            for row in rows
            if row.get("plan_selected_device_count")
        ]
        planned_devices = "n/a" if not device_values else str(int(statistics.median(device_values)))
        cells = [
            algorithm,
            str(len(rows)),
            str(failed[algorithm]),
            median_p95("ttft_s"),
            median_p95("total_s"),
            median_p95("decode_tokens_per_second"),
            generated_range,
            median_only("distributed_overhead_s"),
            median_only("remote_compute_s"),
            layout.replace("|", "/"),
            risk,
            planned_devices,
            plan_pair("plan_mean_ttft_ms", "plan_tail_ttft_ms"),
            plan_pair("plan_mean_tpot_ms", "plan_tail_tpot_ms"),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    output = "\n".join(lines) + "\n"
    print(output, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
