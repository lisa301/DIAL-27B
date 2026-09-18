#!/usr/bin/env python3
"""Run reproducible non-streaming DIAL planner benchmarks and write raw CSV."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import mimetypes
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ALGORITHMS = (
    "dial",
    "edgeshard-latency",
    "edgeshard-throughput",
    "static-even",
    "solo",
)
METRIC_FIELDS = (
    "ttft_s",
    "total_s",
    "tokens_per_second",
    "decode_tokens_per_second",
    "generated_tokens",
    "distributed_overhead_s",
    "remote_compute_s",
    "remote_requests",
)
CSV_FIELDS = (
    "timestamp_utc",
    "algorithm",
    "run",
    "prompt_index",
    "prompt_id",
    "prompt_chars",
    "has_image",
    "configured_sample_len",
    "client_wall_s",
    *METRIC_FIELDS,
    "response_chars",
    "response_sha256",
    "plan_score",
    "plan_score_definition",
    "plan_segments",
    "plan_risk_weight",
    "plan_selected_device_count",
    "plan_mean_ttft_ms",
    "plan_mean_tpot_ms",
    "plan_tail_ttft_ms",
    "plan_tail_tpot_ms",
    "plan_sha256",
    "tag_json",
    "status",
    "error",
)


@dataclass(frozen=True)
class PromptCase:
    prompt_id: str
    prompt: str
    image: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark one running DIAL planner configuration. Restart all DIAL "
            "processes between algorithms; --algorithm is a result label only."
        )
    )
    parser.add_argument(
        "--api",
        required=True,
        help="DIAL API base URL or full /api/v1/chat/completions URL",
    )
    parser.add_argument("--algorithm", required=True, choices=ALGORITHMS)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", help="One prompt string reused for every run")
    source.add_argument("--prompt-file", type=Path, help="UTF-8 file containing one prompt")
    source.add_argument(
        "--prompts-jsonl",
        type=Path,
        help='JSONL cases: a string, or {"id":"...","prompt":"...","image":"..."}',
    )
    parser.add_argument("--image", type=Path, help="Image for --prompt/--prompt-file")
    parser.add_argument("--system", default="You are a helpful AI assistant.")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=900.0, help="Seconds per request")
    parser.add_argument(
        "--sample-len",
        type=int,
        required=True,
        help="Metadata only; must equal the server's --sample-len value",
    )
    parser.add_argument("--plan-report", type=Path, help="Master --auto-plan-output JSON")
    parser.add_argument("--tag", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing compatible CSV instead of refusing to overwrite it",
    )
    args = parser.parse_args()
    if args.runs <= 0:
        parser.error("--runs must be greater than zero")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    if args.sample_len <= 0:
        parser.error("--sample-len must be greater than zero")
    if args.image and args.prompts_jsonl:
        parser.error("put per-case image paths in --prompts-jsonl; do not combine it with --image")
    return args


def normalize_endpoint(raw: str) -> str:
    endpoint = raw.strip()
    if not endpoint.startswith(("http://", "https://")):
        endpoint = "http://" + endpoint
    endpoint = endpoint.rstrip("/")
    suffix = "/api/v1/chat/completions"
    return endpoint if endpoint.endswith(suffix) else endpoint + suffix


def load_prompt_cases(args: argparse.Namespace) -> list[PromptCase]:
    if args.prompt is not None:
        cases = [PromptCase("prompt-1", args.prompt, args.image)]
    elif args.prompt_file is not None:
        text = args.prompt_file.read_text(encoding="utf-8")
        cases = [PromptCase(args.prompt_file.stem, text, args.image)]
    else:
        cases = []
        source_dir = args.prompts_jsonl.resolve().parent
        with args.prompts_jsonl.open(encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                raw = raw.strip()
                if not raw or raw.startswith("#"):
                    continue
                value = json.loads(raw)
                if isinstance(value, str):
                    cases.append(PromptCase(f"prompt-{line_number}", value))
                    continue
                if not isinstance(value, dict) or not isinstance(value.get("prompt"), str):
                    raise ValueError(
                        f"{args.prompts_jsonl}:{line_number}: expected a string or an object with a string prompt"
                    )
                image = value.get("image")
                image_path = None
                if image is not None:
                    if not isinstance(image, str):
                        raise ValueError(
                            f"{args.prompts_jsonl}:{line_number}: image must be a path string"
                        )
                    image_path = Path(image)
                    if not image_path.is_absolute():
                        image_path = source_dir / image_path
                cases.append(
                    PromptCase(str(value.get("id", f"prompt-{line_number}")), value["prompt"], image_path)
                )
    if not cases:
        raise ValueError("the prompt source contains no benchmark cases")
    for case in cases:
        if not case.prompt:
            raise ValueError(f"prompt {case.prompt_id!r} is empty")
        if case.image is not None and not case.image.is_file():
            raise ValueError(f"image does not exist: {case.image}")
    return cases


def parse_tags(values: list[str]) -> dict[str, str]:
    tags: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid --tag {value!r}; expected KEY=VALUE")
        key, tag_value = value.split("=", 1)
        if not key:
            raise ValueError("--tag key cannot be empty")
        tags[key] = tag_value
    return tags


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_plan(path: Path | None, expected_algorithm: str) -> dict[str, str]:
    empty = {
        "plan_score": "",
        "plan_score_definition": "",
        "plan_segments": "",
        "plan_risk_weight": "",
        "plan_selected_device_count": "",
        "plan_mean_ttft_ms": "",
        "plan_mean_tpot_ms": "",
        "plan_tail_ttft_ms": "",
        "plan_tail_tpot_ms": "",
        "plan_sha256": "",
    }
    if path is None:
        return empty
    value = json.loads(path.read_text(encoding="utf-8"))
    actual = value.get("algorithm")
    if actual != expected_algorithm:
        raise ValueError(
            f"plan report algorithm is {actual!r}, expected {expected_algorithm!r}"
        )
    segments = value.get("segments", [])
    layout = " | ".join(
        f"{item['device']}:{item['start_layer']}-{item['end_layer']}" for item in segments
    )
    return {
        "plan_score": str(value.get("cost", {}).get("score", "")),
        "plan_score_definition": str(value.get("score_definition", "")),
        "plan_segments": layout,
        "plan_risk_weight": str(value.get("risk_weight", "")),
        "plan_selected_device_count": str(value.get("selected_device_count", "")),
        "plan_mean_ttft_ms": str(value.get("cost", {}).get("ttft_ms", "")),
        "plan_mean_tpot_ms": str(value.get("cost", {}).get("tpot_ms", "")),
        "plan_tail_ttft_ms": str(value.get("cost", {}).get("tail_ttft_ms", "")),
        "plan_tail_tpot_ms": str(value.get("cost", {}).get("tail_tpot_ms", "")),
        "plan_sha256": file_sha256(path),
    }


def image_part(path: Path) -> dict[str, Any]:
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_base64", "media_type": media_type, "data": data}


def request_body(case: PromptCase, system: str) -> bytes:
    if case.image is None:
        content: Any = case.prompt
    else:
        content = [image_part(case.image), {"type": "text", "text": case.prompt}]
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})
    return json.dumps({"messages": messages, "stream": False}).encode("utf-8")


def extract_response_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("API response has no choices")
    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


def invoke(endpoint: str, body: bytes, timeout: float) -> tuple[dict[str, Any], float]:
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail[:500]}") from error
    wall_s = time.perf_counter() - started
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("API response is not a JSON object")
    return value, wall_s


def percentile_nearest_rank(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * percentile + 0.999999) - 1))
    return ordered[index]


def prepare_writer(path: Path, append: bool) -> tuple[Any, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    if exists and not append:
        raise FileExistsError(f"output already exists (use --append): {path}")
    if exists:
        with path.open(newline="", encoding="utf-8") as existing:
            header = next(csv.reader(existing), None)
        if header != list(CSV_FIELDS):
            raise ValueError(f"existing CSV has an incompatible header: {path}")
    handle = path.open("a" if append else "x", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
    if not exists:
        writer.writeheader()
    return handle, writer


def main() -> int:
    args = parse_args()
    try:
        endpoint = normalize_endpoint(args.api)
        cases = load_prompt_cases(args)
        tags = parse_tags(args.tag)
        plan = load_plan(args.plan_report, args.algorithm)
        handle, writer = prepare_writer(args.output, args.append)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    bodies = [request_body(case, args.system) for case in cases]
    print(
        f"algorithm={args.algorithm} endpoint={endpoint} cases={len(cases)} "
        f"warmup={args.warmup} runs={args.runs}"
    )

    try:
        for index in range(args.warmup):
            case_index = index % len(cases)
            print(f"warmup {index + 1}/{args.warmup}: {cases[case_index].prompt_id}", flush=True)
            invoke(endpoint, bodies[case_index], args.timeout)
    except Exception as error:
        handle.close()
        print(f"warmup failed; no measured rows were written: {error}", file=sys.stderr)
        return 1

    tag_json = json.dumps(tags, ensure_ascii=False, sort_keys=True)
    successful: list[dict[str, Any]] = []
    failures = 0
    try:
        for run_index in range(args.runs):
            case_index = run_index % len(cases)
            case = cases[case_index]
            row: dict[str, Any] = {
                field: "" for field in CSV_FIELDS
            }
            row.update(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "algorithm": args.algorithm,
                    "run": run_index + 1,
                    "prompt_index": case_index,
                    "prompt_id": case.prompt_id,
                    "prompt_chars": len(case.prompt),
                    "has_image": case.image is not None,
                    "configured_sample_len": args.sample_len,
                    "tag_json": tag_json,
                    **plan,
                }
            )
            try:
                response, wall_s = invoke(endpoint, bodies[case_index], args.timeout)
                response_text = extract_response_text(response)
                row.update(
                    {
                        "client_wall_s": wall_s,
                        "response_chars": len(response_text),
                        "response_sha256": hashlib.sha256(
                            response_text.encode("utf-8")
                        ).hexdigest(),
                        "status": "ok",
                    }
                )
                for field in METRIC_FIELDS:
                    value = response.get(field)
                    row[field] = "" if value is None else value
                successful.append(row)
                print(
                    f"run {run_index + 1}/{args.runs} {case.prompt_id}: "
                    f"ttft={row['ttft_s']}s total={row['total_s']}s "
                    f"decode_tps={row['decode_tokens_per_second']}",
                    flush=True,
                )
            except Exception as error:
                failures += 1
                row.update({"status": "error", "error": str(error)[:1000]})
                print(f"run {run_index + 1}/{args.runs} failed: {error}", file=sys.stderr)
            writer.writerow(row)
            handle.flush()
    finally:
        handle.close()

    if successful:
        print("summary (successful runs only):")
        for field in ("ttft_s", "total_s", "decode_tokens_per_second", "client_wall_s"):
            values = [float(row[field]) for row in successful if row[field] != ""]
            if values:
                print(
                    f"  {field}: median={statistics.median(values):.6f} "
                    f"p95={percentile_nearest_rank(values, 0.95):.6f} n={len(values)}"
                )
    print(f"wrote {args.output} ({len(successful)} successful, {failures} failed)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
