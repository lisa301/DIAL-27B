#!/usr/bin/env python3
"""Create RKLLM input_embed calibration data for a Qwen3-VL text suffix model.

For a suffix split starting at language layer L, RKLLM sees the hidden state at
the input of layer L, not raw token embeddings. This script can either run a
new-enough original Qwen3-VL HuggingFace model, or a generated text-only prefix
model containing layers `0..L-1`.

It writes RKLLM dataset records:

    [{"input_embed": [[[...]]], "target": "..."}]

`hidden_states[L]` is used because HF hidden_states[0] is the embedding output,
hidden_states[1] is after layer 0, and so on.

When `--prefix-model-dir` is used, the script captures the hidden state before
the prefix model final norm. That tensor is exactly the input to suffix layer L.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List

import torch


def load_records(path: str, text_key: str, target_key: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            return data["data"]
        raise ValueError(f"unsupported JSON shape in {path}; expected list or {{'data': list}}")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                records.append({text_key: text, target_key: ""})
    return records


def record_text(tokenizer, record: Dict[str, Any], text_key: str, apply_chat_template: bool) -> str:
    if isinstance(record.get("messages"), list):
        return tokenizer.apply_chat_template(
            record["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )

    text = str(record.get(text_key, ""))
    if not text:
        raise ValueError(f"record has no '{text_key}' text: {record}")

    if apply_chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return text


def resolve_dtype(raw: str):
    raw = raw.lower()
    if raw == "auto":
        return "auto"
    if raw == "float32":
        return torch.float32
    if raw == "float16":
        return torch.float16
    if raw == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {raw}")


def load_model(model_dir: str, device: str, dtype: str, trust_remote_code: bool):
    import transformers
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    torch_dtype = resolve_dtype(dtype)

    class_names = [
        "Qwen3VLForConditionalGeneration",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModelForCausalLM",
        "AutoModel",
    ]
    last_err = None
    for class_name in class_names:
        cls = getattr(transformers, class_name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(
                model_dir,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=True,
            )
            model.eval()
            if device != "cpu":
                model.to(device)
            return tokenizer, model
        except Exception as err:  # noqa: BLE001 - keep trying compatible HF classes.
            last_err = err
    raise RuntimeError(f"failed to load model with supported HF auto classes: {last_err}")


def load_prefix_model(model_dir: str, device: str, dtype: str, trust_remote_code: bool):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    model = AutoModel.from_pretrained(
        model_dir,
        torch_dtype=resolve_dtype(dtype),
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.eval()
    if device != "cpu":
        model.to(device)
    return tokenizer, model


def find_hidden_states(outputs) -> Iterable[torch.Tensor]:
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is not None:
        return hidden_states
    language_outputs = getattr(outputs, "language_model_outputs", None)
    if language_outputs is not None and getattr(language_outputs, "hidden_states", None) is not None:
        return language_outputs.hidden_states
    raise RuntimeError("model output does not contain hidden_states")


def prefix_boundary_hidden(model, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    captured: Dict[str, torch.Tensor] = {}

    def capture_norm_input(_module, module_inputs):
        captured["hidden"] = module_inputs[0].detach()

    handle = model.norm.register_forward_pre_hook(capture_norm_input)
    try:
        with torch.inference_mode():
            model(**inputs, use_cache=False)
    finally:
        handle.remove()
    if "hidden" not in captured:
        raise RuntimeError("failed to capture prefix boundary hidden before final norm")
    return captured["hidden"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", help="Original Qwen3-VL model directory")
    ap.add_argument(
        "--prefix-model-dir",
        help="Generated text-only prefix model directory for layers 0..L-1",
    )
    ap.add_argument("--samples", required=True, help="Text samples: .json, .jsonl, or one prompt per line")
    ap.add_argument("--out", required=True, help="Output RKLLM dataset JSON")
    ap.add_argument("--start-layer", type=int, required=True, help="Suffix start layer L")
    ap.add_argument("--text-key", default="input")
    ap.add_argument("--target-key", default="target")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, ...")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--apply-chat-template", action="store_true")
    ap.add_argument("--output-dtype", default="float16", choices=["float32", "float16"])
    args = ap.parse_args()

    if not args.model_dir and not args.prefix_model_dir:
        raise ValueError("set --model-dir or --prefix-model-dir")

    model_dir = os.path.abspath(args.prefix_model_dir or args.model_dir)
    records = load_records(args.samples, args.text_key, args.target_key)
    if args.max_samples > 0:
        records = records[: args.max_samples]
    if not records:
        raise ValueError("no calibration records loaded")

    if args.prefix_model_dir:
        tokenizer, model = load_prefix_model(model_dir, args.device, args.dtype, args.trust_remote_code)
        use_prefix_capture = True
    else:
        tokenizer, model = load_model(model_dir, args.device, args.dtype, args.trust_remote_code)
        use_prefix_capture = False
    out_dtype = torch.float16 if args.output_dtype == "float16" else torch.float32
    dataset = []

    for idx, record in enumerate(records):
        text = record_text(tokenizer, record, args.text_key, args.apply_chat_template)
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        if use_prefix_capture:
            if int(model.config.num_hidden_layers) != args.start_layer:
                raise ValueError(
                    f"prefix model has {model.config.num_hidden_layers} layers, expected start_layer {args.start_layer}"
                )
            hidden = prefix_boundary_hidden(model, inputs).to(out_dtype).cpu()
        else:
            with torch.inference_mode():
                outputs = model(**inputs, output_hidden_states=True, use_cache=False)
            hidden_states = list(find_hidden_states(outputs))
            if args.start_layer >= len(hidden_states):
                raise IndexError(
                    f"start_layer {args.start_layer} out of range for hidden_states len {len(hidden_states)}"
                )
            hidden = hidden_states[args.start_layer].detach().to(out_dtype).cpu()
        dataset.append(
            {
                "input_embed": hidden.tolist(),
                "target": str(record.get(args.target_key, "")),
            }
        )
        print(f"[INFO] sample {idx + 1}/{len(records)} hidden_shape={tuple(hidden.shape)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False)
        f.write("\n")
    print(f"[RESULT] wrote {len(dataset)} records to {args.out}")


if __name__ == "__main__":
    main()
