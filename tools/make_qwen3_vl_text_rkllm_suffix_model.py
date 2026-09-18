#!/usr/bin/env python3
"""Build a text-only Qwen3 suffix model from a Qwen3-VL checkpoint.

The output is a HuggingFace-style `Qwen3ForCausalLM` directory containing
language layers `start_layer..last_layer` from the Qwen3-VL text branch.

This is intended for the first RKLLM split-deployment milestone:

    GPU/Candle layers 0..start_layer-1 -> RKLLM suffix layers -> existing lm_head

The suffix model keeps the real final RMSNorm, so `end_layer` is intentionally
fixed to the last language layer. For arbitrary middle segments, use a custom
model without final norm instead.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple

from safetensors import safe_open
from safetensors.torch import save_file


COMMON_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "added_tokens.json",
    "chat_template.json",
    "generation_config.json",
]

EMBED_SRC = "model.language_model.embed_tokens.weight"
EMBED_DST = "model.embed_tokens.weight"
NORM_SRC = "model.language_model.norm.weight"
NORM_DST = "model.norm.weight"
LM_HEAD_SRC_CANDIDATES = [
    "lm_head.weight",
    "model.language_model.lm_head.weight",
]
LM_HEAD_DST = "lm_head.weight"
OPTIONAL_PREFIX_RENAMES = [
    ("model.language_model.rotary_emb.", "model.rotary_emb."),
]


@dataclass(frozen=True)
class TensorRef:
    src_key: str
    src_path: str
    dst_key: str


def parse_size(text: str) -> int:
    raw = text.strip().lower()
    units = [
        ("gib", 1024**3),
        ("gb", 1000**3),
        ("mib", 1024**2),
        ("mb", 1000**2),
        ("kib", 1024),
        ("kb", 1000),
        ("b", 1),
    ]
    for suffix, scale in units:
        if raw.endswith(suffix):
            return int(float(raw[: -len(suffix)].strip()) * scale)
    return int(raw)


def human_size(num_bytes: int) -> str:
    return f"{num_bytes / (1024**3):.2f} GiB"


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Mapping) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def resolve_weight_map(model_dir: str) -> Tuple[Dict[str, str], List[str]]:
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    single_path = os.path.join(model_dir, "model.safetensors")

    if os.path.exists(index_path):
        index = load_json(index_path)
        weight_map = index.get("weight_map", {})
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError(f"invalid or empty weight_map in {index_path}")
        abs_map = {k: os.path.join(model_dir, v) for k, v in weight_map.items()}
        shard_files = sorted(set(abs_map.values()))
        missing = [p for p in shard_files if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"missing safetensors shards: {missing}")
        return abs_map, shard_files

    if os.path.exists(single_path):
        return {}, [single_path]

    raise FileNotFoundError(
        f"cannot find model.safetensors.index.json or model.safetensors under {model_dir}"
    )


def all_tensor_keys(weight_map: Mapping[str, str], shard_files: Iterable[str]) -> List[str]:
    if weight_map:
        return sorted(weight_map.keys())

    keys: List[str] = []
    for path in shard_files:
        with safe_open(path, framework="pt", device="cpu") as f:
            keys.extend(f.keys())
    return sorted(keys)


def find_key_path(key: str, weight_map: Mapping[str, str], shard_files: Iterable[str]) -> str | None:
    if key in weight_map:
        return weight_map[key]
    for path in shard_files:
        with safe_open(path, framework="pt", device="cpu") as f:
            if key in f.keys():
                return path
    return None


def require_tensor_ref(
    *,
    src_key: str,
    dst_key: str,
    weight_map: Mapping[str, str],
    shard_files: Iterable[str],
) -> TensorRef:
    path = find_key_path(src_key, weight_map, shard_files)
    if path is None:
        raise KeyError(f"required tensor not found: {src_key}")
    return TensorRef(src_key=src_key, src_path=path, dst_key=dst_key)


def choose_lm_head(weight_map: Mapping[str, str], shard_files: Iterable[str]) -> Tuple[str, str] | None:
    for key in LM_HEAD_SRC_CANDIDATES:
        path = find_key_path(key, weight_map, shard_files)
        if path is not None:
            return key, path
    return None


def make_qwen3_config(src_cfg: Mapping, start_layer: int) -> dict:
    text_cfg = dict(src_cfg.get("text_config", {}))
    if not text_cfg:
        raise KeyError("source config.json does not contain text_config")

    total_layers = int(text_cfg["num_hidden_layers"])
    if start_layer < 0 or start_layer >= total_layers:
        raise ValueError(f"start_layer must be in [0, {total_layers - 1}], got {start_layer}")

    out = dict(text_cfg)
    out["architectures"] = ["Qwen3ForCausalLM"]
    out["model_type"] = "qwen3"
    out["num_hidden_layers"] = total_layers - start_layer
    out["use_cache"] = True
    out["tie_word_embeddings"] = bool(
        text_cfg.get("tie_word_embeddings", src_cfg.get("tie_word_embeddings", False))
    )

    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if key not in out and key in src_cfg:
            out[key] = src_cfg[key]
    if "head_dim" not in out:
        out["head_dim"] = int(out["hidden_size"]) // int(out["num_attention_heads"])
    if "hidden_act" not in out:
        out["hidden_act"] = "silu"
    if "torch_dtype" not in out:
        out["torch_dtype"] = src_cfg.get("torch_dtype", "bfloat16")
    if "transformers_version" in src_cfg and "transformers_version" not in out:
        out["transformers_version"] = src_cfg["transformers_version"]
    if "rope_scaling" not in out:
        out["rope_scaling"] = None
    if isinstance(out.get("rope_scaling"), dict):
        vision_cfg = src_cfg.get("vision_config")
        if isinstance(vision_cfg, Mapping):
            # RKLLM checks this when Qwen3-VL mrope fields are present, even for
            # the text-only suffix model that consumes external embeddings.
            out["vision_config"] = {
                key: vision_cfg[key]
                for key in ("spatial_merge_size", "patch_size", "temporal_patch_size")
                if key in vision_cfg
            }
    if isinstance(out.get("layer_types"), list):
        out["layer_types"] = out["layer_types"][start_layer:]

    return out


def build_tensor_plan(
    *,
    src_cfg: Mapping,
    start_layer: int,
    include_lm_head: str,
    materialize_tied_lm_head: bool,
    weight_map: Mapping[str, str],
    shard_files: Iterable[str],
) -> "OrderedDict[str, TensorRef]":
    text_cfg = src_cfg["text_config"]
    total_layers = int(text_cfg["num_hidden_layers"])
    keys = all_tensor_keys(weight_map, shard_files)

    refs: "OrderedDict[str, TensorRef]" = OrderedDict()

    def add_ref(ref: TensorRef) -> None:
        if ref.dst_key in refs:
            raise RuntimeError(f"duplicate output tensor key: {ref.dst_key}")
        refs[ref.dst_key] = ref

    add_ref(require_tensor_ref(src_key=EMBED_SRC, dst_key=EMBED_DST, weight_map=weight_map, shard_files=shard_files))
    add_ref(require_tensor_ref(src_key=NORM_SRC, dst_key=NORM_DST, weight_map=weight_map, shard_files=shard_files))

    for src_prefix, dst_prefix in OPTIONAL_PREFIX_RENAMES:
        for src_key in keys:
            if src_key.startswith(src_prefix):
                path = find_key_path(src_key, weight_map, shard_files)
                if path is not None:
                    add_ref(TensorRef(src_key=src_key, src_path=path, dst_key=dst_prefix + src_key[len(src_prefix) :]))

    for global_layer in range(start_layer, total_layers):
        src_prefix = f"model.language_model.layers.{global_layer}."
        dst_prefix = f"model.layers.{global_layer - start_layer}."
        selected = [k for k in keys if k.startswith(src_prefix)]
        if not selected:
            raise KeyError(f"no tensors found for source layer {global_layer} ({src_prefix}*)")
        for src_key in selected:
            path = find_key_path(src_key, weight_map, shard_files)
            if path is None:
                raise KeyError(f"tensor listed but not found: {src_key}")
            add_ref(TensorRef(src_key=src_key, src_path=path, dst_key=dst_prefix + src_key[len(src_prefix) :]))

    lm_head = choose_lm_head(weight_map, shard_files)
    tied = bool(text_cfg.get("tie_word_embeddings", src_cfg.get("tie_word_embeddings", False)))
    if include_lm_head == "always" and lm_head is None and not materialize_tied_lm_head:
        raise KeyError("lm_head.weight not found; use --materialize-tied-lm-head if the model ties embeddings")
    if include_lm_head == "never":
        return refs
    if lm_head is not None:
        src_key, path = lm_head
        add_ref(TensorRef(src_key=src_key, src_path=path, dst_key=LM_HEAD_DST))
    elif materialize_tied_lm_head:
        embed = require_tensor_ref(src_key=EMBED_SRC, dst_key=LM_HEAD_DST, weight_map=weight_map, shard_files=shard_files)
        add_ref(embed)
    elif include_lm_head == "auto" and not tied:
        raise KeyError("lm_head.weight not found and source config does not declare tied embeddings")

    return refs


def ensure_output_dir(path: str, force: bool) -> None:
    if os.path.exists(path):
        if not force:
            raise FileExistsError(f"{path} already exists (use --force to overwrite)")
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def copy_common_files(src_dir: str, out_dir: str) -> List[str]:
    copied = []
    for name in COMMON_FILES:
        src = os.path.join(src_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, name))
            copied.append(name)
    return copied


def tensor_nbytes(tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def read_tensor(ref: TensorRef):
    with safe_open(ref.src_path, framework="pt", device="cpu") as f:
        if ref.src_key not in f.keys():
            raise KeyError(f"{ref.src_key} not found in {ref.src_path}")
        return f.get_tensor(ref.src_key).contiguous()


def write_sharded_safetensors(
    refs: "OrderedDict[str, TensorRef]",
    out_dir: str,
    max_shard_bytes: int,
) -> Tuple[Dict[str, str], int, List[str]]:
    weight_map: Dict[str, str] = {}
    shard_names: List[str] = []
    pending = {}
    pending_bytes = 0
    total_bytes = 0
    shard_idx = 1

    def flush() -> None:
        nonlocal pending, pending_bytes, shard_idx
        if not pending:
            return
        name = f"model-{shard_idx:05d}.safetensors"
        save_file(pending, os.path.join(out_dir, name))
        for key in pending:
            weight_map[key] = name
        shard_names.append(name)
        pending = {}
        pending_bytes = 0
        shard_idx += 1

    for dst_key, ref in refs.items():
        tensor = read_tensor(ref)
        size = tensor_nbytes(tensor)
        if pending and pending_bytes + size > max_shard_bytes:
            flush()
        pending[dst_key] = tensor
        pending_bytes += size
        total_bytes += size
        if pending_bytes >= max_shard_bytes:
            flush()
    flush()
    return weight_map, total_bytes, shard_names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-model-dir", required=True, help="Original Qwen3-VL HF model directory")
    ap.add_argument("--out-model-dir", required=True, help="Output Qwen3 suffix model directory")
    ap.add_argument(
        "--start-layer",
        type=int,
        required=True,
        help="First original language layer to put into the RKLLM suffix model",
    )
    ap.add_argument(
        "--include-lm-head",
        choices=["auto", "always", "never"],
        default="auto",
        help="Whether to include lm_head.weight in the generated model",
    )
    ap.add_argument(
        "--materialize-tied-lm-head",
        action="store_true",
        help="When lm_head.weight is absent, write lm_head.weight from embed_tokens.weight",
    )
    ap.add_argument("--max-shard-size", default="2GiB", help="Output safetensors shard size")
    ap.add_argument("--force", action="store_true", help="Overwrite output directory")
    ap.add_argument("--dry-run", action="store_true", help="Print the tensor plan and exit")
    args = ap.parse_args()

    src_dir = os.path.abspath(args.src_model_dir)
    out_dir = os.path.abspath(args.out_model_dir)
    src_cfg_path = os.path.join(src_dir, "config.json")
    if not os.path.exists(src_cfg_path):
        raise FileNotFoundError(src_cfg_path)

    src_cfg = load_json(src_cfg_path)
    text_cfg = src_cfg.get("text_config", {})
    total_layers = int(text_cfg.get("num_hidden_layers", 0))
    if total_layers <= 0:
        raise ValueError("source text_config.num_hidden_layers must be positive")
    if args.start_layer >= total_layers:
        raise ValueError(f"start_layer {args.start_layer} exceeds last layer {total_layers - 1}")

    weight_map, shard_files = resolve_weight_map(src_dir)
    refs = build_tensor_plan(
        src_cfg=src_cfg,
        start_layer=args.start_layer,
        include_lm_head=args.include_lm_head,
        materialize_tied_lm_head=args.materialize_tied_lm_head,
        weight_map=weight_map,
        shard_files=shard_files,
    )
    out_cfg = make_qwen3_config(src_cfg, args.start_layer)

    print("[PLAN]")
    print(f"src_model_dir: {src_dir}")
    print(f"out_model_dir: {out_dir}")
    print(f"source_layers: {args.start_layer}..{total_layers - 1}")
    print(f"output_num_hidden_layers: {out_cfg['num_hidden_layers']}")
    print(f"tensor_count: {len(refs)}")
    print(f"max_shard_size: {args.max_shard_size}")
    for dst_key, ref in refs.items():
        print(f"  - {ref.src_key} -> {dst_key}")

    if args.dry_run:
        return

    ensure_output_dir(out_dir, args.force)
    write_json(os.path.join(out_dir, "config.json"), out_cfg)
    copied = copy_common_files(src_dir, out_dir)
    out_weight_map, total_bytes, shard_names = write_sharded_safetensors(
        refs, out_dir, parse_size(args.max_shard_size)
    )
    index = {
        "metadata": {
            "generated_by": "tools/make_qwen3_vl_text_rkllm_suffix_model.py",
            "source_model_dir": src_dir,
            "source_layers": f"{args.start_layer}..{total_layers - 1}",
            "total_size": str(total_bytes),
        },
        "weight_map": out_weight_map,
    }
    write_json(os.path.join(out_dir, "model.safetensors.index.json"), index)

    print("\n[RESULT]")
    print(f"wrote_config: {os.path.join(out_dir, 'config.json')}")
    print(f"wrote_shards({len(shard_names)}): {shard_names}")
    print(f"tensor_bytes: {human_size(total_bytes)}")
    print(f"copied_files({len(copied)}): {copied}")


if __name__ == "__main__":
    main()
