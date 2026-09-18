#!/usr/bin/env python3
"""Build a minimal Qwen3-VL model directory for Dial runtime.

This script extracts only the tensors required by current runtime in
`full-npu` text mode:
- model.language_model.embed_tokens.weight
- model.language_model.norm.weight
- lm_head.weight (required when tie_word_embeddings is false)

It writes:
- model-min.safetensors
- model.safetensors.index.json (pointing only to model-min.safetensors)
- essential tokenizer/config files
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Dict, List, Tuple

from safetensors import safe_open
from safetensors.torch import save_file


ESSENTIAL_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.json",
    "generation_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
]

EMBED_KEY = "model.language_model.embed_tokens.weight"
NORM_KEY = "model.language_model.norm.weight"
LM_HEAD_KEYS = ["lm_head.weight", "model.language_model.lm_head.weight"]


def human_size(num_bytes: int) -> str:
    gib = num_bytes / (1024**3)
    gb = num_bytes / (1000**3)
    return f"{num_bytes} B ({gib:.2f} GiB, {gb:.2f} GB)"


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_weight_map(model_dir: str) -> Tuple[Dict[str, str], List[str]]:
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    single_path = os.path.join(model_dir, "model.safetensors")

    if os.path.exists(index_path):
        idx = load_json(index_path)
        weight_map = idx.get("weight_map", {})
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError(f"invalid or empty weight_map in {index_path}")
        shard_files = sorted({os.path.join(model_dir, rel) for rel in weight_map.values()})
        missing = [p for p in shard_files if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"missing shard files: {missing}")
        return {k: os.path.join(model_dir, v) for k, v in weight_map.items()}, shard_files

    if os.path.exists(single_path):
        return {}, [single_path]

    raise FileNotFoundError(
        f"cannot find model.safetensors.index.json or model.safetensors under {model_dir}"
    )


def find_key_in_files(key: str, files: List[str]) -> str | None:
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as f:
            if key in f.keys():
                return path
    return None


def load_tensor(key: str, path: str):
    with safe_open(path, framework="pt", device="cpu") as f:
        if key not in f.keys():
            raise KeyError(f"{key} not found in {path}")
        return f.get_tensor(key)


def ensure_dir(path: str, force: bool) -> None:
    if os.path.exists(path):
        if not force:
            raise FileExistsError(f"{path} already exists (use --force to overwrite)")
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def choose_lm_head_key(weight_map: Dict[str, str], files: List[str]) -> str | None:
    for key in LM_HEAD_KEYS:
        if key in weight_map:
            return key
    for key in LM_HEAD_KEYS:
        if find_key_in_files(key, files) is not None:
            return key
    return None


def copy_essential_files(src_dir: str, out_dir: str) -> List[str]:
    copied = []
    for name in ESSENTIAL_FILES:
        src = os.path.join(src_dir, name)
        if os.path.exists(src):
            dst = os.path.join(out_dir, name)
            shutil.copy2(src, dst)
            copied.append(name)
    return copied


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-model-dir", required=True, help="Original Qwen3-VL model directory")
    ap.add_argument("--out-model-dir", required=True, help="Output minimal model directory")
    ap.add_argument(
        "--include-lm-head",
        choices=["auto", "always", "never"],
        default="auto",
        help="auto: include lm_head when present or required; always/never: force behavior",
    )
    ap.add_argument("--force", action="store_true", help="Overwrite output directory if exists")
    ap.add_argument("--dry-run", action="store_true", help="Print plan and exit")
    args = ap.parse_args()

    src_dir = os.path.abspath(args.src_model_dir)
    out_dir = os.path.abspath(args.out_model_dir)

    cfg_path = os.path.join(src_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(cfg_path)
    cfg = load_json(cfg_path)

    text_cfg = cfg.get("text_config", {})
    tie_root = bool(cfg.get("tie_word_embeddings", False))
    tie_text = bool(text_cfg.get("tie_word_embeddings", False))
    tied = tie_root or tie_text

    weight_map, shard_files = resolve_weight_map(src_dir)

    planned_keys = [EMBED_KEY, NORM_KEY]
    lm_key = choose_lm_head_key(weight_map, shard_files)

    if args.include_lm_head == "always":
        if lm_key is None:
            raise KeyError("lm_head.weight not found, but --include-lm-head=always was set")
        planned_keys.append(lm_key)
    elif args.include_lm_head == "never":
        if not tied:
            raise RuntimeError(
                "model is not tied embeddings, but --include-lm-head=never was set; "
                "runtime would fail without lm_head.weight"
            )
    else:
        # auto
        if lm_key is not None:
            planned_keys.append(lm_key)
        elif not tied:
            raise RuntimeError(
                "lm_head.weight not found and model is not tied embeddings; cannot build minimal package"
            )

    key_to_file: Dict[str, str] = {}
    for key in planned_keys:
        path = weight_map.get(key)
        if path is None:
            path = find_key_in_files(key, shard_files)
        if path is None:
            raise KeyError(f"required tensor '{key}' not found in safetensors shards")
        key_to_file[key] = path

    print("[PLAN]")
    print(f"src_model_dir: {src_dir}")
    print(f"out_model_dir: {out_dir}")
    print(f"tie_word_embeddings(root/text): {tie_root}/{tie_text}")
    print(f"selected_tensor_keys: {planned_keys}")
    for k in planned_keys:
        print(f"  - {k} <- {os.path.basename(key_to_file[k])}")

    if args.dry_run:
        return

    ensure_dir(out_dir, args.force)

    tensors = {}
    tensor_bytes = 0
    for key in planned_keys:
        t = load_tensor(key, key_to_file[key])
        tensors[key] = t
        tensor_bytes += t.numel() * t.element_size()

    out_tensor_name = "model-min.safetensors"
    out_tensor_path = os.path.join(out_dir, out_tensor_name)
    save_file(tensors, out_tensor_path)

    out_index = {
        "metadata": {
            "generated_by": "tools/make_min_model_dir.py",
            "source_model_dir": src_dir,
            "tensor_count": str(len(planned_keys)),
        },
        "weight_map": {k: out_tensor_name for k in planned_keys},
    }
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump(out_index, f, ensure_ascii=False, indent=2)
        f.write("\n")

    copied_files = copy_essential_files(src_dir, out_dir)
    out_size = os.path.getsize(out_tensor_path)

    print("\n[RESULT]")
    print(f"saved: {out_tensor_path}")
    print(f"tensor_bytes(uncompressed estimate): {human_size(tensor_bytes)}")
    print(f"saved_safetensors_size: {human_size(out_size)}")
    print(f"copied_files({len(copied_files)}): {copied_files}")
    print(f"index: {os.path.join(out_dir, 'model.safetensors.index.json')}")


if __name__ == "__main__":
    main()
