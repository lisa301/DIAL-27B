#!/usr/bin/env python3
"""Build a text-only Qwen3 prefix model from a Qwen3-VL checkpoint.

The output contains language layers `0..end_layer` from the Qwen3-VL text
branch. It is mainly used to generate boundary hidden states for RKLLM suffix
quantization:

    prefix layers 0..L-1 -> hidden at input of suffix layer L

The prefix model still includes a final norm because HuggingFace Qwen3Model
expects it, but calibration code should capture the hidden state before that
norm.
"""

from __future__ import annotations

import argparse
import os
from collections import OrderedDict

from make_qwen3_vl_text_rkllm_suffix_model import (
    EMBED_DST,
    EMBED_SRC,
    LM_HEAD_DST,
    NORM_DST,
    NORM_SRC,
    TensorRef,
    all_tensor_keys,
    choose_lm_head,
    copy_common_files,
    ensure_output_dir,
    find_key_path,
    human_size,
    load_json,
    parse_size,
    read_tensor,
    require_tensor_ref,
    resolve_weight_map,
    tensor_nbytes,
    write_json,
    write_sharded_safetensors,
)


def make_qwen3_prefix_config(src_cfg: dict, end_layer: int) -> dict:
    text_cfg = dict(src_cfg.get("text_config", {}))
    if not text_cfg:
        raise KeyError("source config.json does not contain text_config")

    total_layers = int(text_cfg["num_hidden_layers"])
    if end_layer < 0 or end_layer >= total_layers:
        raise ValueError(f"end_layer must be in [0, {total_layers - 1}], got {end_layer}")

    out = dict(text_cfg)
    out["architectures"] = ["Qwen3Model"]
    out["model_type"] = "qwen3"
    out["num_hidden_layers"] = end_layer + 1
    out["use_cache"] = False
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
        if isinstance(vision_cfg, dict):
            out["vision_config"] = {
                key: vision_cfg[key]
                for key in ("spatial_merge_size", "patch_size", "temporal_patch_size")
                if key in vision_cfg
            }
    if isinstance(out.get("layer_types"), list):
        out["layer_types"] = out["layer_types"][: end_layer + 1]

    return out


def make_qwen3_prefix_causal_lm_config(src_cfg: dict, end_layer: int) -> dict:
    out = make_qwen3_prefix_config(src_cfg, end_layer)
    out["architectures"] = ["Qwen3ForCausalLM"]
    out["use_cache"] = True
    return out


def build_prefix_tensor_plan(
    *,
    src_cfg: dict,
    end_layer: int,
    include_lm_head: bool,
    weight_map: dict[str, str],
    shard_files,
) -> "OrderedDict[str, TensorRef]":
    keys = all_tensor_keys(weight_map, shard_files)
    refs: "OrderedDict[str, TensorRef]" = OrderedDict()

    def add_ref(ref: TensorRef) -> None:
        if ref.dst_key in refs:
            raise RuntimeError(f"duplicate output tensor key: {ref.dst_key}")
        refs[ref.dst_key] = ref

    add_ref(require_tensor_ref(src_key=EMBED_SRC, dst_key=EMBED_DST, weight_map=weight_map, shard_files=shard_files))
    add_ref(require_tensor_ref(src_key=NORM_SRC, dst_key=NORM_DST, weight_map=weight_map, shard_files=shard_files))

    for layer in range(0, end_layer + 1):
        src_prefix = f"model.language_model.layers.{layer}."
        dst_prefix = f"model.layers.{layer}."
        selected = [k for k in keys if k.startswith(src_prefix)]
        if not selected:
            raise KeyError(f"no tensors found for source layer {layer} ({src_prefix}*)")
        for src_key in selected:
            path = find_key_path(src_key, weight_map, shard_files)
            if path is None:
                raise KeyError(f"tensor listed but not found: {src_key}")
            add_ref(TensorRef(src_key=src_key, src_path=path, dst_key=dst_prefix + src_key[len(src_prefix) :]))

    if include_lm_head:
        lm_head = choose_lm_head(weight_map, shard_files)
        if lm_head is None:
            raise KeyError("lm_head.weight not found in source model")
        src_key, path = lm_head
        add_ref(TensorRef(src_key=src_key, src_path=path, dst_key=LM_HEAD_DST))

    return refs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-model-dir", required=True, help="Original Qwen3-VL HF model directory")
    ap.add_argument("--out-model-dir", required=True, help="Output Qwen3 prefix model directory")
    ap.add_argument("--end-layer", type=int, required=True, help="Last original language layer to include")
    ap.add_argument(
        "--causal-lm",
        action="store_true",
        help="Write a Qwen3ForCausalLM config and include lm_head.weight for RKLLM OUTPUT parsing",
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
    if args.end_layer >= total_layers:
        raise ValueError(f"end_layer {args.end_layer} exceeds last layer {total_layers - 1}")

    weight_map, shard_files = resolve_weight_map(src_dir)
    refs = build_prefix_tensor_plan(
        src_cfg=src_cfg,
        end_layer=args.end_layer,
        include_lm_head=args.causal_lm,
        weight_map=weight_map,
        shard_files=shard_files,
    )
    out_cfg = (
        make_qwen3_prefix_causal_lm_config(src_cfg, args.end_layer)
        if args.causal_lm
        else make_qwen3_prefix_config(src_cfg, args.end_layer)
    )

    print("[PLAN]")
    print(f"src_model_dir: {src_dir}")
    print(f"out_model_dir: {out_dir}")
    print(f"source_layers: 0..{args.end_layer}")
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
            "generated_by": "tools/make_qwen3_vl_text_prefix_model.py",
            "source_model_dir": src_dir,
            "source_layers": f"0..{args.end_layer}",
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
