#!/usr/bin/env python3
"""Build a Qwen3 text transformer-block-only model directory.

The output intentionally excludes embed_tokens, final norm and lm_head. It is
used to test whether RKLLM custom_config can export a pure hidden-state chunk:

    hidden_states -> layers[start..end] -> hidden_states
"""

from __future__ import annotations

import argparse
import os
from collections import OrderedDict

import torch

from make_qwen3_vl_text_rkllm_suffix_model import (
    TensorRef,
    all_tensor_keys,
    copy_common_files,
    ensure_output_dir,
    find_key_path,
    human_size,
    load_json,
    parse_size,
    resolve_weight_map,
    write_json,
    write_sharded_safetensors,
)


def make_config(src_cfg: dict, start_layer: int, end_layer: int) -> dict:
    text_cfg = dict(src_cfg.get("text_config", {}))
    if not text_cfg:
        raise KeyError("source config.json does not contain text_config")
    total_layers = int(text_cfg["num_hidden_layers"])
    if start_layer < 0 or end_layer < start_layer or end_layer >= total_layers:
        raise ValueError(
            f"layer range must be within [0,{total_layers - 1}], got {start_layer}..{end_layer}"
        )

    out = dict(text_cfg)
    out["architectures"] = ["Qwen3Model"]
    out["model_type"] = "qwen3"
    out["num_hidden_layers"] = end_layer - start_layer + 1
    out["use_cache"] = True
    out["tie_word_embeddings"] = False
    out["vocab_size"] = 1
    if "head_dim" not in out:
        out["head_dim"] = int(out["hidden_size"]) // int(out["num_attention_heads"])
    if "hidden_act" not in out:
        out["hidden_act"] = "silu"
    if "torch_dtype" not in out:
        out["torch_dtype"] = src_cfg.get("torch_dtype", "bfloat16")
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
        out["layer_types"] = out["layer_types"][start_layer : end_layer + 1]
    return out


def build_refs(src_dir: str, start_layer: int, end_layer: int):
    weight_map, shard_files = resolve_weight_map(src_dir)
    keys = all_tensor_keys(weight_map, shard_files)
    refs: "OrderedDict[str, TensorRef]" = OrderedDict()

    def add_ref(ref: TensorRef) -> None:
        if ref.dst_key in refs:
            raise RuntimeError(f"duplicate output tensor key: {ref.dst_key}")
        refs[ref.dst_key] = ref

    for global_layer in range(start_layer, end_layer + 1):
        src_prefix = f"model.language_model.layers.{global_layer}."
        dst_prefix = f"model.layers.{global_layer - start_layer}."
        selected = [k for k in keys if k.startswith(src_prefix)]
        if not selected:
            raise KeyError(f"no tensors found for source layer {global_layer} ({src_prefix}*)")
        for src_key in selected:
            path = find_key_path(src_key, weight_map, shard_files)
            if path is None:
                raise KeyError(f"tensor listed but not found: {src_key}")
            add_ref(
                TensorRef(
                    src_key=src_key,
                    src_path=path,
                    dst_key=dst_prefix + src_key[len(src_prefix) :],
                )
            )
    return refs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-model-dir", required=True)
    ap.add_argument("--out-model-dir", required=True)
    ap.add_argument("--start-layer", type=int, required=True)
    ap.add_argument("--end-layer", type=int, required=True)
    ap.add_argument("--max-shard-size", default="2GiB")
    ap.add_argument(
        "--dummy-io",
        action="store_true",
        help="Add tiny embed_tokens/norm/lm_head tensors to satisfy RKLLM parsers.",
    )
    ap.add_argument(
        "--dummy-vocab-size",
        type=int,
        default=1,
        help="Vocabulary size for --dummy-io tensors. RKLLM export requires at least 48.",
    )
    ap.add_argument(
        "--causal-lm",
        action="store_true",
        help="Declare the output config as Qwen3ForCausalLM instead of Qwen3Model.",
    )
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src_dir = os.path.abspath(args.src_model_dir)
    out_dir = os.path.abspath(args.out_model_dir)
    src_cfg = load_json(os.path.join(src_dir, "config.json"))
    out_cfg = make_config(src_cfg, args.start_layer, args.end_layer)
    if args.causal_lm:
        out_cfg["architectures"] = ["Qwen3ForCausalLM"]
    refs = build_refs(src_dir, args.start_layer, args.end_layer)

    print("[PLAN]")
    print(f"src_model_dir: {src_dir}")
    print(f"out_model_dir: {out_dir}")
    print(f"source_layers: {args.start_layer}..{args.end_layer}")
    print(f"output_num_hidden_layers: {out_cfg['num_hidden_layers']}")
    print(f"tensor_count: {len(refs)}")
    for dst_key, ref in refs.items():
        print(f"  - {ref.src_key} -> {dst_key}")
    if args.dry_run:
        return

    ensure_output_dir(out_dir, args.force)
    if args.dummy_io:
        if args.dummy_vocab_size <= 0:
            raise ValueError("--dummy-vocab-size must be positive")
        out_cfg["vocab_size"] = args.dummy_vocab_size
        # RKLLM's quantization path still touches special token ids even when
        # the runtime input is RKLLM_INPUT_EMBED, so keep them inside the tiny
        # dummy vocabulary.
        out_cfg["bos_token_id"] = 0
        out_cfg["eos_token_id"] = 1 if args.dummy_vocab_size > 1 else 0
        out_cfg["pad_token_id"] = out_cfg["eos_token_id"]
    write_json(os.path.join(out_dir, "config.json"), out_cfg)
    copied = copy_common_files(src_dir, out_dir)
    weight_map, total_bytes, shard_names = write_sharded_safetensors(
        refs, out_dir, parse_size(args.max_shard_size)
    )
    if args.dummy_io:
        hidden_size = int(out_cfg["hidden_size"])
        dummy_name = f"model-{len(shard_names) + 1:05d}.safetensors"
        dummy_tensors = {
            "model.embed_tokens.weight": torch.zeros(
                (args.dummy_vocab_size, hidden_size), dtype=torch.bfloat16
            ),
            "model.norm.weight": torch.ones((hidden_size,), dtype=torch.bfloat16),
            "lm_head.weight": torch.zeros(
                (args.dummy_vocab_size, hidden_size), dtype=torch.bfloat16
            ),
        }
        from safetensors.torch import save_file

        save_file(dummy_tensors, os.path.join(out_dir, dummy_name))
        for key in dummy_tensors:
            weight_map[key] = dummy_name
        shard_names.append(dummy_name)
        total_bytes += sum(t.numel() * t.element_size() for t in dummy_tensors.values())
    write_json(
        os.path.join(out_dir, "model.safetensors.index.json"),
        {
            "metadata": {
                "generated_by": "tools/make_qwen3_vl_text_block_model.py",
                "source_model_dir": src_dir,
                "source_layers": f"{args.start_layer}..{args.end_layer}",
                "total_size": str(total_bytes),
            },
            "weight_map": weight_map,
        },
    )
    print("\n[RESULT]")
    print(f"wrote_config: {os.path.join(out_dir, 'config.json')}")
    print(f"wrote_shards({len(shard_names)}): {shard_names}")
    print(f"tensor_bytes: {human_size(total_bytes)}")
    print(f"copied_files({len(copied)}): {copied}")


if __name__ == "__main__":
    main()
