#!/usr/bin/env python3
"""Export Qwen3-VL-8B text decode chunks to ONNX.

This exporter is intentionally scoped to the Qwen3-VL-8B text decoder and
targets the "decode" path only:

- input hidden states are fixed to `(1, 1, 4096)`
- each exported ONNX contains a contiguous chunk of transformer layers
- KV cache is modeled as explicit inputs/outputs
- RoPE cosine/sine tables are provided as inputs to avoid rebuilding them in-graph

Two export modes are provided:

- dynamic: past KV sequence dimension is dynamic
- static:  past KV sequence dimension is fixed to `--static-past-len`

Dynamic export is the most faithful representation of decode semantics.
Static export is useful for later RKNN bucket experiments, but note that its
outputs grow to `past_len + 1`, so it is not yet a full runtime cache-update
solution by itself.

KV output mode:
- full:  output full `present_k/v` (legacy runtime behavior)
- delta: output only newly generated `k/v` (seq=1) for local append
"""

from __future__ import annotations

import argparse
import json
import math
import os
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from safetensors.torch import safe_open


@dataclass(frozen=True)
class TextCfg:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int


def load_text_cfg(model_dir: str) -> TextCfg:
    cfg_path = os.path.join(model_dir, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if cfg.get("model_type") != "qwen3_vl":
        raise ValueError(f"unsupported model_type in {cfg_path}: {cfg.get('model_type')}")

    tc = cfg["text_config"]
    out = TextCfg(
        hidden_size=int(tc["hidden_size"]),
        intermediate_size=int(tc["intermediate_size"]),
        num_hidden_layers=int(tc["num_hidden_layers"]),
        num_attention_heads=int(tc["num_attention_heads"]),
        num_key_value_heads=int(tc["num_key_value_heads"]),
        head_dim=int(tc.get("head_dim", tc["hidden_size"] // tc["num_attention_heads"])),
        rms_norm_eps=float(tc["rms_norm_eps"]),
        rope_theta=float(tc["rope_theta"]),
        max_position_embeddings=int(tc["max_position_embeddings"]),
    )

    # This script is intentionally specialized to 8B to keep the graph layout
    # and runtime assumptions narrow.
    if (
        out.hidden_size != 4096
        or out.intermediate_size != 12288
        or out.num_hidden_layers != 36
        or out.num_attention_heads != 32
        or out.num_key_value_heads != 8
        or out.head_dim != 128
    ):
        raise ValueError(
            "this exporter only supports Qwen3-VL-8B text config "
            f"(got hidden={out.hidden_size}, inter={out.intermediate_size}, "
            f"layers={out.num_hidden_layers}, heads={out.num_attention_heads}, "
            f"kv_heads={out.num_key_value_heads}, head_dim={out.head_dim})"
        )

    return out


def _load_safetensors_paths_from_index(index_path: str) -> List[str]:
    with open(index_path, "r", encoding="utf-8") as f:
        idx = json.load(f)
    weight_map = idx.get("weight_map", {})
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError(f"invalid or empty weight_map in {index_path}")

    parent = os.path.dirname(index_path)
    seen = set()
    paths: List[str] = []
    for rel in weight_map.values():
        p = os.path.join(parent, rel)
        if p not in seen:
            seen.add(p)
            paths.append(p)

    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"missing shard files from index {index_path}: {missing}")
    return paths


def resolve_safetensors_paths(model_dir: str, safetensors_arg: str | None) -> List[str]:
    if safetensors_arg:
        if not os.path.exists(safetensors_arg):
            raise FileNotFoundError(safetensors_arg)
        if safetensors_arg.endswith(".index.json"):
            return _load_safetensors_paths_from_index(safetensors_arg)
        return [safetensors_arg]

    single = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single):
        return [single]

    index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        return _load_safetensors_paths_from_index(index)

    raise FileNotFoundError(
        f"cannot find model.safetensors or model.safetensors.index.json under {model_dir}"
    )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.view(1, 1, 1, -1).float()
    sin = sin.view(1, 1, 1, -1).float()
    q = q.float()
    k = k.float()
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


class RmsNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight.float()


class QwenMlp(nn.Module):
    def __init__(self, cfg: TextCfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class QwenAttention(nn.Module):
    def __init__(self, cfg: TextCfg, kv_output: str):
        super().__init__()
        if kv_output not in {"full", "delta"}:
            raise ValueError(f"unsupported kv_output: {kv_output}")
        self.kv_output = kv_output

        q_size = cfg.num_attention_heads * cfg.head_dim
        kv_size = cfg.num_key_value_heads * cfg.head_dim

        self.q_proj = nn.Linear(cfg.hidden_size, q_size, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, kv_size, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, kv_size, bias=False)
        self.o_proj = nn.Linear(q_size, cfg.hidden_size, bias=False)

        self.q_norm = RmsNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.k_norm = RmsNorm(cfg.head_dim, cfg.rms_norm_eps)

        self.num_attention_heads = cfg.num_attention_heads
        self.num_key_value_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.num_kv_groups = cfg.num_attention_heads // cfg.num_key_value_heads

    def repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        return x.repeat_interleave(self.num_kv_groups, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_k: torch.Tensor,
        past_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, hidden_size = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2).contiguous()
        k = self.k_proj(x).view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.v_proj(x).view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2).contiguous()

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rotary(q, k, cos, sin)

        present_k = torch.cat([past_k.float(), k], dim=2)
        present_v = torch.cat([past_v.float(), v.float()], dim=2)

        att_k = self.repeat_kv(present_k)
        att_v = self.repeat_kv(present_v)

        att = torch.matmul(q, att_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = torch.softmax(att, dim=-1)
        y = torch.matmul(att, att_v)
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, hidden_size)
        y = self.o_proj(y)
        if self.kv_output == "delta":
            kv_k = k.float()
            kv_v = v.float()
        else:
            kv_k = present_k
            kv_v = present_v
        return y, kv_k, kv_v


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TextCfg, kv_output: str):
        super().__init__()
        self.rms_1 = RmsNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn = QwenAttention(cfg, kv_output)
        self.rms_2 = RmsNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = QwenMlp(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_k: torch.Tensor,
        past_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = x.float()
        h = self.rms_1(x)
        h, present_k, present_v = self.attn(h, cos, sin, past_k, past_v)
        x = h + residual
        x = x + self.mlp(self.rms_2(x))
        return x, present_k, present_v


class DecodeChunk(nn.Module):
    def __init__(self, cfg: TextCfg, layer_ids: Sequence[int], kv_output: str):
        super().__init__()
        if not layer_ids:
            raise ValueError("layer_ids must not be empty")
        if kv_output not in {"full", "delta"}:
            raise ValueError(f"unsupported kv_output: {kv_output}")
        self.cfg = cfg
        self.layer_ids = list(layer_ids)
        self.kv_output = kv_output
        self.blocks = nn.ModuleList([TransformerBlock(cfg, kv_output) for _ in self.layer_ids])

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *past_kv: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if len(past_kv) != 2 * len(self.blocks):
            raise ValueError(
                f"expected {2 * len(self.blocks)} past tensors, got {len(past_kv)}"
            )

        outputs: List[torch.Tensor] = []
        h = x.float()
        for idx, block in enumerate(self.blocks):
            past_k = past_kv[2 * idx]
            past_v = past_kv[2 * idx + 1]
            h, present_k, present_v = block(h, cos, sin, past_k, past_v)
            outputs.append(present_k)
            outputs.append(present_v)
        return (h, *outputs)


def load_chunk_weights(model: DecodeChunk, st_paths: List[str]) -> None:
    sd: Dict[str, torch.Tensor] = {}
    with ExitStack() as stack:
        key_to_reader: Dict[str, object] = {}
        for st_path in st_paths:
            f = stack.enter_context(safe_open(st_path, framework="pt", device="cpu"))
            for key in f.keys():
                if key in key_to_reader:
                    raise RuntimeError(f"duplicate tensor key across shards: {key}")
                key_to_reader[key] = f

        def get_tensor(key: str) -> torch.Tensor:
            reader = key_to_reader.get(key)
            if reader is None:
                raise KeyError(f"tensor not found in provided safetensors shards: {key}")
            return reader.get_tensor(key).float().contiguous()

        for local_idx, global_idx in enumerate(model.layer_ids):
            prefix = f"model.language_model.layers.{global_idx}."
            block = f"blocks.{local_idx}."

            sd[block + "rms_1.weight"] = get_tensor(prefix + "input_layernorm.weight")
            sd[block + "attn.q_proj.weight"] = get_tensor(prefix + "self_attn.q_proj.weight")
            sd[block + "attn.k_proj.weight"] = get_tensor(prefix + "self_attn.k_proj.weight")
            sd[block + "attn.v_proj.weight"] = get_tensor(prefix + "self_attn.v_proj.weight")
            sd[block + "attn.o_proj.weight"] = get_tensor(prefix + "self_attn.o_proj.weight")
            sd[block + "attn.q_norm.weight"] = get_tensor(prefix + "self_attn.q_norm.weight")
            sd[block + "attn.k_norm.weight"] = get_tensor(prefix + "self_attn.k_norm.weight")
            sd[block + "rms_2.weight"] = get_tensor(prefix + "post_attention_layernorm.weight")
            sd[block + "mlp.gate_proj.weight"] = get_tensor(prefix + "mlp.gate_proj.weight")
            sd[block + "mlp.up_proj.weight"] = get_tensor(prefix + "mlp.up_proj.weight")
            sd[block + "mlp.down_proj.weight"] = get_tensor(prefix + "mlp.down_proj.weight")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state_dict mismatch: missing={missing} unexpected={unexpected}")


def build_rotary_cache(position: int, cfg: TextCfg) -> tuple[torch.Tensor, torch.Tensor]:
    if cfg.head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {cfg.head_dim}")
    theta = torch.arange(0, cfg.head_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (cfg.rope_theta ** (theta / float(cfg.head_dim)))
    pos = torch.tensor([float(position)], dtype=torch.float32).view(1, 1)
    freqs = pos * inv_freq.view(1, -1)
    emb = torch.cat([freqs, freqs], dim=-1).contiguous()
    return emb.cos(), emb.sin()


def iter_chunk_ranges(start_layer: int, end_layer: int, chunk_size: int) -> Iterable[tuple[int, int]]:
    curr = start_layer
    while curr <= end_layer:
        stop = min(curr + chunk_size - 1, end_layer)
        yield curr, stop
        curr = stop + 1


def export_chunk(
    model: DecodeChunk,
    out_path: str,
    opset: int,
    mode: str,
    dummy_past_len: int,
    rotary_pos: int,
) -> None:
    cfg = model.cfg
    cos, sin = build_rotary_cache(rotary_pos, cfg)
    x = torch.randn(1, 1, cfg.hidden_size, dtype=torch.float32)

    past_len = int(dummy_past_len)
    if past_len < 0:
        raise ValueError(f"dummy past length must be >= 0, got {past_len}")

    past_inputs: List[torch.Tensor] = []
    for _ in model.layer_ids:
        past_inputs.append(torch.zeros(1, cfg.num_key_value_heads, past_len, cfg.head_dim, dtype=torch.float32))
        past_inputs.append(torch.zeros(1, cfg.num_key_value_heads, past_len, cfg.head_dim, dtype=torch.float32))

    input_names = ["x", "cos", "sin"]
    output_names = ["hidden_out"]
    dynamic_axes = None

    for local_idx, global_idx in enumerate(model.layer_ids):
        input_names.extend([f"past_k_l{global_idx}", f"past_v_l{global_idx}"])
        output_names.extend([f"present_k_l{global_idx}", f"present_v_l{global_idx}"])

    if mode == "dynamic":
        dynamic_axes = {}
        for local_idx, global_idx in enumerate(model.layer_ids):
            dynamic_axes[f"past_k_l{global_idx}"] = {2: "past_seq"}
            dynamic_axes[f"past_v_l{global_idx}"] = {2: "past_seq"}
            if model.kv_output == "full":
                dynamic_axes[f"present_k_l{global_idx}"] = {2: "present_seq"}
                dynamic_axes[f"present_v_l{global_idx}"] = {2: "present_seq"}

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    torch.onnx.export(
        model,
        (x, cos, sin, *past_inputs),
        out_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
        external_data=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-dir",
        required=True,
        help="HF-like model directory containing config.json and safetensors shards",
    )
    ap.add_argument(
        "--safetensors",
        default=None,
        help="Path to model.safetensors OR model.safetensors.index.json "
        "(defaults to auto-detect under --model-dir)",
    )
    ap.add_argument("--out-dir", required=True, help="Directory to write exported ONNX chunks")
    ap.add_argument("--chunk-size", type=int, default=2, help="Number of text layers per ONNX chunk")
    ap.add_argument("--start-layer", type=int, default=0, help="First global layer index to export")
    ap.add_argument("--end-layer", type=int, default=35, help="Last global layer index to export (inclusive)")
    ap.add_argument(
        "--mode",
        choices=["dynamic", "static"],
        default="dynamic",
        help="dynamic keeps past KV seq dim symbolic; static fixes it to --static-past-len",
    )
    ap.add_argument(
        "--static-past-len",
        type=int,
        default=1024,
        help="Used when --mode static; also used as dummy seq length for export tracing",
    )
    ap.add_argument(
        "--dynamic-dummy-past-len",
        type=int,
        default=16,
        help="Dummy past KV length used when --mode dynamic",
    )
    ap.add_argument(
        "--kv-output",
        choices=["full", "delta"],
        default="full",
        help="KV output mode: full=present_k/v (legacy), delta=new token k/v only (recommended with local append runtime).",
    )
    ap.add_argument(
        "--rotary-pos",
        type=int,
        default=0,
        help="Dummy decode position used to build RoPE cos/sin sample inputs during export",
    )
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Load weights and run one eager forward pass per chunk without exporting ONNX",
    )
    args = ap.parse_args()

    cfg = load_text_cfg(args.model_dir)
    st_paths = resolve_safetensors_paths(args.model_dir, args.safetensors)

    if args.chunk_size != 2:
        print(f"[WARN] recommended chunk_size for 8B is 2, got {args.chunk_size}")
    if args.start_layer < 0 or args.end_layer >= cfg.num_hidden_layers or args.start_layer > args.end_layer:
        raise ValueError(
            f"invalid layer range [{args.start_layer}, {args.end_layer}] for {cfg.num_hidden_layers} layers"
        )

    if args.mode == "static":
        dummy_past_len = args.static_past_len
    else:
        dummy_past_len = args.dynamic_dummy_past_len

    os.makedirs(args.out_dir, exist_ok=True)
    exported: List[str] = []

    for layer_start, layer_end in iter_chunk_ranges(args.start_layer, args.end_layer, args.chunk_size):
        layer_ids = list(range(layer_start, layer_end + 1))
        model = DecodeChunk(cfg, layer_ids, kv_output=args.kv_output).eval()
        load_chunk_weights(model, st_paths)

        if args.dry_run:
            cos, sin = build_rotary_cache(args.rotary_pos, cfg)
            x = torch.randn(1, 1, cfg.hidden_size, dtype=torch.float32)
            past_inputs: List[torch.Tensor] = []
            for _ in layer_ids:
                past_inputs.append(
                    torch.zeros(1, cfg.num_key_value_heads, dummy_past_len, cfg.head_dim, dtype=torch.float32)
                )
                past_inputs.append(
                    torch.zeros(1, cfg.num_key_value_heads, dummy_past_len, cfg.head_dim, dtype=torch.float32)
                )
            outputs = model(x, cos, sin, *past_inputs)
            print(
                f"[DRY-RUN] layers {layer_start:02d}-{layer_end:02d}: "
                f"hidden_out={tuple(outputs[0].shape)} "
                f"present_seq={outputs[1].shape[2]}"
            )
            continue

        mode_tag = "dynamic" if args.mode == "dynamic" else f"static_p{dummy_past_len}"
        out_name = (
            f"qwen3_vl_8b_text_decode_l{layer_start:02d}_l{layer_end:02d}_"
            f"{mode_tag}_kv{args.kv_output}.onnx"
        )
        out_path = os.path.join(args.out_dir, out_name)
        export_chunk(
            model=model,
            out_path=out_path,
            opset=args.opset,
            mode=args.mode,
            dummy_past_len=dummy_past_len,
            rotary_pos=args.rotary_pos,
        )
        exported.append(out_path)
        print(f"exported: {out_path}")

    print(f"mode: {args.mode}")
    print(f"kv_output: {args.kv_output}")
    print(f"chunk_size: {args.chunk_size}")
    print(f"layers: {args.start_layer}-{args.end_layer}")
    print(f"safetensors_files: {len(st_paths)}")
    if args.mode == "dynamic":
        print(f"dynamic_dummy_past_len: {dummy_past_len}")
    else:
        print(f"static_past_len: {dummy_past_len}")

    if exported:
        sample = exported[0]
        print("sample convert command:")
        print(
            "python /home/seaway/sdb/ljl/Dial_llama/tools/convert_onnx_to_rknn.py "
            f"--onnx {sample} --out {sample[:-5]}.rknn --target-platform rk3588"
        )
        if args.kv_output == "delta":
            print("note: this ONNX uses delta KV outputs; runtime must append KV locally.")


if __name__ == "__main__":
    main()
