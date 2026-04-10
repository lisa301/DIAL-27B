#!/usr/bin/env python3
# Export Qwen3-VL vision tower (ViT + merger + deepstack side outputs) to ONNX
# with a static input shape.
#
# This exporter is aligned with the current Rust visual path:
# - nearest-resized 2D positional embeddings
# - vision rotary attention
# - deepstack side outputs for text-layer injection
#
# Notes:
# - Input expects normalized RGB float32 tensor: (1, 3, H, W) where H=W=--side.
#   Normalization should be done outside the model: (x/255 - 0.5) / 0.5
# - Qwen3-VL vision uses a temporal_patch_size (often 2). The Rust path folds the
#   duplicated-frame behavior into the patch projection weights. We do the same here.

from __future__ import annotations

import argparse
import json
import math
import os
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from safetensors.torch import safe_open


@dataclass(frozen=True)
class VisionCfg:
    depth: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    in_channels: int
    patch_size: int
    temporal_patch_size: int
    num_position_embeddings: int
    spatial_merge_size: int
    out_hidden_size: int
    deepstack_visual_indexes: List[int]


def load_vision_cfg(model_dir: str) -> VisionCfg:
    cfg_path = os.path.join(model_dir, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    vc = cfg["vision_config"]
    return VisionCfg(
        depth=int(vc["depth"]),
        hidden_size=int(vc["hidden_size"]),
        intermediate_size=int(vc["intermediate_size"]),
        num_heads=int(vc["num_heads"]),
        in_channels=int(vc["in_channels"]),
        patch_size=int(vc["patch_size"]),
        temporal_patch_size=int(vc["temporal_patch_size"]),
        num_position_embeddings=int(vc["num_position_embeddings"]),
        spatial_merge_size=int(vc["spatial_merge_size"]),
        out_hidden_size=int(vc["out_hidden_size"]),
        deepstack_visual_indexes=[int(v) for v in vc.get("deepstack_visual_indexes", [])],
    )


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


def rotate_half_vision(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    in_q_dtype = q.dtype
    in_k_dtype = k.dtype
    q = q.float()
    k = k.float()
    cos = cos.unsqueeze(0).unsqueeze(0).float()
    sin = sin.unsqueeze(0).unsqueeze(0).float()
    q = (q * cos + rotate_half_vision(q) * sin).to(in_q_dtype)
    k = (k * cos + rotate_half_vision(k) * sin).to(in_k_dtype)
    return q, k


class VitAttention(nn.Module):
    def __init__(self, hidden: int, num_heads: int):
        super().__init__()
        if hidden % num_heads != 0:
            raise ValueError(f"hidden {hidden} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=True)
        self.proj = nn.Linear(hidden, hidden, bias=True)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, seq, hidden = x.shape
        qkv = self.qkv(x).view(b, seq, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(1, 2).contiguous()
        k = qkv[:, :, 1].transpose(1, 2).contiguous()
        v = qkv[:, :, 2].transpose(1, 2).contiguous()

        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        in_dtype = q.dtype
        q = q.float()
        k = k.float()
        v = v.float()
        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = torch.softmax(att, dim=-1)
        y = torch.matmul(att, v).to(in_dtype)
        y = y.transpose(1, 2).contiguous().view(b, seq, hidden)
        return self.proj(y)


class VitMlp(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, intermediate, bias=True)
        self.fc2 = nn.Linear(intermediate, hidden, bias=True)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class VitBlock(nn.Module):
    def __init__(self, cfg: VisionCfg):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.hidden_size, eps=1e-5, elementwise_affine=True)
        self.attn = VitAttention(cfg.hidden_size, cfg.num_heads)
        self.norm2 = nn.LayerNorm(cfg.hidden_size, eps=1e-5, elementwise_affine=True)
        self.mlp = VitMlp(cfg.hidden_size, cfg.intermediate_size)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class Merger(nn.Module):
    def __init__(self, token_hidden: int, spatial_merge_size: int, out_hidden: int):
        super().__init__()
        m = int(spatial_merge_size)
        if m < 1:
            raise ValueError(f"invalid spatial_merge_size: {m}")
        self.spatial_merge_size = m
        merged_hidden = token_hidden * m * m
        self.norm = nn.LayerNorm(token_hidden, eps=1e-5, elementwise_affine=True)
        self.fc1 = nn.Linear(merged_hidden, merged_hidden, bias=True)
        self.fc2 = nn.Linear(merged_hidden, out_hidden, bias=True)
        self.act = nn.GELU(approximate="tanh")

    def spatial_merge(self, x: torch.Tensor, hp: int, wp: int) -> torch.Tensor:
        b, _seq, hidden = x.shape
        m = self.spatial_merge_size
        if m == 1:
            return x
        if hp % m != 0 or wp % m != 0:
            raise ValueError(f"patch grid {hp}x{wp} not divisible by spatial_merge_size {m}")
        x = x.view(b, hp, wp, hidden)
        x = x.view(b, hp // m, m, wp // m, m, hidden)
        x = x.transpose(2, 3).contiguous()
        return x.view(b, (hp // m) * (wp // m), hidden * m * m)

    def forward(self, x: torch.Tensor, hp: int, wp: int) -> torch.Tensor:
        x = self.norm(x)
        x = self.spatial_merge(x, hp, wp)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class DeepMerger(nn.Module):
    def __init__(self, token_hidden: int, spatial_merge_size: int, out_hidden: int):
        super().__init__()
        m = int(spatial_merge_size)
        if m < 1:
            raise ValueError(f"invalid spatial_merge_size: {m}")
        self.spatial_merge_size = m
        merged_hidden = token_hidden * m * m
        self.norm = nn.LayerNorm(merged_hidden, eps=1e-5, elementwise_affine=True)
        self.fc1 = nn.Linear(merged_hidden, merged_hidden, bias=True)
        self.fc2 = nn.Linear(merged_hidden, out_hidden, bias=True)
        self.act = nn.GELU(approximate="tanh")

    def spatial_merge(self, x: torch.Tensor, hp: int, wp: int) -> torch.Tensor:
        b, _seq, hidden = x.shape
        m = self.spatial_merge_size
        if m == 1:
            return x
        if hp % m != 0 or wp % m != 0:
            raise ValueError(f"patch grid {hp}x{wp} not divisible by spatial_merge_size {m}")
        x = x.view(b, hp, wp, hidden)
        x = x.view(b, hp // m, m, wp // m, m, hidden)
        x = x.transpose(2, 3).contiguous()
        return x.view(b, (hp // m) * (wp // m), hidden * m * m)

    def forward(self, x: torch.Tensor, hp: int, wp: int) -> torch.Tensor:
        x = self.spatial_merge(x, hp, wp)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class VisionTower(nn.Module):
    def __init__(self, cfg: VisionCfg, side: int):
        super().__init__()
        self.cfg = cfg
        self.side = int(side)
        if self.side % cfg.patch_size != 0:
            raise ValueError(f"side {side} not divisible by patch_size {cfg.patch_size}")
        self.hp = self.side // cfg.patch_size
        self.wp = self.side // cfg.patch_size
        self.head_dim = cfg.hidden_size // cfg.num_heads
        m = max(1, cfg.spatial_merge_size)
        if self.hp % m != 0 or self.wp % m != 0:
            raise ValueError(
                f"patch grid {self.hp}x{self.wp} not divisible by spatial_merge_size {m}"
            )

        base_side = int(math.isqrt(cfg.num_position_embeddings))
        if base_side * base_side != cfg.num_position_embeddings:
            raise ValueError("num_position_embeddings is not a square")
        if self.hp > base_side or self.wp > base_side:
            raise ValueError(f"patch grid {self.hp}x{self.wp} exceeds base {base_side}x{base_side}")

        self.patch = nn.Conv2d(
            cfg.in_channels,
            cfg.hidden_size,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
            bias=True,
        )
        self.pos_embed = nn.Embedding(cfg.num_position_embeddings, cfg.hidden_size)
        self.blocks = nn.ModuleList([VitBlock(cfg) for _ in range(cfg.depth)])
        self.merger = Merger(cfg.hidden_size, cfg.spatial_merge_size, cfg.out_hidden_size)
        self.deepstack_mergers = nn.ModuleList(
            [
                DeepMerger(cfg.hidden_size, cfg.spatial_merge_size, cfg.out_hidden_size)
                for _ in cfg.deepstack_visual_indexes
            ]
        )
        self.deepstack_layer_to_output = {
            layer_idx: output_idx
            for output_idx, layer_idx in enumerate(cfg.deepstack_visual_indexes)
        }

        pos_ids = self._build_resized_pos_ids(base_side)
        rope_cos, rope_sin = self._build_vision_rotary()
        self.register_buffer("pos_ids", pos_ids, persistent=False)
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

    def _build_resized_pos_ids(self, base_side: int) -> torch.Tensor:
        idx_grid = torch.arange(
            self.cfg.num_position_embeddings, dtype=torch.float32
        ).view(1, 1, base_side, base_side)
        if self.hp != base_side or self.wp != base_side:
            idx_grid = F.interpolate(idx_grid, size=(self.hp, self.wp), mode="nearest")
        return idx_grid.view(-1).round().to(torch.long)

    def _build_vision_rotary(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.head_dim % 2 != 0:
            raise ValueError(f"vision head_dim must be even, got {self.head_dim}")

        rotary_dim = self.head_dim // 2
        inv_freq = 1.0 / (
            10000.0
            ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / float(rotary_dim))
        )

        rows = torch.arange(self.hp, dtype=torch.float32).view(self.hp, 1).expand(self.hp, self.wp)
        cols = torch.arange(self.wp, dtype=torch.float32).view(1, self.wp).expand(self.hp, self.wp)
        rows = rows.reshape(-1, 1)
        cols = cols.reshape(-1, 1)

        row_freqs = rows * inv_freq.view(1, -1)
        col_freqs = cols * inv_freq.view(1, -1)
        rotary = torch.cat((row_freqs, col_freqs), dim=-1)
        emb = torch.cat((rotary, rotary), dim=-1)
        return emb.cos(), emb.sin()

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.patch(image.to(self.patch.weight.dtype))
        x = x.flatten(2).transpose(1, 2).contiguous()
        pos = self.pos_embed(self.pos_ids).unsqueeze(0)
        x = x + pos

        deep_outputs: List[torch.Tensor] = []
        for layer_idx, blk in enumerate(self.blocks):
            x = blk(x, self.rope_cos, self.rope_sin)
            output_idx = self.deepstack_layer_to_output.get(layer_idx)
            if output_idx is not None:
                deep_outputs.append(self.deepstack_mergers[output_idx](x, self.hp, self.wp))

        main = self.merger(x, self.hp, self.wp)
        return (main, *deep_outputs)


def load_weights(model: VisionTower, st_paths: List[str]) -> None:
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
            return reader.get_tensor(key)

        w5 = get_tensor("model.visual.patch_embed.proj.weight")
        sd["patch.weight"] = w5.sum(dim=2).contiguous()
        sd["patch.bias"] = get_tensor("model.visual.patch_embed.proj.bias")

        sd["pos_embed.weight"] = get_tensor("model.visual.pos_embed.weight")

        for i in range(model.cfg.depth):
            p = f"model.visual.blocks.{i}."
            sd[f"blocks.{i}.norm1.weight"] = get_tensor(p + "norm1.weight")
            sd[f"blocks.{i}.norm1.bias"] = get_tensor(p + "norm1.bias")
            sd[f"blocks.{i}.attn.qkv.weight"] = get_tensor(p + "attn.qkv.weight")
            sd[f"blocks.{i}.attn.qkv.bias"] = get_tensor(p + "attn.qkv.bias")
            sd[f"blocks.{i}.attn.proj.weight"] = get_tensor(p + "attn.proj.weight")
            sd[f"blocks.{i}.attn.proj.bias"] = get_tensor(p + "attn.proj.bias")
            sd[f"blocks.{i}.norm2.weight"] = get_tensor(p + "norm2.weight")
            sd[f"blocks.{i}.norm2.bias"] = get_tensor(p + "norm2.bias")
            sd[f"blocks.{i}.mlp.fc1.weight"] = get_tensor(p + "mlp.linear_fc1.weight")
            sd[f"blocks.{i}.mlp.fc1.bias"] = get_tensor(p + "mlp.linear_fc1.bias")
            sd[f"blocks.{i}.mlp.fc2.weight"] = get_tensor(p + "mlp.linear_fc2.weight")
            sd[f"blocks.{i}.mlp.fc2.bias"] = get_tensor(p + "mlp.linear_fc2.bias")

        sd["merger.norm.weight"] = get_tensor("model.visual.merger.norm.weight")
        sd["merger.norm.bias"] = get_tensor("model.visual.merger.norm.bias")
        sd["merger.fc1.weight"] = get_tensor("model.visual.merger.linear_fc1.weight")
        sd["merger.fc1.bias"] = get_tensor("model.visual.merger.linear_fc1.bias")
        sd["merger.fc2.weight"] = get_tensor("model.visual.merger.linear_fc2.weight")
        sd["merger.fc2.bias"] = get_tensor("model.visual.merger.linear_fc2.bias")

        for i, _layer_idx in enumerate(model.cfg.deepstack_visual_indexes):
            p = f"model.visual.deepstack_merger_list.{i}."
            sd[f"deepstack_mergers.{i}.norm.weight"] = get_tensor(p + "norm.weight")
            sd[f"deepstack_mergers.{i}.norm.bias"] = get_tensor(p + "norm.bias")
            sd[f"deepstack_mergers.{i}.fc1.weight"] = get_tensor(p + "linear_fc1.weight")
            sd[f"deepstack_mergers.{i}.fc1.bias"] = get_tensor(p + "linear_fc1.bias")
            sd[f"deepstack_mergers.{i}.fc2.weight"] = get_tensor(p + "linear_fc2.weight")
            sd[f"deepstack_mergers.{i}.fc2.bias"] = get_tensor(p + "linear_fc2.bias")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state_dict mismatch: missing={missing} unexpected={unexpected}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-dir",
        required=True,
        help="HF-like model directory containing config.json and safetensors files",
    )
    ap.add_argument(
        "--safetensors",
        default=None,
        help="Path to model.safetensors OR model.safetensors.index.json "
        "(defaults to auto-detect under --model-dir)",
    )
    ap.add_argument(
        "--side",
        type=int,
        default=448,
        help="Fixed square input side length in pixels",
    )
    ap.add_argument("--out", required=True, help="Output ONNX path")
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument(
        "--weight-dtype",
        default="float16",
        choices=["float16", "float32"],
        help="ONNX stored weight dtype. float16 keeps 8B multi-output export in a single file.",
    )
    args = ap.parse_args()

    cfg = load_vision_cfg(args.model_dir)
    st_paths = resolve_safetensors_paths(args.model_dir, args.safetensors)

    model = VisionTower(cfg, side=args.side).eval()
    load_weights(model, st_paths)
    if args.weight_dtype == "float16":
        model = model.half()
    else:
        model = model.float()
    dummy = torch.randn(1, 3, args.side, args.side, dtype=torch.float32)

    output_names = ["vision_embeds"] + [
        f"deepstack_{i}" for i in range(len(cfg.deepstack_visual_indexes))
    ]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        args.out,
        input_names=["image"],
        output_names=output_names,
        opset_version=args.opset,
        do_constant_folding=True,
        external_data=False,
    )

    print("exported:", args.out)
    print(f"safetensors_files: {len(st_paths)}")
    print(f"weight_dtype: {args.weight_dtype}")
    print(f"input: (1,3,{args.side},{args.side})")
    print(f"patch_grid: {model.hp}x{model.wp} seq={model.hp * model.wp}")
    merged_seq = (model.hp // max(1, cfg.spatial_merge_size)) * (
        model.wp // max(1, cfg.spatial_merge_size)
    )
    print(f"output[0]: (1,{merged_seq},{cfg.out_hidden_size})")
    for i, layer_idx in enumerate(cfg.deepstack_visual_indexes):
        print(f"output[{i + 1}]: deepstack layer {layer_idx} -> (1,{merged_seq},{cfg.out_hidden_size})")


if __name__ == "__main__":
    main()
