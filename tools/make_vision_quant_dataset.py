#!/usr/bin/env python3
"""Create RKNN calibration tensors for Qwen3-VL vision RKNN quantization.

The exported vision ONNX expects normalized RGB float32 input in NCHW:
    image: (1, 3, side, side)
with normalization:
    (x / 255 - 0.5) / 0.5

This script resizes each image with letterbox padding to a fixed square, writes
one .npy tensor per image, and creates the RKNN dataset list file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True, help="Image files or directories")
    ap.add_argument("--out-dir", required=True, help="Directory for generated .npy tensors")
    ap.add_argument("--dataset", required=True, help="Output dataset txt path")
    ap.add_argument("--side", type=int, default=448)
    ap.add_argument("--max-images", type=int, default=128)
    return ap.parse_args()


def collect_images(paths: list[str], max_images: int) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    out: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            for item in sorted(path.rglob("*")):
                if item.suffix.lower() in exts:
                    out.append(item)
                    if max_images > 0 and len(out) >= max_images:
                        return out
        elif path.is_file() and path.suffix.lower() in exts:
            out.append(path)
            if max_images > 0 and len(out) >= max_images:
                return out
    return out


def letterbox_rgb(path: Path, side: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    w, h = image.size
    scale = min(side / w, side / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    image = image.resize((nw, nh), Image.Resampling.BICUBIC)

    canvas = Image.new("RGB", (side, side), (128, 128, 128))
    left = (side - nw) // 2
    top = (side - nh) // 2
    canvas.paste(image, (left, top))

    arr = np.asarray(canvas).astype(np.float32)
    arr = (arr / 255.0 - 0.5) / 0.5
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(arr, dtype=np.float32)


def main() -> int:
    args = parse_args()
    images = collect_images(args.images, args.max_images)
    if not images:
        raise SystemExit("no images found")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = Path(args.dataset)
    dataset.parent.mkdir(parents=True, exist_ok=True)

    rows: list[str] = []
    for i, image_path in enumerate(images):
        tensor = letterbox_rgb(image_path, args.side)
        out = out_dir / f"vision_calib_{i:04d}.npy"
        np.save(out, tensor)
        rows.append(str(out.resolve()))

    dataset.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} tensors to {out_dir}")
    print(f"dataset: {dataset}")
    print(f"input shape: (1,3,{args.side},{args.side})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
