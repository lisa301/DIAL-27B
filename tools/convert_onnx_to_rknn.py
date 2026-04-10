#!/usr/bin/env python3
"""Convert a static-shape ONNX model to RKNN.

Default is FP16 (no quantization), suitable for Qwen3-VL vision ONNX exported by
`export_qwen3_vl_vision_onnx.py` with input name `image` and shape `(1,3,448,448)`.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import List


def parse_shape_3d(text: str) -> List[int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"input shape must be 3D like '3,448,448', got: {text}")
    vals = [int(x) for x in parts]
    if any(v <= 0 for v in vals):
        raise ValueError(f"input shape must be positive, got: {vals}")
    return vals


def build_dynamic_input_from_onnx(onnx_path: str, symbolic_dim_value: int) -> List[List[List[int]]]:
    """Build RKNN `dynamic_input` from ONNX graph input shapes.

    RKNN expects:
      dynamic_input = [
        [shape_input0, shape_input1, ...],   # profile 0
        [shape_input0, shape_input1, ...],   # profile 1 (optional)
      ]

    Here we build a single profile and replace all symbolic dims with
    `symbolic_dim_value`.
    """
    if symbolic_dim_value <= 0:
        raise ValueError(f"symbolic_dim_value must be > 0, got {symbolic_dim_value}")

    import onnx

    model = onnx.load(onnx_path)
    input_shapes: List[List[int]] = []
    for value_info in model.graph.input:
        tensor_type = value_info.type.tensor_type
        if tensor_type.elem_type == 0:
            raise ValueError(f"unsupported non-tensor input: {value_info.name}")
        dims: List[int] = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value") and dim.dim_value > 0:
                dims.append(int(dim.dim_value))
            elif dim.HasField("dim_param") and dim.dim_param:
                dims.append(symbolic_dim_value)
            else:
                raise ValueError(
                    f"input '{value_info.name}' has unknown dim; cannot build dynamic_input automatically"
                )
        input_shapes.append(dims)

    if not input_shapes:
        raise ValueError(f"no graph inputs found in {onnx_path}")
    return [input_shapes]


def convert_one(
    *,
    onnx: str,
    out: str,
    target_platform: str,
    input_name: str,
    input_size: List[int],
    manual_input: bool,
    float_dtype: str,
    optimization_level: int,
    quantize: bool,
    dataset: str | None,
    quantized_dtype: str,
    verbose: bool,
    dynamic_input_seq_len: int,
) -> int:
    from rknn.api import RKNN

    if not os.path.exists(onnx):
        raise FileNotFoundError(onnx)

    out_dir = os.path.dirname(os.path.abspath(out)) or "."
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] converting: {onnx}")
    print(f"[INFO] output: {out}")

    rknn = RKNN(verbose=verbose)
    try:
        dynamic_input = None
        if dynamic_input_seq_len > 0:
            dynamic_input = build_dynamic_input_from_onnx(onnx, dynamic_input_seq_len)
            print(
                f"[INFO] dynamic_input enabled (symbolic dims -> {dynamic_input_seq_len}), "
                f"profiles={len(dynamic_input)}"
            )

        ret = rknn.config(
            target_platform=target_platform,
            float_dtype=float_dtype,
            optimization_level=optimization_level,
            quantized_dtype=quantized_dtype,
            # Keep None because this ONNX expects already-normalized float input.
            mean_values=None,
            std_values=None,
            dynamic_input=dynamic_input,
        )
        if ret != 0:
            print(f"[ERR] rknn.config failed: {ret}")
            return ret or 1

        # For static ONNX, let RKNN parse input metadata directly.
        # Passing input_size_list to static ONNX may reinterpret rank and break Conv shape inference.
        if manual_input:
            ret = rknn.load_onnx(
                model=onnx,
                inputs=[input_name],
                input_size_list=[input_size],
            )
        else:
            ret = rknn.load_onnx(model=onnx)
        if ret != 0:
            print(f"[ERR] rknn.load_onnx failed: {ret}")
            return ret or 1

        ret = rknn.build(
            do_quantization=quantize,
            dataset=dataset if quantize else None,
        )
        if ret != 0:
            print(f"[ERR] rknn.build failed: {ret}")
            return ret or 1

        ret = rknn.export_rknn(out)
        if ret != 0:
            print(f"[ERR] rknn.export_rknn failed: {ret}")
            return ret or 1

        print("exported:", out)
        print(f"target_platform: {target_platform}")
        print(f"quantize: {quantize}")
        if manual_input:
            print(f"manual_input: True, input=[1,{','.join(map(str, input_size))}] name={input_name}")
        else:
            print("manual_input: False (use ONNX embedded input shape)")
        return 0
    finally:
        rknn.release()


def main() -> int:
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--onnx", help="Path to input ONNX (single-file mode)")
    group.add_argument("--onnx-dir", help="Directory containing ONNX files (batch mode)")

    ap.add_argument("--out", help="Path to output RKNN (required in single-file mode)")
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for batch mode (default: same as --onnx-dir)",
    )
    ap.add_argument(
        "--pattern",
        default="*.onnx",
        help="Glob pattern used in batch mode, e.g. '*.onnx'",
    )
    ap.add_argument(
        "--suffix",
        default=".rknn",
        help="Output suffix in batch mode (default: .rknn)",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files whose output RKNN already exists (batch mode)",
    )
    ap.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue converting next files after an error (batch mode)",
    )
    ap.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Convert at most N files in batch mode (0 means no limit)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned conversions and exit (batch mode)",
    )

    ap.add_argument("--target-platform", default="rk3588", help="RKNN target platform, e.g. rk3588")
    ap.add_argument("--input-name", default="image", help="ONNX input name (used with --manual-input)")
    ap.add_argument(
        "--input-shape",
        default="3,448,448",
        help="Input C,H,W for batch=1 (used with --manual-input), e.g. 3,448,448",
    )
    ap.add_argument(
        "--manual-input",
        action="store_true",
        help="Manually pass inputs/input_size_list to load_onnx. "
        "For static ONNX models, keep this OFF.",
    )
    ap.add_argument(
        "--float-dtype",
        default="float16",
        choices=["float16", "float32"],
        help="Float precision for non-quantized build",
    )
    ap.add_argument(
        "--optimization-level",
        type=int,
        default=3,
        choices=[0, 1, 2, 3],
        help="RKNN optimization level",
    )
    ap.add_argument("--quantize", action="store_true", help="Enable quantization build")
    ap.add_argument(
        "--dataset",
        default=None,
        help="Calibration dataset list file (required when --quantize)",
    )
    ap.add_argument(
        "--quantized-dtype",
        default="w8a8",
        help="Quantized dtype used by RKNN build, e.g. w8a8",
    )
    ap.add_argument("--verbose", action="store_true", help="Enable RKNN verbose logs")
    ap.add_argument(
        "--dynamic-input-seq-len",
        type=int,
        default=0,
        help="If >0, auto-build RKNN dynamic_input from ONNX and replace symbolic dims (e.g. past_seq) with this value.",
    )
    args = ap.parse_args()

    if args.quantize and not args.dataset:
        raise ValueError("--dataset is required when --quantize is enabled")
    if args.dataset and not os.path.exists(args.dataset):
        raise FileNotFoundError(args.dataset)

    input_size = parse_shape_3d(args.input_shape)

    if args.onnx:
        if not args.out:
            raise ValueError("--out is required in single-file mode")
        return convert_one(
            onnx=args.onnx,
            out=args.out,
            target_platform=args.target_platform,
            input_name=args.input_name,
            input_size=input_size,
            manual_input=args.manual_input,
            float_dtype=args.float_dtype,
            optimization_level=args.optimization_level,
            quantize=args.quantize,
            dataset=args.dataset,
            quantized_dtype=args.quantized_dtype,
            verbose=args.verbose,
            dynamic_input_seq_len=args.dynamic_input_seq_len,
        )

    onnx_dir = args.onnx_dir
    if onnx_dir is None:
        raise ValueError("internal error: onnx-dir is empty")
    if not os.path.isdir(onnx_dir):
        raise NotADirectoryError(onnx_dir)
    if args.out:
        raise ValueError("--out cannot be used in batch mode, use --out-dir")

    files = sorted(glob.glob(os.path.join(onnx_dir, args.pattern)))
    files = [f for f in files if os.path.isfile(f)]
    if args.max_files and args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"no ONNX files matched: dir={onnx_dir}, pattern={args.pattern}")

    out_dir = args.out_dir or onnx_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] batch mode: {len(files)} files")
    print(f"[INFO] onnx_dir: {onnx_dir}")
    print(f"[INFO] out_dir: {out_dir}")
    print(f"[INFO] pattern: {args.pattern}")

    planned = []
    for onnx in files:
        name = os.path.splitext(os.path.basename(onnx))[0]
        out = os.path.join(out_dir, f"{name}{args.suffix}")
        planned.append((onnx, out))

    if args.dry_run:
        for i, (onnx, out) in enumerate(planned, start=1):
            print(f"[DRY-RUN] {i:02d}: {onnx} -> {out}")
        return 0

    success = 0
    failed = 0
    skipped = 0

    for i, (onnx, out) in enumerate(planned, start=1):
        print(f"\n[INFO] [{i}/{len(planned)}]")
        if args.skip_existing and os.path.exists(out):
            print(f"[SKIP] exists: {out}")
            skipped += 1
            continue

        try:
            ret = convert_one(
                onnx=onnx,
                out=out,
                target_platform=args.target_platform,
                input_name=args.input_name,
                input_size=input_size,
                manual_input=args.manual_input,
                float_dtype=args.float_dtype,
                optimization_level=args.optimization_level,
                quantize=args.quantize,
                dataset=args.dataset,
                quantized_dtype=args.quantized_dtype,
                verbose=args.verbose,
                dynamic_input_seq_len=args.dynamic_input_seq_len,
            )
            if ret == 0:
                success += 1
            else:
                failed += 1
                if not args.continue_on_error:
                    print("[ERR] stop on first failure (use --continue-on-error to proceed)")
                    break
        except Exception as e:  # pylint: disable=broad-except
            failed += 1
            print(f"[ERR] exception: {e}")
            if not args.continue_on_error:
                print("[ERR] stop on first failure (use --continue-on-error to proceed)")
                break

    print("\n[SUMMARY]")
    print(f"success: {success}")
    print(f"failed: {failed}")
    print(f"skipped: {skipped}")
    print(f"total: {len(planned)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
