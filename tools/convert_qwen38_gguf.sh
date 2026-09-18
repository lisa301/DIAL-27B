#!/usr/bin/env bash
set -euo pipefail
if (( $# != 3 )); then
    echo 'Usage: bash tools/convert_qwen38_gguf.sh LLAMA_CPP_B10837 HF_BF16_MODEL_DIR OUTPUT_DIR' >&2
    exit 2
fi
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
llama_dir=$(realpath "$1")
model_dir=$(realpath "$2")
output_dir=$3
python_bin=${DIAL_CONVERT_PYTHON:-python3}
quantize_bin=${DIAL_QUANTIZE_BIN:-"$llama_dir/build/bin/llama-quantize"}
[[ -f "$llama_dir/convert_hf_to_gguf.py" && -f "$model_dir/config.json" ]] || {
    echo 'Missing llama.cpp converter or HF config.json' >&2; exit 1;
}
[[ -x "$quantize_bin" ]] || {
    echo "Build llama-quantize first: cmake --build $llama_dir/build --target llama-quantize -j2" >&2; exit 1;
}
"$python_bin" -c '
import json, pathlib, sys
c = json.loads((pathlib.Path(sys.argv[1]) / "config.json").read_text())
if c.get("quantization_config") or c.get("text_config", {}).get("quantization_config"):
    sys.exit("Use the original BF16/F16 checkpoint, not NVFP4/FP8 compressed-tensors/ModelOpt. No implicit requantization.")
if c.get("model_type") != "qwen3_5":
    sys.exit("Expected Qwen3.8 model_type=qwen3_5")
' "$model_dir"
# Reuse the CMake source-pin check without compiling. Separate from GPU builds.
cmake -S "$repo_dir/backends/qwen38_ggml" -B "$repo_dir/build/qwen38-ggml-source-check" \
    "-DLLAMA_CPP_DIR=$llama_dir" -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
mkdir -p "$output_dir"
bf16_path="$output_dir/Qwen3.8-27B-BF16.gguf"
q4_path="$output_dir/Qwen3.8-27B-Q4_K_M.gguf"
if [[ -e "$bf16_path" || -e "$q4_path" ]]; then
    echo "Output already exists in $output_dir; no files overwritten. Use another output directory or quantize an existing BF16 GGUF manually." >&2
    exit 1
fi
echo 'Requires space for both temporary BF16 GGUF (~52 GB) and Q4_K_M; use the external disk, not a nearly-full root disk.'
"$python_bin" "$llama_dir/convert_hf_to_gguf.py" "$model_dir" \
    --outfile "$bf16_path" --outtype bf16 --no-mtp
"$quantize_bin" "$bf16_path" "$q4_path" Q4_K_M
echo "GGUF: $q4_path"
echo 'Keep the HF config.json, tokenizer.json and generation_config.json for DIAL; Worker also needs the same GGUF file.'
echo 'BF16 GGUF is retained for recovery/requantization; nothing was deleted.'
