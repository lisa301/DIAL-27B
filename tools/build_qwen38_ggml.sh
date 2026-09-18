#!/usr/bin/env bash
set -euo pipefail
if (( $# != 2 )); then
    echo 'Usage: bash tools/build_qwen38_ggml.sh /path/to/llama.cpp-b10837 87|110|native|cpu' >&2
    exit 2
fi
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
llama_dir=$(realpath "$1")
arch=$2
build_dir=${DIAL_GGML_BUILD_DIR:-"$repo_dir/build/qwen38-ggml"}
jobs=${DIAL_GGML_JOBS:-2}
if [[ ! -f "$llama_dir/ggml/include/ggml.h" ]]; then
    echo "Missing llama.cpp b10837 source: $llama_dir" >&2
    exit 1
fi
if [[ ! "$jobs" =~ ^[1-9][0-9]*$ ]]; then
    echo 'DIAL_GGML_JOBS must be a positive integer' >&2
    exit 2
fi
case "$arch" in
    87|110|native)
        # Orin=87, Thor=110. Compile locally with the node's own CUDA toolkit.
        cuda_options=(-DGGML_CUDA=ON -DGGML_CUDA_FA=ON
            "-DGGML_CUDA_GRAPHS=${DIAL_GGML_CUDA_GRAPHS:-ON}"
            "-DCMAKE_CUDA_ARCHITECTURES=$arch")
        ;;
    cpu) cuda_options=(-DGGML_CUDA=OFF) ;;
    *) echo 'Architecture must be 87 (Orin), 110 (Thor), native, or cpu' >&2; exit 2 ;;
esac
cmake -S "$repo_dir/backends/qwen38_ggml" -B "$build_dir" \
    "-DLLAMA_CPP_DIR=$llama_dir" -DCMAKE_BUILD_TYPE=Release "${cuda_options[@]}"
cmake --build "$build_dir" --target dial_qwen38_ggml --parallel "$jobs"
echo "Adapter: $build_dir/lib/libdial_qwen38_ggml.so"
echo 'Next: cargo build --release --features cuda (on each GPU node)'
