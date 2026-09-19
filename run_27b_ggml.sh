#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mode=${1:-}
if [[ "$mode" != master && "$mode" != worker && "$mode" != client ]]; then
    echo 'Usage: bash run_27b_ggml.sh master|worker|client [extra dial-cli options]' >&2
    exit 2
fi
shift
# Board-local defaults. Edit these when moving a model; an existing environment
# value still takes precedence, so temporary overrides keep working.
case "$mode" in
    worker)
        HF_MODEL_DIR=${HF_MODEL_DIR:-/media/nvidia/Elements/Qwen3.8-27B}
        GGUF_MODEL=${GGUF_MODEL:-/media/nvidia/Elements/Qwen3.8-27B-Q4_K_M.gguf}
        DIAL_WORKER_BIND=${DIAL_WORKER_BIND:-192.168.2.88:10128}
        ;;
    master)
        HF_MODEL_DIR=${HF_MODEL_DIR:-/home/nvidia/models/Qwen3.8-27B-NVFP4}
        GGUF_MODEL=${GGUF_MODEL:-/home/nvidia/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf}
        DIAL_API_BIND=${DIAL_API_BIND:-0.0.0.0:8082}
        ;;
esac
bin=${DIAL_BIN:-"$repo_dir/target/release/dial-cli"}
[[ -x "$bin" ]] || { echo "DIAL executable missing: $bin; rebuild cargo build --release --features cuda" >&2; exit 1; }
cd "$repo_dir"
if [[ "$mode" == client ]]; then
    exec "$bin" --model-size 27b --api-client "${DIAL_API_URL:-http://127.0.0.1:8082}" \
        --ask "${DIAL_ASK:-只回答一个数字：1加1等于多少？}" "$@"
fi

# Stable performance defaults. Every value except GGML_CUDA_DISABLE_GRAPHS can
# still be overridden on the command line for diagnostics.
export SPM_COMPACT_BATCH=${SPM_COMPACT_BATCH:-1}
export SPM_TRACE_TRANSFER=${SPM_TRACE_TRANSFER:-0}
export DIAL_GGML_FUSED=${DIAL_GGML_FUSED:-1}
export RUST_LOG=${RUST_LOG:-info}

# Upstream disables CUDA graphs when this variable merely exists, even when it
# is set to 0. Use DIAL_GGML_CUDA_GRAPHS=0 for an intentional diagnostic run.
if [[ "${DIAL_GGML_CUDA_GRAPHS:-1}" == 0 ]]; then
    export GGML_CUDA_DISABLE_GRAPHS=1
else
    unset GGML_CUDA_DISABLE_GRAPHS
fi

if [[ "$mode" == worker ]]; then
    # The previous 4-core affinity reduced Orin request scheduling throughput.
    export DIAL_DISABLE_CPU_AFFINITY=${DIAL_DISABLE_CPU_AFFINITY:-1}

    # Jetson clocks are not guaranteed to remain locked after reboot. Configure
    # them before loading weights. Set DIAL_MAX_PERF=0 to skip sudo on restarts.
    if [[ "${DIAL_MAX_PERF:-1}" != 0 ]]; then
        command -v nvpmodel >/dev/null || {
            echo 'nvpmodel not found; set DIAL_MAX_PERF=0 to skip Jetson tuning' >&2
            exit 1
        }
        command -v jetson_clocks >/dev/null || {
            echo 'jetson_clocks not found; set DIAL_MAX_PERF=0 to skip Jetson tuning' >&2
            exit 1
        }
        echo "Configuring Jetson maximum-performance mode ${DIAL_NVP_MODE:-0}..."
        sudo nvpmodel -m "${DIAL_NVP_MODE:-0}"
        sudo jetson_clocks
    fi
fi

: "${HF_MODEL_DIR:?Set HF_MODEL_DIR to the matching HF config/tokenizer directory on THIS node}"
: "${GGUF_MODEL:?Set GGUF_MODEL to the matching Qwen3.8 Q4_K_M GGUF on THIS node}"
ggml_lib=${GGML_LIB:-"$repo_dir/build/qwen38-ggml/lib/libdial_qwen38_ggml.so"}
topology=${DIAL_TOPOLOGY:-"$repo_dir/topology_qwen38.yml"}
for file in "$HF_MODEL_DIR/config.json" "$GGUF_MODEL" "$ggml_lib" "$topology"; do
    [[ -f "$file" ]] || { echo "Required file missing: $file" >&2; exit 1; }
done
# Refuse an old binary instead of silently launching its safetensors path.
help_output=$("$bin" --help)
[[ "$help_output" == *qwen38-ggml* && "$help_output" == *--qwen38-ggml-lib* ]] || {
    echo 'Old dial-cli binary: copy the updated source to BOTH nodes and rebuild.' >&2; exit 1;
}
common=(--model-size 27b --inference-backend qwen38-ggml --mode "$mode"
    --model "$HF_MODEL_DIR" --qwen38-gguf "$GGUF_MODEL" --qwen38-ggml-lib "$ggml_lib"
    --topology "$topology" --device "${DIAL_DEVICE:-0}" --dtype f16
    --kv-cache-max-len "${DIAL_CONTEXT:-4096}")
if [[ "$mode" == worker ]]; then
    exec "$bin" "${common[@]}" --name "${DIAL_WORKER_NAME:-worker0}" \
        --address "${DIAL_WORKER_BIND:-192.168.2.88:10128}" "$@"
else
    exec "$bin" "${common[@]}" --api "${DIAL_API_BIND:-0.0.0.0:8082}" \
        --qwen38-thinking false --temperature 0 --sample-len "${DIAL_SAMPLE_LEN:-128}" "$@"
fi
