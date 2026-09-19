#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mode=${1:-}

if [[ "$mode" == worker ]]; then
    echo 'Low-latency mode runs all 64 decoder layers on Thor; do not start the Orin worker.' >&2
    exit 2
fi

# A single autoregressive stream cannot overlap an Orin prefix with the Thor
# suffix: both execute serially for every token. Keep the decode path local.
export DIAL_TOPOLOGY=${DIAL_TOPOLOGY:-"$repo_dir/topology_qwen38_thor.yml"}
export SPM_COMPACT_BATCH=${SPM_COMPACT_BATCH:-1}

exec "$repo_dir/run_27b_ggml.sh" "$@"
