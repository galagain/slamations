#!/usr/bin/env bash
set -euo pipefail

sequence="${1:-${TUM_SEQUENCE:-}}"
[[ -n "$sequence" ]] || { echo "usage: $0 /path/to/tum_sequence" >&2; exit 2; }

method_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$method_dir/../.." && pwd)"

export ANTSLAM_OUTPUT="${ANTSLAM_OUTPUT:-${ANTSLAM_FAST_CUDA_OUTPUT:-$method_dir/results/run}}"
export ANTSLAM_CUDA_FRONTEND="${ANTSLAM_CUDA_FRONTEND:-1}"
export ANTSLAM_CPU_BOOTSTRAP_FRAMES="${ANTSLAM_CPU_BOOTSTRAP_FRAMES:-0}"
export ANTSLAM_CUDA_LOCAL_MATCHER="${ANTSLAM_CUDA_LOCAL_MATCHER:-0}"

exec "$root_dir/methods/antslam/run_tum.sh" "$sequence"
