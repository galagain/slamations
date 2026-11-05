#!/usr/bin/env bash
set -euo pipefail

sequence="${1:-${TUM_SEQUENCE:-}}"
[[ -n "$sequence" ]] || { echo "usage: $0 /path/to/tum_rgbd_sequence [version]" >&2; exit 2; }

method_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
version="${2:-dynslam_profile_$(date +%Y%m%d_%H%M%S)}"

DYNSLAM_OUTPUT="${DYNSLAM_OUTPUT:-$method_dir/results}" \
  "$method_dir/run_tum.sh" "$sequence" "$version"
