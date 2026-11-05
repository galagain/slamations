#!/usr/bin/env bash
set -euo pipefail
sequence="${1:-${TUM_SEQUENCE:-}}"
[[ -n "$sequence" ]] || { echo "usage: $0 /path/to/tum_sequence" >&2; exit 2; }
method_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$method_dir/../.." && pwd)"
"$root_dir/common/run_docker_tum.sh" "$method_dir" FASTLOOP_IMAGE "thesis/fastloop:jetson-r36.3-tum" FASTLOOP_OUTPUT /usr/local/bin/run_tum_inside.sh "$sequence"
