#!/usr/bin/env bash
set -euo pipefail
sequence="${1:-${TUM_SEQUENCE:-}}"
[[ -n "$sequence" ]] || { echo "usage: $0 /path/to/tum_sequence" >&2; exit 2; }
method_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$method_dir/../.." && pwd)"
"$root_dir/common/profile_tum_method.sh" turbomap "$method_dir" TURBOMAP_OUTPUT "$sequence" "$method_dir/run_tum.sh"
