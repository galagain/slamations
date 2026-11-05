#!/usr/bin/env bash
set -euo pipefail
method_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$method_dir/../.." && pwd)"
source_dir="${FASTTRACK_SOURCE:-$root_dir/third_party/FastTrack}"
pangolin_dir="${PANGOLIN_SOURCE:-$root_dir/third_party/Pangolin-v0.5}"
image="${FASTTRACK_IMAGE:-thesis/fasttrack:jetson-r36.3-tum}"
"$root_dir/common/stage_context.sh" "$method_dir" "$source_dir" "$pangolin_dir"
docker build --build-arg BUILD_JOBS="${BUILD_JOBS:-$(nproc)}" -t "$image" "$method_dir/.docker_context"
echo "$image"
