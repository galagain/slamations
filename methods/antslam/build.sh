#!/usr/bin/env bash
set -euo pipefail

method_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$method_dir/../.." && pwd)"
source_dir="${ANTSLAM_SOURCE:-$root_dir/third_party/AntSLAM}"
pangolin_dir="${PANGOLIN_SOURCE:-$root_dir/third_party/Pangolin-v0.5}"
image="${ANTSLAM_IMAGE:-thesis/antslam:jetson-r36.3-tum}"

if [[ ! -f "$source_dir/CMakeLists.txt" ]]; then
  echo "Expected a fresh ORB-SLAM3-derived GitHub clone at $source_dir." >&2
  exit 1
fi

"$root_dir/common/stage_context.sh" "$method_dir" "$source_dir" "$pangolin_dir"
docker build --build-arg BUILD_JOBS="${BUILD_JOBS:-$(nproc)}" -t "$image" "$method_dir/.docker_context"
echo "$image"
