#!/usr/bin/env bash
set -euo pipefail

method_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$method_dir/../.." && pwd)"
source_dir="${ANTSSLAM_SOURCE:-$root_dir/third_party/AntSSLAM}"
pangolin_dir="${PANGOLIN_SOURCE:-$root_dir/third_party/Pangolin-v0.5}"
image="${ANTSSLAM_IMAGE:-thesis/antsslam:jetson-r36.3-tum}"

if [[ ! -f "$source_dir/CMakeLists.txt" ]]; then
  echo "Expected an ORB-SLAM3-derived AntSSLAM source tree at $source_dir." >&2
  exit 1
fi

"$root_dir/common/stage_context.sh" "$method_dir" "$source_dir" "$pangolin_dir"
docker build \
  --build-arg BUILD_JOBS="${BUILD_JOBS:-$(nproc)}" \
  --build-arg ANTSSLAM_TENSORRT="${ANTSSLAM_TENSORRT:-OFF}" \
  -t "$image" "$method_dir/.docker_context"
echo "$image"
