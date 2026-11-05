#!/usr/bin/env bash
set -euo pipefail

method_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$method_dir/../.." && pwd)"
source_dir="${ORB_SLAM3_SOURCE:-$root_dir/third_party/ORB_SLAM3}"
pangolin_dir="${PANGOLIN_SOURCE:-$root_dir/third_party/Pangolin-v0.5}"
image="${ORB_SLAM3_IMAGE:-thesis/orbslam3:jetson-r36.3-profile}"

if [[ ! -f "$source_dir/CMakeLists.txt" ]]; then
  echo "Expected a GitHub clone at $source_dir." >&2
  echo "Run: git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git $source_dir" >&2
  exit 1
fi

if [[ ! -d "$pangolin_dir" ]]; then
  echo "Expected a local Pangolin source tree at $pangolin_dir." >&2
  exit 1
fi

"$root_dir/common/stage_context.sh" "$method_dir" "$source_dir" "$pangolin_dir"
docker build -t "$image" "$method_dir/.docker_context"
echo "$image"
