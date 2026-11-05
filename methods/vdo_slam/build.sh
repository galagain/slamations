#!/usr/bin/env bash
set -euo pipefail

method_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$method_dir/../.." && pwd)"
source_dir="${VDO_SLAM_SOURCE:-$root_dir/third_party/semantic/VDO_SLAM}"
image="${VDO_SLAM_IMAGE:-thesis/vdo_slam:tum-rgbd}"

if [[ ! -d "$source_dir" ]]; then
  source_dir="${root_dir}/third_party/semantic/VDO_SLAM"
  if [[ ! -d "$source_dir" ]]; then
    git clone --depth 1 https://github.com/halajun/VDO_SLAM.git "$source_dir"
  fi
fi

"$root_dir/common/stage_context.sh" "$method_dir" "$source_dir"
docker build -t "$image" "$method_dir/.docker_context"
echo "$image"
