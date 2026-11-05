#!/usr/bin/env bash
set -euo pipefail

method_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$method_dir/../.." && pwd)"
source_dir="${ULTIMATE_SLAM_SOURCE:-$root_dir/third_party/rpg_ultimate_slam_open}"
image="${ULTIMATE_SLAM_IMAGE:-thesis/ultimate_slam:ros-noetic-tum}"

if [[ ! -f "$source_dir/CMakeLists.txt" ]]; then
  git clone https://github.com/uzh-rpg/rpg_ultimate_slam_open.git "$source_dir"
fi

"$root_dir/common/stage_context.sh" "$method_dir" "$source_dir"
if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete "$method_dir/calibration"/ "$method_dir/.docker_context/calibration"/
else
  mkdir -p "$method_dir/.docker_context/calibration"
  cp -a "$method_dir/calibration"/. "$method_dir/.docker_context/calibration"/
fi

docker build -t "$image" "$method_dir/.docker_context"
echo "$image"
