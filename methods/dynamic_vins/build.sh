#!/usr/bin/env bash
set -euo pipefail

method_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$method_dir/../.." && pwd)"
source_dir="${DYNAMIC_VINS_SOURCE:-$root_dir/third_party/semantic/dynamic_vins_context}"
yolo_dir="${YOLO_ROS_SOURCE:-$root_dir/third_party/semantic/yolo_ros}"
image="${DYNAMIC_VINS_IMAGE:-thesis/dynamic_vins:tum-rgbd}"

if [[ ! -d "$source_dir/Dynamic-VINS" ]]; then
  source_dir="$root_dir/third_party/semantic/dynamic_vins_context"
  mkdir -p "$source_dir"
  if [[ ! -d "$source_dir/Dynamic-VINS" ]]; then
    git clone --depth 1 https://github.com/HITSZ-NRSL/Dynamic-VINS.git "$source_dir/Dynamic-VINS"
  fi
fi

if [[ ! -d "$yolo_dir" ]]; then
  git clone --depth 1 https://github.com/jianhengLiu/yolo_ros.git "$yolo_dir"
fi

"$root_dir/common/stage_context.sh" "$method_dir" "$source_dir"
if command -v rsync >/dev/null 2>&1; then
  rsync -a --exclude='.git' "$yolo_dir"/ "$method_dir/.docker_context/yolo_ros"/
else
  cp -a "$yolo_dir" "$method_dir/.docker_context/yolo_ros"
fi
docker build -t "$image" "$method_dir/.docker_context"
echo "$image"
