#!/usr/bin/env bash
set -euo pipefail

method_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$method_dir/../.." && pwd)"
source_dir="${PANOPTIC_SLAM_SOURCE:-$root_dir/third_party/semantic/Panoptic-SLAM}"
image="${PANOPTIC_SLAM_IMAGE:-thesis/panoptic_slam:tum-rgbd}"

if [[ ! -d "$source_dir" ]]; then
  source_dir="$root_dir/third_party/semantic/Panoptic-SLAM"
  if [[ ! -d "$source_dir" ]]; then
    git clone --depth 1 https://github.com/iit-DLSLab/Panoptic-SLAM.git "$source_dir"
  fi
fi

"$root_dir/common/stage_context.sh" "$method_dir" "$source_dir"

wheel_context="$method_dir/.docker_context/wheels"
mkdir -p "$wheel_context"
torch_wheel="${PANOPTIC_TORCH_WHEEL:-/home/nvidia/Desktop/g2tm-segmenter/third_party/jetson-wheels/torch-2.3.0a0+6ddf5cf85e.nv24.04.14026654-cp310-cp310-linux_aarch64.whl}"
torchvision_wheel="${PANOPTIC_TORCHVISION_WHEEL:-/home/nvidia/Desktop/g2tm-segmenter/third_party/jetson-wheels/torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl}"
if [[ -f "$torch_wheel" && -f "$torchvision_wheel" ]]; then
  cp -L "$torch_wheel" "$torchvision_wheel" "$wheel_context/"
fi

docker build \
  --build-arg INSTALL_DETECTRON2="${PANOPTIC_INSTALL_DETECTRON2:-1}" \
  -t "$image" "$method_dir/.docker_context"
echo "$image"
