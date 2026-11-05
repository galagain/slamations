#!/usr/bin/env bash
set -euo pipefail

method_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$method_dir/../.." && pwd)"
source_dir="${RDS_SLAM_SOURCE:-$root_dir/third_party/semantic/rds_slam_context}"
image="${RDS_SLAM_IMAGE:-thesis/rds_slam:tum-rgbd}"

if [[ ! -d "$source_dir/RDS-SLAM" ]]; then
  source_dir="${root_dir}/third_party/semantic/rds_slam_context"
  mkdir -p "$source_dir"
  if [[ ! -d "$source_dir/RDS-SLAM" ]]; then
    git clone --depth 1 https://github.com/yubaoliu/RDS-SLAM.git "$source_dir/RDS-SLAM"
  fi
  if [[ -f /home/nvidia/Documents/dockers/rds_slam/segnet_pascal.caffemodel ]]; then
    cp /home/nvidia/Documents/dockers/rds_slam/segnet_pascal.caffemodel "$source_dir/"
  fi
fi

"$root_dir/common/stage_context.sh" "$method_dir" "$source_dir"
docker build \
  --build-arg BASE_IMAGE="${RDS_BASE_IMAGE:-nvcr.io/nvidia/l4t-base:r35.1.0}" \
  --build-arg INSTALL_TF_JETSON="${RDS_INSTALL_TF_JETSON:-1}" \
  -t "$image" "$method_dir/.docker_context"
echo "$image"
