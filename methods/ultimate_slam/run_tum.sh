#!/usr/bin/env bash
set -euo pipefail

sequence="${1:-${TUM_SEQUENCE:-}}"
if [[ -z "$sequence" ]]; then
  echo "usage: $0 /path/to/rgbd_dataset_freiburgX_sequence" >&2
  exit 2
fi

method_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
image="${ULTIMATE_SLAM_IMAGE:-thesis/ultimate_slam:ros-noetic-tum}"
output="${ULTIMATE_SLAM_OUTPUT:-$method_dir/results/run}"
sequence="$(readlink -f "$sequence")"
container_sequence="/data/$(basename "$sequence")"
mkdir -p "$output"

docker_flags=(--rm --network host)
if [[ -t 0 && -t 1 ]]; then
  docker_flags+=(-it)
fi
if [[ -n "${ULTIMATE_SLAM_MAX_FRAMES:-}" ]]; then
  docker_flags+=(-e "ULTIMATE_SLAM_MAX_FRAMES=$ULTIMATE_SLAM_MAX_FRAMES")
fi

docker run "${docker_flags[@]}" \
  -v "$sequence:$container_sequence:ro" \
  -v "$method_dir/scripts:/opt/ultimate_slam_runtime_scripts:ro" \
  -v "$output:/output" \
  "$image" /opt/ultimate_slam_runtime_scripts/run_tum_inside.sh "$container_sequence"
