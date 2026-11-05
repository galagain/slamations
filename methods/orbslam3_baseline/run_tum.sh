#!/usr/bin/env bash
set -euo pipefail

sequence="${1:-${TUM_SEQUENCE:-}}"
if [[ -z "$sequence" ]]; then
  echo "usage: $0 /path/to/rgbd_dataset_freiburgX_sequence" >&2
  exit 2
fi

method_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
image="${ORB_SLAM3_IMAGE:-thesis/orbslam3:jetson-r36.3-profile}"
output="${ORB_SLAM3_OUTPUT:-$method_dir/results/run}"
sequence="$(readlink -f "$sequence")"
container_sequence="/data/$(basename "$sequence")"
mkdir -p "$output"

docker_flags=(--rm --runtime nvidia --network host)
if [[ -t 0 && -t 1 ]]; then
  docker_flags+=(-it)
fi

docker run "${docker_flags[@]}" \
  -e DISPLAY="${DISPLAY:-}" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$sequence:$container_sequence:ro" \
  -v "$output:/output" \
  "$image" /usr/local/bin/run_tum_inside.sh "$container_sequence"
