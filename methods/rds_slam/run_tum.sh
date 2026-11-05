#!/usr/bin/env bash
set -euo pipefail

variant="${1:-segnet}"
sequence="${2:-${TUM_SEQUENCE:-}}"
if [[ -z "$sequence" || ! "$variant" =~ ^(segnet|maskrcnn)$ ]]; then
  echo "usage: $0 segnet|maskrcnn /path/to/rgbd_dataset_freiburgX_name" >&2
  exit 2
fi

method_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$method_dir/../.." && pwd)"
image="${RDS_SLAM_IMAGE:-thesis/rds_slam:tum-rgbd}"
output="${RDS_SLAM_OUTPUT:-$method_dir/results/run}"
sequence="$(readlink -f "$sequence")"
container_sequence="/data/$(basename "$sequence")"
mkdir -p "$output"

cidfile_dir="$(mktemp -d "${TMPDIR:-/tmp}/campagne_rds_slam.XXXXXX")"
cidfile="$cidfile_dir/container.cid"
cleanup_container() {
  status=$?
  if [[ -s "$cidfile" ]]; then
    docker rm -f "$(cat "$cidfile")" >/dev/null 2>&1 || true
  fi
  rm -rf "$cidfile_dir"
  exit "$status"
}
trap cleanup_container EXIT INT TERM

docker_flags=(--rm --cidfile "$cidfile" --runtime nvidia --privileged --network host)
if [[ -t 0 && -t 1 ]]; then
  docker_flags+=(-it)
fi

cuda11_mounts=()
if [[ -d /home/nvidia/miniconda3/pkgs ]]; then
  cuda11_mounts+=(-v /home/nvidia/miniconda3/pkgs:/host_pkgs:ro)
fi

model_mounts=()
for model_dir in \
  "$root_dir/semantic_models/segnet" \
  "$root_dir/../semantic_models/segnet" \
  "/home/nvidia/Desktop/kalvici/semantic_models/segnet" \
  "/home/nvidia/Desktop/kalvici/segmentation/models"; do
  if [[ -d "$model_dir" ]]; then
    model_mounts+=(-v "$model_dir:/opt/rds_models/segnet:ro")
    break
  fi
done

docker run "${docker_flags[@]}" \
  -e DISPLAY="${DISPLAY:-}" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$sequence:$container_sequence:ro" \
  -v "$root_dir/common:/opt/slam_common:ro" \
  -v "$method_dir/scripts/run_tum_inside.sh:/usr/local/bin/run_tum_inside.sh:ro" \
  "${cuda11_mounts[@]}" \
  "${model_mounts[@]}" \
  -v "$output:/output" \
  "$image" /usr/local/bin/run_tum_inside.sh "$variant" "$container_sequence"
