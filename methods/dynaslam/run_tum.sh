#!/usr/bin/env bash
set -euo pipefail

sequence="${1:-${TUM_SEQUENCE:-}}"
if [[ -z "$sequence" ]]; then
  echo "usage: $0 /path/to/rgbd_dataset_freiburgX_name" >&2
  exit 2
fi

method_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output="${DYNASLAM_OUTPUT:-$method_dir/results/run}"
sequence="$(readlink -f "$sequence")"
container_sequence="/data/$(basename "$sequence")"
mkdir -p "$output"

viewer_requested=0
case "${DYNASLAM_VIEWER:-0}" in
  1|true|TRUE|yes|YES|on|ON) viewer_requested=1 ;;
esac

if [[ -n "${DYNASLAM_IMAGE:-}" ]]; then
  image="$DYNASLAM_IMAGE"
elif [[ "$viewer_requested" -eq 1 ]]; then
  image="${DYNASLAM_VIEWER_IMAGE:-campagne/dynaslam:tum-rgbd-viewer}"
else
  image="thesis/dynaslam:tum-rgbd"
fi

mask_mounts=()
if [[ -n "${DYNASLAM_MASKS:-}" ]]; then
  dynaslam_masks="$(readlink -f "$DYNASLAM_MASKS")"
  if [[ ! -d "$dynaslam_masks" ]]; then
    echo "DYNASLAM_MASKS does not exist or is not a directory: $DYNASLAM_MASKS" >&2
    exit 2
  fi
  mask_mounts+=(-v "$dynaslam_masks:/masks:ro" -e DYNASLAM_MASKS=/masks)
fi

cidfile_dir="$(mktemp -d "${TMPDIR:-/tmp}/campagne_dynaslam.XXXXXX")"
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

docker_flags=(--rm --cidfile "$cidfile" --network host)
if [[ -t 0 && -t 1 ]]; then
  docker_flags+=(-it)
fi
if [[ "$viewer_requested" -eq 1 ]]; then
  docker_flags+=(--gpus all --privileged)
fi

docker run "${docker_flags[@]}" \
  -e DISPLAY="${DISPLAY:-}" \
  -e NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}" \
  -e TUM_MAX_FRAMES="${TUM_MAX_FRAMES:-}" \
  -e DYNASLAM_VIEWER="${DYNASLAM_VIEWER:-0}" \
  -e LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-0}" \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$sequence:$container_sequence:ro" \
  "${mask_mounts[@]}" \
  -v "$output:/output" \
  "$image" /usr/local/bin/run_tum_inside.sh "$container_sequence"
