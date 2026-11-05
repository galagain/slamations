#!/usr/bin/env bash
set -euo pipefail

sequence="${1:-${TUM_SEQUENCE:-}}"
if [[ -z "$sequence" ]]; then
  echo "usage: $0 /path/to/rgbd_dataset_freiburg3_name" >&2
  exit 2
fi

method_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
image="${DYNAMIC_VINS_IMAGE:-thesis/dynamic_vins:tum-rgbd}"
output="${DYNAMIC_VINS_OUTPUT:-$method_dir/results/run}"
sequence="$(readlink -f "$sequence")"
container_sequence="/data/$(basename "$sequence")"
mkdir -p "$output"

cidfile_dir="$(mktemp -d "${TMPDIR:-/tmp}/campagne_dynamic_vins.XXXXXX")"
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

docker run "${docker_flags[@]}" \
  -e DISPLAY="${DISPLAY:-}" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$sequence:$container_sequence:ro" \
  -v "$output:/output" \
  "$image" /usr/local/bin/run_tum_inside.sh "$container_sequence"
