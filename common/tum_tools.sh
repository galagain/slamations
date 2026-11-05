#!/usr/bin/env bash
set -euo pipefail

tum_sequence_name() {
  basename "$1"
}

tum_family() {
  local seq
  seq="$(tum_sequence_name "$1")"
  case "$seq" in
    *freiburg1*|fr1*) echo "TUM1" ;;
    *freiburg2*|fr2*) echo "TUM2" ;;
    *freiburg3*|fr3*) echo "TUM3" ;;
    *)
      echo "Could not infer TUM family from '$seq'." >&2
      echo "Use a directory name containing freiburg1, freiburg2, or freiburg3." >&2
      return 2
      ;;
  esac
}

tum_association_file() {
  local sequence_dir="$1"
  local output_dir="${2:-/output}"
  local seq
  seq="$(tum_sequence_name "$sequence_dir")"
  mkdir -p "$output_dir/associations"
  echo "$output_dir/associations/${seq}.txt"
}

ensure_tum_associations() {
  local sequence_dir="$1"
  local output_dir="${2:-/output}"
  local assoc
  assoc="$(tum_association_file "$sequence_dir" "$output_dir")"
  if [[ ! -s "$assoc" ]]; then
    python3 /opt/slam_common/tum_associate.py \
      "$sequence_dir/rgb.txt" \
      "$sequence_dir/depth.txt" > "$assoc"
  fi
  echo "$assoc"
}
