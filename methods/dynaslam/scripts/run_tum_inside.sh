#!/usr/bin/env bash
set -euo pipefail

source /opt/slam_common/tum_tools.sh

sequence="${1:-/data/sequence}"
family="$(tum_family "$sequence")"
assoc="$(ensure_tum_associations "$sequence" /output)"
out_dir="/output/$(basename "$sequence")"
mkdir -p "$out_dir"

if [[ -n "${TUM_MAX_FRAMES:-}" ]]; then
  subset_assoc="$out_dir/associations_first_${TUM_MAX_FRAMES}.txt"
  head -n "$TUM_MAX_FRAMES" "$assoc" > "$subset_assoc"
  assoc="$subset_assoc"
fi

cd "$DYNASLAM_ROOT"
cmd=(./Examples/RGB-D/rgbd_tum \
  Vocabulary/ORBvoc.txt \
  "Examples/RGB-D/${family}.yaml" \
  "$sequence" \
  "$assoc")

if [[ -n "${DYNASLAM_MASKS:-}" ]]; then
  cmd+=("$DYNASLAM_MASKS")
fi

"${cmd[@]}"

cp -f CameraTrajectory.txt KeyFrameTrajectory.txt "$out_dir/" 2>/dev/null || true
echo "DynaSLAM results: $out_dir"
