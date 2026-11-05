#!/usr/bin/env bash
set -euo pipefail

source /opt/slam_common/tum_tools.sh

sequence="${1:-/data/sequence}"
family="$(tum_family "$sequence")"
assoc="$(ensure_tum_associations "$sequence" /output)"
out_dir="/output/$(basename "$sequence")"
mkdir -p "$out_dir"

cd "$ORB_SLAM3_ROOT"

binary="Examples/RGB-D/rgbd_tum"
settings="Examples/RGB-D/${family}.yaml"

if [[ ! -x "$binary" ]]; then
  echo "ORB-SLAM3 RGB-D executable not found: $binary" >&2
  exit 1
fi

run_exit=0
if [[ "${ORB_SLAM3_USE_XVFB:-1}" == "1" && -x /usr/bin/xvfb-run ]]; then
  xvfb-run -a --server-args="-screen 0 640x480x24" \
    "$binary" Vocabulary/ORBvoc.txt "$settings" "$sequence" "$assoc" || run_exit=$?
else
  "$binary" Vocabulary/ORBvoc.txt "$settings" "$sequence" "$assoc" || run_exit=$?
fi

for file in \
  CameraTrajectory.txt \
  KeyFrameTrajectory.txt \
  TrackingTimeStats.txt \
  LocalMapTimeStats.txt \
  LBA_Stats.txt \
  SessionInfo.txt \
  ExecMean.txt; do
  cp -f "$file" "$out_dir/" 2>/dev/null || true
done

echo "ORB-SLAM3 results: $out_dir"
exit "$run_exit"
