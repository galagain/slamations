#!/usr/bin/env bash
set -euo pipefail

source /opt/slam_common/tum_tools.sh

sequence="${1:-/data/sequence}"
family="$(tum_family "$sequence")"
assoc="$(ensure_tum_associations "$sequence" /output)"
out_dir="/output/$(basename "$sequence")"
mkdir -p "$out_dir"

cd "$PANOPTIC_SLAM_ROOT"
. .venv/bin/activate
mkdir -p Output

settings="Examples/RGB-D/${family}.yaml"
vocabulary="Vocabulary/ORBvoc.txt"
[[ -f "$vocabulary" ]] || vocabulary="Vocabulary/ORBvoc.txt.tar.gz"

./panoptic_slam "$vocabulary" "$settings" "$sequence" "$assoc"
cp -f CameraTrajectory.txt KeyFrameTrajectory.txt "$out_dir/" 2>/dev/null || true
cp -f Output/* "$out_dir/" 2>/dev/null || true
echo "Panoptic-SLAM results: $out_dir"
