#!/usr/bin/env bash
set -euo pipefail

source /opt/slam_common/tum_tools.sh

sequence="${1:-/data/sequence}"
assoc="$(ensure_tum_associations "$sequence" /output)"
prepared="/output/prepared/$(basename "$sequence")"
out_dir="/output/results/$(basename "$sequence")"
mkdir -p "$out_dir"

python3 /opt/slam_common/prepare_vdo_tum.py "$sequence" "$assoc" "$prepared"

cd "$VDO_SLAM_ROOT"
./example/vdo_slam example/tum_rgbd.yaml "$prepared"
cp -f /app/* "$out_dir/" 2>/dev/null || true
cp -f *.txt "$out_dir/" 2>/dev/null || true
echo "VDO-SLAM prepared data: $prepared"
echo "VDO-SLAM results: $out_dir"
