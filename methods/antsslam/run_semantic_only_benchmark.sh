#!/usr/bin/env bash
set -euo pipefail

sequence="${1:-${TUM_SEQUENCE:-}}"
backend="${2:-all}"
[[ -n "$sequence" ]] || { echo "usage: $0 /path/to/tum_sequence [tokenmask|segnet|maskrcnn|all]" >&2; exit 2; }

method_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$method_dir/../.." && pwd)"
image="${ANTSSLAM_IMAGE:-thesis/antsslam:jetson-r36.3-tum}"
output="${ANTSSLAM_SEMANTIC_ONLY_OUTPUT:-$method_dir/results/semantic_only_$(date +%Y%m%d_%H%M%S)}"
sequence="$(readlink -f "$sequence")"
sequence_name="$(basename "$sequence")"

mkdir -p "$output/associations"
assoc="$output/associations/${sequence_name}.txt"
python3 "$root_dir/common/tum_associate.py" "$sequence/rgb.txt" "$sequence/depth.txt" > "$assoc"

settings_for_backend() {
  case "$1" in
    tokenmask) echo "/opt/method_configs/tum_rgbd_semantic_tokenmask_stride16_exact/$(case "$sequence_name" in *freiburg1*) echo TUM1 ;; *freiburg2*) echo TUM2 ;; *) echo TUM3 ;; esac).yaml" ;;
    segnet) echo "/opt/method_configs/tum_rgbd_semantic_segnet_latest/$(case "$sequence_name" in *freiburg1*) echo TUM1 ;; *freiburg2*) echo TUM2 ;; *) echo TUM3 ;; esac).yaml" ;;
    maskrcnn) echo "/opt/method_configs/tum_rgbd_semantic_maskrcnn_latest/$(case "$sequence_name" in *freiburg1*) echo TUM1 ;; *freiburg2*) echo TUM2 ;; *) echo TUM3 ;; esac).yaml" ;;
    *) echo "unknown backend '$1'" >&2; return 2 ;;
  esac
}

semantic_segmentation_dir="${ANTSSLAM_SEGMENTATION_DIR:-${ANTSLAM_SEGMENTATION_DIR:-/home/nvidia/Desktop/kalvici/segmentation}}"
semantic_trt_cache_dir="${ANTSSLAM_TRT_CACHE_DIR:-${ANTSLAM_TRT_CACHE_DIR:-/home/nvidia/Desktop/kalvici/trt_cache}}"
semantic_tokenmask_models_dir="${ANTSSLAM_TOKENMASK_MODELS_DIR:-${ANTSLAM_TOKENMASK_MODELS_DIR:-/home/nvidia/Desktop/kalvici/neww/Jetson-SLAM/models}}"

run_one() {
  local name="$1"
  local settings
  settings="$(settings_for_backend "$name")"
  echo "===== semantic only $name ====="
  mkdir -p "$output/$name"
  set +e
  docker run --rm --runtime nvidia --network host \
    -e NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}" \
    -v "$sequence:/data/$sequence_name:ro" \
    -v "$output:/output" \
    -v "$method_dir/configs:/opt/method_configs:ro" \
    -v "$semantic_segmentation_dir:/opt/semantic/segmentation:ro" \
    -v "$semantic_trt_cache_dir:/opt/semantic/trt_cache:ro" \
    -v "$semantic_tokenmask_models_dir:/opt/semantic/tokenmask_models:ro" \
    "$image" -lc \
      "mkdir -p /output/$name && /opt/antsslam/Examples/RGB-D/semantic_tum_benchmark $settings /data/$sequence_name /output/associations/${sequence_name}.txt /output/$name"
  local status=$?
  set -e
  if [[ "$status" -ne 0 && ! -s "$output/$name/semantic_only_summary.txt" ]]; then
    return "$status"
  fi
  cat "$output/$name/semantic_only_summary.txt"
  if [[ "$status" -ne 0 ]]; then
    echo "warning: backend '$name' returned status $status after writing the summary" >&2
  fi
}

if [[ "$backend" == "all" ]]; then
  run_one tokenmask
  run_one segnet
  run_one maskrcnn
else
  run_one "$backend"
fi

echo "Results: $output"
