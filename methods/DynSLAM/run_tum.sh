#!/usr/bin/env bash
set -euo pipefail

method_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base_config_dir="$method_dir/configs/tum_rgbd_dyneomt_stride16_exact"

method="dynslam_ft11000_dyneomt_fast"
fasttrack="1"
turbomap="0"
fastloop="0"
ft_mask="11000"
tm_mask="0000"
fl_mask="00000"
ad_mask="00000"
semantic_frame_stride="${DYNSLAM_SEMANTIC_FRAME_STRIDE:-16}"
semantic_max_mask_age="${DYNSLAM_SEMANTIC_MAX_MASK_AGE:-0}"

if ! [[ "$semantic_frame_stride" =~ ^[0-9]+$ ]] || [[ "$semantic_frame_stride" -lt 1 ]]; then
  echo "DYNSLAM_SEMANTIC_FRAME_STRIDE must be an integer >= 1" >&2
  exit 2
fi
if ! [[ "$semantic_max_mask_age" =~ ^[0-9]+$ ]]; then
  echo "DYNSLAM_SEMANTIC_MAX_MASK_AGE must be an integer >= 0" >&2
  exit 2
fi

if [[ -n "${DYNSLAM_CONFIG_DIR:-}" ]]; then
  config_dir="$(readlink -f "$DYNSLAM_CONFIG_DIR")"
elif [[ "$semantic_frame_stride" == "16" && "$semantic_max_mask_age" == "0" ]]; then
  config_dir="$base_config_dir"
else
  config_dir="$method_dir/cache/runtime_configs/stride${semantic_frame_stride}_age${semantic_max_mask_age}"
fi

prepare_config_dir() {
  if [[ -n "${DYNSLAM_CONFIG_DIR:-}" ]]; then
    return
  fi
  if [[ "$config_dir" == "$base_config_dir" ]]; then
    return
  fi
  mkdir -p "$config_dir"
  python3 - "$base_config_dir" "$config_dir" "$semantic_frame_stride" "$semantic_max_mask_age" <<'PY'
import re
import shutil
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
frame_stride = sys.argv[3]
max_mask_age = sys.argv[4]

for src in source.glob("TUM*.yaml"):
    dst = target / src.name
    text = src.read_text(encoding="utf-8")
    text = re.sub(r"Semantic\.frame_stride:\s*\d+", f"Semantic.frame_stride: {frame_stride}", text)
    text = re.sub(r"Semantic\.max_mask_age:\s*\d+", f"Semantic.max_mask_age: {max_mask_age}", text)
    dst.write_text(text, encoding="utf-8")

for src in source.iterdir():
    if src.is_file() and not src.name.startswith("TUM"):
        shutil.copy2(src, target / src.name)
PY
}

show_config() {
  cat <<EOF
DynSLAM fast preset
  method: $method
  FT: $fasttrack / $ft_mask
  TM: $turbomap / $tm_mask
  FL: $fastloop / $fl_mask
  AD: $ad_mask
  semantic backend: dyneomt
  semantic frame stride: $semantic_frame_stride
  semantic max mask age: $semantic_max_mask_age
  semantic configs: $config_dir
EOF
}

usage() {
  cat >&2 <<'USAGE'
usage:
  ./methods/DynSLAM/run_tum.sh /path/to/tum_rgbd_sequence [version]
  ./methods/DynSLAM/run_tum.sh --show-config
USAGE
}

if [[ "${1:-}" == "--show-config" || "${1:-}" == "--print-config" ]]; then
  show_config
  exit 0
fi

sequence="${1:-${TUM_SEQUENCE:-}}"
if [[ -z "$sequence" ]]; then
  usage
  exit 2
fi

sequence="$(readlink -f "$sequence")"
[[ -d "$sequence" ]] || { echo "TUM RGB-D sequence not found: $sequence" >&2; exit 1; }
[[ -f "$sequence/rgb.txt" && -f "$sequence/depth.txt" ]] || {
  echo "Missing rgb.txt/depth.txt in $sequence" >&2
  exit 1
}
prepare_config_dir
[[ -d "$config_dir" ]] || { echo "DynSLAM config directory not found: $config_dir" >&2; exit 1; }
[[ -f "$method_dir/scripts/run_tum_inside.sh" ]] || {
  echo "Missing container entrypoint: $method_dir/scripts/run_tum_inside.sh" >&2
  exit 1
}

version="${2:-dynslam_ft11000_$(date +%Y%m%d_%H%M%S)}"
image="${DYNSLAM_IMAGE:-fastsem/dynslam:ft11000-dyneomt}"
output="$(readlink -m "${DYNSLAM_OUTPUT:-$method_dir/results}")"
seq_name="$(basename "$sequence")"
container_sequence="/data/tum_rgbd/$seq_name"
associate_script="${DYNSLAM_ASSOCIATE_SCRIPT:-$method_dir/../../common/tum_associate.py}"
associate_script="$(readlink -f "$associate_script")"
assoc_cache="$(readlink -m "${DYNSLAM_ASSOC_CACHE:-$method_dir/cache/associations}")"
full_assoc="$assoc_cache/${seq_name}.full.txt"
run_assoc="$output/associations/${seq_name}.txt"
max_frames="${DYNSLAM_MAX_FRAMES:-0}"
mask_cache="$(readlink -m "${DYNSLAM_MASK_CACHE:-$method_dir/cache/dyneomt_masks/$seq_name/$version}")"
mask_script="$method_dir/scripts/dyneomt_mask_from_cache.sh"
dyneomt_repo="$(readlink -f "${DYNSLAM_DYNEOMT_REPO:-$method_dir/external/DynEOMT}")"
dyneomt_image="${DYNSLAM_DYNEOMT_IMAGE:-dyneomt-benchmark:jp6}"
dyneomt_checkpoint_root="$(readlink -f "${DYNSLAM_DYNEOMT_CHECKPOINT_ROOT:-/home/nvidia/dyneomt_jetson/checkpoints}")"
dyneomt_checkpoint_rel="${DYNSLAM_DYNEOMT_CHECKPOINT_REL:-vipseg_vitl_full.pth}"
dyneomt_config_rel="${DYNSLAM_DYNEOMT_CONFIG_REL:-configs/DynEoMT/VIPSeg/dinov2/vit-large/dyneomt_online_vitl.yaml}"
dyneomt_short_side="${DYNSLAM_DYNEOMT_SHORT_SIDE:-384}"
dyneomt_max_side="${DYNSLAM_DYNEOMT_MAX_SIDE:-640}"
dyneomt_threshold="${DYNSLAM_DYNEOMT_DYNAMIC_THRESHOLD:-0.5}"
dyneomt_frame_stride="${DYNSLAM_DYNEOMT_FRAME_STRIDE:-$semantic_frame_stride}"
mkdir -p "$output" "$(dirname "$run_assoc")" "$assoc_cache"

[[ -f "$associate_script" ]] || { echo "Association script not found: $associate_script" >&2; exit 1; }
if [[ ! -s "$full_assoc" ]]; then
  python3 "$associate_script" "$sequence/rgb.txt" "$sequence/depth.txt" > "$full_assoc"
fi
if [[ "$max_frames" =~ ^[0-9]+$ && "$max_frames" -gt 0 ]]; then
  head -n "$max_frames" "$full_assoc" > "$run_assoc"
else
  cp "$full_assoc" "$run_assoc"
fi

if [[ "${DYNSLAM_SKIP_DYNEOMT_PRECOMPUTE:-0}" != "1" ]]; then
  [[ -d "$dyneomt_repo" ]] || { echo "DynEOMT repo not found: $dyneomt_repo" >&2; exit 1; }
  [[ -f "$dyneomt_repo/$dyneomt_config_rel" ]] || { echo "DynEOMT config not found: $dyneomt_repo/$dyneomt_config_rel" >&2; exit 1; }
  [[ -f "$dyneomt_checkpoint_root/$dyneomt_checkpoint_rel" ]] || { echo "DynEOMT checkpoint not found: $dyneomt_checkpoint_root/$dyneomt_checkpoint_rel" >&2; exit 1; }
  mkdir -p "$mask_cache"
  echo "DynEOMT mask precompute"
  echo "  image: $dyneomt_image"
  echo "  cache: $mask_cache"
  docker run --rm --runtime nvidia --network host --ipc host \
    -e NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}" \
    -v "$dyneomt_repo:/opt/dyneomt:ro" \
    -v "$sequence:$container_sequence:ro" \
    -v "$run_assoc:/association.txt:ro" \
    -v "$mask_cache:/masks" \
    -v "$method_dir/scripts/precompute_dyneomt_masks.py:/usr/local/bin/precompute_dyneomt_masks.py:ro" \
    -v "$dyneomt_checkpoint_root:/checkpoints:ro" \
    "$dyneomt_image" bash -lc \
      "python3 /usr/local/bin/precompute_dyneomt_masks.py \
        --repo /opt/dyneomt \
        --config /opt/dyneomt/$dyneomt_config_rel \
        --checkpoint /checkpoints/$dyneomt_checkpoint_rel \
        --sequence $container_sequence \
        --association /association.txt \
        --output-dir /masks \
        --short-side $dyneomt_short_side \
        --max-side $dyneomt_max_side \
        --frame-stride $dyneomt_frame_stride \
        --dynamic-threshold $dyneomt_threshold"
else
  [[ -f "$mask_cache/index.json" ]] || { echo "DynEOMT mask cache missing index.json: $mask_cache" >&2; exit 1; }
fi

docker_flags=(--rm --runtime nvidia --network host --ipc host)
if [[ -t 0 && -t 1 ]]; then
  docker_flags+=(-it)
fi
if [[ -n "${DYNSLAM_SEMANTIC_TRACE:-}" ]]; then
  docker_flags+=(-e "DYNSLAM_SEMANTIC_TRACE=$DYNSLAM_SEMANTIC_TRACE")
fi

show_config
exec docker run "${docker_flags[@]}" \
  -e NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}" \
  -e DYNSLAM_TUM_SEQUENCE="$container_sequence" \
  -e DYNSLAM_METHOD_NAME="$method" \
  -e DYNSLAM_VERSION="$version" \
  -e DYNSLAM_FASTTRACK="$fasttrack" \
  -e DYNSLAM_TURBOMAP="$turbomap" \
  -e DYNSLAM_FASTLOOP="$fastloop" \
  -e DYNSLAM_KERNEL_FT="$ft_mask" \
  -e DYNSLAM_KERNEL_TM="$tm_mask" \
  -e DYNSLAM_KERNEL_FL="$fl_mask" \
  -e DYNSLAM_KERNEL_AD="$ad_mask" \
  -e SLAM_SETTINGS_DIR="/opt/dynslam_configs" \
  -e SLAM_RGBD_BINARY="${SLAM_RGBD_BINARY:-}" \
  -e SLAM_VOCABULARY="${SLAM_VOCABULARY:-}" \
  -e DYNSLAM_DYNEOMT_COMMAND="${DYNSLAM_DYNEOMT_COMMAND:-/usr/local/bin/dyneomt_predict_mask.sh}" \
  -e DYNSLAM_DYNEOMT_MASK_CACHE="/dyneomt_masks" \
  -e NITRO_REALTIME_SLEEP="${NITRO_REALTIME_SLEEP:-0}" \
  -v "$sequence:$container_sequence:ro" \
  -v "$config_dir:/opt/dynslam_configs:ro" \
  -v "$method_dir/scripts/run_tum_inside.sh:/usr/local/bin/dynslam_run_tum_inside.sh:ro" \
  -v "$mask_script:/usr/local/bin/dyneomt_predict_mask.sh:ro" \
  -v "$mask_cache:/dyneomt_masks:ro" \
  -v "$output:/output" \
  "$image" /usr/local/bin/dynslam_run_tum_inside.sh
