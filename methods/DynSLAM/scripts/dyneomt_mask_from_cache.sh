#!/usr/bin/env bash
set -euo pipefail

input="${DYN_SLAM_INPUT:?DYN_SLAM_INPUT is required}"
output="${DYN_SLAM_OUTPUT:?DYN_SLAM_OUTPUT is required}"
cache="${DYNSLAM_DYNEOMT_MASK_CACHE:-/dyneomt_masks}"
index="$cache/index.json"

[[ -f "$input" ]] || { echo "DynEOMT cache input not found: $input" >&2; exit 1; }
[[ -d "$cache" ]] || { echo "DynEOMT mask cache not found: $cache" >&2; exit 1; }

src=""
if [[ -f "$index" ]]; then
  src_rel="$(
    python3 - "$input" "$index" <<'PY'
import hashlib
import json
import sys

try:
    import cv2
except Exception as exc:
    print("", end="")
    sys.exit(0)

image_path, index_path = sys.argv[1], sys.argv[2]
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
if image is None:
    print("", end="")
    sys.exit(0)

pixel_hash = hashlib.sha1(image.tobytes()).hexdigest()
with open(index_path, "r", encoding="utf-8") as handle:
    index = json.load(handle)
print(index.get("pixel_sha1_to_mask", {}).get(pixel_hash, ""), end="")
PY
  )"
  if [[ -n "$src_rel" && -f "$cache/$src_rel" ]]; then
    src="$cache/$src_rel"
  fi
fi

if [[ -z "$src" ]]; then
  base="$(basename "$input")"
  frame_id="${base#frame_}"
  frame_id="${frame_id%_rgb.*}"
  candidate="$cache/frame_${frame_id}_dynamic.png"
  if [[ -f "$candidate" ]]; then
    src="$candidate"
  fi
fi

[[ -n "$src" ]] || {
  echo "No DynEOMT mask found in $cache for $input" >&2
  exit 1
}

cp "$src" "$output"
