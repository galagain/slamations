#!/usr/bin/env bash
set -uo pipefail

source /opt/slam_common/tum_tools.sh

sequence="${1:-/data/sequence}"
family="$(tum_family "$sequence")"
assoc="$(ensure_tum_associations "$sequence" /output)"
out_dir="/output/$(basename "$sequence")"
mkdir -p "$out_dir"

root="${SLAM_ROOT:-/opt/slam}"
cd "$root"

binary="${SLAM_RGBD_BINARY:-Examples/RGB-D/rgbd_tum}"
if [[ ! -x "$binary" && -x build/Examples/RGB-D/rgbd_tum ]]; then
  binary="build/Examples/RGB-D/rgbd_tum"
fi
if [[ ! -x "$binary" ]]; then
  found="$(find "$root" -type f -name rgbd_tum -perm -111 2>/dev/null | head -1)"
  if [[ -n "$found" ]]; then
    binary="$found"
  fi
fi
if [[ ! -x "$binary" ]]; then
  echo "No RGB-D TUM executable found under $root." >&2
  exit 1
fi

settings_dir="${SLAM_SETTINGS_DIR:-Examples/RGB-D}"
settings="${settings_dir}/${family}.yaml"
if [[ ! -f "$settings" && -f "Examples_old/RGB-D/${family}.yaml" ]]; then
  settings="Examples_old/RGB-D/${family}.yaml"
fi
if [[ ! -f "$settings" ]]; then
  echo "No settings file found for $family under $root." >&2
  exit 1
fi

vocab="${SLAM_VOCABULARY:-Vocabulary/ORBvoc.txt}"
if [[ ! -f "$vocab" ]]; then
  echo "No vocabulary found: $root/$vocab" >&2
  exit 1
fi

run_exit=0
if [[ "${SLAM_USE_XVFB:-1}" == "1" && -x /usr/bin/xvfb-run ]]; then
  xvfb-run -a --server-args="-screen 0 640x480x24" \
    "$binary" "$vocab" "$settings" "$sequence" "$assoc" || run_exit=$?
else
  "$binary" "$vocab" "$settings" "$sequence" "$assoc" || run_exit=$?
fi

cp -f CameraTrajectory.txt KeyFrameTrajectory.txt "$out_dir/" 2>/dev/null || true
find . -maxdepth 1 -type f \( \
    -name '*Trajectory*.txt' -o \
    -name '*Time*.txt' -o \
    -name '*time*.txt' -o \
    -name '*Stats*.txt' -o \
    -name 'Exec*.txt' \
  \) -exec cp -f {} "$out_dir/" \; 2>/dev/null || true

echo "${SLAM_METHOD_NAME:-SLAM} results: $out_dir"
exit "$run_exit"
