#!/usr/bin/env bash
set -euo pipefail

sequence="${1:-${TUM_SEQUENCE:-}}"
if [[ -z "$sequence" ]]; then
  echo "usage: $0 /path/to/rgbd_dataset_freiburgX_sequence" >&2
  exit 2
fi

method_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$method_dir/../.." && pwd)"
sequence="$(readlink -f "$sequence")"
seq_name="$(basename "$sequence")"
timestamp="$(date +%Y%m%d_%H%M%S)"
run_root="${ORB_SLAM3_PROFILE_ROOT:-$method_dir/results/profiles}"
run_dir="$run_root/${seq_name}_${timestamp}"
output_parent="$run_dir/docker_output"
sequence_output="$output_parent/$seq_name"
mkdir -p "$run_dir" "$output_parent"

{
  echo "date=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "sequence=$sequence"
  echo "kernel=$(uname -a)"
  echo "jetson_model=$(tr '\0' '\n' < /proc/device-tree/model 2>/dev/null || true)"
  echo "nv_tegra_release=$(cat /etc/nv_tegra_release 2>/dev/null || true)"
  echo "nvpmodel="
  nvpmodel -q 2>/dev/null || true
  echo "jetson_clocks="
  jetson_clocks --show 2>/dev/null || true
  echo "orbslam3_source_commit=$(git -C "$root_dir/third_party/ORB_SLAM3" rev-parse HEAD 2>/dev/null || true)"
  echo "pangolin_source_commit=$(git -C "$root_dir/third_party/Pangolin-v0.5" rev-parse HEAD 2>/dev/null || true)"
  echo "orbslam3_source_status="
  git -C "$root_dir/third_party/ORB_SLAM3" status --short 2>/dev/null || true
} > "$run_dir/environment.txt"

tegrastats_pid=""
if command -v tegrastats >/dev/null 2>&1; then
  tegrastats --interval "${TEGRASTATS_INTERVAL_MS:-1000}" --logfile "$run_dir/tegrastats.log" >/dev/null 2>&1 &
  tegrastats_pid="$!"
fi

cleanup() {
  if [[ -n "$tegrastats_pid" ]]; then
    kill "$tegrastats_pid" >/dev/null 2>&1 || true
    wait "$tegrastats_pid" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

set +e
ORB_SLAM3_OUTPUT="$output_parent" "$method_dir/run_tum.sh" "$sequence" 2>&1 | tee "$run_dir/orbslam3.log"
run_exit="${PIPESTATUS[0]}"
set -e
echo "$run_exit" > "$run_dir/run_exit_code.txt"

cleanup
trap - EXIT INT TERM

if [[ -d "$sequence_output" ]]; then
  find "$sequence_output" -maxdepth 1 -type f -exec cp -f {} "$run_dir/" \;
fi

if [[ -f "$sequence/groundtruth.txt" && -f "$run_dir/CameraTrajectory.txt" ]]; then
  python3 "$root_dir/common/evaluate_tum_trajectory.py" \
    --groundtruth "$sequence/groundtruth.txt" \
    --estimate "$run_dir/CameraTrajectory.txt" \
    --output "$run_dir/evaluation.json" \
    > "$run_dir/evaluation.stdout" || true
fi

python3 "$root_dir/common/summarize_orbslam3_profile.py" "$run_dir" \
  --frame-period-ms "${ORB_SLAM3_FRAME_PERIOD_MS:-33.333}" \
  --warmup "${ORB_SLAM3_WARMUP_FRAMES:-5}" \
  --output-json "$run_dir/bottlenecks.json" \
  --output-md "$run_dir/bottlenecks.md" \
  > "$run_dir/bottlenecks.stdout"

echo "$run_dir"
