#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 4 ]]; then
  echo "usage: $0 METHOD_NAME METHOD_DIR OUTPUT_ENV_VAR /path/to/tum_sequence [run_script]" >&2
  exit 2
fi

method_name="$1"
method_dir="$(readlink -f "$2")"
output_env="$3"
sequence="$(readlink -f "$4")"
run_script="${5:-$method_dir/run_tum.sh}"

root_dir="$(cd "$method_dir/../.." && pwd)"
seq_name="$(basename "$sequence")"
timestamp="$(date +%Y%m%d_%H%M%S)"
run_root="$(realpath -m "${THESIS_PROFILE_ROOT:-$method_dir/results/profiles}")"
run_dir="$run_root/${seq_name}_${timestamp}"
output_parent="$run_dir/docker_output"
sequence_output="$output_parent/$seq_name"
mkdir -p "$run_dir" "$output_parent"

{
  echo "date=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "method=$method_name"
  echo "sequence=$sequence"
  echo "kernel=$(uname -a)"
  echo "jetson_model=$(tr '\0' '\n' < /proc/device-tree/model 2>/dev/null || true)"
  echo "nv_tegra_release=$(cat /etc/nv_tegra_release 2>/dev/null || true)"
  echo "nvpmodel="
  nvpmodel -q 2>/dev/null || true
  echo "jetson_clocks="
  jetson_clocks --show 2>/dev/null || true
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

start_ns="$(date +%s%N)"
set +e
env "$output_env=$output_parent" "$run_script" "$sequence" 2>&1 | tee "$run_dir/${method_name}.log"
run_exit="${PIPESTATUS[0]}"
set -e
end_ns="$(date +%s%N)"
wall_seconds="$(awk -v s="$start_ns" -v e="$end_ns" 'BEGIN { printf "%.9f", (e-s)/1000000000 }')"
echo "$run_exit" > "$run_dir/run_exit_code.txt"
echo "$wall_seconds" > "$run_dir/wall_seconds.txt"

cleanup
trap - EXIT INT TERM

output_candidates=(
  "$sequence_output"
  "$output_parent/results/$seq_name"
  "$output_parent/segnet/$seq_name"
  "$output_parent/maskrcnn/$seq_name"
)
for candidate in "${output_candidates[@]}"; do
  if [[ -d "$candidate" ]]; then
    echo "$candidate" > "$run_dir/output_source_dir.txt"
    find "$candidate" -maxdepth 1 -type f -exec cp -f {} "$run_dir/" \;
    break
  fi
done

if [[ -f "$sequence/groundtruth.txt" && -f "$run_dir/CameraTrajectory.txt" ]]; then
  python3 "$root_dir/common/evaluate_tum_trajectory.py" \
    --groundtruth "$sequence/groundtruth.txt" \
    --estimate "$run_dir/CameraTrajectory.txt" \
    --output "$run_dir/evaluation.json" \
    > "$run_dir/evaluation.stdout" || true
fi

python3 "$root_dir/common/profile_tum_run.py" \
  --method "$method_name" \
  --sequence "$sequence" \
  --run-dir "$run_dir" \
  --wall-seconds "$wall_seconds" \
  --run-exit-code "$run_exit" \
  --log "$run_dir/${method_name}.log" \
  --output-json "$run_dir/profile_summary.json" \
  --output-md "$run_dir/profile_summary.md" \
  > "$run_dir/profile_summary.stdout"

echo "$run_dir"
exit "$run_exit"
