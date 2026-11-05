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
run_root="${ULTIMATE_SLAM_PROFILE_ROOT:-$method_dir/results/profiles}"
run_dir="$run_root/${seq_name}_${timestamp}"
output_parent="$run_dir/docker_output"
sequence_output="$output_parent/$seq_name"
mkdir -p "$run_dir" "$output_parent"

{
  echo "date=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "sequence=$sequence"
  echo "kernel=$(uname -a)"
  echo "ultimate_slam_source_commit=$(git -C "$root_dir/third_party/rpg_ultimate_slam_open" rev-parse HEAD 2>/dev/null || true)"
  echo "ultimate_slam_source_status="
  git -C "$root_dir/third_party/rpg_ultimate_slam_open" status --short 2>/dev/null || true
  echo "note=TUM RGB-D has no event stream and no IMU; this wrapper uses RGB frames plus synthetic stationary IMU."
} > "$run_dir/environment.txt"

start_ns="$(date +%s%N)"
set +e
ULTIMATE_SLAM_OUTPUT="$output_parent" "$method_dir/run_tum.sh" "$sequence" 2>&1 | tee "$run_dir/ultimate_slam.log"
run_exit="${PIPESTATUS[0]}"
set -e
end_ns="$(date +%s%N)"
awk -v s="$start_ns" -v e="$end_ns" 'BEGIN { printf "%.6f\n", (e-s)/1000000000 }' > "$run_dir/wall_seconds.txt"
echo "$run_exit" > "$run_dir/run_exit_code.txt"

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

python3 - "$run_dir" <<'PY'
import csv, json, math, pathlib, sys

run_dir = pathlib.Path(sys.argv[1])
summary = {
    "method": "ultimate_slam",
    "run_exit_code": int((run_dir / "run_exit_code.txt").read_text().strip()),
    "wall_seconds": float((run_dir / "wall_seconds.txt").read_text().strip()),
    "tum_validity": "non_comparable_synthetic_imu_no_events_no_depth",
}
traj = run_dir / "CameraTrajectory.txt"
if traj.exists():
    rows = [ln for ln in traj.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    summary["trajectory_poses"] = len(rows)
if (run_dir / "evaluation.json").exists():
    summary["evaluation"] = json.loads((run_dir / "evaluation.json").read_text())
if (run_dir / "traj_es.csv").exists():
    with (run_dir / "traj_es.csv").open() as f:
        n = sum(1 for line in f if line.strip() and not line.startswith("timestamp"))
    summary["raw_traj_es_rows"] = n
(run_dir / "profile_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

lines = [
    "# UltimateSLAM TUM profile",
    "",
    f"- run_exit_code: {summary['run_exit_code']}",
    f"- wall_seconds: {summary['wall_seconds']:.3f}",
    f"- validity: {summary['tum_validity']}",
]
if "trajectory_poses" in summary:
    lines.append(f"- trajectory_poses: {summary['trajectory_poses']}")
if "evaluation" in summary:
    ate = summary["evaluation"]["ate_m"]["rmse"]
    lines.append(f"- ate_rmse_m: {ate:.6f}")
(run_dir / "profile_summary.md").write_text("\n".join(lines) + "\n")
PY

echo "$run_dir"
