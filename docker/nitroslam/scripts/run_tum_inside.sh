#!/usr/bin/env bash
set -euo pipefail

root="${SLAM_ROOT:-/opt/nitroslam}"
sequence="${NITRO_TUM_SEQUENCE:-/data/sequence}"
seq_name="$(basename "$sequence")"
method="${NITRO_METHOD_NAME:-nitroslam}"
version="${NITRO_VERSION:-manual}"
fasttrack="${NITRO_FASTTRACK:-0}"
turbomap="${NITRO_TURBOMAP:-0}"
fastloop="${NITRO_FASTLOOP:-0}"
kernel_ft="${NITRO_KERNEL_FT:-11110}"
kernel_tm="${NITRO_KERNEL_TM:-1111}"
kernel_fl="${NITRO_KERNEL_FL:-11111}"
kernel_ad="${NITRO_KERNEL_AD:-00000}"

run_dir="/output/tum_rgbd/${seq_name}/${version}/${method}"
stats_dir="${run_dir}/stats"
mkdir -p "$run_dir" "$stats_dir" /output/associations

family=""
case "$seq_name" in
  *freiburg1*|fr1*) family="TUM1" ;;
  *freiburg2*|fr2*) family="TUM2" ;;
  *freiburg3*|fr3*) family="TUM3" ;;
  *) echo "Could not infer TUM family from $seq_name" >&2; exit 2 ;;
esac

assoc="/output/associations/${seq_name}.txt"
if [[ ! -s "$assoc" ]]; then
  python3 /usr/local/bin/tum_associate.py "$sequence/rgb.txt" "$sequence/depth.txt" > "$assoc"
fi

binary="${SLAM_RGBD_BINARY:-$root/Examples/RGB-D/rgbd_tum}"
if [[ ! -x "$binary" && -x "$root/build/Examples/RGB-D/rgbd_tum" ]]; then
  binary="$root/build/Examples/RGB-D/rgbd_tum"
fi
[[ -x "$binary" ]] || { echo "Missing rgbd_tum binary: $binary" >&2; exit 1; }

settings="${SLAM_SETTINGS:-$root/Examples/RGB-D/${family}.yaml}"
[[ -f "$settings" ]] || { echo "Missing settings: $settings" >&2; exit 1; }

vocab="${SLAM_VOCABULARY:-$root/Vocabulary/ORBvoc.txt}"
[[ -f "$vocab" ]] || { echo "Missing vocabulary: $vocab" >&2; exit 1; }

cd "$root"
rm -f CameraTrajectory.txt KeyFrameTrajectory.txt

echo "Nitro-SLAM TUM RGB-D run"
echo "  method: $method"
echo "  sequence: $seq_name"
echo "  flags: FT=$fasttrack TM=$turbomap FL=$fastloop"
echo "  masks: FT=$kernel_ft TM=$kernel_tm FL=$kernel_fl AD=$kernel_ad"
echo "  run_dir: $run_dir"

start_ns="$(date +%s%N)"
set +e
CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-EAGER}" \
NITRO_REALTIME_SLEEP="${NITRO_REALTIME_SLEEP:-0}" \
  "$binary" "$vocab" "$settings" "$sequence" "$assoc" "$stats_dir" \
  "$fasttrack" "$turbomap" "$fastloop" "$kernel_ft" "$kernel_tm" "$kernel_fl" "$kernel_ad" \
  2>&1 | tee "$run_dir/run.log"
run_exit="${PIPESTATUS[0]}"
set -e
end_ns="$(date +%s%N)"
wall_seconds="$(awk -v s="$start_ns" -v e="$end_ns" 'BEGIN { printf "%.9f", (e-s)/1000000000 }')"

cp -f CameraTrajectory.txt KeyFrameTrajectory.txt "$run_dir/" 2>/dev/null || true
if [[ -f "$sequence/groundtruth.txt" && -f "$run_dir/CameraTrajectory.txt" ]]; then
  python3 /usr/local/bin/evaluate_tum_rgbd.py \
    --groundtruth "$sequence/groundtruth.txt" \
    --estimate "$run_dir/CameraTrajectory.txt" \
    --output "$run_dir/evaluation.json" \
    > "$run_dir/evaluation.stdout" || true
fi

python3 /usr/local/bin/summarize_tum_rgbd_run.py \
  --method "$method" \
  --sequence "$sequence" \
  --run-dir "$run_dir" \
  --wall-seconds "$wall_seconds" \
  --run-exit-code "$run_exit" \
  > "$run_dir/profile_summary.stdout"

echo "Nitro-SLAM TUM RGB-D results: $run_dir"
exit "$run_exit"
