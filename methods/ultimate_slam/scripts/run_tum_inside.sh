#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/slam_common/tum_tools.sh

set +u
source /opt/ros/noetic/setup.bash
source /root/uslam_ws/devel/setup.bash
set -u

sequence="${1:-/data/sequence}"
family="$(tum_family "$sequence")"
seq_name="$(basename "$sequence")"
out_dir="/output/$seq_name"
bag="/output/bags/${seq_name}_ultimate_slam_synthetic_imu.bag"
mkdir -p "$out_dir" /output/bags

export ROS_MASTER_URI="${ROS_MASTER_URI:-http://127.0.0.1:11311}"
roscore_pid=""
cleanup() {
  if [[ -n "$roscore_pid" ]]; then
    kill "$roscore_pid" >/dev/null 2>&1 || true
    wait "$roscore_pid" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if ! rostopic list >/dev/null 2>&1; then
  roscore > "$out_dir/roscore.log" 2>&1 &
  roscore_pid=$!
  for _ in $(seq 1 80); do
    if rostopic list >/dev/null 2>&1; then
      break
    fi
    sleep 0.1
  done
fi

if ! rostopic list >/dev/null 2>&1; then
  echo "ROS master did not become available at $ROS_MASTER_URI" >&2
  exit 3
fi

case "$family" in
  TUM1) calib="TUM1.yaml" ;;
  TUM2) calib="TUM2.yaml" ;;
  TUM3) calib="TUM3.yaml" ;;
  *) echo "Unsupported TUM family: $family" >&2; exit 2 ;;
esac

if [[ ! -s "$bag" ]]; then
  "$script_dir/tum_to_uslam_bag.py" "$sequence" "$bag"
fi

cd "$USLAM_ROOT"
rm -f "$out_dir"/traj_es.csv "$out_dir"/CameraTrajectory.txt

run_exit=0
rosrun ze_vio_ceres ze_vio_ceres_node \
  --data_source=1 \
  --bag_filename="$bag" \
  --num_cams=1 \
  --num_dvs=0 \
  --num_imus=1 \
  --topic_cam0=/dvs/image_raw \
  --topic_imu0=/dvs/imu \
  --vio_use_events=0 \
  --vio_use_events_and_images=0 \
  --calib_filename="$USLAM_ROOT/calibration/$calib" \
  --flagfile="$USLAM_ROOT/applications/ze_vio_ceres/cfg/vio_base.conf" \
  --timeshift_cam_imu=0 \
  --vio_trace_pose=true \
  --log_dir="$out_dir" \
  --vio_descriptor_use_dummy=true \
  --vmodule=feature_tracker=0 \
  --colorlogtostderr=1 \
  --logtostderr=1 \
  --alsologtostderr=1 \
  --vio_viz_level=0 \
  --vio_viz_feature_tracks=false \
  --vio_viz_show_image=false \
  --data_source_stop_after_n_frames="${ULTIMATE_SLAM_MAX_FRAMES:--1}" || run_exit=$?

if [[ -f "$out_dir/traj_es.csv" ]]; then
  "$script_dir/traj_es_to_tum.py" "$out_dir/traj_es.csv" "$out_dir/CameraTrajectory.txt"
fi

cat > "$out_dir/run_metadata.txt" <<EOF
sequence=$sequence
family=$family
bag=$bag
calibration=$calib
input_modalities=rgb_images_plus_synthetic_stationary_imu
missing_modalities=tum_rgbd_has_no_events_no_real_imu_depth_not_used_by_ultimate_slam
EOF

echo "UltimateSLAM results: $out_dir"
exit "$run_exit"
