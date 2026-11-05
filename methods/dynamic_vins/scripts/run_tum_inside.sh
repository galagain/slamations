#!/usr/bin/env bash
set -euo pipefail

source /opt/slam_common/tum_tools.sh

sequence="${1:-/data/sequence}"
assoc="$(ensure_tum_associations "$sequence" /output)"
bag="/output/bags/$(basename "$sequence").bag"
mkdir -p /output/bags /output/dynamic_vins

export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"
export ROS_HOSTNAME="${ROS_HOSTNAME:-localhost}"
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash

if [[ ! -s "$bag" ]]; then
  python3 /opt/slam_common/tum_to_rosbag.py "$sequence" "$assoc" "$bag"
fi

roslaunch vins_estimator tum_rgbd_pytorch.launch >/tmp/dynamic_vins.launch.log 2>&1 &
launch_pid=$!
trap 'kill "$launch_pid" 2>/dev/null || true' EXIT
sleep "${DYNAMIC_VINS_WARMUP:-8}"

rosbag play "$bag"
sleep 2
kill "$launch_pid" 2>/dev/null || true
cp -f /tmp/dynamic_vins.launch.log /output/dynamic_vins/launch.log 2>/dev/null || true
echo "Dynamic-VINS output: /output/dynamic_vins"
