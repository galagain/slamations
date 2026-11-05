#!/usr/bin/env bash
set -euo pipefail

variant="${1:-segnet}"
sequence="${2:-/data/sequence}"

source /opt/slam_common/tum_tools.sh
family="$(tum_family "$sequence")"
assoc="$(ensure_tum_associations "$sequence" /output)"
out_dir="/output/${variant}/$(basename "$sequence")"
mkdir -p "$out_dir"

export ROS_DISTRO="${ROS_DISTRO:-noetic}"
export ROS_VERSION="${ROS_VERSION:-1}"
export ROS_PYTHON_VERSION="${ROS_PYTHON_VERSION:-3}"
export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"
export ROS_HOSTNAME="${ROS_HOSTNAME:-localhost}"
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
default_cuda11_ld="/host_pkgs/cuda-cudart-11.7.99-0/lib:/host_pkgs/libcublas-11.10.3.66-0/lib:/host_pkgs/libcufft-10.7.2.124-h9126a7e_0/lib:/host_pkgs/libcurand-10.3.10.19-0/lib:/host_pkgs/libcusolver-11.4.0.1-0/lib:/host_pkgs/libcusparse-11.7.4.91-0/lib"
if [[ -d /host_pkgs/cuda-cudart-11.7.99-0/lib ]]; then
  export LD_LIBRARY_PATH="${RDS_CUDA11_LD:-$default_cuda11_ld}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export PYTHONPATH="$RDS_SLAM_ROOT/MaskRCNN_ROS/include/MaskRCNN:$RDS_SLAM_ROOT/MaskRCNN_ROS/include/MaskRCNN/samples/coco:/root/catkin_ws/devel/lib/python3/dist-packages:/opt/ros/noetic/lib/python3/dist-packages"

if [[ -d /opt/rds_models/segnet ]]; then
  mkdir -p /root/semseg/LUT /root/semseg/SegNet/Example_Models
  [[ -f /opt/rds_models/segnet/pascal.png ]] && ln -sf /opt/rds_models/segnet/pascal.png /root/semseg/LUT/pascal.png
  [[ -f /opt/rds_models/segnet/segnet_pascal.prototxt ]] && ln -sf /opt/rds_models/segnet/segnet_pascal.prototxt /root/semseg/SegNet/Example_Models/segnet_pascal.prototxt
fi

roscore >/tmp/roscore.log 2>&1 &
roscore_pid=$!
copy_logs() {
  cp -f /tmp/roscore.log "$out_dir/roscore.log" 2>/dev/null || true
  cp -f /tmp/rds_segnet.log "$out_dir/rds_segnet.log" 2>/dev/null || true
  cp -f /tmp/rds_maskrcnn.log "$out_dir/rds_maskrcnn.log" 2>/dev/null || true
}
trap 'copy_logs; kill "$roscore_pid" 2>/dev/null || true' EXIT
sleep 3

if [[ "$variant" == "segnet" ]]; then
  cat >/tmp/segnet_action_server_cli.py <<'PY'
#!/usr/bin/env python3
import os
import subprocess
import tempfile

import actionlib
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from segnet_ros.msg import segnetAction, segnetFeedback, segnetResult


def image_msg_to_bgr(msg):
    channels_by_encoding = {
        "bgr8": 3,
        "rgb8": 3,
        "mono8": 1,
        "8UC1": 1,
        "8UC3": 3,
    }
    channels = channels_by_encoding.get(msg.encoding, 3)
    array = np.frombuffer(msg.data, dtype=np.uint8)
    if channels == 1:
        image = array.reshape((msg.height, msg.step))[:, :msg.width]
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    row_pixels = msg.step // channels
    image = array.reshape((msg.height, row_pixels, channels))[:, :msg.width, :]
    if msg.encoding == "rgb8":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image.copy()


def mono_image_msg(label, header):
    msg = Image()
    msg.header = header
    msg.height, msg.width = label.shape[:2]
    msg.encoding = "mono8"
    msg.is_bigendian = 0
    msg.step = msg.width
    msg.data = label.astype(np.uint8, copy=False).tobytes()
    return msg


def color_image_msg(color, header):
    msg = Image()
    msg.header = header
    msg.height, msg.width = color.shape[:2]
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = color.astype(np.uint8, copy=False).tobytes()
    return msg


class SegnetCliActionServer:
    def __init__(self):
        self.prototxt = rospy.get_param("/segnet/prototxt", "/root/semseg/SegNet/Example_Models/segnet_pascal.prototxt")
        self.caffemodel = rospy.get_param("/segnet/caffemodel", "/root/cnnmodel/segnet_pascal.caffemodel")
        self.lut = rospy.get_param("/segnet/lut", "/root/semseg/LUT/pascal.png")
        self.action_name = rospy.get_param("/segnet/action_name", "/segnet_action")
        self.dynamic_labels = os.environ.get("RDS_SEGNET_DYNAMIC_LABELS", "15")
        self.cli = "/root/catkin_ws/devel/lib/segnet_ros/segnet_cli"
        self.server = actionlib.SimpleActionServer(
            self.action_name, segnetAction, execute_cb=self.execute, auto_start=False
        )
        self.server.start()
        print("SegNet CLI action server ready", flush=True)

    def execute(self, goal):
        feedback = segnetFeedback()
        result = segnetResult()
        result.id = goal.id
        try:
            image = image_msg_to_bgr(goal.image)
            with tempfile.TemporaryDirectory(prefix="rds_segnet_") as tmpdir:
                in_path = os.path.join(tmpdir, "rgb.png")
                mask_path = os.path.join(tmpdir, "feature_mask.png")
                cv2.imwrite(in_path, image)
                completed = subprocess.run(
                    [self.cli, self.prototxt, self.caffemodel, self.lut, in_path, mask_path, self.dynamic_labels, "0"],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                print(completed.stdout, flush=True)
                if completed.returncode != 0:
                    raise RuntimeError(f"segnet_cli failed with exit code {completed.returncode}")

                feature_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if feature_mask is None:
                    raise RuntimeError("segnet_cli did not write a mask")
                if feature_mask.shape[:2] != image.shape[:2]:
                    feature_mask = cv2.resize(feature_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

            label = np.zeros(feature_mask.shape, dtype=np.uint8)
            label[feature_mask == 0] = 15
            color = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
            color[label == 15] = (0, 0, 255)
            result.label = mono_image_msg(label, goal.image.header)
            result.label_color = color_image_msg(color, goal.image.header)
            feedback.complete = True
            self.server.publish_feedback(feedback)
            self.server.set_succeeded(result)
        except Exception as exc:
            rospy.logerr("SegNet CLI action failed: %s", exc)
            feedback.complete = False
            self.server.publish_feedback(feedback)
            self.server.set_aborted(result, str(exc))


if __name__ == "__main__":
    rospy.init_node("segnet_action_server_cli")
    SegnetCliActionServer()
    rospy.spin()
PY
  chmod +x /tmp/segnet_action_server_cli.py
  python3 /tmp/segnet_action_server_cli.py >/tmp/rds_segnet.log 2>&1 &
  init_delay=100
  frame_delay=100
else
  sed -i \
    -e '1s|^#!.*python.*$|#!/usr/bin/env python3|' \
    -e "s|/root/catkin_ws/src/MaskRCNN_ROS/include/MaskRCNN/|$RDS_SLAM_ROOT/MaskRCNN_ROS/include/MaskRCNN/|g" \
    "$RDS_SLAM_ROOT/MaskRCNN_ROS/script/action_server.py"
  roslaunch maskrcnn_ros action_server.launch >/tmp/rds_maskrcnn.log 2>&1 &
  init_delay=30000
  frame_delay=30000
fi
server_pid=$!
trap 'copy_logs; kill "$server_pid" "$roscore_pid" 2>/dev/null || true' EXIT
sleep "${RDS_SERVER_WARMUP:-10}"

rds_opencv_preload="/usr/lib/aarch64-linux-gnu/libopencv_core.so.4.5:/usr/lib/aarch64-linux-gnu/libopencv_imgproc.so.4.5:/usr/lib/aarch64-linux-gnu/libopencv_imgcodecs.so.4.5:/usr/lib/aarch64-linux-gnu/libopencv_calib3d.so.4.5:/usr/lib/aarch64-linux-gnu/libopencv_features2d.so.4.5:/usr/lib/aarch64-linux-gnu/libopencv_flann.so.4.5"
LD_PRELOAD="${rds_opencv_preload}${LD_PRELOAD:+:$LD_PRELOAD}" rosrun rds_slam Dynamic_RGBD \
  "$RDS_SLAM_ROOT/SLAM/Vocabulary/ORBvoc.txt" \
  "$RDS_SLAM_ROOT/SLAM/config/RGB-D/${family}.yaml" \
  "$sequence" \
  "$assoc" \
  "$out_dir" \
  _cnn_method:="$variant" \
  _init_delay:="$init_delay" \
  _frame_delay:="$frame_delay" \
  _init_frames:=2

cp -f CameraTrajectory.txt KeyFrameTrajectory.txt "$out_dir/" 2>/dev/null || true
echo "RDS-SLAM ${variant} results: $out_dir"
