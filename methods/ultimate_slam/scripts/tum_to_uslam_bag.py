#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np
import rosbag
import rospy
from sensor_msgs.msg import Image, Imu


def read_tum_list(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        stamp, rel = line.split()[:2]
        rows.append((float(stamp), rel))
    return rows


def make_image_msg(stamp, image):
    msg = Image()
    msg.header.stamp = rospy.Time.from_sec(stamp)
    msg.header.frame_id = "dvs"
    msg.height, msg.width = image.shape[:2]
    msg.encoding = "mono8"
    msg.is_bigendian = False
    msg.step = msg.width
    msg.data = np.ascontiguousarray(image).tobytes()
    return msg


def make_imu_msg(stamp):
    msg = Imu()
    msg.header.stamp = rospy.Time.from_sec(stamp)
    msg.header.frame_id = "imu"
    msg.orientation_covariance[0] = -1.0
    msg.angular_velocity.x = 0.0
    msg.angular_velocity.y = 0.0
    msg.angular_velocity.z = 0.0
    msg.linear_acceleration.x = 0.0
    msg.linear_acceleration.y = 0.0
    msg.linear_acceleration.z = 9.81
    return msg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence")
    parser.add_argument("bag")
    parser.add_argument("--imu-rate", type=float, default=200.0)
    args = parser.parse_args()

    sequence = Path(args.sequence)
    rgb_rows = read_tum_list(sequence / "rgb.txt")
    if not rgb_rows:
        raise SystemExit(f"no RGB frames found in {sequence / 'rgb.txt'}")

    t0 = rgb_rows[0][0] - 0.05
    t1 = rgb_rows[-1][0] + 0.05
    dt = 1.0 / args.imu_rate
    imu_times = []
    t = t0
    while t <= t1 + 0.5 * dt:
        imu_times.append(t)
        t += dt

    events = [(t, "imu", None) for t in imu_times]
    events.extend((stamp, "image", rel) for stamp, rel in rgb_rows)
    events.sort(key=lambda x: (x[0], 0 if x[1] == "imu" else 1))

    Path(args.bag).parent.mkdir(parents=True, exist_ok=True)
    with rosbag.Bag(args.bag, "w") as bag:
        for stamp, kind, rel in events:
            rostime = rospy.Time.from_sec(stamp)
            if kind == "imu":
                bag.write("/dvs/imu", make_imu_msg(stamp), rostime)
                continue
            image_path = sequence / rel
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise SystemExit(f"failed to read image {image_path}")
            bag.write("/dvs/image_raw", make_image_msg(stamp, image), rostime)


if __name__ == "__main__":
    main()
