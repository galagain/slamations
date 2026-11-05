FROM ros:noetic-ros-base-focal

ENV DEBIAN_FRONTEND=noninteractive \
    USLAM_WS=/root/uslam_ws \
    USLAM_ROOT=/root/uslam_ws/src/rpg_ultimate_slam_open

RUN apt-get update && apt-get install -y --no-install-recommends \
    autoconf automake build-essential ca-certificates cmake git \
    libblas-dev liblapack-dev libtool wget \
    libopencv-dev libyaml-cpp-dev python3-catkin-tools python3-empy \
    python3-numpy python3-opencv python3-pip python3-vcstool python3-yaml \
    ros-noetic-cv-bridge ros-noetic-geometry-msgs ros-noetic-image-transport \
    ros-noetic-message-generation ros-noetic-message-runtime ros-noetic-nav-msgs \
    ros-noetic-rosbag ros-noetic-roscpp ros-noetic-sensor-msgs ros-noetic-std-msgs \
    ros-noetic-tf ros-noetic-visualization-msgs \
 && rm -rf /var/lib/apt/lists/*

COPY common /opt/slam_common
COPY scripts /usr/local/bin
COPY calibration /opt/ultimate_slam_calibration
COPY src ${USLAM_ROOT}

WORKDIR ${USLAM_WS}/src
RUN sed -E 's#git@github.com:#https://github.com/#g; s#\\.git$##g' \
      ${USLAM_ROOT}/dependencies.yaml > /tmp/ultimate_slam_dependencies.yaml \
 && vcs import --recursive . < /tmp/ultimate_slam_dependencies.yaml

WORKDIR ${USLAM_ROOT}
RUN cp -f /opt/ultimate_slam_calibration/*.yaml calibration/ \
 && python3 - <<'PY'
from pathlib import Path

# OpenCV 4 no longer exposes CV_* constants in the same way as OpenCV 3.
for p in Path('/root/uslam_ws/src').rglob('*.cpp'):
    s = p.read_text(errors='ignore')
    ns = s.replace('CV_GRAY2BGR', 'cv::COLOR_GRAY2BGR')
    ns = ns.replace('CV_BGR2GRAY', 'cv::COLOR_BGR2GRAY')
    ns = ns.replace('CV_RGB2GRAY', 'cv::COLOR_RGB2GRAY')
    ns = ns.replace('CV_GRAY2RGB', 'cv::COLOR_GRAY2RGB')
    if ns != s:
        p.write_text(ns)

# A few synchronizer loops in the released code walk one element past the
# vector end. The bug is latent with DAVIS bags too, and fatal with ASAN/debug.
p = Path('/root/uslam_ws/src/rpg_ultimate_slam_open/common/ze_data_provider/src/camera_imu_synchronizer_base.cpp')
s = p.read_text()
s = s.replace('for (size_t i = 0; i <= image_buffer_.size(); ++i)', 'for (size_t i = 0; i < image_buffer_.size(); ++i)')
p.write_text(s)

# The upstream fast_neon package switches to x86 SSE unless ARM_ARCHITECTURE is
# set, then adds ARMv7-only flags. On Jetson/aarch64 we keep the NEON source but
# remove the ARMv7 compiler switches.
p = Path('/root/uslam_ws/src/fast_neon/CMakeLists.txt')
s = p.read_text()
s = s.replace('-O3 -mfpu=neon -march=armv7-a', '-O3 -fomit-frame-pointer -D__ARM_NEON__=1')
p.write_text(s)

p = Path('/root/uslam_ws/src/fast_neon/src/faster_corner_9_neon.cpp')
s = p.read_text()
s = s.replace('#ifndef __ARM_NEON__', '#if !defined(__ARM_NEON__) && !defined(__ARM_NEON)')
p.write_text(s)

p = Path('/root/uslam_ws/src/fast_neon/include/fast/faster_corner_utilities.h')
s = p.read_text()
s = s.replace('#if __ARM_NEON__', '#if defined(__ARM_NEON__) || defined(__ARM_NEON)')
s = s.replace('#elif __SSE2__', '#elif defined(__SSE2__)')
p.write_text(s)

p = Path('/root/uslam_ws/src/rpg_ultimate_slam_open/common/ze_cmake/cmake/modules/ze_setup.cmake')
s = p.read_text()
s = s.replace(' -mfpu=neon -march=armv7-a', ' -fomit-frame-pointer -D__ARM_NEON__=1')
p.write_text(s)
PY

WORKDIR ${USLAM_WS}
RUN chmod +x /usr/local/bin/*.sh /usr/local/bin/*.py /opt/slam_common/*.py /opt/slam_common/*.sh || true \
 && /bin/bash -lc "source /opt/ros/noetic/setup.bash && catkin config --extend /opt/ros/noetic --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=14" \
 && /bin/bash -lc "source /opt/ros/noetic/setup.bash && export ARM_ARCHITECTURE=1 && catkin build ze_vio_ceres --no-status --summarize"

RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc \
 && echo "source ${USLAM_WS}/devel/setup.bash" >> /root/.bashrc

WORKDIR /workspace
ENTRYPOINT ["/bin/bash"]
