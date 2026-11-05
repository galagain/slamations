FROM ros:noetic-ros-base-focal

ENV DEBIAN_FRONTEND=noninteractive \
    DYNAMIC_VINS_ROOT=/root/catkin_ws/src/Dynamic-VINS

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates cmake git libatlas-base-dev libboost-all-dev \
    libeigen3-dev libgflags-dev libgoogle-glog-dev libsuitesparse-dev \
    python3-dev python3-pip python3-numpy python3-opencv wget \
    ros-noetic-actionlib ros-noetic-cv-bridge ros-noetic-image-transport \
    ros-noetic-message-filters ros-noetic-nav-msgs ros-noetic-nodelet \
    ros-noetic-ros-numpy ros-noetic-rosbag ros-noetic-sensor-msgs \
    ros-noetic-tf ros-noetic-visualization-msgs \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip wheel setuptools \
 && python3 -m pip install numpy opencv-python pyyaml rospkg \
 && python3 -m pip install --ignore-installed empy==3.3.4

WORKDIR /opt
RUN git clone --branch 2.0.0 --depth 1 https://ceres-solver.googlesource.com/ceres-solver \
 && cd ceres-solver \
 && cmake -S . -B ceres-bin -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF \
 && cmake --build ceres-bin -j"$(nproc)" \
 && cmake --install ceres-bin

RUN git clone https://github.com/strasdat/Sophus.git \
 && cd Sophus \
 && git checkout a621ff \
 && python3 - <<'PY'
from pathlib import Path
p = Path("sophus/so2.cpp")
s = p.read_text()
s = s.replace("SO2::SO2() : unit_complex_(1, 0) {}", "SO2::SO2()\\n{\\n  unit_complex_.real(1.0);\\n  unit_complex_.imag(0.0);\\n}")
s = s.replace("unit_complex_.real() = 1.;", "unit_complex_.real(1.);")
s = s.replace("unit_complex_.imag() = 0.;", "unit_complex_.imag(0.);")
p.write_text(s)
PY
RUN cmake -S /opt/Sophus -B /opt/Sophus/build -DCMAKE_BUILD_TYPE=Release \
 && cmake --build /opt/Sophus/build -j"$(nproc)" \
 && cmake --install /opt/Sophus/build

COPY common /opt/slam_common
COPY scripts /usr/local/bin
COPY src/Dynamic-VINS ${DYNAMIC_VINS_ROOT}
COPY yolo_ros /root/catkin_ws/src/yolo_ros

WORKDIR /root/catkin_ws/src/yolo_ros
RUN python3 -m pip install torch torchvision || true \
 && python3 -m pip install -r requirements.txt || true \
 && python3 -m pip install 'numpy<1.24' \
 && sed -i 's/device: gpu/device: cpu/' config/config.yaml \
 && sed -i 's/self.model = self.model.cuda() if self.device != "cpu" else self.model/self.model = self.model.cuda() if self.device.type != "cpu" else self.model/' \
      src/yolo_bridge/yolo_bridge.py

WORKDIR ${DYNAMIC_VINS_ROOT}
RUN sed -i 's#output_path: .*#output_path: "/output/dynamic_vins"#' config/tum_rgbd/tum_fr3.yaml \
 && sed -i 's#pose_graph_save_path: .*#pose_graph_save_path: "/output/dynamic_vins/pose_graph"#' config/tum_rgbd/tum_fr3.yaml

WORKDIR /root/catkin_ws
RUN chmod +x /usr/local/bin/*.sh /opt/slam_common/*.py /opt/slam_common/*.sh || true \
 && /bin/bash -lc "source /opt/ros/noetic/setup.bash && catkin_make -DCMAKE_BUILD_TYPE=Release"

RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc \
 && echo "source /root/catkin_ws/devel/setup.bash" >> /root/.bashrc

WORKDIR /workspace
ENTRYPOINT ["/bin/bash"]
