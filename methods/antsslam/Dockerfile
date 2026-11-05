FROM nvcr.io/nvidia/l4t-jetpack:r36.3.0

ARG BUILD_JOBS=2
ARG ANTSSLAM_TENSORRT=OFF
ENV DEBIAN_FRONTEND=noninteractive \
    SLAM_ROOT=/opt/antsslam \
    SLAM_METHOD_NAME=antsslam \
    SLAM_USE_XVFB=0 \
    ANTSLAM_VIEWER=0 \
    ANTSLAM_REALTIME_SLEEP=0 \
    ANTSSLAM_SEMANTIC_ROOT=/opt/semantic/segmentation

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates cmake gfortran git libatlas-base-dev \
    libavcodec-dev libavdevice-dev libavformat-dev libavutil-dev \
    libboost-serialization-dev libdc1394-dev libeigen3-dev \
    libegl1-mesa-dev libglew-dev libgl1-mesa-dev libgtk-3-dev \
    libjpeg-dev libopenexr-dev libopencv-dev nvidia-opencv-dev libpng-dev libssl-dev \
    libswscale-dev libtbb-dev libtbb12 libtiff-dev libv4l-dev \
    libwayland-dev libx264-dev libxkbcommon-dev libxvidcore-dev \
    mesa-utils ninja-build perl pkg-config python3 python3-dev \
    python3-numpy tar wayland-protocols xvfb \
 && rm -rf /var/lib/apt/lists/*

COPY pangolin /opt/Pangolin-v0.5
WORKDIR /opt/Pangolin-v0.5
RUN perl -0pi -e 's@find_package\(FFMPEG QUIET\).*?message\(STATUS "ffmpeg Found and Enabled"\)\nendif\(\)\n@@s' src/CMakeLists.txt \
 && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TOOLS=OFF \
 && cmake --build build -j"${BUILD_JOBS}" \
 && cmake --install build \
 && ldconfig

COPY common /opt/slam_common
COPY scripts /usr/local/bin
COPY src ${SLAM_ROOT}

WORKDIR ${SLAM_ROOT}
RUN chmod +x /usr/local/bin/*.sh /usr/local/bin/*.py /opt/slam_common/*.py /opt/slam_common/*.sh || true
RUN if [ -f Vocabulary/ORBvoc.txt.tar.gz ] && [ ! -f Vocabulary/ORBvoc.txt ]; then tar -xf Vocabulary/ORBvoc.txt.tar.gz -C Vocabulary; fi
RUN cmake -S Thirdparty/DBoW2 -B Thirdparty/DBoW2/build -DCMAKE_BUILD_TYPE=Release \
 && cmake --build Thirdparty/DBoW2/build -j"${BUILD_JOBS}"
RUN cmake -S Thirdparty/g2o -B Thirdparty/g2o/build -DCMAKE_BUILD_TYPE=Release \
 && cmake --build Thirdparty/g2o/build -j"${BUILD_JOBS}"
RUN cmake -S Thirdparty/Sophus -B Thirdparty/Sophus/build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-DREGISTER_TIMES" \
      -DANTSSLAM_USE_TENSORRT_SEMANTICS="${ANTSSLAM_TENSORRT}" \
 && cmake --build build --target rgbd_tum semantic_tum_benchmark -j"${BUILD_JOBS}"

WORKDIR /workspace
ENTRYPOINT ["/bin/bash"]
