FROM nvcr.io/nvidia/l4t-jetpack:r36.3.0

ARG BUILD_JOBS=2
ENV DEBIAN_FRONTEND=noninteractive \
    SLAM_ROOT=/opt/orb_slam2_cuda \
    SLAM_METHOD_NAME=orb_slam2_cuda \
    OpenCV_DIR=/opt/opencv-cuda/lib/cmake/opencv4 \
    LD_LIBRARY_PATH=/opt/opencv-cuda/lib:${LD_LIBRARY_PATH}

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

# ORB_SLAM2_CUDA requires OpenCV CUDA modules that are not consistently present
# in JetPack apt packages, so this image builds a local OpenCV once at image build time.
RUN git clone --branch 4.8.0 --depth 1 https://github.com/opencv/opencv.git /opt/opencv-4.8.0 \
 && git clone --branch 4.8.0 --depth 1 https://github.com/opencv/opencv_contrib.git /opt/opencv_contrib-4.8.0 \
 && cmake -S /opt/opencv-4.8.0 -B /opt/opencv-4.8.0/build -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/opencv-cuda \
      -DOPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib-4.8.0/modules \
      -DBUILD_LIST=cudev,core,imgproc,imgcodecs,calib3d,features2d,flann,highgui,video,videoio \
      -DWITH_CUDA=ON \
      -DCUDA_ARCH_BIN=8.7 \
      -DCUDA_ARCH_PTX="" \
      -DENABLE_FAST_MATH=ON \
      -DCUDA_FAST_MATH=ON \
      -DWITH_CUBLAS=ON \
      -DBUILD_TESTS=OFF \
      -DBUILD_PERF_TESTS=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_opencv_python2=OFF \
      -DBUILD_opencv_python3=OFF \
      -DOPENCV_GENERATE_PKGCONFIG=ON \
 && cmake --build /opt/opencv-4.8.0/build -j"${BUILD_JOBS}" \
 && cmake --install /opt/opencv-4.8.0/build \
 && rm -rf /opt/opencv-4.8.0 /opt/opencv_contrib-4.8.0 \
 && ldconfig

COPY common /opt/slam_common
COPY scripts /usr/local/bin
COPY src ${SLAM_ROOT}

WORKDIR ${SLAM_ROOT}
RUN chmod +x /usr/local/bin/*.sh /usr/local/bin/*.py /opt/slam_common/*.py /opt/slam_common/*.sh || true \
 && python3 /usr/local/bin/patch_orb_slam2_cuda.py \
 && cmake -S Thirdparty/DBoW2 -B Thirdparty/DBoW2/build -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR="${OpenCV_DIR}" \
 && cmake --build Thirdparty/DBoW2/build -j"${BUILD_JOBS}" \
 && cmake -S Thirdparty/g2o -B Thirdparty/g2o/build -DCMAKE_BUILD_TYPE=Release \
 && cmake --build Thirdparty/g2o/build -j"${BUILD_JOBS}" \
 && if [ -f Vocabulary/ORBvoc.txt.tar.gz ] && [ ! -f Vocabulary/ORBvoc.txt ]; then tar -xf Vocabulary/ORBvoc.txt.tar.gz -C Vocabulary; fi \
 && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR="${OpenCV_DIR}" \
 && cmake --build build --target rgbd_tum -j"${BUILD_JOBS}"

WORKDIR /workspace
ENTRYPOINT ["/bin/bash"]
