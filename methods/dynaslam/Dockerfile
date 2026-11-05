FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive \
    DYNASLAM_ROOT=/opt/DynaSLAM

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates cmake git libboost-all-dev libeigen3-dev \
    libgl1-mesa-dev libgl1-mesa-dri mesa-utils libglew-dev libgtk2.0-dev libopencv-dev \
    libpython2.7 libpython2.7-dev libqt5opengl5-dev qtbase5-dev \
    python python-numpy \
    python2.7 python2.7-dev python-pip unzip wget \
 && rm -rf /var/lib/apt/lists/*

RUN git clone --branch v0.5 --depth 1 https://github.com/stevenlovegrove/Pangolin.git /opt/Pangolin-v0.5 \
 && mkdir -p /opt/Pangolin-v0.5/build \
 && cd /opt/Pangolin-v0.5/build \
 && cmake .. -DCMAKE_BUILD_TYPE=Release \
 && make -j"$(nproc)" \
 && make install \
 && ldconfig

COPY common /opt/slam_common
COPY scripts /usr/local/bin
COPY src ${DYNASLAM_ROOT}

WORKDIR ${DYNASLAM_ROOT}
RUN chmod +x /usr/local/bin/*.sh /opt/slam_common/*.py /opt/slam_common/*.sh || true \
 && python -m pip install --upgrade pip==20.3.4 setuptools wheel \
 && python -m pip install numpy==1.16.6 \
 && python - <<'PY'
import os
import subprocess

p = "CMakeLists.txt"
with open(p, "r") as f:
    s = f.read()
numpy_include = subprocess.check_output([
    "python", "-c", "import numpy; print(numpy.get_include())"
]).strip()
numpy_include = os.path.join(numpy_include, "numpy")
s = s.replace(
    "find_package(PythonLibs REQUIRED)",
    "set(PYTHON_EXECUTABLE /usr/bin/python2.7)\n"
    "set(PYTHON_INCLUDE_DIR /usr/include/python2.7)\n"
    "set(PYTHON_LIBRARY /usr/lib/aarch64-linux-gnu/libpython2.7.so)\n"
    "find_package(PythonLibs 2.7 REQUIRED)"
)
s = s.replace("/usr/lib/x86_64-linux-gnu/libpython2.7.so", "${PYTHON_LIBRARIES}")
s = s.replace("/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy", numpy_include)
s = s.replace("/usr/lib/python2.7/dist-packages/numpy/core/include/numpy/", numpy_include + "/")
s = s.replace(
    "\nadd_executable(mono_carla\nExamples/Monocular/mono_carla.cc)\n"
    "target_link_libraries(mono_carla ${PROJECT_NAME})\n",
    "\n"
)
with open(p, "w") as f:
    f.write(s)

p = "Examples/RGB-D/rgbd_tum.cc"
with open(p, "r") as f:
    s = f.read()
s = s.replace(
    "#include <unistd.h>\n",
    "#include <unistd.h>\n#include <cstdlib>\n"
)
s = s.replace(
    "ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::RGBD,true);",
    "bool useViewer = false;\n"
    "    const char* viewerEnv = std::getenv(\"DYNASLAM_VIEWER\");\n"
    "    if(viewerEnv != NULL && std::string(viewerEnv) != \"0\" && std::string(viewerEnv) != \"false\") useViewer = true;\n"
    "    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::RGBD,useViewer);"
)
s = s.replace(
    "    // Stop all threads\n"
    "    SLAM.Shutdown();",
    "    // Save once before shutdown so long cleanup cannot discard the TUM trajectory.\n"
    "    SLAM.SaveTrajectoryTUM(\"CameraTrajectory.txt\");\n"
    "    SLAM.SaveKeyFrameTrajectoryTUM(\"KeyFrameTrajectory.txt\");\n\n"
    "    // Stop all threads\n"
    "    SLAM.Shutdown();"
)
s = s.replace(
    "    for(int ni=0; ni<nImages; ni++)\n"
    "    {\n",
    "    for(int ni=0; ni<nImages; ni++)\n"
    "    {\n"
    "        if(ni % 50 == 0) std::cout << \"DynaSLAM frame \" << ni << \"/\" << nImages << std::endl;\n"
)
with open(p, "w") as f:
    f.write(s)

p = "src/Tracking.cc"
with open(p, "r") as f:
    s = f.read()
s = s.replace(
    "    mGeometry.GeometricModelCorrection(mCurrentFrame,mImGray,imMask);\n",
    "    mGeometry.GeometricModelCorrection(mCurrentFrame,imDepth,imMask);\n"
)
with open(p, "w") as f:
    f.write(s)
PY

RUN if [ -f Vocabulary/ORBvoc.txt.tar.gz ] && [ ! -f Vocabulary/ORBvoc.txt ]; then tar -xf Vocabulary/ORBvoc.txt.tar.gz -C Vocabulary; fi \
 && chmod +x build.sh \
 && ./build.sh

WORKDIR /workspace
ENTRYPOINT ["/bin/bash"]
