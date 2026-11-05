FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive \
    VDO_SLAM_ROOT=/opt/VDO_SLAM

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates cmake git libatlas-base-dev \
    libeigen3-dev libgtk2.0-dev libopencv-contrib-dev libopencv-dev \
    libsuitesparse-dev python3 python3-opencv python3-numpy unzip vim wget \
 && rm -rf /var/lib/apt/lists/*

COPY common /opt/slam_common
COPY scripts /usr/local/bin
COPY src ${VDO_SLAM_ROOT}

WORKDIR ${VDO_SLAM_ROOT}
RUN chmod +x /usr/local/bin/*.sh /opt/slam_common/*.py /opt/slam_common/*.sh || true \
 && mkdir -p /app \
 && sed -i 's/-Wall//g; s/ -W / /g; s/-march=native//g' dependencies/g2o/CMakeLists.txt \
 && sed -i 's/CMAKE_C_FLAGS} /CMAKE_C_FLAGS} -w /g; s/CMAKE_CXX_FLAGS} /CMAKE_CXX_FLAGS} -w /g' dependencies/g2o/CMakeLists.txt

RUN mkdir -p dependencies/g2o/build \
 && cd dependencies/g2o/build \
 && cmake .. -DCMAKE_BUILD_TYPE=Release \
 && make -j1

RUN cp /usr/local/bin/tum_rgbd.yaml example/tum_rgbd.yaml \
 && sed -i 's/CV_LOAD_IMAGE_UNCHANGED/cv::IMREAD_UNCHANGED/g' example/vdo_slam.cc \
 && sed -i 's/CV_RGB2GRAY/cv::COLOR_RGB2GRAY/g; s/CV_BGR2GRAY/cv::COLOR_BGR2GRAY/g; s/CV_RGBA2GRAY/cv::COLOR_RGBA2GRAY/g; s/CV_BGRA2GRAY/cv::COLOR_BGRA2GRAY/g; s/CV_GRAY2RGB/cv::COLOR_GRAY2RGB/g; s/CV_FILLED/cv::FILLED/g' src/Tracking.cc \
 && sed -i 's/cv::SOLVEPNP_AP3P/cv::SOLVEPNP_P3P/g' src/Tracking.cc \
 && sed -i '/opencv2\/xfeatures2d.hpp/d' src/Frame.cc \
 && sed -i 's/-Wall//g; s/ -W / /g; s/-march=native//g' CMakeLists.txt \
 && sed -i 's/CMAKE_C_FLAGS} /CMAKE_C_FLAGS} -w /g; s/CMAKE_CXX_FLAGS} /CMAKE_CXX_FLAGS} -w /g' CMakeLists.txt

RUN python3 - <<'PY'
from pathlib import Path
import re
p = Path("src/cvplot/highgui.cc")
s = p.read_text()
s = re.sub(r"Rect selectROI\(const std::string &windowName, void \*img, bool showCrosshair,\n               bool fromCenter\) \{.*?\n\}\n\nRect selectROI", "Rect selectROI(const std::string &windowName, void *img, bool showCrosshair,\n               bool fromCenter) {\n  return Rect(-1, -1, -1, -1);\n}\n\nRect selectROI", s, flags=re.S)
s = re.sub(r"Rect selectROI\(void \*img, bool showCrosshair, bool fromCenter\) \{.*?\n\}\n\nvoid selectROIs", "Rect selectROI(void *img, bool showCrosshair, bool fromCenter) {\n  return Rect(-1, -1, -1, -1);\n}\n\nvoid selectROIs", s, flags=re.S)
s = re.sub(r"void selectROIs\(const std::string &windowName, void \*img,\n                std::vector<Rect> &boundingBoxes, bool showCrosshair,\n                bool fromCenter\) \{.*?\n\}\n\nvoid setMouseCallback", "void selectROIs(const std::string &windowName, void *img,\n                std::vector<Rect> &boundingBoxes, bool showCrosshair,\n                bool fromCenter) {\n}\n\nvoid setMouseCallback", s, flags=re.S)
p.write_text(s)

p = Path("src/Tracking.cc")
s = p.read_text()
needle = "    camera_mat.at<double>(2, 2) = 1.0;\n\n    // output\n"
cam_guard = """    camera_mat.at<double>(2, 2) = 1.0;

    if (N < 4)
    {
        if (mVelocity.empty())
            return cv::Mat::eye(4,4,CV_32F)*mLastFrame.mTcw;
        return mVelocity*mLastFrame.mTcw;
    }

    // output
"""
s = s.replace(needle, cam_guard, 1)
obj_guard = """    camera_mat.at<double>(2, 2) = 1.0;

    if (N < 4)
    {
        ObjId_sub.clear();
        return Mod;
    }

    // output
"""
s = s.replace(needle, obj_guard, 1)
p.write_text(s)
PY

RUN mkdir -p build \
 && cd build \
 && cmake .. -DCMAKE_BUILD_TYPE=Release \
 && make VERBOSE=1 -j1

WORKDIR /workspace
ENTRYPOINT ["/bin/bash"]
