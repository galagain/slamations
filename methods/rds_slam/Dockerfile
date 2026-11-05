ARG BASE_IMAGE=nvcr.io/nvidia/l4t-base:r35.1.0
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    RDS_SLAM_ROOT=/root/catkin_ws/src/RDS-SLAM

RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-transport-https ca-certificates curl gnupg2 lsb-release software-properties-common \
    automake autotools-dev bison build-essential byacc cmake g++ gdb git gfortran \
    hdf5-tools libatlas-base-dev libavcodec-dev libavformat-dev libboost-all-dev \
    libdc1394-22-dev libeigen3-dev libevent-dev libglew-dev libgoogle-glog-dev \
    libgtk2.0-dev libhdf5-dev libjpeg-dev libleveldb-dev liblmdb-dev libopencv-dev \
    libpng-dev libprotobuf-dev libsnappy-dev libssl-dev libswscale-dev libtbb-dev \
    libtiff-dev libv4l-dev libx264-dev libxvidcore-dev pkg-config protobuf-compiler \
    python3-dev python3-pip python3-numpy python3-opencv sudo tmux unzip vim wget zip \
 && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - \
 && echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros1.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
    ros-noetic-desktop-full python3-rosdep python3-rosinstall python3-vcstools \
    python3-rosinstall-generator python3-wstool \
 && rm -rf /var/lib/apt/lists/* \
 && rosdep init || true \
 && rosdep update

RUN python3 -m pip install --upgrade pip wheel setuptools \
 && python3 -m pip install 'numpy<1.24' 'Cython<3' scipy Pillow matplotlib scikit-image imageio==2.6.1 imgaug pandas future keras==2.3.1 \
 && H5PY_SETUP_REQUIRES=0 python3 -m pip install h5py==2.10.0 \
 && python3 -m pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

ARG INSTALL_TF_JETSON=1
RUN if [ "$INSTALL_TF_JETSON" = "1" ]; then \
      wget -q https://developer.download.nvidia.com/compute/redist/jp/v50/tensorflow/tensorflow-1.15.5+nv22.06-cp38-cp38-linux_aarch64.whl && \
      python3 -m pip install tensorflow-1.15.5+nv22.06-cp38-cp38-linux_aarch64.whl && \
      rm tensorflow-1.15.5+nv22.06-cp38-cp38-linux_aarch64.whl; \
    fi \
 && python3 -m pip install --ignore-installed 'numpy==1.23.5'

COPY common /opt/slam_common
COPY scripts /usr/local/bin
COPY src/RDS-SLAM ${RDS_SLAM_ROOT}
COPY src/segnet_pascal.caffemodel /tmp/segnet_pascal.caffemodel

WORKDIR /root
RUN chmod +x /usr/local/bin/*.sh /opt/slam_common/*.py /opt/slam_common/*.sh || true \
 && mkdir -p /root/cnnmodel \
 && mv /tmp/segnet_pascal.caffemodel /root/cnnmodel/segnet_pascal.caffemodel \
 && wget -q -nc -P /root/cnnmodel https://github.com/matterport/Mask_RCNN/releases/download/v1.0/mask_rcnn_coco.h5

# Build caffe-segnet for the SegNet ROS action server.
WORKDIR ${RDS_SLAM_ROOT}/caffe-segnet
RUN sed -i 's/CV_LOAD_IMAGE_COLOR/cv::IMREAD_COLOR/g; s/CV_LOAD_IMAGE_GRAYSCALE/cv::IMREAD_GRAYSCALE/g' \
      src/caffe/layers/window_data_layer.cpp src/caffe/util/io.cpp \
 && sed -i 's/CV_BGR2GRAY/cv::COLOR_BGR2GRAY/g; s/CV_BGRA2GRAY/cv::COLOR_BGRA2GRAY/g; s/CV_BGRA2BGR/cv::COLOR_BGRA2BGR/g; s/CV_GRAY2BGR/cv::COLOR_GRAY2BGR/g' \
      examples/cpp_classification/classification.cpp \
 && cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCPU_ONLY=ON \
      -DHDF5_INCLUDE_DIRS=/usr/include/hdf5/serial \
      -DHDF5_LIBRARIES="/usr/lib/aarch64-linux-gnu/libhdf5_serial.so;/usr/lib/aarch64-linux-gnu/libhdf5_serial_hl.so" \
      -DHDF5_HL_LIBRARIES="/usr/lib/aarch64-linux-gnu/libhdf5_serial_hl.so" \
 && cmake --build build -j"$(nproc)" \
 && cmake --install build \
 && ldconfig

WORKDIR /root/catkin_ws
RUN find ${RDS_SLAM_ROOT}/SLAM -type f \( -name '*.cc' -o -name '*.cpp' -o -name '*.h' \) \
      -exec sed -i \
        -e 's/CV_LOAD_IMAGE_UNCHANGED/cv::IMREAD_UNCHANGED/g' \
        -e 's/CV_GRAY2BGR/cv::COLOR_GRAY2BGR/g' \
        -e 's/CV_REDUCE_SUM/cv::REDUCE_SUM/g' {} + \
 && sed -i 's/std::thread(spin_thread);/ros::AsyncSpinner spinner(1); spinner.start();/' \
      ${RDS_SLAM_ROOT}/SLAM/Examples/ros_tum_rgbd.cpp \
	 && sed -i 's/ORB_SLAM3::System SLAM(argv\[1\], argv\[2\], ORB_SLAM3::System::RGBD, true);/ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::RGBD, false);/' \
	      ${RDS_SLAM_ROOT}/SLAM/Examples/ros_tum_rgbd.cpp \
	 && perl -0pi -e 's#    // Stop all threads\n    SLAM.Shutdown\(\);#    // Save trajectories before semantic cleanup, so a late shutdown failure still leaves usable TUM output.\n    if (argc >= 6) {\n        const std::string resultPath(argv[5]);\n        SLAM.SaveTrajectoryTUM(resultPath + "/CameraTrajectory.txt");\n        SLAM.SaveKeyFrameTrajectoryTUM(resultPath + "/KeyFrameTrajectory.txt");\n    }\n    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");\n    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");\n\n    // Stop all threads\n    SLAM.Shutdown();#' \
	      ${RDS_SLAM_ROOT}/SLAM/Examples/ros_tum_rgbd.cpp \
	 && sed -i 's#SLAM.TrackRGBD(imRGB, imD, tframe);#SLAM.TrackRGBD(imRGB, imD, tframe, std::string(argv[3]) + "/" + vstrImageFilenamesD[ni]);#' \
	      ${RDS_SLAM_ROOT}/SLAM/Examples/ros_tum_rgbd.cpp \
	 && find ${RDS_SLAM_ROOT}/SLAM ${RDS_SLAM_ROOT}/SegNet_ROS -type f \( -name '*.cc' -o -name '*.cpp' -o -name '*.h' \) \
      -exec sed -i -E \
        -e 's#^([[:space:]]*)cv::(namedWindow|imshow|waitKey)\(#\1// cv::\2(#' {} + \
 && sed -i \
      -e 's/cv_bridge::toCvCopy(mResult->label)/cv_bridge::toCvCopy(mResult->label, sensor_msgs::image_encodings::MONO8)/g' \
      -e 's/cv_bridge::toCvCopy(result->label)/cv_bridge::toCvCopy(result->label, sensor_msgs::image_encodings::MONO8)/g' \
      -e 's/cv_bridge::toCvCopy(result->label_color)/cv_bridge::toCvCopy(result->label_color, sensor_msgs::image_encodings::BGR8)/g' \
      ${RDS_SLAM_ROOT}/SegNet_ROS/include/segnet_ros/segnet_client.h \
      ${RDS_SLAM_ROOT}/SegNet_ROS/src/action_client_node.cpp \
 && perl -0pi -e 's#            cv::Mat label = cv_bridge::toCvCopy\(mResult->label, sensor_msgs::image_encodings::MONO8\)->image;#            const sensor_msgs::Image& label_msg = mResult->label;\n            cv::Mat label;\n            if (label_msg.height > 0 && label_msg.width > 0 && !label_msg.data.empty()) {\n                label = cv::Mat(label_msg.height, label_msg.width, CV_8UC1, const_cast<unsigned char*>(label_msg.data.data()), label_msg.step).clone();\n            } else {\n                label = cv::Mat::zeros(in_images[i].rows, in_images[i].cols, CV_8UC1);\n            }#' \
      ${RDS_SLAM_ROOT}/SegNet_ROS/include/segnet_ros/segnet_client.h \
 && sed -i '/mbFinishRequested = true;/a\    { std::unique_lock<std::mutex> lockKFs(mMutexNewKFs); mlNewKeyFrames.clear(); }\n    { std::unique_lock<std::mutex> lockTrack(mMutexSemanticTrack); mlSemanticTrack.clear(); }' \
      ${RDS_SLAM_ROOT}/SLAM/src/Semantic.cpp \
 && sed -i '/mKernel = getStructuringElement/,/cv::Point(mDilation_size, mDilation_size));/c\    mKernel = cv::Mat::ones(2 * mDilation_size + 1, 2 * mDilation_size + 1, CV_8U);' \
      ${RDS_SLAM_ROOT}/SLAM/src/Semantic.cpp \
 && sed -i \
      's#find_package(OpenCV 3.3.1)#find_package(OpenCV 4.5 REQUIRED PATHS /usr/lib/aarch64-linux-gnu/cmake/opencv4 NO_DEFAULT_PATH)#' \
      ${RDS_SLAM_ROOT}/SLAM/CMakeLists.txt \
 && sed -i \
      's#find_package(OpenCV 3.0 REQUIRED)#find_package(OpenCV 4.5 REQUIRED PATHS /usr/lib/aarch64-linux-gnu/cmake/opencv4 NO_DEFAULT_PATH)#' \
      ${RDS_SLAM_ROOT}/MaskRCNN_ROS/CMakeLists.txt \
 && /bin/bash -lc "source /opt/ros/noetic/setup.bash && \
    cd ${RDS_SLAM_ROOT}/MaskRCNN_ROS/include/MaskRCNN && python3 setup.py install && \
    cd ${RDS_SLAM_ROOT}/SLAM && chmod +x build_thirdparty.sh && ./build_thirdparty.sh && \
    cd /root/catkin_ws && catkin_make -DCMAKE_BUILD_TYPE=Release"

RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc \
 && echo "source /root/catkin_ws/devel/setup.bash" >> /root/.bashrc

WORKDIR /workspace
ENTRYPOINT ["/bin/bash"]
