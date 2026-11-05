#!/usr/bin/env python3
from pathlib import Path
import os

root = Path(os.environ.get("SLAM_ROOT", "/opt/orb_slam2_cuda"))


def replace_file(path, old, new):
    p = root / path
    text = p.read_text()
    if old in text:
        p.write_text(text.replace(old, new))


def insert_after(path, anchor, addition):
    p = root / path
    text = p.read_text()
    if addition in text:
        return
    if anchor not in text:
        raise SystemExit(f"anchor not found in {p}: {anchor!r}")
    p.write_text(text.replace(anchor, anchor + addition, 1))


def instrument_frame_times(path):
    p = root / path
    text = p.read_text()
    if "FrameTrackTimes.txt" in text:
        return
    text = text.replace("#include<fstream>\n", "#include<fstream>\n#include<iomanip>\n", 1)
    block = """    {
        ofstream fFrameTimes("FrameTrackTimes.txt");
        fFrameTimes << "# frame timestamp_s tracking_ms input_period_ms live_fps\\n";
        fFrameTimes << fixed << setprecision(9);
        for(int i=0; i<nImages; i++)
        {
            double T=0;
            if(i<nImages-1)
                T = vTimestamps[i+1]-vTimestamps[i];
            else if(i>0)
                T = vTimestamps[i]-vTimestamps[i-1];
            const double tracking_s = vTimesTrack[i];
            const double tracking_ms = 1000.0 * tracking_s;
            const double live_fps = tracking_s > 0.0 ? 1.0 / tracking_s : 0.0;
            fFrameTimes << i << " " << vTimestamps[i] << " " << tracking_ms << " " << 1000.0*T << " " << live_fps << endl;
        }
    }

"""
    anchor = "    // Stop all threads\n"
    if anchor not in text:
        raise SystemExit(f"anchor not found in {p}: {anchor!r}")
    p.write_text(text.replace(anchor, block + anchor, 1))


cmake = root / "CMakeLists.txt"
text = cmake.read_text()
text = text.replace("find_package(OpenCV 3.2.0 REQUIRED)", "find_package(OpenCV REQUIRED)")
text = text.replace("set(EIGEN3_INCLUDE_DIR /usr/local/include/eigen3)", "set(EIGEN3_INCLUDE_DIR /usr/include/eigen3)")
if "find_library(NVTX_LIBRARY nvToolsExt" not in text:
    text = text.replace(
        "find_package(CUDA QUIET REQUIRED)\n",
        """find_package(CUDA QUIET REQUIRED)
find_library(NVTX_LIBRARY nvToolsExt
    PATHS ${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/lib
          /usr/local/cuda/targets/aarch64-linux/lib
          /usr/local/cuda/lib64)
if(NOT NVTX_LIBRARY)
    set(NVTX_LIBRARY "")
    string(REPLACE "-DUSE_NVTX" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    string(REPLACE "-DUSE_NVTX" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()
""",
        1,
    )
text = text.replace("nvToolsExt\n)", "${NVTX_LIBRARY}\n)")
cmake.write_text(text)

(root / "include" / "helper_cuda.h").write_text(
    r"""#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

inline void campaign_check_cuda(cudaError_t result, const char* call, const char* file, int line) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d: %s returned %s\n", file, line, call, cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}

#define checkCudaErrors(call) campaign_check_cuda((call), #call, __FILE__, __LINE__)
"""
)

replace_file("src/Sim3Solver.cc", "CV_REDUCE_SUM", "cv::REDUCE_SUM")
replace_file("Examples/RGB-D/rgbd_tum.cc", "CV_LOAD_IMAGE_UNCHANGED", "cv::IMREAD_UNCHANGED")
replace_file("include/ORBextractor.h", "#include <opencv/cv.h>", "#include <opencv2/opencv.hpp>")
replace_file("include/ORBextractor.h", "#include <opencv2/cudafilters.hpp>\n", "")
replace_file("include/ORBextractor.h", "    cv::Ptr<cv::cuda::Filter> mpGaussianFilter;\n", "")
replace_file("src/ORBextractor.cc", "#include <opencv2/cudafeatures2d.hpp>\n", "")
replace_file("src/ORBextractor.cc", "#include <opencv2/cudawarping.hpp>\n", "")
replace_file("src/ORBextractor.cc", "#include <opencv2/cudaarithm.hpp>\n", "#include <opencv2/imgproc.hpp>\n")
replace_file(
    "src/ORBextractor.cc",
    "          mpGaussianFilter->apply(gMat, gMat, ic_angle.cvStream());",
    "          ic_angle.cvStream().waitForCompletion();\n          cv::Mat hMat;\n          gMat.download(hMat);\n          cv::GaussianBlur(hMat, hMat, cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);\n          gMat.upload(hMat);",
)
replace_file(
    "src/ORBextractor.cc",
    "    mpGaussianFilter->apply(gMat, gMat, ic_angle.cvStream());",
    "    ic_angle.cvStream().waitForCompletion();\n    cv::Mat hMatLast;\n    gMat.download(hMatLast);\n    cv::GaussianBlur(hMatLast, hMatLast, cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);\n    gMat.upload(hMatLast);",
)
replace_file(
    "src/ORBextractor.cc",
    "    mpGaussianFilter = cv::cuda::createGaussianFilter(mvImagePyramid[0].type(), mvImagePyramid[0].type(), Size(7, 7), 2, 2, BORDER_REFLECT_101);\n",
    "",
)
replace_file(
    "src/ORBextractor.cc",
    """    // Compute the resized image
    if (level != 0) {
      cuda::resize(mvImagePyramid[level-1], mvImagePyramid[level], sz, 0, 0, INTER_LINEAR, mcvStream);
      cuda::copyMakeBorder(mvImagePyramid[level], target, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                            BORDER_REFLECT_101, cv::Scalar(), mcvStream);
    } else {
      cuda::GpuMat gpuImg(image);
      cuda::copyMakeBorder(gpuImg, target, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                            BORDER_REFLECT_101, cv::Scalar(), mcvStream);
    }
""",
    """    // Compute the resized image. JetPack 6 OpenCV packages do not ship CUDA
    // imgproc modules, so use CPU imgproc and upload to GpuMat for the CUDA ORB kernels.
    cv::Mat levelImg;
    if (level != 0) {
      cv::Mat prevLevel;
      mvImagePyramid[level-1].download(prevLevel);
      cv::resize(prevLevel, levelImg, sz, 0, 0, cv::INTER_LINEAR);
    } else {
      levelImg = image;
    }
    cv::Mat bordered;
    cv::copyMakeBorder(levelImg, bordered, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                       cv::BORDER_REFLECT_101);
    mvImagePyramidBorder[level].upload(bordered);
""",
)
replace_file("src/cuda/Fast_gpu.cu", "    cvStream = StreamAccessor::wrapStream(stream);\n", "")
replace_file("src/cuda/Fast_gpu.cu", "    cvStream.~Stream();\n", "")
replace_file("src/cuda/Fast_gpu.cu", "    _cvStream = StreamAccessor::wrapStream(stream);\n", "")
replace_file("src/cuda/Fast_gpu.cu", "    _cvStream.~Stream();\n", "")
replace_file("src/cuda/Fast_gpu.cu", "    scoreMat.setTo(Scalar::all(0), cvStream);", "    scoreMat.setTo(Scalar::all(0));")
replace_file("src/cuda/Orb_gpu.cu", "    cvStream = StreamAccessor::wrapStream(stream);\n", "")
replace_file("src/cuda/Orb_gpu.cu", "    cvStream.~Stream();\n", "")
replace_file("src/cuda/Orb_gpu.cu", "    desc.setTo(Scalar::all(0), cvStream);", "    desc.setTo(Scalar::all(0));")
replace_file("src/cuda/Orb_gpu.cu", "    desc.download(_descriptors, cvStream);", "    desc.download(_descriptors);")
replace_file(
    "include/LoopClosing.h",
    "Eigen::aligned_allocator<std::pair<const KeyFrame*, g2o::Sim3> >",
    "Eigen::aligned_allocator<std::pair<KeyFrame* const, g2o::Sim3> >",
)
insert_after("include/PnPsolver.h", "#include <opencv2/core/core.hpp>\n", "#include <opencv2/core/core_c.h>\n")

insert_after("src/Tracking.cc", "#include <opencv2/features2d/features2d.hpp>\n", "#include <opencv2/imgproc/types_c.h>\n")
insert_after("src/FrameDrawer.cc", "#include <opencv2/highgui/highgui.hpp>\n", "#include <opencv2/imgproc/types_c.h>\n")
insert_after("Examples/RGB-D/rgbd_tum.cc", "#include<opencv2/core/core.hpp>\n", "#include <opencv2/imgcodecs.hpp>\n")
instrument_frame_times("Examples/RGB-D/rgbd_tum.cc")
