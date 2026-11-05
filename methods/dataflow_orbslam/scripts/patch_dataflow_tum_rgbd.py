#!/usr/bin/env python3
from pathlib import Path
import os

root = Path(os.environ.get("SLAM_ROOT", "/opt/dataflow_orbslam"))


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
text = text.replace(
    "find_package(OpenCV 3.4.3 REQUIRED HINTS /usr/local/opencv-3.4.3)",
    "find_package(OpenCV REQUIRED)",
)
text = text.replace(
    """if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()
""",
    """if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV not found.")
endif()
""",
)
rgbd_marker = "# Campaign TUM RGB-D executable"
if rgbd_marker not in text:
    text += f"""

{rgbd_marker}
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${{PROJECT_SOURCE_DIR}}/bin/build${{CUSTOM}}${{EXECUTOR}}${{PIPELINE}}${{WAIT}}_${{SYNC_TYPE}}/RGB-D)
cuda_add_executable(rgbd_tum
Examples/RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd_tum ${{PROJECT_NAME}})
"""
cmake.write_text(text)

dbow2_cmake = root / "Thirdparty" / "DBoW2" / "CMakeLists.txt"
text = dbow2_cmake.read_text()
text = text.replace("find_package(OpenCV 3.0 QUIET)", "find_package(OpenCV REQUIRED)")
text = text.replace(
    """if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()
""",
    """if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV not found.")
endif()
""",
)
dbow2_cmake.write_text(text)

replace_file("src/Sim3Solver.cc", "CV_REDUCE_SUM", "cv::REDUCE_SUM")
replace_file("Examples/RGB-D/rgbd_tum.cc", "CV_LOAD_IMAGE_UNCHANGED", "cv::IMREAD_UNCHANGED")
replace_file(
    "Examples/RGB-D/rgbd_tum.cc",
    "ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::RGBD,true);",
    "ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::RGBD,false);",
)
replace_file("Examples/RGB-D/rgbd_tum.cc", "SLAM.TrackRGBD(imRGB,imD,tframe);", "SLAM.TrackRGBD(imRGB,imD,tframe,0);")
replace_file("include/ORBextractor.h", "#include <opencv/cv.h>", "#include <opencv2/opencv.hpp>")
replace_file(
    "include/LoopClosing.h",
    "Eigen::aligned_allocator<std::pair<const KeyFrame*, g2o::Sim3> >",
    "Eigen::aligned_allocator<std::pair<KeyFrame* const, g2o::Sim3> >",
)
insert_after("include/PnPsolver.h", "#include <opencv2/core/core.hpp>\n", "#include <opencv2/core/core_c.h>\n")
replace_file("custom_openvx/kernels.hpp", "#include <opencv2/cudafilters.hpp>\n", "")
replace_file("custom_openvx/kernels.hpp", "\tcv::Ptr<cv::cuda::Filter> filter;\n", "")
replace_file(
    "custom_openvx/kernels.hpp",
    """\t\tfilter = cv::cuda::createGaussianFilter(dst->imgDevice.type(),
\t\t\t\t\tdst->imgDevice.type(), cv::Size(7,7), 2, 2);
""",
    "",
)
replace_file("custom_openvx/scale.cc", "#include <opencv2/cudawarping.hpp>\n", "")
replace_file("custom_openvx/scale.cc", "    cv::cuda::Stream stream2 = cv::cuda::StreamAccessor::wrapStream(stream);\n", "")
replace_file("custom_openvx/gaussian.cc", "cv::cuda::Stream stream2 = cv::cuda::StreamAccessor::wrapStream(stream);\n\t", "")
replace_file(
    "custom_openvx/scale.cc",
    "//scale_gpu(d_img, src->cols, src->rows, src->step, d_img_scaled, dst->cols, dst->rows, dst->step, invX, invY, stream, computationFinished);\n\t\tcv::cuda::resize(imgsrc, imgdst, imgdst.size(), 0, 0, cv::INTER_LINEAR, stream2);",
    "scale_gpu(d_img, src->cols, src->rows, src->step, d_img_scaled, dst->cols, dst->rows, dst->step, invX, invY, stream, computationFinished);",
)
replace_file(
    "custom_openvx/gaussian.cc",
    "//gaussian_gpu(d_img, src->cols, src->rows, src->step, d_img_scaled, stream, computationFinished);\n\t\tfilter->apply(imgsrc, imgdst, stream2);",
    "gaussian_gpu(d_img, src->cols, src->rows, src->step, d_img_scaled, stream, computationFinished);",
)
replace_file(
    "Examples/RGB-D/rgbd_tum.cc",
    'SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");',
    'SLAM.SaveTrajectoryTUM("CameraTrajectory.txt", 1.0);',
)
replace_file(
    "Examples/RGB-D/rgbd_tum.cc",
    'SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");',
    'SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt", 1.0);',
)

insert_after("src/Tracking.cc", "#include<opencv2/features2d/features2d.hpp>\n", "#include <opencv2/imgproc/types_c.h>\n")
insert_after("src/FrameDrawer.cc", "#include <opencv2/highgui/highgui.hpp>\n", "#include <opencv2/imgproc/types_c.h>\n")
insert_after("Examples/RGB-D/rgbd_tum.cc", "#include<opencv2/core/core.hpp>\n", "#include <opencv2/imgcodecs.hpp>\n")
insert_after("Examples/RGB-D/rgbd_tum.cc", "#include<chrono>\n", "#include <unistd.h>\n")
instrument_frame_times("Examples/RGB-D/rgbd_tum.cc")
