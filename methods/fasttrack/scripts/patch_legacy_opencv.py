#!/usr/bin/env python3
from pathlib import Path
import os

root = Path(os.environ.get("SLAM_ROOT", "/opt/fasttrack"))


def instrument_frame_times(path):
    p = root / path
    text = p.read_text()
    if "FrameTrackTimes.txt" in text:
        return text
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


p = root / "src" / "LoopClosing.cc"
text = p.read_text()
text = text.replace("CV_LOAD_IMAGE_UNCHANGED", "cv::IMREAD_UNCHANGED")
text = text.replace("CV_GRAY2BGR", "cv::COLOR_GRAY2BGR")
text = text.replace("CV_FONT_HERSHEY_DUPLEX", "cv::FONT_HERSHEY_DUPLEX")
if "#include <opencv2/imgcodecs.hpp>" not in text:
    text = text.replace("#include\"LoopClosing.h\"\n", "#include\"LoopClosing.h\"\n#include <opencv2/imgcodecs.hpp>\n#include <opencv2/imgproc.hpp>\n", 1)
p.write_text(text)

p = root / "Examples" / "RGB-D" / "rgbd_tum.cc"
text = p.read_text()
text = text.replace(
    "ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::RGBD,true);",
    "ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::RGBD,false);",
)
p.write_text(text)
instrument_frame_times("Examples/RGB-D/rgbd_tum.cc")

cmake = root / "CMakeLists.txt"
text = cmake.read_text()
old_examples = text.find("#Old examples")
if old_examples != -1:
    text = text[:old_examples]
cmake.write_text(text)
