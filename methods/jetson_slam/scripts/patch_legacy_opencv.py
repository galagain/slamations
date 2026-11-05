#!/usr/bin/env python3
from pathlib import Path
import os

root = Path(os.environ.get("SLAM_ROOT", "/opt/jetson_slam"))


def replace(path, old, new):
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


replace("src/Sim3Solver.cpp", "CV_REDUCE_SUM", "cv::REDUCE_SUM")
replace("src/System.cpp", "setprecision(0) << *lT", "setprecision(6) << *lT")
replace("Examples/RGB-D/rgbd_tum.cpp", "CV_LOAD_IMAGE_UNCHANGED", "cv::IMREAD_UNCHANGED")
replace("include/ORBextractor.h", "using namespace cv;\n", "")

insert_after(
    "src/FrameDrawer.cpp",
    "#include <opencv2/highgui/highgui.hpp>\n",
    "#include <opencv2/imgproc/types_c.h>\n",
)
insert_after(
    "src/Tracking.cpp",
    "#include <opencv2/imgproc/imgproc.hpp>\n",
    "#include <opencv2/imgproc/types_c.h>\n",
)
insert_after(
    "Examples/RGB-D/rgbd_tum.cpp",
    "#include<opencv2/core/core.hpp>\n",
    "#include <opencv2/imgcodecs.hpp>\n",
)

for yaml_name in ("TUM1.yaml", "TUM2.yaml", "TUM3.yaml"):
    p = root / "Examples" / "RGB-D" / yaml_name
    text = p.read_text()
    if "gpu.use_gpu:" not in text:
        text = text.replace(
            "# Camera Parameters. Adjust them!\n",
            "# GPU Parameters\n"
            "gpu.use_gpu: 1\n"
            "gpu.is_jetson: 0\n\n"
            "# Camera Parameters. Adjust them!\n",
            1,
        )
    extra = {
        "ORBextractor.tile_h": "15",
        "ORBextractor.tile_w": "15",
        "ORBextractor.fixed_multi_scale_tile_size": "0",
        "ORBextractor.apply_nms_ms": "0",
        "ORBextractor.nms_ms_mode_gpu": "1",
        "ORBextractor.th_FAST_MIN": "7",
        "ORBextractor.th_FAST_MAX": "20",
        "ORBextractor.FAST_N_MIN": "9",
        "ORBextractor.FAST_N_MAX": "14",
    }
    if "ORBextractor.th_FAST_MAX:" not in text:
        insert = "\n# Jetson-SLAM GPU ORB parameters\n" + "\n".join(f"{k}: {v}" for k, v in extra.items()) + "\n"
        text = text.replace("ORBextractor.minThFAST: 7\n", "ORBextractor.minThFAST: 7\n" + insert, 1)
    p.write_text(text)

instrument_frame_times("Examples/RGB-D/rgbd_tum.cpp")
