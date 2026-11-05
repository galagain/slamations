#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import re
import sys
import textwrap


TIMESTAMP_H = r"""
/*
 * File: Timestamp.h
 * Author: Dorian Galvez-Lopez
 * Date: March 2009
 * Description: timestamping functions
 * License: see the LICENSE.txt file
 */

#ifndef __D_TIMESTAMP__
#define __D_TIMESTAMP__

#include <iostream>
using namespace std;

namespace DUtils {

class Timestamp
{
public:
  enum tOptions
  {
    NONE = 0,
    CURRENT_TIME = 0x1,
    ZERO = 0x2
  };

public:
  Timestamp(Timestamp::tOptions option = NONE);
  virtual ~Timestamp(void);
  bool empty() const;
  void setToCurrentTime();

  inline void setTime(unsigned long secs, unsigned long usecs)
  {
    m_secs = secs;
    m_usecs = usecs;
  }

  inline void getTime(unsigned long &secs, unsigned long &usecs) const
  {
    secs = m_secs;
    usecs = m_usecs;
  }

  void setTime(const string &stime);
  void setTime(double s);
  double getFloatTime() const;
  string getStringTime() const;
  double operator- (const Timestamp &t) const;
  Timestamp plus(unsigned long s, unsigned long us) const;
  Timestamp minus(unsigned long s, unsigned long us) const;
  Timestamp& operator+= (double s);
  Timestamp& operator-= (double s);
  Timestamp operator+ (double s) const;
  Timestamp operator- (double s) const;
  bool operator> (const Timestamp &t) const;
  bool operator>= (const Timestamp &t) const;
  bool operator== (const Timestamp &t) const;
  bool operator< (const Timestamp &t) const;
  bool operator<= (const Timestamp &t) const;
  string Format(bool machine_friendly = false) const;
  static string Format(double s);

protected:
  unsigned long m_secs;
  unsigned long m_usecs;
};

}

#endif
"""


TIMESTAMP_CPP = r"""
/*
 * File: Timestamp.cpp
 * Author: Dorian Galvez-Lopez
 * Date: March 2009
 * Description: timestamping functions
 * License: see the LICENSE.txt file
 */

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <sstream>
#include <iomanip>

#ifdef _WIN32
#ifndef WIN32
#define WIN32
#endif
#endif

#ifdef WIN32
#include <sys/timeb.h>
#define sprintf sprintf_s
#else
#include <sys/time.h>
#endif

#include "Timestamp.h"

using namespace std;
using namespace DUtils;

Timestamp::Timestamp(Timestamp::tOptions option)
{
  if(option & CURRENT_TIME)
    setToCurrentTime();
  else if(option & ZERO)
    setTime(0.);
}

Timestamp::~Timestamp(void)
{
}

bool Timestamp::empty() const
{
  return m_secs == 0 && m_usecs == 0;
}

void Timestamp::setToCurrentTime()
{
#ifdef WIN32
  struct __timeb32 timebuffer;
  _ftime32_s(&timebuffer);
  m_secs = timebuffer.time;
  m_usecs = timebuffer.millitm * 1000;
#else
  struct timeval now;
  gettimeofday(&now, NULL);
  m_secs = now.tv_sec;
  m_usecs = now.tv_usec;
#endif
}

void Timestamp::setTime(const string &stime)
{
  string::size_type p = stime.find('.');
  if(p == string::npos)
  {
    m_secs = atol(stime.c_str());
    m_usecs = 0;
  }
  else
  {
    m_secs = atol(stime.substr(0, p).c_str());
    string s_usecs = stime.substr(p + 1, 6);
    m_usecs = atol(stime.substr(p + 1).c_str());
    m_usecs *= (unsigned long)pow(10.0, double(6 - s_usecs.length()));
  }
}

void Timestamp::setTime(double s)
{
  m_secs = (unsigned long)s;
  m_usecs = (s - (double)m_secs) * 1e6;
}

double Timestamp::getFloatTime() const
{
  return double(m_secs) + double(m_usecs) / 1000000.0;
}

string Timestamp::getStringTime() const
{
  char buf[32];
  sprintf(buf, "%.6lf", this->getFloatTime());
  return string(buf);
}

double Timestamp::operator- (const Timestamp &t) const
{
  return this->getFloatTime() - t.getFloatTime();
}

Timestamp& Timestamp::operator+= (double s)
{
  *this = *this + s;
  return *this;
}

Timestamp& Timestamp::operator-= (double s)
{
  *this = *this - s;
  return *this;
}

Timestamp Timestamp::operator+ (double s) const
{
  unsigned long secs = (long)floor(s);
  unsigned long usecs = (long)((s - (double)secs) * 1e6);
  return this->plus(secs, usecs);
}

Timestamp Timestamp::plus(unsigned long secs, unsigned long usecs) const
{
  Timestamp t;
  const unsigned long max = 1000000ul;
  if(m_usecs + usecs >= max)
    t.setTime(m_secs + secs + 1, m_usecs + usecs - max);
  else
    t.setTime(m_secs + secs, m_usecs + usecs);
  return t;
}

Timestamp Timestamp::operator- (double s) const
{
  unsigned long secs = (long)floor(s);
  unsigned long usecs = (long)((s - (double)secs) * 1e6);
  return this->minus(secs, usecs);
}

Timestamp Timestamp::minus(unsigned long secs, unsigned long usecs) const
{
  Timestamp t;
  const unsigned long max = 1000000ul;
  if(m_usecs < usecs)
    t.setTime(m_secs - secs - 1, max - (usecs - m_usecs));
  else
    t.setTime(m_secs - secs, m_usecs - usecs);
  return t;
}

bool Timestamp::operator> (const Timestamp &t) const
{
  if(m_secs > t.m_secs) return true;
  else if(m_secs == t.m_secs) return m_usecs > t.m_usecs;
  else return false;
}

bool Timestamp::operator>= (const Timestamp &t) const
{
  if(m_secs > t.m_secs) return true;
  else if(m_secs == t.m_secs) return m_usecs >= t.m_usecs;
  else return false;
}

bool Timestamp::operator< (const Timestamp &t) const
{
  if(m_secs < t.m_secs) return true;
  else if(m_secs == t.m_secs) return m_usecs < t.m_usecs;
  else return false;
}

bool Timestamp::operator<= (const Timestamp &t) const
{
  if(m_secs < t.m_secs) return true;
  else if(m_secs == t.m_secs) return m_usecs <= t.m_usecs;
  else return false;
}

bool Timestamp::operator== (const Timestamp &t) const
{
  return(m_secs == t.m_secs && m_usecs == t.m_usecs);
}

string Timestamp::Format(bool machine_friendly) const
{
  struct tm tm_time;
  time_t t = (time_t)getFloatTime();
#ifdef WIN32
  localtime_s(&tm_time, &t);
#else
  localtime_r(&t, &tm_time);
#endif

  char buffer[128];
  if(machine_friendly)
    strftime(buffer, 128, "%Y%m%d_%H%M%S", &tm_time);
  else
    strftime(buffer, 128, "%c", &tm_time);
  return string(buffer);
}

string Timestamp::Format(double s)
{
  int days = int(s / (24. * 3600.0));
  s -= days * (24. * 3600.0);
  int hours = int(s / 3600.0);
  s -= hours * 3600;
  int minutes = int(s / 60.0);
  s -= minutes * 60;
  int seconds = int(s);
  int ms = int((s - seconds) * 1e6);

  stringstream ss;
  ss.fill('0');
  bool b;
  if((b = (days > 0))) ss << days << "d ";
  if((b = (b || hours > 0))) ss << setw(2) << hours << ":";
  if((b = (b || minutes > 0))) ss << setw(2) << minutes << ":";
  if(b) ss << setw(2);
  ss << seconds;
  if(!b) ss << "." << setw(6) << ms;

  return ss.str();
}
"""


RGBD_TUM_CC = r"""
/**
* This file is part of ORB-SLAM3 / Nitro-SLAM.
*/

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unistd.h>

#include <opencv2/core/core.hpp>

#include <System.h>
#include <Stats/TrackingStats.h>
#include <Stats/LocalMappingStats.h>
#include <Stats/LoopClosingStats.h>
#include "Kernels/TrackingKernelController.h"
#include "Kernels/MappingKernelController.h"
#include "Kernels/LoopClosingKernelController.h"

using namespace std;

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);

static bool env_flag_enabled(const char *name, bool default_value)
{
    const char *value = std::getenv(name);
    if(!value)
        return default_value;
    return string(value) != "0";
}

int main(int argc, char **argv)
{
    if(argc != 5 && argc != 12 && argc != 13)
    {
        cerr << endl
             << "Usage: ./rgbd_tum path_to_vocabulary path_to_settings path_to_sequence path_to_association "
             << "[strStatsFile FastTrack[0|1] TurboMap[0|1] FastLoop[0|1] kernel_status_FT kernel_status_TM kernel_status_FL adaptive_status_AD]"
             << endl;
        return 1;
    }

    const bool has_flags = (argc == 12 || argc == 13);
    const string strStatsFile = has_flags ? string(argv[5]) : string();
    const bool run_FastTrack = has_flags && (strcmp(argv[6], "1") == 0);
    const bool run_TurboMap = has_flags && (strcmp(argv[7], "1") == 0);
    const bool run_FastLoop = has_flags && (strcmp(argv[8], "1") == 0);

    if(!run_FastTrack && !run_TurboMap && !run_FastLoop)
        cout << "Running the original ORB-SLAM3 RGB-D code..." << endl;
    else
        cout << "Running FastTrack->" << run_FastTrack
             << ", TurboMap->" << run_TurboMap
             << ", FastLoop->" << run_FastLoop << endl;

    if(has_flags)
    {
        const string kernelFT = argv[9];
        const string kernelTM = argv[10];
        const string kernelFL = argv[11];
        const string kernelAD = (argc == 13) ? string(argv[12]) : string("00000");

        if(kernelFT.size() < 5 || kernelTM.size() < 4 || kernelFL.size() < 5 || kernelAD.size() < 5)
        {
            cerr << "Invalid kernel mask length." << endl;
            return 1;
        }

        const bool AD_featureBudgetEnabled = (kernelAD[0] == '1');
        const bool AD_poseSchedulerEnabled = (kernelAD[1] == '1');
        const bool AD_warmupRescueEnabled = (kernelAD[2] == '1');
        const bool AD_lowSupportLockEnabled = (kernelAD[3] == '1');
        const bool AD_cudaFallbackEnabled = (kernelAD[4] == '1');

        setenv("NITRO_ADAPTIVE_MASK", kernelAD.c_str(), 1);
        setenv("NITRO_ADAPTIVE_FEATURE_BUDGET", AD_featureBudgetEnabled ? "1" : "0", 1);
        setenv("NITRO_ADAPTIVE_POSE_SCHEDULER", AD_poseSchedulerEnabled ? "1" : "0", 1);
        setenv("NITRO_ADAPTIVE_WARMUP_RESCUE", AD_warmupRescueEnabled ? "1" : "0", 1);
        setenv("NITRO_ADAPTIVE_LOW_SUPPORT_LOCK", AD_lowSupportLockEnabled ? "1" : "0", 1);
        setenv("NITRO_ADAPTIVE_CUDA_FALLBACK", AD_cudaFallbackEnabled ? "1" : "0", 1);
        cout << "Activated Adaptive mask: " << kernelAD << endl;

        if(run_FastTrack)
        {
            const bool FT_orbExtractionEnabled = (kernelFT[0] == '1');
            const bool FT_stereoMatchEnabled = (kernelFT[1] == '1');
            const bool FT_searchLocalPointsEnabled = (kernelFT[2] == '1');
            const bool FT_poseEstimationEnabled = (kernelFT[3] == '1');
            const bool FT_poseOptimizationEnabled = (kernelFT[4] == '1');

            TrackingKernelController::activate();
            TrackingKernelController::setGPURunMode(
                FT_orbExtractionEnabled,
                FT_stereoMatchEnabled,
                FT_searchLocalPointsEnabled,
                FT_poseEstimationEnabled,
                FT_poseOptimizationEnabled
            );
            cout << "Activated FastTrack mask: " << kernelFT << endl;
        }

        if(run_TurboMap)
        {
            const bool TM_searchForTriangulationEnabled = (kernelTM[0] == '1');
            const bool TM_fuseEnabled = (kernelTM[1] == '1');
            const bool TM_keyframeCullingEnabled = (kernelTM[2] == '1');
            const bool TM_LBAEnabled = (kernelTM[3] == '1');

            MappingKernelController::activate();
            MappingKernelController::setGPURunMode(
                TM_searchForTriangulationEnabled,
                TM_fuseEnabled,
                TM_keyframeCullingEnabled,
                TM_LBAEnabled
            );
            cout << "Activated TurboMap mask: " << kernelTM << endl;
        }

        if(run_FastLoop)
        {
            const bool FL_mergedSearchByProjectionEnabled = (kernelFL[0] == '1');
            const bool FL_merged3SearchByProjectionEnabled = (kernelFL[1] == '1');
            const bool FL_searchAndFuseEnabled = (kernelFL[2] == '1');
            const bool FL_singleSearchByProjectionEnabled = (kernelFL[3] == '1');
            const bool FL_graphOptimizationEnabled = (kernelFL[4] == '1');

            LoopClosingKernelController::activate();
            LoopClosingKernelController::setGPURunMode(
                FL_mergedSearchByProjectionEnabled,
                FL_merged3SearchByProjectionEnabled,
                FL_searchAndFuseEnabled,
                FL_singleSearchByProjectionEnabled,
                FL_graphOptimizationEnabled
            );
            cout << "Activated FastLoop mask: " << kernelFL << endl;
        }
    }

    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;
    string strAssociationFilename = string(argv[4]);
    LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);

    const int nImages = vstrImageFilenamesRGB.size();
    if(vstrImageFilenamesRGB.empty())
    {
        cerr << endl << "No images found in provided path." << endl;
        return 1;
    }
    else if(vstrImageFilenamesD.size() != vstrImageFilenamesRGB.size())
    {
        cerr << endl << "Different number of images for rgb and depth." << endl;
        return 1;
    }

    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::RGBD, false);
    float imageScale = SLAM.GetImageScale();

    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    const bool realtime_sleep = env_flag_enabled("NITRO_REALTIME_SLEEP", false);
    cv::Mat imRGB, imD;
    for(int ni = 0; ni < nImages; ni++)
    {
        imRGB = cv::imread(string(argv[3]) + "/" + vstrImageFilenamesRGB[ni], cv::IMREAD_UNCHANGED);
        imD = cv::imread(string(argv[3]) + "/" + vstrImageFilenamesD[ni], cv::IMREAD_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(imRGB.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenamesRGB[ni] << endl;
            return 1;
        }

        if(imageScale != 1.f)
        {
            int width = imRGB.cols * imageScale;
            int height = imRGB.rows * imageScale;
            cv::resize(imRGB, imRGB, cv::Size(width, height));
            cv::resize(imD, imD, cv::Size(width, height));
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        SLAM.TrackRGBD(imRGB, imD, tframe);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        vTimesTrack[ni] = ttrack;

#ifdef REGISTER_TRACKING_STATS
        double ttrack_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(t2 - t1).count();
        TrackingStats::getInstance().tracking_time.emplace_back((long unsigned int)ni, ttrack_ms);
#endif

        double T = 0;
        if(ni < nImages - 1)
            T = vTimestamps[ni + 1] - tframe;
        else if(ni > 0)
            T = tframe - vTimestamps[ni - 1];

        if(realtime_sleep && ttrack < T)
            usleep((T - ttrack) * 1e6);
    }

    SLAM.Shutdown();

    if(has_flags && !strStatsFile.empty())
    {
#ifdef REGISTER_TRACKING_STATS
        TrackingStats::getInstance().saveStats(strStatsFile);
#endif
#ifdef REGISTER_LOCAL_MAPPING_STATS
        LocalMappingStats::getInstance().saveStats(strStatsFile);
#endif
#ifdef REGISTER_LOOP_CLOSING_STATS
        LoopClosingStats::getInstance().saveStats(strStatsFile);
#endif
    }

    sort(vTimesTrack.begin(), vTimesTrack.end());
    float totaltime = 0;
    for(int ni = 0; ni < nImages; ni++)
        totaltime += vTimesTrack[ni];

    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages / 2] << endl;
    cout << "mean tracking time: " << totaltime / nImages << endl;

    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    return 0;
}

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps)
{
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while(!fAssociation.eof())
    {
        string s;
        getline(fAssociation, s);
        if(!s.empty() && s[0] != '#')
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);
        }
    }
}
"""


def patch_graphite_cmake(root: pathlib.Path) -> None:
    cmake = root / "Thirdparty" / "graphite" / "CMakeLists.txt"
    text = cmake.read_text()
    text = re.sub(r"\nfind_package\(cudss 0\.7\.0 REQUIRED\)", "", text)
    text = text.replace(" CUDA::cublas Eigen3::Eigen cudss)", " CUDA::cublas Eigen3::Eigen)")
    cmake.write_text(text)


def patch_nitro_cmake(root: pathlib.Path) -> None:
    cmake = root / "CMakeLists.txt"
    text = cmake.read_text()
    text = text.replace("add_subdirectory(Thirdparty/graphite)\nset(BUILD_EXAMPLES OFF)", "set(BUILD_EXAMPLES OFF CACHE BOOL \"\" FORCE)\nset(BUILD_DOCS OFF CACHE BOOL \"\" FORCE)\nset(BUILD_TESTS OFF CACHE BOOL \"\" FORCE)\nadd_subdirectory(Thirdparty/graphite)")
    cmake.write_text(text)


def patch_compute_engine_cmake(root: pathlib.Path) -> None:
    cmake = root / "Thirdparty" / "g2o" / "compute-engine" / "CMakeLists.txt"
    text = cmake.read_text()
    text = re.sub(
        r"\n# Benchmarks\nadd_executable\(sbm_bench.*?gtest_discover_tests\(tests PROPERTIES DISCOVERY_TIMEOUT 600\)\n",
        "\n# Benchmarks and tests are disabled in the Docker launcher build.\n",
        text,
        flags=re.S,
    )
    cmake.write_text(text)


def patch_rgbd_tum(root: pathlib.Path) -> None:
    source = root / "Examples" / "RGB-D" / "rgbd_tum.cc"
    if source.exists():
        source.write_text(textwrap.dedent(RGBD_TUM_CC).lstrip())


def replace_once(text: str, old: str, new: str, path: pathlib.Path) -> str:
    if old not in text:
        raise RuntimeError(f"Patch anchor not found in {path}: {old[:80]!r}")
    return text.replace(old, new, 1)


def patch_rgbd_cuda_context(root: pathlib.Path) -> None:
    tracking = root / "src" / "Tracking.cc"
    text = tracking.read_text()
    text = replace_once(
        text,
        """    if(mSensor==System::STEREO || mSensor==System::IMU_STEREO) {
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST, settings->newImSize().width, settings->newImSize().height);
        CudaUtils::loadSetting(nFeatures, nLevels, false, fScaleFactor, settings->newImSize().width, settings->newImSize().height, cameraIsFisheye);
    }
    if(mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR){
        mpIniORBextractor = new ORBextractor(5*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST, settings->newImSize().width, settings->newImSize().height);
        CudaUtils::loadSetting(5*nFeatures, nLevels, true, fScaleFactor, settings->newImSize().width, settings->newImSize().height, cameraIsFisheye);
    }
""",
        """    if(mSensor==System::STEREO || mSensor==System::IMU_STEREO) {
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST, settings->newImSize().width, settings->newImSize().height);
        CudaUtils::loadSetting(nFeatures, nLevels, false, fScaleFactor, settings->newImSize().width, settings->newImSize().height, cameraIsFisheye);
    }
    if(mSensor==System::RGBD || mSensor==System::IMU_RGBD) {
        CudaUtils::loadSetting(nFeatures, nLevels, false, fScaleFactor, settings->newImSize().width, settings->newImSize().height, cameraIsFisheye);
    }
    if(mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR){
        mpIniORBextractor = new ORBextractor(5*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST, settings->newImSize().width, settings->newImSize().height);
        CudaUtils::loadSetting(5*nFeatures, nLevels, true, fScaleFactor, settings->newImSize().width, settings->newImSize().height, cameraIsFisheye);
    }
""",
        tracking,
    )
    tracking.write_text(text)

    cuda_frame = root / "src" / "Kernels" / "CudaWrappers" / "CudaFrame.cu"
    text = cuda_frame.read_text()
    text = replace_once(
        text,
        """#include "Kernels/CudaWrappers/CudaFrame.h"
#include <cstdio>
#include <vector>
""",
        """#include "Kernels/CudaWrappers/CudaFrame.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>
""",
        cuda_frame,
    )
    text = replace_once(
        text,
        """    CudaFrame::CudaFrame() {
        initializeMemory();
    }
""",
        """    CudaFrame::CudaFrame()
        : mvKeysIsOnGpu(false), mvKeysRightIsOnGpu(false), mDescriptorsIsOnGpu(false) {
        initializeMemory();
    }
""",
        cuda_frame,
    )
    text = replace_once(
        text,
        """        checkCudaError(cudaMemcpy(mvuRight, F.mvuRight.data(), mvuRight_size * sizeof(float), cudaMemcpyHostToDevice), "CudaFrame:: Failed to copy mvuRight to gpu");
""",
        """        if (mvuRight_size > 0)
            checkCudaError(cudaMemcpy(mvuRight, F.mvuRight.data(), mvuRight_size * sizeof(float), cudaMemcpyHostToDevice), "CudaFrame:: Failed to copy mvuRight to gpu");
""",
        cuda_frame,
    )
    text = replace_once(
        text,
        """            checkCudaError(cudaMemcpy((void*) mDescriptors, F.mDescriptors.data,  F.mDescriptors.rows * DESCRIPTOR_SIZE * sizeof(uint8_t), cudaMemcpyHostToDevice), "CudaFrame:: Failed to copy mDescriptors to gpu"); 
""",
        """            if (F.mDescriptors.rows > 0)
                checkCudaError(cudaMemcpy((void*) mDescriptors, F.mDescriptors.data,  F.mDescriptors.rows * DESCRIPTOR_SIZE * sizeof(uint8_t), cudaMemcpyHostToDevice), "CudaFrame:: Failed to copy mDescriptors to gpu"); 
""",
        cuda_frame,
    )
    text = replace_once(
        text,
        """            checkCudaError(cudaMemcpy((void*) mvKeys, tmp_mvKeys.data(), mvKeys_size * sizeof(CudaKeyPoint), cudaMemcpyHostToDevice), "CudaFrame:: Failed to copy mvKeys to gpu");
""",
        """            if (mvKeys_size > 0)
                checkCudaError(cudaMemcpy((void*) mvKeys, tmp_mvKeys.data(), mvKeys_size * sizeof(CudaKeyPoint), cudaMemcpyHostToDevice), "CudaFrame:: Failed to copy mvKeys to gpu");
""",
        cuda_frame,
    )
    text = replace_once(
        text,
        """            checkCudaError(cudaMemcpy((void*) mvKeysRight, tmp_mvKeysRight.data(), mvKeysRight_size * sizeof(CudaKeyPoint), cudaMemcpyHostToDevice), "CudaFrame:: Failed to copy mvKeysRight to gpu");
""",
        """            if (mvKeysRight_size > 0)
                checkCudaError(cudaMemcpy((void*) mvKeysRight, tmp_mvKeysRight.data(), mvKeysRight_size * sizeof(CudaKeyPoint), cudaMemcpyHostToDevice), "CudaFrame:: Failed to copy mvKeysRight to gpu");
""",
        cuda_frame,
    )
    text = replace_once(
        text,
        """        checkCudaError(cudaMemcpy(mvpMapPoints, tmp_mvpMapPoints.data(), tmp_mvpMapPoints.size() * sizeof(CudaMapPoint), cudaMemcpyHostToDevice), "CudaFrame:: Failed to copy mvpMapPoints to gpu");
""",
        """        if (!tmp_mvpMapPoints.empty())
            checkCudaError(cudaMemcpy(mvpMapPoints, tmp_mvpMapPoints.data(), tmp_mvpMapPoints.size() * sizeof(CudaMapPoint), cudaMemcpyHostToDevice), "CudaFrame:: Failed to copy mvpMapPoints to gpu");
""",
        cuda_frame,
    )
    text = replace_once(
        text,
        """        checkCudaError(cudaMemcpy(mvKeysUn, tmp_mvKeysUn.data(), mvKeysUn_size * sizeof(CudaKeyPoint), cudaMemcpyHostToDevice), "CudaFrame:: Failed to copy mvKeysUn to gpu");
""",
        """        if (mvKeysUn_size > 0)
            checkCudaError(cudaMemcpy(mvKeysUn, tmp_mvKeysUn.data(), mvKeysUn_size * sizeof(CudaKeyPoint), cudaMemcpyHostToDevice), "CudaFrame:: Failed to copy mvKeysUn to gpu");
""",
        cuda_frame,
    )
    text = replace_once(
        text,
        """                size_t num_keypoints = F.mGrid[i][j].size();
                if (num_keypoints > 0) {
                    std::memcpy(&flatMGrid[(i * FRAME_GRID_ROWS + j) * keypoints_per_cell], F.mGrid[i][j].data(), num_keypoints * sizeof(std::size_t));
                }
                flatMGrid_size[i * FRAME_GRID_ROWS + j] = num_keypoints;
""",
        """                size_t num_keypoints = std::min(F.mGrid[i][j].size(), static_cast<size_t>(keypoints_per_cell));
                if (num_keypoints > 0) {
                    std::memcpy(&flatMGrid[(i * FRAME_GRID_ROWS + j) * keypoints_per_cell], F.mGrid[i][j].data(), num_keypoints * sizeof(std::size_t));
                }
                flatMGrid_size[i * FRAME_GRID_ROWS + j] = num_keypoints;
""",
        cuda_frame,
    )
    text = replace_once(
        text,
        """                    size_t num_keypoints = F.mGridRight[i][j].size();
                    if (num_keypoints > 0) {
                        std::memcpy(&flatMGridRight[(i * FRAME_GRID_ROWS + j) * KEYPOINTS_PER_CELL], F.mGridRight[i][j].data(), num_keypoints * sizeof(std::size_t));
                    }
                    flatMGridRight_size[i * FRAME_GRID_ROWS + j] = num_keypoints;
""",
        """                    size_t num_keypoints = std::min(F.mGridRight[i][j].size(), static_cast<size_t>(KEYPOINTS_PER_CELL));
                    if (num_keypoints > 0) {
                        std::memcpy(&flatMGridRight[(i * FRAME_GRID_ROWS + j) * KEYPOINTS_PER_CELL], F.mGridRight[i][j].data(), num_keypoints * sizeof(std::size_t));
                    }
                    flatMGridRight_size[i * FRAME_GRID_ROWS + j] = num_keypoints;
""",
        cuda_frame,
    )
    text = replace_once(
        text,
        """        checkCudaError(cudaMemcpy(mvbOutlier, intVec.data(),  intVec.size() * sizeof(uint8_t), cudaMemcpyHostToDevice), "CudaFrame:: Failed to copy mvbOutlier to gpu");
""",
        """        if (!intVec.empty())
            checkCudaError(cudaMemcpy(mvbOutlier, intVec.data(),  intVec.size() * sizeof(uint8_t), cudaMemcpyHostToDevice), "CudaFrame:: Failed to copy mvbOutlier to gpu");
""",
        cuda_frame,
    )
    cuda_frame.write_text(text)

    cuda_keyframe = root / "src" / "Kernels" / "CudaWrappers" / "CudaKeyFrame.cu"
    text = cuda_keyframe.read_text()
    text = replace_once(
        text,
        """#include "Kernels/CudaWrappers/CudaKeyFrame.h"
""",
        """#include "Kernels/CudaWrappers/CudaKeyFrame.h"
#include <algorithm>
#include <cstring>
""",
        cuda_keyframe,
    )
    text = replace_once(
        text,
        """    CudaKeyFrame::CudaKeyFrame() {
        initializeMemory();
    }
""",
        """    CudaKeyFrame::CudaKeyFrame()
        : isEmpty(false), mnId(0), Nleft(-1), gpuAddr(nullptr), mFeatCount(0) {
        initializeMemory();
    }
""",
        cuda_keyframe,
    )
    text = replace_once(
        text,
        """        checkCudaError(cudaMemcpy(mvDepth, KF->mvDepth.data(), KF->mvDepth.size() * sizeof(float), cudaMemcpyHostToDevice), "CudaKeyFrame:: Failed to copy mvDepth to gpu");
""",
        """        if (!KF->mvDepth.empty())
            checkCudaError(cudaMemcpy(mvDepth, KF->mvDepth.data(), KF->mvDepth.size() * sizeof(float), cudaMemcpyHostToDevice), "CudaKeyFrame:: Failed to copy mvDepth to gpu");
""",
        cuda_keyframe,
    )
    text = replace_once(
        text,
        """        checkCudaError(cudaMemcpy(mvScaleFactors, KF->mvScaleFactors.data(), mvScaleFactors_size * sizeof(float), cudaMemcpyHostToDevice), "CudaKeyFrame:: Failed to copy mvScaleFactors to gpu");
""",
        """        if (mvScaleFactors_size > 0)
            checkCudaError(cudaMemcpy(mvScaleFactors, KF->mvScaleFactors.data(), mvScaleFactors_size * sizeof(float), cudaMemcpyHostToDevice), "CudaKeyFrame:: Failed to copy mvScaleFactors to gpu");
""",
        cuda_keyframe,
    )
    text = replace_once(
        text,
        """        checkCudaError(cudaMemcpy(mvInvLevelSigma2, KF->mvInvLevelSigma2.data(), mvInvLevelSigma2_size * sizeof(float), cudaMemcpyHostToDevice), "CudaKeyFrame:: Failed to copy mvInvLevelSigma2 to gpu");
""",
        """        if (mvInvLevelSigma2_size > 0)
            checkCudaError(cudaMemcpy(mvInvLevelSigma2, KF->mvInvLevelSigma2.data(), mvInvLevelSigma2_size * sizeof(float), cudaMemcpyHostToDevice), "CudaKeyFrame:: Failed to copy mvInvLevelSigma2 to gpu");
""",
        cuda_keyframe,
    )
    text = replace_once(
        text,
        """        checkCudaError(cudaMemcpy(mvuRight, KF->mvuRight.data(), mvuRight_size * sizeof(float), cudaMemcpyHostToDevice), "CudaKeyFrame:: Failed to copy mvuRight to gpu");
""",
        """        if (mvuRight_size > 0)
            checkCudaError(cudaMemcpy(mvuRight, KF->mvuRight.data(), mvuRight_size * sizeof(float), cudaMemcpyHostToDevice), "CudaKeyFrame:: Failed to copy mvuRight to gpu");
""",
        cuda_keyframe,
    )
    text = replace_once(
        text,
        """        checkCudaError(cudaMemcpy((void*) mDescriptors, KF->mDescriptors.data,  KF->mDescriptors.rows * DESCRIPTOR_SIZE * sizeof(uint8_t), cudaMemcpyHostToDevice), "CudaKeyFrame:: Failed to copy mDescriptors to gpu"); 
""",
        """        if (KF->mDescriptors.rows > 0)
            checkCudaError(cudaMemcpy((void*) mDescriptors, KF->mDescriptors.data,  KF->mDescriptors.rows * DESCRIPTOR_SIZE * sizeof(uint8_t), cudaMemcpyHostToDevice), "CudaKeyFrame:: Failed to copy mDescriptors to gpu"); 
""",
        cuda_keyframe,
    )
    text = replace_once(
        text,
        """        checkCudaError(cudaMemcpy((void*) mvKeys, tmp_mvKeys.data(), mvKeys_size * sizeof(CudaKeyPoint), cudaMemcpyHostToDevice), "CudaKeyFrame:: Failed to copy mvKeys to gpu");
""",
        """        if (mvKeys_size > 0)
            checkCudaError(cudaMemcpy((void*) mvKeys, tmp_mvKeys.data(), mvKeys_size * sizeof(CudaKeyPoint), cudaMemcpyHostToDevice), "CudaKeyFrame:: Failed to copy mvKeys to gpu");
""",
        cuda_keyframe,
    )
    text = replace_once(
        text,
        """        checkCudaError(cudaMemcpy((void*) mvKeysRight, tmp_mvKeysRight.data(), mvKeysRight_size * sizeof(CudaKeyPoint), cudaMemcpyHostToDevice), "CudaKeyFrame:: Failed to copy mvKeysRight to gpu");
""",
        """        if (mvKeysRight_size > 0)
            checkCudaError(cudaMemcpy((void*) mvKeysRight, tmp_mvKeysRight.data(), mvKeysRight_size * sizeof(CudaKeyPoint), cudaMemcpyHostToDevice), "CudaKeyFrame:: Failed to copy mvKeysRight to gpu");
""",
        cuda_keyframe,
    )
    text = replace_once(
        text,
        """        checkCudaError(cudaMemcpy(mvKeysUn, tmp_mvKeysUn.data(), mvKeysUn_size * sizeof(CudaKeyPoint), cudaMemcpyHostToDevice), "CudaKeyFrame:: Failed to copy mvKeysUn to gpu");
""",
        """        if (mvKeysUn_size > 0)
            checkCudaError(cudaMemcpy(mvKeysUn, tmp_mvKeysUn.data(), mvKeysUn_size * sizeof(CudaKeyPoint), cudaMemcpyHostToDevice), "CudaKeyFrame:: Failed to copy mvKeysUn to gpu");
""",
        cuda_keyframe,
    )
    text = replace_once(
        text,
        """                size_t num_keypoints = KF->getMGrid()[i][j].size();
                if (num_keypoints > 0) {
                    std::memcpy(&flatMGrid[(i * mnGridRows + j) * keypoints_per_cell], KF->getMGrid()[i][j].data(), num_keypoints * sizeof(std::size_t));
                }
                flatMGrid_size[i * mnGridRows + j] = num_keypoints;
""",
        """                size_t num_keypoints = std::min(KF->getMGrid()[i][j].size(), static_cast<size_t>(keypoints_per_cell));
                if (num_keypoints > 0) {
                    std::memcpy(&flatMGrid[(i * mnGridRows + j) * keypoints_per_cell], KF->getMGrid()[i][j].data(), num_keypoints * sizeof(std::size_t));
                }
                flatMGrid_size[i * mnGridRows + j] = num_keypoints;
""",
        cuda_keyframe,
    )
    text = replace_once(
        text,
        """                    size_t num_keypoints = KF->mGridRight[i][j].size();
                    if (num_keypoints > 0) {
                        std::memcpy(&flatMGridRight[(i * mnGridRows + j) * KEYPOINTS_PER_CELL], KF->mGridRight[i][j].data(), num_keypoints * sizeof(std::size_t));
                    }
                    flatMGridRight_size[i * mnGridRows + j] = num_keypoints;
""",
        """                    size_t num_keypoints = std::min(KF->mGridRight[i][j].size(), static_cast<size_t>(KEYPOINTS_PER_CELL));
                    if (num_keypoints > 0) {
                        std::memcpy(&flatMGridRight[(i * mnGridRows + j) * KEYPOINTS_PER_CELL], KF->mGridRight[i][j].data(), num_keypoints * sizeof(std::size_t));
                    }
                    flatMGridRight_size[i * mnGridRows + j] = num_keypoints;
""",
        cuda_keyframe,
    )
    text = replace_once(
        text,
        """        mFeatCount = featVec.size();
        unsigned int tmp_mFeatVec[mFeatCount * MAX_FEAT_PER_WORD];
        int tmp_mFeatVecStartIndexes[mFeatCount];
""",
        """        mFeatCount = featVec.size();
        if (mFeatCount == 0)
            return;
        unsigned int tmp_mFeatVec[mFeatCount * MAX_FEAT_PER_WORD];
        int tmp_mFeatVecStartIndexes[mFeatCount];
""",
        cuda_keyframe,
    )
    cuda_keyframe.write_text(text)


def patch_adaptive_controller(root: pathlib.Path) -> None:
    orb_header = root / "include" / "ORBextractor.h"
    text = orb_header.read_text()
    text = replace_once(
        text,
        """    int inline GetNFeatures() {
        return nfeatures;}

    uchar* GetGPUPyramid() {
""",
        """    int inline GetNFeatures() {
        return nfeatures;}

    void SetFeatureBudget(int featureBudget);

    uchar* GetGPUPyramid() {
""",
        orb_header,
    )
    orb_header.write_text(text)

    orb_source = root / "src" / "ORBextractor.cc"
    text = orb_source.read_text()
    text = replace_once(
        text,
        """#include <vector>
#include <iostream>
""",
        """#include <vector>
#include <algorithm>
#include <cstdlib>
#include <string>
#include <iostream>
""",
        orb_source,
    )
    text = replace_once(
        text,
        """        int nkeypoints = 0;

        if (TrackingKernelController::orbExtractionKernelRunStatus == 1) {
""",
        """        int nkeypoints = 0;
        bool useGpuExtraction = (TrackingKernelController::orbExtractionKernelRunStatus == 1);

        if (useGpuExtraction) {
""",
        orb_source,
    )
    text = replace_once(
        text,
        """        Mat descriptors;
""",
        """        const char *fallbackEnv = std::getenv("NITRO_ADAPTIVE_CUDA_FALLBACK");
        const bool allowCudaFallback = fallbackEnv && std::string(fallbackEnv) != "0";
        const int fallbackMinKeypoints = std::max(1, nfeatures / 3);
        if(useGpuExtraction && allowCudaFallback && nkeypoints < fallbackMinKeypoints)
        {
            cout << "[NitroAdaptive] CUDA ORB fallback to CPU: keypoints="
                 << nkeypoints << " threshold=" << fallbackMinKeypoints << endl;
            useGpuExtraction = false;
            allKeypointsGPU.clear();
            allKeypoints.clear();
            nkeypoints = 0;
            ComputePyramid(image);
            ComputeKeyPointsOctTree(allKeypoints);
            for (int level = 0; level < nlevels; ++level)
                nkeypoints += (int)allKeypoints[level].size();
        }

        Mat descriptors;
""",
        orb_source,
    )
    text = replace_once(
        text,
        """        if (TrackingKernelController::orbExtractionKernelRunStatus == 1) {
            for (int level = 0; level < nlevels; ++level)
""",
        """        if (useGpuExtraction) {
            for (int level = 0; level < nlevels; ++level)
""",
        orb_source,
    )
    text = replace_once(
        text,
        """    static void computeOrientation(const Mat& image, vector<KeyPoint>& keypoints, const vector<int>& umax)
""",
        """    void ORBextractor::SetFeatureBudget(int featureBudget)
    {
        featureBudget = std::max(1, featureBudget);
        if(featureBudget == nfeatures)
            return;

        nfeatures = featureBudget;
        mnFeaturesPerLevel.assign(nlevels, 0);
        float factor = 1.0f / scaleFactor;
        float nDesiredFeaturesPerScale = nfeatures * (1 - factor) /
            (1 - (float)pow((double)factor, (double)nlevels));

        int sumFeatures = 0;
        for(int level = 0; level < nlevels - 1; level++)
        {
            mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
            sumFeatures += mnFeaturesPerLevel[level];
            nDesiredFeaturesPerScale *= factor;
        }
        mnFeaturesPerLevel[nlevels - 1] = std::max(nfeatures - sumFeatures, 0);
    }

    static void computeOrientation(const Mat& image, vector<KeyPoint>& keypoints, const vector<int>& umax)
""",
        orb_source,
    )
    orb_source.write_text(text)

    tracking_header = root / "include" / "Tracking.h"
    text = tracking_header.read_text()
    text = replace_once(
        text,
        """    bool TrackLocalMap();
    void SearchLocalPoints();

    bool NeedNewKeyFrame();
""",
        """    bool TrackLocalMap();
    void SearchLocalPoints();
    void ConfigureNitroAdaptiveController(int defaultFeatures);
    void ApplyNitroAdaptiveFeatureBudget();
    bool NitroAdaptiveShouldRunPoseOptimization(int prePoseMatches);

    bool NeedNewKeyFrame();
""",
        tracking_header,
    )
    text = replace_once(
        text,
        """    //Current matches in frame
    int mnMatchesInliers;
""",
        """    //Current matches in frame
    int mnMatchesInliers;

    // Nitro adaptive controller, inspired by AntSLAM scheduling ablations.
    bool mbNitroAdaptiveEnabled;
    bool mbNitroAdaptiveFeatureBudget;
    bool mbNitroAdaptivePoseScheduler;
    bool mbNitroAdaptiveWarmupRescue;
    bool mbNitroAdaptiveLowSupportLock;
    bool mbNitroAdaptiveInitialized;
    bool mbNitroAdaptiveSupportClassified;
    bool mbNitroAdaptiveLowSupportLocked;
    int mnNitroAdaptiveDefaultFeatures;
    int mnNitroAdaptiveFeatureBudget;
    int mnNitroAdaptivePoseInterval;
    int mnNitroAdaptiveMode;
    int mnNitroAdaptiveWarmupFrames;
    unsigned long mnNitroAdaptiveLastModeChangeFrame;
    unsigned long mnNitroAdaptiveSamples;
    double mdNitroAdaptiveEmaRatio;
    double mdNitroAdaptiveFeatureSum;
    double mdNitroAdaptiveInlierSum;
""",
        tracking_header,
    )
    tracking_header.write_text(text)

    tracking = root / "src" / "Tracking.cc"
    text = tracking.read_text()
    text = replace_once(
        text,
        """#include <iostream>
""",
        """#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
""",
        tracking,
    )
    text = replace_once(
        text,
        """namespace ORB_SLAM3
{

ORBextractor *Tracking::mpORBextractorLeft, *Tracking::mpORBextractorRight;
""",
        """namespace ORB_SLAM3
{

namespace
{
enum NitroAdaptiveMode
{
    NITRO_ADAPT_BOOTSTRAP = 0,
    NITRO_ADAPT_RESCUE = 1,
    NITRO_ADAPT_LOOP = 2,
    NITRO_ADAPT_FAST = 3,
    NITRO_ADAPT_STRONG = 4
};

bool NitroEnvBool(const char *name, bool defaultValue)
{
    const char *value = std::getenv(name);
    if(!value)
        return defaultValue;
    return std::string(value) != "0";
}

int NitroEnvInt(const char *name, int defaultValue)
{
    const char *value = std::getenv(name);
    if(!value)
        return defaultValue;
    try
    {
        return std::stoi(value);
    }
    catch(...)
    {
        return defaultValue;
    }
}

const char *NitroAdaptiveModeName(int mode)
{
    switch(mode)
    {
        case NITRO_ADAPT_RESCUE: return "rescue";
        case NITRO_ADAPT_LOOP: return "loop/low-support";
        case NITRO_ADAPT_FAST: return "dynamic/fast";
        case NITRO_ADAPT_STRONG: return "strong/lean";
        case NITRO_ADAPT_BOOTSTRAP:
        default: return "bootstrap";
    }
}
}

ORBextractor *Tracking::mpORBextractorLeft, *Tracking::mpORBextractorRight;
""",
        tracking,
    )
    text = replace_once(
        text,
        """    if(mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR){
        mpIniORBextractor = new ORBextractor(5*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST, settings->newImSize().width, settings->newImSize().height);
        CudaUtils::loadSetting(5*nFeatures, nLevels, true, fScaleFactor, settings->newImSize().width, settings->newImSize().height, cameraIsFisheye);
    }
    //IMU parameters
""",
        """    if(mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR){
        mpIniORBextractor = new ORBextractor(5*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST, settings->newImSize().width, settings->newImSize().height);
        CudaUtils::loadSetting(5*nFeatures, nLevels, true, fScaleFactor, settings->newImSize().width, settings->newImSize().height, cameraIsFisheye);
    }
    ConfigureNitroAdaptiveController(nFeatures);
    //IMU parameters
""",
        tracking,
    )
    text = replace_once(
        text,
        """    if(mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR)
        mpIniORBextractor = new ORBextractor(5*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST,imageWidth,imageHeight);

    cout << endl << "ORB Extractor Parameters: " << endl;
""",
        """    if(mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR)
        mpIniORBextractor = new ORBextractor(5*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST,imageWidth,imageHeight);

    ConfigureNitroAdaptiveController(nFeatures);

    cout << endl << "ORB Extractor Parameters: " << endl;
""",
        tracking,
    )
    text = replace_once(
        text,
        """bool Tracking::ParseCamParamFile(cv::FileStorage &fSettings)
""",
        """void Tracking::ConfigureNitroAdaptiveController(int defaultFeatures)
{
    mbNitroAdaptiveFeatureBudget = NitroEnvBool("NITRO_ADAPTIVE_FEATURE_BUDGET", false);
    mbNitroAdaptivePoseScheduler = NitroEnvBool("NITRO_ADAPTIVE_POSE_SCHEDULER", false);
    mbNitroAdaptiveWarmupRescue = NitroEnvBool("NITRO_ADAPTIVE_WARMUP_RESCUE", false);
    mbNitroAdaptiveLowSupportLock = NitroEnvBool("NITRO_ADAPTIVE_LOW_SUPPORT_LOCK", false);
    mbNitroAdaptiveEnabled = mbNitroAdaptiveFeatureBudget ||
                              mbNitroAdaptivePoseScheduler ||
                              mbNitroAdaptiveWarmupRescue ||
                              mbNitroAdaptiveLowSupportLock;
    mbNitroAdaptiveInitialized = false;
    mbNitroAdaptiveSupportClassified = false;
    mbNitroAdaptiveLowSupportLocked = false;
    mnNitroAdaptiveDefaultFeatures = std::max(1, defaultFeatures);
    mnNitroAdaptiveFeatureBudget = mnNitroAdaptiveDefaultFeatures;
    mnNitroAdaptivePoseInterval = 1;
    mnNitroAdaptiveMode = NITRO_ADAPT_BOOTSTRAP;
    mnNitroAdaptiveWarmupFrames = NitroEnvInt("NITRO_ADAPTIVE_WARMUP_FRAMES", 200);
    mnNitroAdaptiveLastModeChangeFrame = 0;
    mnNitroAdaptiveSamples = 0;
    mdNitroAdaptiveEmaRatio = 0.0;
    mdNitroAdaptiveFeatureSum = 0.0;
    mdNitroAdaptiveInlierSum = 0.0;

    if(mbNitroAdaptiveEnabled)
    {
        cout << "[NitroAdaptive] enabled mask="
             << (std::getenv("NITRO_ADAPTIVE_MASK") ? std::getenv("NITRO_ADAPTIVE_MASK") : "env")
             << " default_features=" << mnNitroAdaptiveDefaultFeatures << endl;
    }
}

void Tracking::ApplyNitroAdaptiveFeatureBudget()
{
    if(!mbNitroAdaptiveEnabled || !mbNitroAdaptiveFeatureBudget || !mpORBextractorLeft)
        return;

    const int cappedBudget = std::max(1, std::min(mnNitroAdaptiveFeatureBudget, mnNitroAdaptiveDefaultFeatures));
    mpORBextractorLeft->SetFeatureBudget(cappedBudget);
    if(mpORBextractorRight)
        mpORBextractorRight->SetFeatureBudget(cappedBudget);
}

bool Tracking::NitroAdaptiveShouldRunPoseOptimization(int prePoseMatches)
{
    if(!mbNitroAdaptiveEnabled)
        return true;

    const int featureCount = std::max(1, mCurrentFrame.N);
    prePoseMatches = std::max(0, prePoseMatches);
    const double ratio = std::min(1.0, static_cast<double>(prePoseMatches) / static_cast<double>(featureCount));
    const double alpha = std::max(0.01, std::min(1.0, static_cast<double>(NitroEnvInt("NITRO_ADAPTIVE_EMA_ALPHA_PERMILLE", 120)) / 1000.0));
    if(!mbNitroAdaptiveInitialized)
        mdNitroAdaptiveEmaRatio = ratio;
    else
        mdNitroAdaptiveEmaRatio = alpha * ratio + (1.0 - alpha) * mdNitroAdaptiveEmaRatio;

    mnNitroAdaptiveSamples++;
    mdNitroAdaptiveFeatureSum += static_cast<double>(featureCount);
    mdNitroAdaptiveInlierSum += static_cast<double>(prePoseMatches);

    const bool warmedUp = mCurrentFrame.mnId >= static_cast<unsigned long>(mnNitroAdaptiveWarmupFrames);
    const int lowSupportFeatureMin = NitroEnvInt("NITRO_ADAPTIVE_LOW_SUPPORT_FEATURE_MIN", 650);
    const double lowSupportRatio = static_cast<double>(NitroEnvInt("NITRO_ADAPTIVE_LOW_SUPPORT_RATIO_PERMILLE", 180)) / 1000.0;
    if(mbNitroAdaptiveLowSupportLock && !mbNitroAdaptiveSupportClassified && warmedUp)
    {
        const double meanFeatures = mdNitroAdaptiveFeatureSum / static_cast<double>(std::max<unsigned long>(1ul, mnNitroAdaptiveSamples));
        const double meanRatio = mdNitroAdaptiveFeatureSum > 0.0 ? mdNitroAdaptiveInlierSum / mdNitroAdaptiveFeatureSum : ratio;
        mbNitroAdaptiveLowSupportLocked = meanFeatures >= static_cast<double>(lowSupportFeatureMin) && meanRatio <= lowSupportRatio;
        mbNitroAdaptiveSupportClassified = true;
        cout << "[NitroAdaptive] support="
             << (mbNitroAdaptiveLowSupportLocked ? "low-support" : "nominal")
             << " frame=" << mCurrentFrame.mnId
             << " mean_features=" << meanFeatures
             << " mean_ratio=" << meanRatio << endl;
    }

    const int criticalMatches = NitroEnvInt("NITRO_ADAPTIVE_CRITICAL_MATCHES", 60);
    const double criticalRatio = static_cast<double>(NitroEnvInt("NITRO_ADAPTIVE_CRITICAL_RATIO_PERMILLE", 90)) / 1000.0;
    const bool criticalSupport = mbNitroAdaptiveWarmupRescue &&
                                 (prePoseMatches < criticalMatches ||
                                  ratio < criticalRatio ||
                                  mState == RECENTLY_LOST ||
                                  mState == LOST);

    int nextMode = NITRO_ADAPT_LOOP;
    if(mbNitroAdaptiveWarmupRescue && !warmedUp)
        nextMode = NITRO_ADAPT_BOOTSTRAP;
    else if(criticalSupport)
        nextMode = NITRO_ADAPT_RESCUE;
    else if(mbNitroAdaptiveLowSupportLock && mbNitroAdaptiveLowSupportLocked)
        nextMode = NITRO_ADAPT_LOOP;
    else
    {
        const double loopEnter = static_cast<double>(NitroEnvInt("NITRO_ADAPTIVE_LOOP_ENTER_PERMILLE", 180)) / 1000.0;
        const double fastEnter = static_cast<double>(NitroEnvInt("NITRO_ADAPTIVE_FAST_ENTER_PERMILLE", 280)) / 1000.0;
        const double strongEnter = static_cast<double>(NitroEnvInt("NITRO_ADAPTIVE_STRONG_ENTER_PERMILLE", 380)) / 1000.0;
        if(mdNitroAdaptiveEmaRatio >= strongEnter)
            nextMode = NITRO_ADAPT_STRONG;
        else if(mdNitroAdaptiveEmaRatio >= fastEnter)
            nextMode = NITRO_ADAPT_FAST;
        else if(mdNitroAdaptiveEmaRatio <= loopEnter || mnNitroAdaptiveMode == NITRO_ADAPT_LOOP)
            nextMode = NITRO_ADAPT_LOOP;
        else
            nextMode = NITRO_ADAPT_LOOP;
    }

    const int minDwellFrames = NitroEnvInt("NITRO_ADAPTIVE_MIN_DWELL_FRAMES", 30);
    if(mbNitroAdaptiveInitialized &&
       nextMode != mnNitroAdaptiveMode &&
       nextMode != NITRO_ADAPT_RESCUE &&
       !criticalSupport &&
       minDwellFrames > 0)
    {
        const unsigned long framesSinceModeChange =
            mCurrentFrame.mnId >= mnNitroAdaptiveLastModeChangeFrame ?
            mCurrentFrame.mnId - mnNitroAdaptiveLastModeChangeFrame :
            static_cast<unsigned long>(minDwellFrames);
        if(framesSinceModeChange < static_cast<unsigned long>(minDwellFrames))
            nextMode = mnNitroAdaptiveMode;
    }

    const int bootstrapFeatures = NitroEnvInt("NITRO_ADAPTIVE_BOOTSTRAP_FEATURES", mnNitroAdaptiveDefaultFeatures);
    const int rescueFeatures = NitroEnvInt("NITRO_ADAPTIVE_RESCUE_FEATURES", bootstrapFeatures);
    const int loopFeatures = NitroEnvInt("NITRO_ADAPTIVE_LOOP_FEATURES", 700);
    const int fastFeatures = NitroEnvInt("NITRO_ADAPTIVE_FAST_FEATURES", 900);
    const int strongFeatures = NitroEnvInt("NITRO_ADAPTIVE_STRONG_FEATURES", 800);
    const int bootstrapInterval = NitroEnvInt("NITRO_ADAPTIVE_BOOTSTRAP_POSE_INTERVAL", 1);
    const int rescueInterval = NitroEnvInt("NITRO_ADAPTIVE_RESCUE_POSE_INTERVAL", 1);
    const int loopInterval = NitroEnvInt("NITRO_ADAPTIVE_LOOP_POSE_INTERVAL", 10);
    const int fastInterval = NitroEnvInt("NITRO_ADAPTIVE_FAST_POSE_INTERVAL", 0);
    const int strongInterval = NitroEnvInt("NITRO_ADAPTIVE_STRONG_POSE_INTERVAL", 0);

    int nextFeatureBudget = mnNitroAdaptiveDefaultFeatures;
    int nextPoseInterval = 1;
    if(nextMode == NITRO_ADAPT_RESCUE)
    {
        nextFeatureBudget = rescueFeatures;
        nextPoseInterval = rescueInterval;
    }
    else if(nextMode == NITRO_ADAPT_LOOP)
    {
        nextFeatureBudget = loopFeatures;
        nextPoseInterval = loopInterval;
    }
    else if(nextMode == NITRO_ADAPT_FAST)
    {
        nextFeatureBudget = fastFeatures;
        nextPoseInterval = fastInterval;
    }
    else if(nextMode == NITRO_ADAPT_STRONG)
    {
        nextFeatureBudget = strongFeatures;
        nextPoseInterval = strongInterval;
    }
    else
    {
        nextFeatureBudget = bootstrapFeatures;
        nextPoseInterval = bootstrapInterval;
    }

    nextFeatureBudget = std::max(1, std::min(nextFeatureBudget, mnNitroAdaptiveDefaultFeatures));
    const bool modeChanged = !mbNitroAdaptiveInitialized ||
                             nextMode != mnNitroAdaptiveMode ||
                             nextFeatureBudget != mnNitroAdaptiveFeatureBudget ||
                             nextPoseInterval != mnNitroAdaptivePoseInterval;
    if(modeChanged)
    {
        mnNitroAdaptiveLastModeChangeFrame = mCurrentFrame.mnId;
        cout << "[NitroAdaptive] mode=" << NitroAdaptiveModeName(nextMode)
             << " frame=" << mCurrentFrame.mnId
             << " features=" << nextFeatureBudget
             << " pose_interval=" << nextPoseInterval
             << " ratio=" << ratio
             << " ema=" << mdNitroAdaptiveEmaRatio << endl;
    }

    mnNitroAdaptiveMode = nextMode;
    mnNitroAdaptiveFeatureBudget = nextFeatureBudget;
    mnNitroAdaptivePoseInterval = nextPoseInterval;
    mbNitroAdaptiveInitialized = true;

    if(!mbNitroAdaptivePoseScheduler)
        return true;
    if(nextPoseInterval <= 0)
        return criticalSupport || (mbNitroAdaptiveWarmupRescue && !warmedUp);
    if(criticalSupport)
        return true;
    return (mCurrentFrame.mnId % static_cast<unsigned long>(nextPoseInterval)) == 0;
}

bool Tracking::ParseCamParamFile(cv::FileStorage &fSettings)
""",
        tracking,
    )
    text = replace_once(
        text,
        """    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    if (mSensor == System::RGBD)
""",
        """    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    ApplyNitroAdaptiveFeatureBudget();

    if (mSensor == System::RGBD)
""",
        tracking,
    )
    text = replace_once(
        text,
        """    if (TrackingKernelController::poseOptimizationRunStatus == 1) {
        int inliers;
""",
        """    const bool nitroRunPoseOptimization = NitroAdaptiveShouldRunPoseOptimization(aux1 - aux2);
    if (TrackingKernelController::poseOptimizationRunStatus == 1 && nitroRunPoseOptimization) {
        int inliers;
""",
        tracking,
    )
    tracking.write_text(text)


def patch_stats_header(root: pathlib.Path) -> None:
    cuda_utils = root / "include" / "Kernels" / "CudaUtils.h"
    if not cuda_utils.exists():
        return

    text = cuda_utils.read_text()
    text = replace_once(
        text,
        """// Flag to activate the measurement of time in each kernel.
#define REGISTER_TRACKING_STATS
#define REGISTER_LOCAL_MAPPING_STATS
#define REGISTER_LOOP_CLOSING_STATS
""",
        """// Flag to activate the measurement of time in each kernel.
#ifndef NITRO_DISABLE_STATS
#define REGISTER_TRACKING_STATS
#define REGISTER_LOCAL_MAPPING_STATS
#define REGISTER_LOOP_CLOSING_STATS
#endif
""",
        cuda_utils,
    )
    cuda_utils.write_text(text)


def restore_missing_dbow2_timestamp(root: pathlib.Path) -> None:
    dutils = root / "Thirdparty" / "DBoW2" / "DUtils"
    dutils.mkdir(parents=True, exist_ok=True)
    header = dutils / "Timestamp.h"
    source = dutils / "Timestamp.cpp"
    if not header.exists():
        header.write_text(textwrap.dedent(TIMESTAMP_H).lstrip())
    if not source.exists():
        source.write_text(textwrap.dedent(TIMESTAMP_CPP).lstrip())


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: patch_nitroslam_context.py /opt/nitroslam", file=sys.stderr)
        return 2
    root = pathlib.Path(sys.argv[1])
    restore_missing_dbow2_timestamp(root)
    patch_graphite_cmake(root)
    patch_compute_engine_cmake(root)
    patch_nitro_cmake(root)
    patch_stats_header(root)
    patch_rgbd_tum(root)
    patch_rgbd_cuda_context(root)
    patch_adaptive_controller(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
