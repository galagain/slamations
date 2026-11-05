#ifndef ANTSLAM_CUDA_LOCAL_MATCHER_H
#define ANTSLAM_CUDA_LOCAL_MATCHER_H

#ifdef ORB_SLAM3_USE_CUDA_FRONTEND

#include <vector>

#include "Frame.h"
#include "MapPoint.h"

namespace ORB_SLAM3
{

class AntSlamCudaLocalMatcher
{
public:
    static bool Enabled();
    static bool Match(Frame &F,
                      const std::vector<MapPoint*> &vpMapPoints,
                      const float th,
                      const bool bFarPoints,
                      const float thFarPoints,
                      std::vector<int> &bestLevel,
                      std::vector<int> &bestLevel2,
                      std::vector<int> &bestDist,
                      std::vector<int> &bestDist2,
                      std::vector<int> &bestIdx,
                      std::vector<int> &bestLevelR,
                      std::vector<int> &bestLevelR2,
                      std::vector<int> &bestDistR,
                      std::vector<int> &bestDistR2,
                      std::vector<int> &bestIdxR);
    static void Shutdown();
};

} // namespace ORB_SLAM3

#endif

#endif
