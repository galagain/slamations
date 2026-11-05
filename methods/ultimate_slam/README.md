# UltimateSLAM

This method wraps `uzh-rpg/rpg_ultimate_slam_open` in a separate ROS Noetic
Docker image.

UltimateSLAM is an event-frame-inertial VIO system. TUM RGB-D does not provide
events or IMU, and its depth channel is not consumed by UltimateSLAM. The TUM
wrapper therefore runs only as a binary smoke test using grayscale RGB frames and
a synthetic stationary IMU. Its TUM numbers are marked non-comparable.

```bash
methods/ultimate_slam/build.sh
methods/ultimate_slam/run_tum_profile.sh /home/nvidia/data/tum_rgbd/rgbd_dataset_freiburg1_xyz
```

Current TUM smoke-test results and the ORB-SLAM3 comparison are summarized in
`methods/ultimate_slam/ANALYSIS.md`.
