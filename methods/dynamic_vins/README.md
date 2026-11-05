# Dynamic-VINS sur TUM RGB-D

Source par defaut: `/home/nvidia/Desktop/SotaSLAM/Dynamic-Vins`.

```bash
./dynamic_vins/build.sh
./dynamic_vins/run_tum.sh /chemin/rgbd_dataset_freiburg3_walking_xyz
```

Le wrapper convertit la sequence TUM en rosbag, puis lance `tum_rgbd_pytorch.launch`. Le service YOLO est configure en CPU dans l'image pour rester portable en arm64; remets `device: gpu` dans `yolo_ros/config/config.yaml` si ton image PyTorch CUDA Jetson est installee.
