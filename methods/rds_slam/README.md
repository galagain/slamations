# RDS-SLAM sur TUM RGB-D

Source par defaut: `/home/nvidia/Documents/dockers/rds_slam`.

```bash
./rds_slam/build.sh
./rds_slam/run_tum.sh segnet /chemin/rgbd_dataset_freiburg3_walking_xyz
./rds_slam/run_tum.sh maskrcnn /chemin/rgbd_dataset_freiburg3_walking_xyz
```

Le Dockerfile reprend l'environnement ROS Noetic/Jetson deja present dans tes dossiers, avec le modele `segnet_pascal.caffemodel` local. Par defaut il part de `nvcr.io/nvidia/l4t-base:r35.1.0`, car le wheel TensorFlow 1.15 utilise par Mask R-CNN est fourni pour JetPack 5.
