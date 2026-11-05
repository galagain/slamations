# Panoptic-SLAM sur TUM RGB-D

Source par defaut: `/home/nvidia/Desktop/panoptic/Panoptic-SLAM`.

```bash
./panoptic_slam/build.sh
./panoptic_slam/run_tum.sh /chemin/rgbd_dataset_freiburg3_walking_xyz
```

Le Dockerfile est adapte a JetPack 6/aarch64. Panoptic-SLAM depend de Detectron2; si l'installation Python change sur ta machine, reconstruis en donnant tes roues PyTorch/Detectron2 dans le contexte ou installe-les dans l'image avant le lancement.
