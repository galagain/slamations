# VDO-SLAM sur TUM RGB-D

Source par defaut: `/home/nvidia/Documents/Formicidae/Myrmicinae/dockers/vdo_slam/VDO_SLAM`.

```bash
./vdo_slam/build.sh
./vdo_slam/run_tum.sh /chemin/rgbd_dataset_freiburg3_walking_xyz
```

Le wrapper convertit TUM vers le format attendu par VDO-SLAM et cree des masques/flows neutres. Pour une vraie evaluation dynamique, remplace `output/vdo_slam/prepared/<sequence>/semantic` et `flow` par des sorties d'instance segmentation et d'optical flow.
