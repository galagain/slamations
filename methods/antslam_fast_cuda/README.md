# AntSLAM Fast CUDA

Variant AntSLAM geometrique sans semantique, avec frontend CUDA force.

Cette methode reutilise l'image Docker `thesis/antslam:jetson-r36.3-tum`
et les configurations TUM RGB-D de `methods/antslam/`, mais elle est separee
dans les campagnes pour eviter de melanger:

- `antslam`: configuration geometrique principale, frontend CUDA desactive par defaut.
- `antslam_fast_cuda`: ablation rapide, `ANTSLAM_CUDA_FRONTEND=1`.
- `antsslam`: variante semantique TokenMask.

Commande:

```bash
methods/antslam_fast_cuda/run_tum_profile.sh /home/nvidia/data/tum_rgbd/rgbd_dataset_freiburg1_xyz
```
