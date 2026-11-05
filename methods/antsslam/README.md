# AntSSLAM

AntSSLAM est le nouveau fork semantique de la these. Il part d'AntSLAM, donc du SLAM geometrique optimise du chapitre precedent, puis ajoute une couche de masquage semantique asynchrone inspiree de l'ancien `LiteSemSLAM`.

Objectif experimental:

- conserver le pipeline ORB-SLAM3/AntSLAM complet;
- ajouter des masques dynamiques `255=keep, 0=reject` avant le tracking;
- comparer trois familles de backends TensorRT deja disponibles dans les anciens travaux: SegNet, Mask R-CNN et TokenMask;
- mesurer separement le cout SLAM et l'effet de la semantique sur les scenes dynamiques TUM RGB-D.

Build geometrique compatible AntSLAM:

```bash
./methods/antsslam/build.sh
```

Build avec backends TensorRT:

```bash
ANTSSLAM_TENSORRT=ON ./methods/antsslam/build.sh
```

Run sans semantique:

```bash
SLAM_SETTINGS_DIR=/opt/method_configs/tum_rgbd_nfeatures_1000 \
./methods/antsslam/run_tum_profile.sh /home/nvidia/data/tum_rgbd/rgbd_dataset_freiburg3_walking_xyz
```

Run avec TokenMask TensorRT:

```bash
ANTSSLAM_SEGMENTATION_DIR=/home/nvidia/Desktop/kalvici/segmentation \
ANTSSLAM_TRT_CACHE_DIR=/home/nvidia/Desktop/kalvici/trt_cache \
ANTSSLAM_TOKENMASK_MODELS_DIR=/home/nvidia/Desktop/kalvici/neww/Jetson-SLAM/models \
SLAM_SETTINGS_DIR=/opt/method_configs/tum_rgbd_semantic_tokenmask_stride16_exact \
./methods/antsslam/run_tum_profile.sh /home/nvidia/data/tum_rgbd/rgbd_dataset_freiburg3_walking_xyz
```

Benchmark TensorRT seul, sans SLAM:

```bash
ANTSSLAM_SEGMENTATION_DIR=/home/nvidia/Desktop/kalvici/segmentation \
ANTSSLAM_TRT_CACHE_DIR=/home/nvidia/Desktop/kalvici/trt_cache \
ANTSSLAM_TOKENMASK_MODELS_DIR=/home/nvidia/Desktop/kalvici/neww/Jetson-SLAM/models \
./methods/antsslam/run_semantic_only_benchmark.sh \
  /home/nvidia/data/tum_rgbd/rgbd_dataset_freiburg3_walking_xyz all
```

Presets fournis:

- `tum_rgbd_nfeatures_1000`: AntSSLAM sans semantique.
- `tum_rgbd_semantic_segnet_latest`: SegNet TensorRT exact, `Semantic.max_mask_age=0`.
- `tum_rgbd_semantic_maskrcnn_latest`: Mask R-CNN TensorRT FP16, `Semantic.max_mask_age=0`.
- `tum_rgbd_semantic_tokenmask_stride16_exact`: TokenMask tiny Cityscapes TensorRT, une inference toutes les 16 images et aucune reutilisation du masque sur les images intermediaires.
- `tum_rgbd_semantic_tokenmask_stride8`: ancienne ablation rapide avec reutilisation non bornee du dernier masque; conservee pour comparaison, mais elle n'est plus le preset recommande.

Les chemins de modeles dans les YAML pointent vers les montages Docker `/opt/semantic/...`; ils sont alimentes par les variables `ANTSSLAM_SEGMENTATION_DIR`, `ANTSSLAM_TRT_CACHE_DIR` et `ANTSSLAM_TOKENMASK_MODELS_DIR`. Les alias historiques `ANTSLAM_SEGMENTATION_DIR`, `ANTSLAM_TRT_CACHE_DIR` et `ANTSLAM_TOKENMASK_MODELS_DIR` sont aussi acceptes.

## Resultats comparables image par image

La comparaison principale utilise le FPS reporte par l'executable dans le
Docker, comme pour AntSLAM: `Tracking FPS = 1000 / mean(total_tracking_ms)`.
La reference correcte pour mesurer le cout de la semantique est la base CUDA
sans semantique, pas l'ancien run CPU.

| Mode sur `fr3_walking_xyz` | Base | Semantique | Tracking moyen | FPS Docker | ATE RMSE | Decision |
|---|---|---|---:|---:|---:|---|
| AntSLAM-like | CPU ORB | aucune | 21.114 ms | 47.36 | 0.0882 m | reference CPU |
| Base CUDA | CUDA ORB | aucune | 10.826 ms | 92.37 | 0.0281 m | reference surcout |
| AntSSLAM final | CUDA ORB | TokenMask stride 16 exact | 14.379 ms | 69.54 | 0.0238 m | retenu |
| SegNet | CUDA ORB | exact | 56.231 ms | 17.78 | 0.0179 m | trop lent |
| Mask R-CNN | CUDA ORB | exact | 126.128 ms | 7.93 | 0.6119 m | rejete |

Lecture rapide : AntSSLAM TokenMask est plus lent que la meme base CUDA sans
semantique. Le gain observe par rapport a l'ancien run CPU vient du front-end
CUDA, pas de la semantique. La semantique est retenue parce qu'elle ameliore la
robustesse sur scene dynamique tout en restant au-dessus de 30 Hz avec une
politique sparse exacte.

Le detail scientifique et les benchmarks TensorRT seuls sont dans
`results/semantic_backend_comparison_20260521.md` et dans `ANALYSIS.md`.
