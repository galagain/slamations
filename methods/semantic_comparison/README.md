# Semantic SLAM Comparison Workspace

Ce dossier organise le travail du chapitre suivant : analyser comment la
semantique est integree dans les SLAMs existants, puis construire AntSSLAM a
partir d'AntSLAM.

## Dossiers de methodes

Chaque methode a son propre dossier Docker dans `methods/` :

| Methode | Dossier | Source amont | Role de la semantique |
|---|---|---|---|
| DynaSLAM | `methods/dynaslam` | `BertaBescos/DynaSLAM` | masque les objets dynamiques avec Mask R-CNN et une verification geometrique multi-vues |
| RDS-SLAM | `methods/rds_slam` | `yubaoliu/RDS-SLAM` | maintient un thread semantique SegNet ou Mask R-CNN pour filtrer les regions dynamiques |
| Panoptic-SLAM | `methods/panoptic_slam` | `iit-DLSLab/Panoptic-SLAM` | combine segmentation panoptique, objets et carte geometrique |
| VDO-SLAM | `methods/vdo_slam` | `halajun/VDO_SLAM` | estime camera et objets dynamiques avec masques/flux optique |
| Dynamic-VINS | `methods/dynamic_vins` | `HITSZ-NRSL/Dynamic-VINS` | rejette des features dynamiques dans un pipeline VIO avec detection d'objets |
| AntSSLAM | `methods/antsslam` | fork local de `third_party/AntSLAM` | applique des masques semantiques asynchrones avant le tracking d'AntSLAM |

Les sources propres doivent etre placees dans `third_party/semantic/` par les
scripts de build, et les modeles lourds doivent etre montes depuis des dossiers
locaux. On evite ainsi de retélécharger ou recopier les poids a chaque
experience.

## Modeles semantiques retrouves dans les anciens travaux

Les anciens travaux de `/home/nvidia/Desktop/kalvici` contiennent deja plusieurs
modeles exportes :

| Modele | Chemin local | Remarque |
|---|---|---|
| Mask R-CNN TensorRT | `/home/nvidia/Desktop/kalvici/segmentation/models/final/maskrcnn_fp16.engine` | environ 122.5 ms par inference dans l'ancien banc TensorRT |
| SegNet TensorRT | `/home/nvidia/Desktop/kalvici/segmentation/models/final/segnet_exact.engine` | environ 27.1 ms par inference, IoU 0.9867 contre Caffe |
| SegNet plugin | `/home/nvidia/Desktop/kalvici/segmentation/plugins/segnet_unpool/libsegnet_trt_plugins.so` | plugin d'unpooling TensorRT requis par SegNet |
| TokenMask tiny Cityscapes | `/home/nvidia/Desktop/kalvici/neww/Jetson-SLAM/models/ours_tiny_city.onnx` | utilise par l'ancien LiteSemSLAM avec cache TensorRT local |

Pour AntSSLAM, les dossiers montables sont :

```bash
ANTSSLAM_SEGMENTATION_DIR=/home/nvidia/Desktop/kalvici/segmentation
ANTSSLAM_TRT_CACHE_DIR=/home/nvidia/Desktop/kalvici/trt_cache
ANTSSLAM_TOKENMASK_MODELS_DIR=/home/nvidia/Desktop/kalvici/neww/Jetson-SLAM/models
```

## Anciennes references experimentales

Les chiffres ci-dessous proviennent de la campagne locale historique. Ils ne
remplacent pas les nouveaux runs reproductibles, mais ils donnent les ordres de
grandeur et les hypotheses de depart.

| Sequence | Methode | FPS approx. | ATE RMSE |
|---|---|---:|---:|
| `fr1_xyz` | ORB-SLAM3 | 17.56 | 0.0110 m |
| `fr1_xyz` | RDS-SLAM SegNet | 5.16 | 0.0100 m |
| `fr1_xyz` | RDS-SLAM MaskRCNN | 5.23 | 0.0104 m |
| `fr1_xyz` | Panoptic-SLAM | 1.93 | 0.0109 m |
| `fr1_xyz` | DynaSLAM | 0.15 | 0.0113 m |
| `fr3_walking_xyz` | ORB-SLAM3 | 15.02 | 0.7212 m |
| `fr3_walking_xyz` | Panoptic-SLAM | 2.20 | 0.0139 m |
| `fr3_walking_xyz` | DynaSLAM | 0.93 | 0.3680 m |
| `fr3_walking_xyz` | RDS-SLAM MaskRCNN | 3.96 | 0.4084 m |
| `fr3_walking_xyz` | RDS-SLAM SegNet | 5.32 | 0.6803 m |

Lecture provisoire : les methodes semantiques lourdes ameliorent souvent la
robustesse dans les scenes dynamiques, mais leur cout d'inference et de
synchronisation reduit fortement le debit. AntSSLAM part de l'hypothese inverse :
garder le SLAM geometrique rapide, puis injecter la semantique comme un signal
asynchrone controle, avec reutilisation temporelle des masques.

## Protocole prevu

Les sequences minimales sont :

- `rgbd_dataset_freiburg1_xyz` : scene statique de controle.
- `rgbd_dataset_freiburg3_walking_xyz` : scene dynamique pour tester l'apport des masques.
- `rgbd_dataset_freiburg2_large_with_loop` : sequence avec boucle pour observer les modules globaux.

Les metriques a conserver pour chaque methode :

- temps image par image du tracking, quand le code le permet ;
- FPS robot = inverse du temps moyen par image traite ;
- temps mur uniquement comme information secondaire ;
- ATE RMSE, RPE translation et RPE rotation ;
- nombre de poses associees ;
- temps de segmentation ou age des masques pour les methodes semantiques ;
- comportement de tracking : pertes, resets, inliers et map points suivis.

## Etat actuel

- `methods/antsslam` compile en mode geometrique dans `thesis/antsslam:jetson-r36.3-tum`.
- Le run de controle sur `fr1_xyz` fonctionne : tracking moyen 21.079 ms, soit 47.44 FPS tracking, ATE RMSE 0.0123 m.
- `methods/antsslam` compile maintenant aussi avec `ANTSSLAM_TENSORRT=ON`; l'image `thesis/antsslam:jetson-r36.3-tum` contient le support TensorRT semantique.
- Le premier run TokenMask TensorRT sur `fr3_walking_xyz` fonctionne :
  - dossier : `methods/antsslam/results/profiles/rgbd_dataset_freiburg3_walking_xyz_20260521_004159`
  - tracking moyen : 29.55 ms, soit 33.83 FPS tracking;
  - P95 tracking : 88.02 ms, 12.7% des frames au-dessus de 33.3 ms;
  - ATE RMSE : 0.0186 m, RPE translation RMSE : 0.0157 m;
  - GPU GR3D moyen : 29.2%, pics a 99%.
- Les runs AntSSLAM SegNet et Mask R-CNN sur `fr3_walking_xyz` fonctionnent aussi :
  - SegNet : 55.58 ms, 17.99 FPS tracking, ATE RMSE 0.0176 m, GPU moyen 54.4%;
  - Mask R-CNN : 46.82 ms, 21.36 FPS tracking, ATE RMSE 0.0168 m, GPU moyen 61.5%.
- Les wrappers historiques `dynaslam`, `rds_slam`, `panoptic_slam`, `vdo_slam` et `dynamic_vins` ont ete rapatries dans `methods/`.
- Les sources locales retrouvees pour DynaSLAM et `yolo_ros` ont ete copiees dans `third_party/semantic`; les autres sources externes seront clonees par leurs scripts de build si aucune copie locale n'est disponible.
- DynaSLAM a ete reconstruit dans le Docker dedie `thesis/dynaslam:tum-rgbd`.
- Smoke test DynaSLAM sur les 80 premieres images de `fr1_xyz` :
  - dossier : `methods/dynaslam/results/profiles/rgbd_dataset_freiburg1_xyz_20260521_005831`;
  - run exit code : 0;
  - temps moyen de tracking parse dans les logs : 0.564 s/image, soit environ 1.77 FPS tracking;
  - ATE RMSE sur les 80 poses estimees : 0.0093 m.
- La prochaine etape est de reconstruire RDS-SLAM, Panoptic-SLAM, VDO-SLAM et Dynamic-VINS dans leurs Docker dedies, puis de relancer DynaSLAM et les autres methodes sur les trois sequences completes du protocole.
