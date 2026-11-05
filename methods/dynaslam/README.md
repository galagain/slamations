# DynaSLAM sur TUM RGB-D

Source par defaut: `third_party/semantic/DynaSLAM` dans le workspace courant
(copie locale rapatriee depuis `/home/nvidia/Desktop/kalvici/campagne/_sources`
si disponible, clone cree sinon).

```bash
./dynaslam/build.sh
./dynaslam/run_tum.sh /chemin/rgbd_dataset_freiburg3_walking_xyz
```

Le lancement par defaut utilise le mode geometrique TUM RGB-D de DynaSLAM, sans Mask R-CNN. La correction geometrique reste active. Pour utiliser des masques Mask R-CNN pre-calcules, fournis un dossier avec les noms d'images TUM sans le prefixe `rgb/`:

```bash
DYNASLAM_MASKS=/chemin/vers/masks ./dynaslam/run_tum.sh /chemin/rgbd_dataset_freiburg3_walking_xyz
```

Smoke test court:

```bash
TUM_MAX_FRAMES=80 ./dynaslam/run_tum.sh /chemin/rgbd_dataset_freiburg3_walking_xyz
```

Viewer Pangolin sur Jetson/JetPack:

```bash
./dynaslam/build_viewer.sh
DYNASLAM_VIEWER=1 ./dynaslam/run_tum.sh /chemin/rgbd_dataset_freiburg3_walking_xyz
```

Quand `DYNASLAM_VIEWER=1`, le wrapper utilise l'image `campagne/dynaslam:tum-rgbd-viewer`, basee sur JetPack/Jammy, pour avoir un contexte OpenGL NVIDIA compatible avec le host.

Validation locale dans ce workspace:

- image `thesis/dynaslam:tum-rgbd` reconstruite;
- smoke test `fr1_xyz` sur 80 images OK;
- dossier de resultat: `methods/dynaslam/results/profiles/rgbd_dataset_freiburg1_xyz_20260521_005831`;
- tracking moyen parse: 0.564 s/image, soit environ 1.77 FPS tracking;
- ATE RMSE sur les 80 poses estimees: 0.0093 m.
