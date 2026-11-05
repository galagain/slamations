# AntSLAM

AntSLAM est un fork local frais d'ORB-SLAM3 destiné aux expérimentations de thèse sur l'optimisation embarquée.

La version actuelle conserve l'architecture ORB-SLAM3 complète, mais rend le benchmark RGB-D plus proche d'un scénario robot en ligne:

- viewer désactivé par défaut avec `ANTSLAM_VIEWER=0`;
- attente artificielle TUM désactivée par défaut avec `ANTSLAM_REALTIME_SLEEP=0`;
- export image par image dans `FrameTrackTimes.txt`;
- instrumentation ORB-SLAM3 fine reprise de la baseline.
- politique automatique par support géométrique: après 200 images, AntSLAM choisit un mode `dynamic/fast` ou `loop/low-support`;
- budget ORB conditionnel: 900 features en mode rapide, 700 features en mode boucle;
- pose optimization espacée uniquement en mode boucle.

Build:

```bash
./methods/antslam/build.sh
```

Run:

```bash
./methods/antslam/run_tum_profile.sh /home/nvidia/data/tum_rgbd/rgbd_dataset_freiburg1_xyz
```

Profil courant:

```bash
./methods/antslam/run_tum_profile.sh /home/nvidia/data/tum_rgbd/rgbd_dataset_freiburg1_xyz
```

Les valeurs par défaut sont définies dans `run_tum.sh`:

```bash
ANTSLAM_POSE_OPT_AUTO=1
ANTSLAM_POSE_OPT_AUTO_INLIER_RATIO_PERMILLE_MAX=180
ANTSLAM_ORB_POLICY_LOOP_FEATURES=700
ANTSLAM_ORB_POLICY_FAST_FEATURES=900
```

Option experimentale de scheduling du local BA:

```bash
export ANTSLAM_LBA_INTERVAL=2
```

Cette option est compilee et reproductible, mais les premiers runs montrent qu'elle n'est pas retenue comme optimisation par defaut. Voir [ANALYSIS.md](ANALYSIS.md).
