# ORB-SLAM3 Baseline

Cette methode sert de reference geometrique pour le chapitre "Optimized SLAM For Embedded Systems". ORB-SLAM3 est reclone depuis GitHub dans `../../third_party/ORB_SLAM3`; les fichiers ne proviennent pas des anciens dossiers locaux.

## Plateforme mesuree

- Carte: NVIDIA Jetson AGX Orin Developer Kit
- JetPack/L4T: R36.3.0
- Mode puissance observe: `MAXN`
- Kernel: `5.15.136-tegra`
- Date des premiers runs: 2026-05-16
- Note: `jetson_clocks --show` demande les privileges root sur cette machine; son etat exact n'a donc pas ete fige par le script.

## Sources

- ORB-SLAM3: `4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4`
- Pangolin v0.5: `7987c9ba6b6c2c46fdcf50baae28bfa4f8352696`
- Image Docker: `thesis/orbslam3:jetson-r36.3-profile`
- Base Docker: `nvcr.io/nvidia/l4t-jetpack:r36.3.0`

Le Dockerfile ne clone pas ORB-SLAM3 pendant le build. `build.sh` prepare un contexte local avec `../../third_party/ORB_SLAM3` et `../../third_party/Pangolin-v0.5`, puis les copie dans l'image. Cela permet de modifier l'instrumentation localement sans retelecharger les sources a chaque build.

## Instrumentation

Le clone ORB-SLAM3 reste un clone Git officiel, avec un diff local limite a l'instrumentation. Le patch exporte est `orbslam3_instrumentation.patch`.

Mesures ajoutees:

- temps total par image dans l'exemple RGB-D TUM;
- temps de tracking par sous-bloc: rectification, resize, ORB, association stereo/RGB-D, IMU, prediction de pose, tracking carte locale, decision keyframe;
- signaux de tracking: nombre de features, inliers, map points suivis, etat du tracker;
- temps de local mapping par sous-bloc: insertion keyframe, culling points, creation points, LBA, culling keyframes;
- taille de file keyframes avant traitement;
- correction du temps LBA enregistre dans les statistiques de mapping.

## Commandes

Construire l'image:

```bash
./methods/orbslam3_baseline/build.sh
```

Lancer un run profile:

```bash
./methods/orbslam3_baseline/run_tum_profile.sh /home/nvidia/data/tum_rgbd/rgbd_dataset_freiburg1_xyz
```

Generer une figure trajectoire/verite terrain pour un run:

```bash
python3 common/plot_tum_trajectory.py methods/orbslam3_baseline/results/profiles/rgbd_dataset_freiburg1_xyz_20260516_020926
```

Les donnees TUM sont montees en lecture seule depuis l'hote. Les sorties sont ecrites dans:

```text
methods/orbslam3_baseline/results/profiles/<sequence>_<timestamp>/
```

## Fichiers produits

- `TrackingTimeStats.txt`: temps tracking et signaux par image.
- `LocalMapTimeStats.txt`: temps local mapping et file de keyframes.
- `LBA_Stats.txt`: complexite du bundle adjustment local.
- `ExecMean.txt`: statistiques globales ORB-SLAM3, dont place recognition et modules globaux.
- `tegrastats.log`: charge CPU/GPU/RAM cote Jetson.
- `evaluation.json`: ATE/RPE TUM.
- `trajectory_overlay.png`: trajectoire estimee alignee, verite terrain et erreur absolue.
- `bottlenecks.md` et `bottlenecks.json`: synthese automatique detaillee.

## Premiers resultats

Une synthese plus detaillee est disponible dans `ANALYSIS.md`.

| Sequence | Tracking moyen | P95 tracking | Images > 33.3 ms | Dominant tracking | Dominant mapping | ATE RMSE |
|---|---:|---:|---:|---|---|---:|
| `rgbd_dataset_freiburg1_xyz` | 30.667 ms | 35.871 ms | 19.8% | ORB extraction, 13.328 ms | LBA, 210.440 ms | 0.010260 m |
| `rgbd_dataset_freiburg3_walking_xyz` | 39.362 ms | 49.673 ms | 57.4% | ORB extraction, 14.342 ms | KF insertion, 63.102 ms | 0.670017 m |
| `rgbd_dataset_freiburg2_large_with_loop` | 26.998 ms | 38.792 ms | 12.8% | ORB extraction, 12.352 ms | LBA, 119.188 ms | 0.154838 m |

Interpretation courte:

- Sur `freiburg1_xyz`, le front-end ORB represente environ 43.5% du temps tracking moyen, et le tracking carte locale environ 31.4%. Le local mapping est domine par le LBA, environ 71.9% du temps local mapping moyen.
- Sur `freiburg3_walking_xyz`, la scene dynamique degrade le suivi: moins d'inliers, moins de map points suivis, 88 frames en etat perdu/relocalisation et plusieurs resets. La file de keyframes monte jusqu'a 4.
- Sur `freiburg2_large_with_loop`, une fermeture de boucle est detectee: loop fusion 122.562 ms, essential graph 1545.942 ms, loop closing total 1673.178 ms, et un full GBA est lance sur 1015 keyframes et 26965 map points.
- La charge GPU `GR3D` reste moderee dans ces runs, ce qui est coherent avec ORB-SLAM3 de reference: le front-end ORB reste principalement CPU. C'est un argument experimental pour tester une acceleration CUDA du front-end, en gardant a l'esprit la future contention avec la segmentation semantique.

## Runs disponibles

- `results/profiles/rgbd_dataset_freiburg1_xyz_20260516_020926/`
- `results/profiles/rgbd_dataset_freiburg3_walking_xyz_20260516_021030/`
- `results/profiles/rgbd_dataset_freiburg2_large_with_loop_20260516_022827/`

Le run dynamique a produit les trajectoires et les statistiques, puis Pangolin/Xvfb a renvoye `BadDrawable` a la fermeture. Les fichiers ont ete post-traites et le code retour original est conserve dans `run_exit_code.txt`.
