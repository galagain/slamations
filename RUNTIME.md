# Ressources et images Docker

## Ce qui reste hors de Git

| Ressource | Préparation |
| --- | --- |
| Séquence TUM RGB-D | Dossier externe avec `rgb.txt`, `depth.txt`, images et, pour évaluer, `groundtruth.txt` |
| Vocabulaire ORB | `python3 scripts/prepare_vocabulary.py --archive /chemin/ORBvoc.txt.tar.gz` ou `--download` |
| TokenMask | `ANTSSLAM_TOKENMASK_MODELS_DIR` ; fichier attendu dans les YAML du preset conservé |
| SegNet / Mask R-CNN | Dossiers `ANTSSLAM_SEGMENTATION_DIR` et `ANTSSLAM_TRT_CACHE_DIR` ; consulter les YAML |
| DynEoMT | `DYNSLAM_DYNEOMT_CHECKPOINT_ROOT` et `DYNSLAM_DYNEOMT_CHECKPOINT_REL` |
| RDS-SLAM | Modèle `segnet_pascal.caffemodel` à placer dans le contexte source indiqué par `RDS_SLAM_SOURCE` avant build |
| Panoptic-SLAM | Wheels Jetson PyTorch/torchvision à fournir via `PANOPTIC_TORCH_WHEEL` et `PANOPTIC_TORCHVISION_WHEEL` |

Les scripts historiques exposent des variables `*_SOURCE`, `*_IMAGE` et `*_OUTPUT`.
Les définir pour remplacer les anciens chemins locaux ou choisir ses images.
Le lanceur ne télécharge aucun dataset ni checkpoint. Les builds Docker accèdent
au réseau pour installer leurs dépendances.

## Conserver les images déjà compilées

Les **Dockerfiles** sont versionnés. Les **images Docker** sont des fichiers
volumineux à transférer séparément ou à héberger dans un registre.
Pour transférer une image existante vers une autre machine compatible :

```bash
# Choisir un dossier externe au dépôt.
docker save fastsem/dynslam:ft11000-dyneomt | gzip > /chemin/externe/dynslam.tar.gz
# Sur l'autre machine :
gunzip -c /chemin/externe/dynslam.tar.gz | docker load
```

DynSLAM utilise aussi `dyneomt-benchmark:jp6` pour générer les masques ; transférer
cette image également pour éviter de reconstruire l'environnement d'inférence.
Vérifier l'architecture et la compatibilité JetPack du matériel cible.

## Recettes de DynSLAM

```bash
# Facultatif : construire les dépendances individuellement.
./docker/nitroslam/build.sh
./docker/dyneomt/build.sh
./methods/DynSLAM/build_image.sh
```

La recette Nitro-SLAM provient du lanceur local retrouvé dans
`Desktop/FastSem/nitroslam_launcher` ; ses sources sont dans
`third_party/Nitro-SLAM`. Elle désactive les statistiques détaillées au build.
L'équivalence binaire avec l'ancienne image `speed-nostats` n'a pas été vérifiée.

La pile DynEoMT part de `dustynv/pytorch:2.7-r36.4.0`, construit Triton 3.3 puis
Detectron2 et les dépendances d'inférence. L'archive LLVM ARM64 est téléchargée
pendant le build, sans être stockée dans Git. Ces compilations sont coûteuses ;
le transfert des images déjà disponibles évite ce travail.

## Configurations conservées

```bash
# AntSSLAM : un backend alternatif au TokenMask par défaut.
SLAM_SETTINGS_DIR=/opt/method_configs/tum_rgbd_semantic_segnet_latest \
  ./slam run antsslam /chemin/sequence_TUM

# DynSLAM : afficher le preset sans lancer de conteneur.
./methods/DynSLAM/run_tum.sh --show-config
```

AntSLAM et AntSSLAM conservent la calibration TUM1, TUM2 et TUM3 dans chaque
preset retenu. Les autres réglages restent accessibles par les variables des
scripts ; les configurations générées et les sorties de profilage ne sont pas
versionnées.
