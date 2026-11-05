# SLAM Methods

Les 24 dossiers de méthodes SLAM du workspace de thèse, préparés pour relancer
les pipelines sans emporter les campagnes de tests et leurs sorties.
Les sources modifiées, licences, Dockerfiles, scripts de build/lancement et
configurations de base sont conservés.

## Démarrage

La plupart des images ciblent **NVIDIA Jetson aarch64, JetPack 6 / R36.3**.
Il faut Docker avec le runtime NVIDIA, Bash, Python 3, Git et rsync sur l'hôte.
Les méthodes ROS historiques ont leurs propres versions dans leurs Dockerfiles.

```bash
./slam list
./slam check

# Préparer une fois le vocabulaire ORB, à partir d'une archive locale :
python3 scripts/prepare_vocabulary.py --archive /chemin/ORBvoc.txt.tar.gz
# Ou le télécharger depuis le commit amont vérifié :
python3 scripts/prepare_vocabulary.py --download

# Construire et lancer une méthode sur UNE séquence TUM RGB-D :
./slam build orbslam3_baseline
./slam run orbslam3_baseline /chemin/rgbd_dataset_freiburg1_xyz
```

Les sorties sont créées à la demande sous `methods/<methode>/results/` et ignorées
par Git. Les commandes directes `methods/<methode>/build.sh`, `run_tum.sh` et
`run_tum_profile.sh` restent disponibles lorsqu'elles existent.
`antslam_fast_cuda` réutilise le build AntSLAM.

## TokenMask et VidEoMT / DynEoMT

```bash
# Le lanceur active la compilation TensorRT pour AntSSLAM.
./slam build antsslam
export ANTSSLAM_TOKENMASK_MODELS_DIR=/chemin/modeles_tokenmask
export ANTSSLAM_SEGMENTATION_DIR=/chemin/segmentation
export ANTSSLAM_TRT_CACHE_DIR=/chemin/cache_tensorrt
./slam run antsslam /chemin/rgbd_dataset_freiburg3_walking_xyz

# Construit aussi les bases Nitro-SLAM et DynEoMT si absentes localement.
./slam build DynSLAM
export DYNSLAM_DYNEOMT_CHECKPOINT_ROOT=/chemin/checkpoints
export DYNSLAM_DYNEOMT_CHECKPOINT_REL=vipseg_vitl_full.pth
./slam run DynSLAM /chemin/rgbd_dataset_freiburg3_walking_xyz
```

Les poids sont à fournir séparément. Voir [les ressources et Docker](RUNTIME.md)
pour les autres dépendances, les variables et le transfert d'images déjà construites.
`./slam list` distingue les pipelines TUM des composants documentés qui demandent
encore une intégration (Kimera, Isaac ROS, gpu-block-solver, iBoW-LCD, etc.).

## Contenu

| Dossier | Contenu |
| --- | --- |
| `methods/` | 24 dossiers, Dockerfiles existants, patches et scripts |
| `common/` | Association TUM, lancement Docker, évaluation et profilage d'un run |
| `third_party/` | Sources tierces et modifications locales, licences conservées |
| `docker/nitroslam/` | Recette de l'image de base utilisée par DynSLAM |
| `docker/dyneomt/` | Recettes PyTorch/Triton et inférence DynEoMT |
| `scripts/` | Catalogue, vérification des prérequis et préparation du vocabulaire |

Configurations conservées :

- AntSLAM : preset géométrique `tum_rgbd_nfeatures_1000`.
- AntSSLAM : base géométrique et presets TokenMask stride 16 exact, SegNet et Mask R-CNN.
- DynSLAM : preset DynEoMT et masque binaire FT=11000 ; calibrations TUM1/2/3.
- Autres méthodes : calibrations et configurations livrées avec leurs sources.

Les variantes de sweep, campagnes, anciens résultats, logs, vidéos, données
exemples volumineuses, poids, images Docker et fichiers compilés sont exclus.
Les recettes de build reconstruisent les contextes Docker à la demande.
Les sources C++ et calibrations d'`Examples_old` référencées par certains CMake
restent disponibles ; leurs données de test sont exclues.

## Provenance et validation

[SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json) indique les origines et commits
récupérés. [EXPORT_POLICY.json](EXPORT_POLICY.json) résume le contenu retenu.
Les README historiques des méthodes peuvent encore citer des campagnes absentes.
Les Dockerfiles et sources ne garantissent pas que toutes les anciennes piles
se reconstruisent sans adaptation : les nouvelles commandes ont été vérifiées
sans lancer une compilation complète ni une nouvelle campagne SLAM.

Chaque composant conserve sa licence. Aucune licence globale supplémentaire
n'est attribuée à cet assemblage.
