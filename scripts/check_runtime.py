#!/usr/bin/env python3
"""Read-only inventory of the runtime requirements; does not build or launch a container."""
import os
import json
import platform
import shutil
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
print('Architecture :', platform.machine(), '(scripts NVIDIA principalement prévus pour Jetson aarch64)')
for command in ('docker', 'rsync', 'python3', 'git'):
    print(f'{command}:', shutil.which(command) or 'MANQUANT')
if shutil.which('docker'):
    result = subprocess.run(['docker', 'info', '--format', '{{json .Runtimes}}'], capture_output=True, text=True)
    print('Runtimes Docker :', ', '.join(sorted(json.loads(result.stdout))) if result.returncode == 0 else 'daemon inaccessible')
    for image in ('fastsem/nitroslam:jetson-r36.3-tumvi-speed-nostats', 'dyneomt-benchmark:jp6', 'fastsem/dynslam:ft11000-dyneomt', 'thesis/antsslam:jetson-r36.3-tum'):
        result = subprocess.run(['docker', 'image', 'inspect', image], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f'Image {image}:', 'présente' if result.returncode == 0 else 'à construire/importer')
vocabularies = list((root / 'third_party').rglob('Vocabulary'))
missing = [p for p in vocabularies if p.is_dir() and not (p / 'ORBvoc.txt').exists() and not (p / 'ORBvoc.txt.tar.gz').exists()]
print(f'Vocabulaires ORB : {len(missing)} dossier(s) à préparer avec scripts/prepare_vocabulary.py')
for name in ('ANTSSLAM_TOKENMASK_MODELS_DIR', 'ANTSSLAM_SEGMENTATION_DIR', 'ANTSSLAM_TRT_CACHE_DIR', 'DYNSLAM_DYNEOMT_CHECKPOINT_ROOT'):
    value = os.environ.get(name)
    print(f'{name}:', 'dossier présent' if value and Path(value).is_dir() else 'à renseigner pour la méthode concernée')
print('Ces contrôles ne valident pas une compilation ni une inférence.')
