#!/usr/bin/env python3
"""Install a shared ORB vocabulary outside Git, from an archive or the pinned upstream."""
import argparse
import hashlib
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
source = parser.add_mutually_exclusive_group(required=True)
source.add_argument('--archive', type=Path, help='ORBvoc.txt.tar.gz local')
source.add_argument('--download', action='store_true', help='Télécharger le vocabulaire ORB-SLAM3 amont')
args = parser.parse_args()
commit = '4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4'
expected_git_blob = '96f5a8184a1b401dae9b51fd8f3aae03de0a8b20'
with tempfile.TemporaryDirectory() as temporary:
    archive = args.archive
    if args.download:
        archive = Path(temporary) / 'ORBvoc.txt.tar.gz'
        url = f'https://raw.githubusercontent.com/UZ-SLAMLab/ORB_SLAM3/{commit}/Vocabulary/ORBvoc.txt.tar.gz'
        urllib.request.urlretrieve(url, archive)
        data = archive.read_bytes()
        actual = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
        if actual != expected_git_blob:
            raise SystemExit('Le vocabulaire téléchargé ne correspond pas au blob amont attendu.')
    with tarfile.open(archive) as contents:
        matches = [m for m in contents.getmembers() if Path(m.name).name == 'ORBvoc.txt' and m.isfile()]
        if len(matches) != 1:
            raise SystemExit('Archive invalide : un fichier ORBvoc.txt est attendu.')
        extracted = Path(temporary) / 'ORBvoc.txt'
        with contents.extractfile(matches[0]) as input_file, extracted.open('wb') as output_file:
            shutil.copyfileobj(input_file, output_file)
    targets = sorted(p for p in (root / 'third_party').rglob('Vocabulary') if p.is_dir())
    for directory in targets:
        shutil.copyfile(extracted, directory / 'ORBvoc.txt')
    print(f'Vocabulaire installé dans {len(targets)} dossiers Vocabulary (ignorés par Git).')
