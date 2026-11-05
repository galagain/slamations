from pathlib import Path
root = Path(__file__).resolve().parents[1]
for method in sorted((root / 'methods').iterdir()):
    if not method.is_dir():
        continue
    build = any((method / name).exists() for name in ('build.sh', 'build_image.sh')) or method.name == 'antslam_fast_cuda'
    run = (method / 'run_tum.sh').exists()
    status = 'build et lancement TUM' if build and run else 'documentation / composant à intégrer'
    print(f'{method.name:28} {status}')
