#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
context="$here/.docker_context"
command -v rsync >/dev/null || { echo 'Installer rsync avant de construire cette image.' >&2; exit 1; }
mkdir -p "$context"
rsync -a --delete --exclude='.git' --exclude='build*' --exclude='*.so*' --exclude='*.o' --exclude='*.a' "$root/third_party/Nitro-SLAM/" "$context/src/"
rsync -a --delete --exclude='.git' --exclude='build*' "$root/third_party/Pangolin-v0.5/" "$context/pangolin/"
rsync -a --delete "$here/scripts/" "$context/scripts/"
cp "$here/Dockerfile" "$context/Dockerfile"
docker build --build-arg BUILD_JOBS="${BUILD_JOBS:-2}" --build-arg NITRO_REGISTER_STATS=0 -t "${DYNSLAM_BASE_IMAGE:-fastsem/nitroslam:jetson-r36.3-tumvi-speed-nostats}" "$context"
