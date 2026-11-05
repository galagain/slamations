#!/usr/bin/env bash
set -euo pipefail

method_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$method_dir/../.." && pwd)"

image="${DYNSLAM_IMAGE:-fastsem/dynslam:ft11000-dyneomt}"
base_image="${DYNSLAM_BASE_IMAGE:-fastsem/nitroslam:jetson-r36.3-tumvi-speed-nostats}"
build_jobs="${DYNSLAM_BUILD_JOBS:-2}"
semantic_src="${DYNSLAM_SEMANTIC_SOURCE:-$repo_root/third_party/AntSSLAM}"
context="$method_dir/.docker_context"

[[ -d "$semantic_src" ]] || { echo "Semantic source not found: $semantic_src" >&2; exit 1; }

rm -rf "$context"
mkdir -p "$context/scripts" "$context/semantic/include" "$context/semantic/src"

cp "$method_dir/docker/Dockerfile" "$context/Dockerfile"
cp "$method_dir/scripts/patch_nitro_with_semantics.py" "$context/scripts/"
cp "$semantic_src/include/SemanticMasker.h" "$context/semantic/include/"
cp "$semantic_src/src/SemanticMasker.cc" "$context/semantic/src/"
cp "$semantic_src/include/Frame.h" "$context/semantic/include/"
cp "$semantic_src/src/Frame.cc" "$context/semantic/src/"
cp "$semantic_src/include/Tracking.h" "$context/semantic/include/"
cp "$semantic_src/src/Tracking.cc" "$context/semantic/src/"

docker build \
  --build-arg BASE_IMAGE="$base_image" \
  --build-arg BUILD_JOBS="$build_jobs" \
  -t "$image" \
  "$context"

echo "Built DynSLAM image: $image"
