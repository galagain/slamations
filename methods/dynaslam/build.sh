#!/usr/bin/env bash
set -euo pipefail

method_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$method_dir/../.." && pwd)"
source_dir="${DYNASLAM_SOURCE:-$root_dir/third_party/semantic/DynaSLAM}"
image="${DYNASLAM_IMAGE:-thesis/dynaslam:tum-rgbd}"

if [[ ! -d "$source_dir" ]]; then
  git clone --depth 1 https://github.com/BertaBescos/DynaSLAM.git "$source_dir"
fi

"$root_dir/common/stage_context.sh" "$method_dir" "$source_dir"
docker build -t "$image" "$method_dir/.docker_context"
echo "$image"
