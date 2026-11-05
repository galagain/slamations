#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: stage_context.sh METHOD_DIR SOURCE [PANGOLIN_SOURCE]" >&2
  exit 2
}

method_dir="${1:-}"
source_dir="${2:-}"
pangolin_source="${3:-}"
[[ -n "$method_dir" && -n "$source_dir" ]] || usage

context_dir="$method_dir/.docker_context"

if [[ ! -d "$source_dir" ]]; then
  echo "Source directory not found: $source_dir" >&2
  exit 1
fi

if [[ -n "$pangolin_source" && ! -d "$pangolin_source" ]]; then
  echo "Pangolin source directory not found: $pangolin_source" >&2
  exit 1
fi

copy_tree() {
  local src="$1"
  local dst="$2"
  mkdir -p "$dst"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a \
      --delete \
      --exclude='.git' \
      --exclude='build' \
      --exclude='build_*' \
      --exclude='*.o' \
      --exclude='*.a' \
      --exclude='*.so' \
      --exclude='*.pdf' \
      --exclude='__pycache__' \
      --exclude='.docker_context' \
      --exclude='Examples_old' \
      --exclude='evaluation' \
      "$src"/ "$dst"/
  else
    rm -rf "$dst"
    mkdir -p "$dst"
    cp -a "$src"/. "$dst"/
    find "$dst" -name .git -type d -prune -exec rm -rf {} +
  fi
}

rm -rf "$context_dir"
mkdir -p "$context_dir"

copy_tree "$source_dir" "$context_dir/src"
if [[ -n "$pangolin_source" ]]; then
  copy_tree "$pangolin_source" "$context_dir/pangolin"
fi
copy_tree "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" "$context_dir/common"

if [[ -d "$method_dir/scripts" ]]; then
  copy_tree "$method_dir/scripts" "$context_dir/scripts"
fi

cp "$method_dir/Dockerfile" "$context_dir/Dockerfile"
