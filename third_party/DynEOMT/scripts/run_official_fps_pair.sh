#!/usr/bin/env bash
set -u

if (( $# < 4 || $# > 5 )); then
  printf 'usage: %s RUN_ID GPU DATASET BACKBONE [WARMUP_ITERS]\n' "$0" >&2
  exit 2
fi

run_id=$1
gpu=$2
dataset=$3
backbone=$4
warmup_iters=${5:-100}
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=/data1/is156025/cg276974/codes/panoptic/envs/miniconda3/envs/eomt2/bin/python
output_dir="$repo_dir/outputs/runtime_official_a100/$run_id"

case "$dataset" in
  vipseg) cfg_dataset=VIPSeg ;;
  ovis) cfg_dataset=ovis ;;
  ytvis2022) cfg_dataset=ytvis22 ;;
  vspw) cfg_dataset=VSPW ;;
  *) printf 'unknown dataset: %s\n' "$dataset" >&2; exit 2 ;;
esac

case "$backbone" in
  vit-small) tag=vits ;;
  vit-base) tag=vitb ;;
  vit-large) tag=vitl ;;
  *) printf 'unknown backbone: %s\n' "$backbone" >&2; exit 2 ;;
esac

weights_for() {
  local method=$1
  if [[ $method == VidEoMT ]]; then
    if [[ $backbone == vit-large ]]; then
      printf 'outputs/videomt_retrain_same_protocol_finalweights/%s/%s/model_final.pth' "$dataset" "$backbone"
    else
      printf 'outputs/videomt_retrain/%s/%s/model_final.pth' "$dataset" "$backbone"
    fi
  elif [[ $dataset == ovis || $dataset == ytvis2022 ]]; then
    printf 'outputs/dyneomt_corrected_temporal_targets/%s/%s/model_final.pth' "$dataset" "$backbone"
  elif [[ $backbone == vit-large ]]; then
    printf '../outputs/dyneomt_models/%s/model_final.pth' "$dataset"
  else
    printf 'outputs/dyneomt/%s/%s/model_final.pth' "$dataset" "$backbone"
  fi
}

cd "$repo_dir"
cache_root="$output_dir/cache/gpu${gpu}"
mkdir -p "$output_dir/logs" "$cache_root/triton" "$cache_root/inductor"

for method in VidEoMT DynEoMT; do
  [[ $method == VidEoMT ]] && prefix=videomt || prefix=dyneomt
  config="configs/$method/$cfg_dataset/dinov2/$backbone/${prefix}_online_${tag}.yaml"
  weights=$(weights_for "$method")
  log_file="$output_dir/logs/${dataset}_${backbone}_${method}.log"

  printf '[start] %s %s %s gpu=%s %s\n' "$dataset" "$backbone" "$method" "$gpu" "$(date -u +%FT%TZ)" | tee "$log_file"
  TRITON_CACHE_DIR="$cache_root/triton" \
  TORCHINDUCTOR_CACHE_DIR="$cache_root/inductor" \
  PYTHONUNBUFFERED=1 \
  CUDA_VISIBLE_DEVICES="$gpu" \
    "$python_bin" "$repo_dir/benchmark.py" \
      --task fps \
      --config-file "$repo_dir/$config" \
      --model-weights "$repo_dir/$weights" \
      --warmup-iters "$warmup_iters" \
      --model-type dinov2 \
      >>"$log_file" 2>&1
  status=$?
  printf '[done] status=%s %s\n' "$status" "$(date -u +%FT%TZ)" | tee -a "$log_file"
done
