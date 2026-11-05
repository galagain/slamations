#!/usr/bin/env bash
set -u

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=/data1/is156025/cg276974/codes/panoptic/envs/miniconda3/envs/eomt2/bin/python
run_id=${1:-$(date -u +%Y%m%d_%H%M%S)}
output_dir="$repo_dir/outputs/runtime_official_a100/$run_id"
cd "$repo_dir"
mkdir -p "$output_dir/logs" "$output_dir/cache"

gpus=(2 3 4 5 6)
datasets=(vipseg ovis ytvis2022 vspw)
backbones=(vit-small vit-base vit-large)

dataset_config() {
  case "$1" in
    vipseg) printf 'VIPSeg' ;;
    ovis) printf 'ovis' ;;
    ytvis2022) printf 'ytvis22' ;;
    vspw) printf 'VSPW' ;;
  esac
}

backbone_tag() {
  case "$1" in
    vit-small) printf 'vits' ;;
    vit-base) printf 'vitb' ;;
    vit-large) printf 'vitl' ;;
  esac
}

warmup_for() {
  if [[ $1 == vspw ]]; then
    printf '200'
  else
    printf '100'
  fi
}

weights_for() {
  local method=$1 dataset=$2 backbone=$3
  if [[ $method == VidEoMT ]]; then
    if [[ $backbone == vit-large ]]; then
      printf 'outputs/videomt_retrain_same_protocol_finalweights/%s/%s/model_final.pth' "$dataset" "$backbone"
    else
      printf 'outputs/videomt_retrain/%s/%s/model_final.pth' "$dataset" "$backbone"
    fi
    return
  fi

  if [[ $dataset == ovis || $dataset == ytvis2022 ]]; then
    printf 'outputs/dyneomt_corrected_temporal_targets/%s/%s/model_final.pth' "$dataset" "$backbone"
  elif [[ $backbone == vit-large ]]; then
    printf '../outputs/dyneomt_models/%s/model_final.pth' "$dataset"
  else
    printf 'outputs/dyneomt/%s/%s/model_final.pth' "$dataset" "$backbone"
  fi
}

run_pair() {
  local gpu=$1 dataset=$2 backbone=$3
  local cfg_dataset tag method prefix config weights log_file cache_root warmup_iters
  cfg_dataset=$(dataset_config "$dataset")
  tag=$(backbone_tag "$backbone")
  warmup_iters=$(warmup_for "$dataset")
  cache_root="$output_dir/cache/gpu${gpu}"
  mkdir -p "$cache_root/triton" "$cache_root/inductor"

  for method in VidEoMT DynEoMT; do
    if [[ $method == VidEoMT ]]; then
      prefix=videomt
    else
      prefix=dyneomt
    fi
    config="configs/$method/$cfg_dataset/dinov2/$backbone/${prefix}_online_${tag}.yaml"
    weights=$(weights_for "$method" "$dataset" "$backbone")
    log_file="$output_dir/logs/${dataset}_${backbone}_${method}.log"

    printf '[start] %s %s %s gpu=%s %s\n' "$dataset" "$backbone" "$method" "$gpu" "$(date -u +%FT%TZ)" | tee "$log_file"
    if [[ ! -f "$repo_dir/$config" || ! -f "$repo_dir/$weights" ]]; then
      printf '[error] missing config or weights: %s %s\n' "$config" "$weights" | tee -a "$log_file"
      continue
    fi

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
}

{
  printf 'run_id=%s\n' "$run_id"
  printf 'python=%s\n' "$python_bin"
  "$python_bin" -c 'import sys, torch, triton; print("python_version=" + sys.version.replace("\n", " ")); print("torch=" + torch.__version__); print("cuda=" + str(torch.version.cuda)); print("triton=" + triton.__version__)'
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,power.limit --format=csv,noheader
  printf 'protocol=official backbone-only FP16 torch.compile batch=1 warmup_iters=100 (VSPW: 200)\n'
} >"$output_dir/environment.txt"

pids=()
pair_index=0
for dataset in "${datasets[@]}"; do
  for backbone in "${backbones[@]}"; do
    gpu=${gpus[$((pair_index % ${#gpus[@]}))]}
    run_pair "$gpu" "$dataset" "$backbone" &
    pids+=("$!")
    pair_index=$((pair_index + 1))
    if (( pair_index % ${#gpus[@]} == 0 )); then
      for pid in "${pids[@]}"; do wait "$pid"; done
      pids=()
    fi
  done
done
for pid in "${pids[@]}"; do wait "$pid"; done

results_csv="$output_dir/results.csv"
printf 'dataset,backbone,method,fps,measured_frames,status,compile_errors,log\n' >"$results_csv"
for log_file in "$output_dir"/logs/*.log; do
  base=$(basename "$log_file" .log)
  dataset=${base%%_*}
  remainder=${base#*_}
  method=${remainder##*_}
  backbone=${remainder%_*}
  fps=$(sed -n 's/^Final online FPS: //p' "$log_file" | tail -1)
  frames=$(sed -n 's/^Measured frames: //p' "$log_file" | tail -1)
  status=$(sed -n 's/^\[done\] status=\([0-9][0-9]*\).*/\1/p' "$log_file" | tail -1)
  compile_errors=$(grep -Ec 'InductorError|PY_SSIZE_T_CLEAN|WON.T CONVERT' "$log_file" || true)
  printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "$dataset" "$backbone" "$method" "$fps" "$frames" "$status" "$compile_errors" "${log_file#$repo_dir/}" >>"$results_csv"
done

printf '%s\n' "$output_dir"
