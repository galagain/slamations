#!/usr/bin/env bash
set -u

DEPLOY_ROOT=${DEPLOY_ROOT:-/home/nvidia/dyneomt_jetson}
IMAGE=${IMAGE:-dyneomt-benchmark:jp6}
RESULT_ROOT="$DEPLOY_ROOT/results/runtime_table_same20"
CODE_ROOT="$DEPLOY_ROOT/code"
CHECKPOINT_ROOT="$DEPLOY_ROOT/checkpoints/runtime_table"

mkdir -p "$RESULT_ROOT/logs"
printf 'dataset\tbackbone\tmodel\tfps\tms_per_frame\tframes\tcuda_time_ms\tstatus\n' \
  > "$RESULT_ROOT/results.tsv"

datasets=(vipseg ovis ytvis22 vspw)
scales=(s b l)
methods=(videomt dyneomt)

declare -A dataset_dir=(
  [vipseg]="VIPSeg"
  [ovis]="ovis"
  [ytvis22]="ytvis22"
  [vspw]="VSPW"
)
declare -A scale_dir=([s]="vit-small" [b]="vit-base" [l]="vit-large")
declare -A scale_name=([s]="vits" [b]="vitb" [l]="vitl")

for dataset in "${datasets[@]}"; do
  for scale in "${scales[@]}"; do
    for method in "${methods[@]}"; do
      case_id="${dataset}_${scale}_${method}"
      log="$RESULT_ROOT/logs/${case_id}.log"
      if [[ "$method" == "videomt" ]]; then
        config="configs/VidEoMT/${dataset_dir[$dataset]}/dinov2/${scale_dir[$scale]}/videomt_online_${scale_name[$scale]}.yaml"
      else
        config="configs/DynEoMT/${dataset_dir[$dataset]}/dinov2/${scale_dir[$scale]}/dyneomt_online_${scale_name[$scale]}.yaml"
      fi
      checkpoint="/deploy/checkpoints/runtime_table/vit${scale}.pth"

      if grep -q '^Final online FPS:' "$log" 2>/dev/null; then
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Reusing completed $case_id"
        status=0
      else
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting $case_id"
        docker run --rm \
          --name "dyneomt_rt_${case_id}" \
          --runtime nvidia \
          --ipc=host \
          --entrypoint /bin/bash \
          -e PYTHONPATH=/deploy/code \
          -e DETECTRON2_DATASETS=/deploy/datasets \
          -e DYNEOMT_SKIP_PRETRAINED=1 \
          -e HF_HUB_OFFLINE=1 \
          -e TORCHINDUCTOR_CACHE_DIR=/deploy/cache/torchinductor \
          -v "$DEPLOY_ROOT:/deploy" \
          "$IMAGE" -lc \
          "cd /deploy/code && python3 benchmark.py \
            --task fps \
            --config-file '$config' \
            --model-weights '$checkpoint' \
            --warmup-iters 20 \
            --measure-same-subset \
            --measure-batches 20 \
            --model-type dinov2" > "$log" 2>&1
        status=$?
      fi

      fps=$(awk -F': ' '/^Final online FPS:/{value=$2} END{print value}' "$log")
      frames=$(awk -F': ' '/^Measured frames:/{value=$2} END{print value}' "$log")
      cuda_ms=$(awk -F': ' '/^Measured CUDA time:/{value=$2} END{print value}' "$log" | awk '{print $1}')
      if [[ -n "$cuda_ms" && -n "$frames" && "$frames" != "0" ]]; then
        ms_per_frame=$(awk -v total="$cuda_ms" -v count="$frames" 'BEGIN{printf "%.2f", total/count}')
      else
        ms_per_frame=""
      fi
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$dataset" "$scale" "$method" "$fps" "$ms_per_frame" \
        "$frames" "$cuda_ms" "$status" >> "$RESULT_ROOT/results.tsv"
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Finished $case_id: status=$status fps=${fps:-NA}"
    done
  done
done

echo "All runtime-table cases finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
