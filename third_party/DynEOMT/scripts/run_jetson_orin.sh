#!/usr/bin/env bash
set -euo pipefail

deployment_root="${DYNEOMT_DEPLOY_ROOT:-/home/nvidia/dyneomt_jetson}"
input_dir="${1:-${deployment_root}/input/vipseg_516}"
output_dir="${2:-${deployment_root}/results/full720}"

docker run --rm --runtime nvidia --ipc=host \
  --entrypoint /bin/bash \
  -v "${deployment_root}:/deploy" \
  -e PYTHONPATH=/deploy/python:/deploy/code:/workspace/Mask2Former \
  -e DYNEOMT_SKIP_PRETRAINED=1 \
  -e HF_HUB_OFFLINE=1 \
  mask2former-orin:jp6 -lc \
  "cd /deploy/code && python3 run_jetson_demo.py \
    --config configs/DynEoMT/VIPSeg/dinov2/vit-large/dyneomt_online_vitl.yaml \
    --checkpoint /deploy/checkpoints/vipseg_vitl_full.pth \
    --input-dir /deploy${input_dir#${deployment_root}} \
    --output-dir /deploy${output_dir#${deployment_root}} \
    --short-side 720 --max-side 1280 --warmup-frames 3 --fast-exit"
