#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/is156025/cg276974/DynEOMT"
REPO="${ROOT}/DynEoMT"
PY="/data1/is156025/cg276974/codes/panoptic/envs/miniconda3/envs/eomt2/bin/python"
VIP_TRUTH="${ROOT}/data/VIPSeg/VIPSeg_720P/panomasksRGB"
VIP_JSON="${ROOT}/data/VIPSeg/VIPSeg_720P/panoptic_gt_VIPSeg_val.json"
LOG_DIR="${REPO}/logs/vipseg_backbone_eval"
MAIN_LOG="${LOG_DIR}/vipseg_backbone_eval.log"

mkdir -p "${LOG_DIR}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${MAIN_LOG}"
}

eval_one() {
  local name="$1"
  local inference_dir="$2"
  local safe_name
  safe_name="$(printf '%s' "${name}" | tr ' /' '__')"

  if [[ ! -d "${inference_dir}" ]]; then
    log "SKIP ${name}: missing inference dir ${inference_dir}"
    return 0
  fi
  if [[ ! -f "${inference_dir}/pred.json" ]]; then
    log "SKIP ${name}: missing ${inference_dir}/pred.json"
    return 0
  fi

  log "START ${name}"

  if [[ -s "${inference_dir}/vpq-final.txt" ]]; then
    log "SKIP VPQ ${name}: ${inference_dir}/vpq-final.txt already exists"
  else
    log "RUN VPQ ${name}"
    "${PY}" "${REPO}/utils/eval_vpq_vspw.py" \
      --submit_dir "${inference_dir}" \
      --truth_dir "${VIP_TRUTH}" \
      --pan_gt_json_file "${VIP_JSON}" \
      --num_processes 16 \
      > "${LOG_DIR}/${safe_name}_vpq.log" 2>&1
    log "DONE VPQ ${name}"
  fi

  if [[ -s "${inference_dir}/stq-final.txt" ]]; then
    log "SKIP STQ ${name}: ${inference_dir}/stq-final.txt already exists"
  else
    log "RUN STQ ${name}"
    "${PY}" "${REPO}/utils/eval_stq_vspw.py" \
      --submit_dir "${inference_dir}" \
      --truth_dir "${VIP_TRUTH}" \
      --pan_gt_json_file "${VIP_JSON}" \
      > "${LOG_DIR}/${safe_name}_stq.log" 2>&1
    log "DONE STQ ${name}"
  fi

  log "DONE ${name}"
}

log "VIPSeg backbone evaluation started"

eval_one "DynEoMT ViT-S" "${REPO}/outputs/dyneomt/vipseg/vit-small/inference"
eval_one "DynEoMT ViT-B" "${REPO}/outputs/dyneomt/vipseg/vit-base/inference"
eval_one "DynEoMT ViT-L" "${ROOT}/outputs/dyneomt_models/vipseg/inference"

eval_one "VidEoMT ViT-S" "${REPO}/outputs/videomt_retrain/vipseg/vit-small/inference"
eval_one "VidEoMT ViT-B" "${REPO}/outputs/videomt_retrain/vipseg/vit-base/inference"
eval_one "VidEoMT ViT-L" "${REPO}/outputs/videomt_retrain_same_protocol_finalweights/vipseg/vit-large/inference"

log "VIPSeg backbone evaluation finished"
