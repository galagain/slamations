#!/usr/bin/env bash
set -euo pipefail

run_id=${1:?usage: collect_official_fps_results.sh RUN_ID}
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output_dir="$repo_dir/outputs/runtime_official_a100/$run_id"
results_csv="$output_dir/results.csv"

printf 'dataset,backbone,method,fps,measured_frames,status,compile_errors,recompile_warnings,log\n' >"$results_csv"

missing=0
failed=0
for dataset in vipseg ovis ytvis2022 vspw; do
  for backbone in vit-small vit-base vit-large; do
    for method in VidEoMT DynEoMT; do
      log_file="$output_dir/logs/${dataset}_${backbone}_${method}.log"
      if [[ ! -f "$log_file" ]]; then
        printf '[missing] %s\n' "$log_file" >&2
        missing=$((missing + 1))
        continue
      fi

      fps=$(sed -n 's/^Final online FPS: //p' "$log_file" | tail -1)
      frames=$(sed -n 's/^Measured frames: //p' "$log_file" | tail -1)
      status=$(sed -n 's/^\[done\] status=\([0-9][0-9]*\).*/\1/p' "$log_file" | tail -1)
      compile_errors=$(grep -Ec 'InductorError|PY_SSIZE_T_CLEAN|WON.T CONVERT|Traceback|CUDA out of memory|Killed' "$log_file" || true)
      recompile_warnings=$(grep -Ec 'hit config.recompile_limit' "$log_file" || true)
      printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$dataset" "$backbone" "$method" "$fps" "$frames" "$status" \
        "$compile_errors" "$recompile_warnings" "${log_file#$repo_dir/}" >>"$results_csv"

      if [[ -z $fps || -z $frames || $status != 0 || $compile_errors != 0 ]]; then
        printf '[invalid] %s fps=%s frames=%s status=%s compile_errors=%s\n' \
          "$log_file" "$fps" "$frames" "$status" "$compile_errors" >&2
        failed=$((failed + 1))
      fi
    done
  done
done

printf 'rows=%s missing=%s invalid=%s\n' "$(($(wc -l <"$results_csv") - 1))" "$missing" "$failed"
(( missing == 0 && failed == 0 ))
