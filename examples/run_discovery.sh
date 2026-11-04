#!/usr/bin/env bash
# Portable Bash runner for an installed KOVAR environment; no scheduler needed.
# Pass the same options as `ko-variation discover`. Example:
# bash examples/run_discovery.sh --fasta sample_ids.fasta --tree core.treefile \
#   --unitigs graph.unitigs --paths graph.paths --unitig-distance unitig_distance \
#   --out results --threads 4 --memory-gb 4
# Set KOVAR_PYTHON=/path/to/environment/bin/python to select an environment.
set -euo pipefail
KOVAR_PYTHON="${KOVAR_PYTHON:-python3}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

args=("$@")
output=""
work=""
resume=0
for ((i=0; i<${#args[@]}; i++)); do
  case "${args[i]}" in
    --out) output="${args[i+1]:-}" ;;
    --out=*) output="${args[i]#--out=}" ;;
    --tmp-dir) work="${args[i+1]:-}" ;;
    --tmp-dir=*) work="${args[i]#--tmp-dir=}" ;;
    --resume) resume=1 ;;
  esac
done

# The main pipeline validates inputs/settings and reuses completed preparation,
# null models, pair decisions, distances and beta fits. Keep persistent work.
if (( resume == 0 )) && {
     [[ -n "$output" && -s "$output/run_identity.json" ]] ||
     [[ -n "$work" && -s "$work/run_identity.json" ]];
   }; then
  args+=(--resume)
  echo "[runner] Resuming the existing checkpoint"
fi
exec "$KOVAR_PYTHON" -m ko_variation.discovery "${args[@]}"
