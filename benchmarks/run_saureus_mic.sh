#!/usr/bin/env bash
# Copy unchanged to scripts/zoo-ardbeg/run_saureus_mic.sh if desired.
# Runs with bash; no scheduler, discovery rerun, eggNOG or old MIC results needed.
set -euo pipefail
STAGE="${1:-all}"
if (( $# )); then shift; fi
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd -- "$HERE/../.." && pwd)}"
if [[ -z "${REPO:-}" ]]; then
  if [[ -f "$HERE/saureus_mic.py" ]]; then REPO="$(cd -- "$HERE/.." && pwd)"
  else REPO="$ROOT/ko-variation"; fi
fi
PYTHON="${KOVAR_PYTHON:-python}"
OUT="${OUT:-$ROOT/kovar_output/saureus/diagnostics/saureus_mic}"
THREADS="${THREADS:-4}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
args=("$STAGE" --out "$OUT" --threads "$THREADS" --random-sets "${RANDOM_SETS:-100}"
      --quinolone "${QUINOLONE:-levofloxacin}" --pcs "${PCS:-10}" --folds "${FOLDS:-5}"
      --groups "${GROUPS_COUNT:-10}" --seed "${SEED:-42}")
case "$STAGE" in
  map|prepare|all)
    if [[ -z "${MAP_DIR:-}" ]]; then
      for candidate in "$ROOT/kovar_output/saureus/plots/functional_map" \
                       "$ROOT/kovar_output/saureus/zoo-ardbeg/plots/functional_map"; do
        if [[ -f "$candidate/map.html" || -f "$candidate/map_saureus.html" ]]; then MAP_DIR="$candidate"; break; fi
      done
    fi
    if [[ -z "${NETWORK:-}" ]]; then
      for candidate in "${MAP_DIR:-}/map.html" "${MAP_DIR:-}/map_saureus.html"; do
        if [[ -s "$candidate" ]]; then NETWORK="$candidate"; break; fi
      done
    fi
    : "${NETWORK:?Set NETWORK to the S. aureus functional-map HTML}"
    args+=(--network "$NETWORK")
    if [[ "$STAGE" != map ]]; then
      DISCOVERY="${DISCOVERY:-$ROOT/kovar_output/saureus/discovery}"
      if [[ ! -d "$DISCOVERY/prepared_inputs" && -d "$ROOT/kovar_output/saureus/zoo-ardbeg/discovery/prepared_inputs" ]]; then
        DISCOVERY="$ROOT/kovar_output/saureus/zoo-ardbeg/discovery"
      fi
      FASTA="${FASTA:-$DISCOVERY/prepared_inputs/saureus_k61.sample_ids.fasta}"
      TREE="${TREE:-$DISCOVERY/prepared_inputs/saureus_core.rooted.treefile}"
      METADATA="${METADATA:-$ROOT/assemblies/saureus_collection_metadatatxt.txt}"
      UNITIGS="${UNITIGS:-$ROOT/pangwes_saureus_output/saureus/saureus_k61.unitigs}"
      ANNOTATION="${ANNOTATION:-${MAP_DIR:-$(dirname -- "$NETWORK")}/annotation/locus_to_gene.tsv}"
      args+=(--fasta "$FASTA" --fasta-columns-are-locus-ids --unitigs "$UNITIGS"
             --tree "$TREE" --metadata "$METADATA" --annotation "$ANNOTATION")
      [[ -z "${CLUSTERS:-}" ]] || args+=(--clusters "$CLUSTERS")
      [[ -z "${SAMPLE_MAP:-}" ]] || args+=(--sample-map "$SAMPLE_MAP")
    fi
    ;;
  audit|predict) ;;
  *) echo 'Stage must be map, prepare, audit, predict or all' >&2; exit 2 ;;
esac
if [[ "$STAGE" == audit || "$STAGE" == all ]]; then
  args+=(--assemblies "${ASSEMBLIES:-$ROOT/assemblies}"
         --gff-dir "${GFF_DIR:-$ROOT/panaroo_saureus_output/panaroo_gff}"
         --amrfinder "${AMRFINDER_EXEC:-amrfinder}")
  [[ -z "${AMRFINDER_DB:-}" ]] || args+=(--amrfinder-db "$AMRFINDER_DB")
  [[ -z "${REFERENCE_PROTEINS:-}" ]] || args+=(--reference-proteins "$REFERENCE_PROTEINS")
fi
printf 'Stage: %s\nOutput: %s\nCPUs: %s\n' "$STAGE" "$OUT" "$THREADS"
"$PYTHON" "$REPO/benchmarks/saureus_mic.py" "${args[@]}" "$@"
