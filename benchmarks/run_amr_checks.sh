#!/usr/bin/env bash
# Standalone checks from upstream PAN-GWES/GFF/tree inputs; no discovery run needed.
# Set RESULTS explicitly only to compare an older run. Python environments may differ.
set -euo pipefail
SPECIES="${1:-saureus}"
STAGE="${2:-both}"
[[ "$SPECIES" == saureus ]] || { echo 'The existing target checks support saureus only.' >&2; exit 2; }
case "$STAGE" in both|all|kovar|pyseer|sccmec|report) ;; *) echo 'Stage: both, all, kovar, pyseer, sccmec or report' >&2; exit 2;; esac
if (( $# > 1 )); then shift 2; elif (( $# )); then shift; fi
if (( $# )) && [[ "$STAGE" == both || "$STAGE" == all ]]; then
  echo 'Extra CLI options require a single stage; use environment variables for a combined run.' >&2; exit 2
fi
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${ROOT:-}" ]]; then
  if [[ -d "$PWD/pangwes_saureus_output" ]]; then ROOT="$PWD"
  elif [[ -d "$HERE/../../pangwes_saureus_output" ]]; then ROOT="$(cd -- "$HERE/../.." && pwd)"
  else ROOT="$PWD"; fi
fi
if [[ -z "${REPO:-}" ]]; then
  for candidate in "$HERE/.." "$ROOT/ko-variation-diagnosis" "$ROOT/ko-variation"; do
    if [[ -f "$candidate/benchmarks/amr_checks.py" ]]; then REPO="$(cd -- "$candidate" && pwd)"; break; fi
  done
fi
: "${REPO:?Set REPO to the checkout containing benchmarks/amr_checks.py.}"
ROOT="$(cd -- "$ROOT" && pwd)"
if [[ -n "${RESULTS:-}" ]]; then
  SPECIES_OUT="$(dirname -- "$(dirname -- "$RESULTS")")"
  DIAGNOSIS="${DIAGNOSIS:-$SPECIES_OUT/diagnostics/amr_targets}"
else
  DIAGNOSIS="${DIAGNOSIS:-${OUT:-$ROOT/amr_checks_output/saureus}/amr_targets}"
fi
THREADS="${THREADS:-4}"
KOVAR_PYTHON="${KOVAR_PYTHON:-python}"
PYSEER_PYTHON="${PYSEER_PYTHON:-python}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
work=()
[[ -z "${SOURCE_WORK:-}" ]] || work=(--work "$SOURCE_WORK")
echo "Stage: $STAGE; diagnostic: $DIAGNOSIS; CPUs: $THREADS"
if [[ "$STAGE" == kovar || "$STAGE" == both || "$STAGE" == all || "$STAGE" == report ]]; then
  args=(--out "$DIAGNOSIS" --threads "$THREADS")
  [[ -z "${RESULTS:-}" ]] || args+=(--results "$RESULTS" "${work[@]}")
  if [[ "$STAGE" == report ]]; then
    args+=(--report-only)
  else
    PANGWES="${PANGWES:-$ROOT/pangwes_saureus_output/saureus}"
    STEM="${STEM:-$PANGWES/saureus_k61}"
    if [[ -n "${RESULTS:-}" ]]; then
      FASTA="${FASTA:-$(dirname -- "$RESULTS")/prepared_inputs/saureus_k61.sample_ids.fasta}"
      MAP_DIR="${MAP_DIR:-$SPECIES_OUT/plots/functional_map}"
    else
      FASTA="${FASTA:-${RAW_FASTA:-$STEM.fasta}}"
      if [[ -z "${TREE:-}" ]]; then
        TREE="${RAW_TREE:-$ROOT/iqtree_saureus_output/saureus/saureus_core.treefile}"
        [[ -s "$TREE" ]] || TREE="$ROOT/iqtree_saureus_output/saureus_core.treefile"
      fi
      args+=(--tree "$TREE" --tree-rooting "${TREE_ROOTING:-midpoint}"
             --tree-missing-samples "${TREE_MISSING_SAMPLES:-error}" --min-maf "${MIN_MAF:-0.05}"
             --alpha "${ALPHA:-0.05}")
      if [[ -n "${SAMPLE_IDS:-}" ]]; then
        args+=(--sample-ids "$SAMPLE_IDS")
      elif [[ -s "${ASSEMBLIES_LIST:-$PANGWES/assemblies_list.txt}" && -s "${GFA:-$STEM.gfa1}" && -s "${CUTTLEFISH_JSON:-$STEM.json}" ]]; then
        args+=(--assemblies-list "${ASSEMBLIES_LIST:-$PANGWES/assemblies_list.txt}"
               --gfa "${GFA:-$STEM.gfa1}" --cuttlefish-json "${CUTTLEFISH_JSON:-$STEM.json}")
      fi
      [[ -z "${EXCLUDE_SAMPLES:-}" ]] || args+=(--exclude-samples "$EXCLUDE_SAMPLES")
      [[ -z "${MEMORY_GB:-}" ]] || args+=(--memory-gb "$MEMORY_GB")
    fi
    args+=(--fasta "$FASTA" --unitigs "${UNITIGS:-$STEM.unitigs}" --original-nulls "${ORIGINAL_NULLS:-auto}")
    if [[ -z "${PANAROO:-}" ]]; then
      for candidate in "$ROOT/panaroo_saureus_output/saureus/gene_presence_absence_roary.csv" "$ROOT/panaroo_saureus_output/saureus/gene_presence_absence.csv"; do
        if [[ -s "$candidate" ]]; then PANAROO="$candidate"; break; fi
      done
    fi
    [[ -z "${PANAROO:-}" ]] || args+=(--panaroo "$PANAROO")
    [[ -z "${MAP_DIR:-}" ]] || args+=(--map-dir "$MAP_DIR")
    if [[ -n "${BAKTA:-}" && -z "${GFF_DIR:-}" ]]; then
      args+=(--bakta "$BAKTA")
    elif [[ -n "${GFF_DIR:-}" || -d "$ROOT/panaroo_saureus_output/panaroo_gff" ]]; then
      args+=(--gff-dir "${GFF_DIR:-$ROOT/panaroo_saureus_output/panaroo_gff}")
    elif [[ -n "${MAP_DIR:-}" && -d "$MAP_DIR/recovered_annotations" ]]; then
      args+=(--bakta "$MAP_DIR/recovered_annotations")
    else
      args+=(--bakta "$ROOT/panaroo_saureus_output/bakta")
    fi
  fi
  echo "KOVAR mode: ${RESULTS:-standalone upstream inputs}; diagnostic output: $DIAGNOSIS"
  "$KOVAR_PYTHON" "$REPO/benchmarks/amr_checks.py" kovar "${args[@]}" "$@"
fi
if [[ "$STAGE" == pyseer || "$STAGE" == both || "$STAGE" == all ]]; then
  "$PYSEER_PYTHON" "$REPO/benchmarks/amr_checks.py" pyseer --diagnosis "$DIAGNOSIS" \
    --out "${PYSEER_OUT:-$(dirname -- "$DIAGNOSIS")/amr_pyseer_lmm}" --threads "$THREADS" "${work[@]}" "$@"
fi
if [[ "$STAGE" == sccmec || "$STAGE" == all ]]; then
  args=(--diagnosis "$DIAGNOSIS" --assemblies "${ASSEMBLIES:-$ROOT/assemblies}"
        --out "${SCCMEC_OUT:-$(dirname -- "$DIAGNOSIS")/sccmec_pyseer}" --threads "$THREADS"
        --sccmec "${SCCMEC_EXEC:-sccmec}" "${work[@]}")
  [[ ! -d "${GFF_DIR:-$ROOT/panaroo_saureus_output/panaroo_gff}" ]] || \
    args+=(--gff-dir "${GFF_DIR:-$ROOT/panaroo_saureus_output/panaroo_gff}")
  "$PYSEER_PYTHON" "$REPO/benchmarks/amr_checks.py" sccmec "${args[@]}" "$@"
fi
