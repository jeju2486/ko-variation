#!/usr/bin/env bash
# One runner for the existing S. aureus KOVAR/pyseer diagnostics.
# Run from the project root, or set ROOT. KOVAR and pyseer may use different Pythons.
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
ROOT="${ROOT:-$PWD}"
if [[ -z "${REPO:-}" ]]; then
  for candidate in "$HERE/.." "$ROOT/ko-variation-diagnosis" "$ROOT/ko-variation"; do
    if [[ -f "$candidate/benchmarks/amr_checks.py" ]]; then REPO="$(cd -- "$candidate" && pwd)"; break; fi
  done
fi
: "${REPO:?Set REPO to the checkout containing benchmarks/amr_checks.py.}"
ROOT="$(cd -- "$ROOT" && pwd)"
if [[ -z "${RESULTS:-}" ]]; then
  for candidate in "$ROOT/kovar_output/saureus/discovery/results" "$ROOT/kovar_output/saureus/zoo-ardbeg/discovery/results"; do
    if [[ -s "$candidate/associations.tsv.gz" ]]; then RESULTS="$candidate"; break; fi
  done
fi
if [[ -n "${RESULTS:-}" ]]; then
  SPECIES_OUT="$(dirname -- "$(dirname -- "$RESULTS")")"
  DIAGNOSIS="${DIAGNOSIS:-$SPECIES_OUT/diagnostics/amr_targets}"
elif [[ -n "${DIAGNOSIS:-}" ]]; then
  SPECIES_OUT="$(dirname -- "$(dirname -- "$DIAGNOSIS")")"
else
  echo 'No completed discovery found. Set RESULTS, or DIAGNOSIS for saved pyseer checks.' >&2; exit 2
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
  : "${RESULTS:?Set RESULTS to the completed discovery results directory.}"
  args=(--results "$RESULTS" --out "$DIAGNOSIS" --threads "$THREADS" "${work[@]}")
  if [[ "$STAGE" == report ]]; then
    args+=(--report-only)
  else
    PANAROO="${PANAROO:-$ROOT/panaroo_saureus_output/saureus/gene_presence_absence_roary.csv}"
    [[ -s "$PANAROO" ]] || PANAROO="$ROOT/panaroo_saureus_output/saureus/gene_presence_absence.csv"
    MAP_DIR="${MAP_DIR:-$SPECIES_OUT/plots/functional_map}"
    args+=(--fasta "${FASTA:-$(dirname -- "$RESULTS")/prepared_inputs/saureus_k61.sample_ids.fasta}"
           --unitigs "${UNITIGS:-$ROOT/pangwes_saureus_output/saureus/saureus_k61.unitigs}"
           --panaroo "$PANAROO" --map-dir "$MAP_DIR" --original-nulls "${ORIGINAL_NULLS:-auto}")
    if [[ -n "${BAKTA:-}" && -z "${GFF_DIR:-}" ]]; then
      args+=(--bakta "$BAKTA")
    elif [[ -n "${GFF_DIR:-}" || -d "$ROOT/panaroo_saureus_output/panaroo_gff" ]]; then
      args+=(--gff-dir "${GFF_DIR:-$ROOT/panaroo_saureus_output/panaroo_gff}")
    elif [[ -d "$MAP_DIR/recovered_annotations" ]]; then
      args+=(--bakta "$MAP_DIR/recovered_annotations")
    else
      args+=(--bakta "$ROOT/panaroo_saureus_output/bakta")
    fi
  fi
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
