#!/usr/bin/env bash
# Portable functional map runner; reuses completed associations/effects.
set -euo pipefail
SPECIES="${1:-saureus}"
case "$SPECIES" in saureus|neisseria|klebsiella) ;; *) echo 'Species: saureus, neisseria, klebsiella' >&2; exit 2;; esac
ROOT="${ROOT:?Set ROOT to the project directory containing the upstream outputs}"
REPO="${REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
# Run names are optional labels, never an inferred execution hostname.
RUN_NAME="${RUN_NAME:-}"
KOVAR_PYTHON="$(command -v -- "${KOVAR_PYTHON:-python}" || true)"
[[ -n "$KOVAR_PYTHON" ]] || { echo 'Activate your Python environment or set KOVAR_PYTHON' >&2; exit 1; }
# Keep Pyseer's older numerical dependencies separate from the KOVAR runtime.
if [[ -z "${PYSEER_PYTHON:-}" && -x "$ROOT/pyseer_annotation_env/bin/python" ]]; then
  PYSEER_PYTHON="$ROOT/pyseer_annotation_env/bin/python"
fi
PYSEER_PYTHON="${PYSEER_PYTHON:-$KOVAR_PYTHON}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
SPECIES_OUT="$ROOT/kovar_output/$SPECIES${RUN_NAME:+/$RUN_NAME}"
RESULTS="${RESULTS:-$SPECIES_OUT/discovery/results}"
MAP_OUT="${MAP_OUT:-$SPECIES_OUT/plots/functional_map}"
# A copied run can retain an inaccessible ARC FASTA path in its provenance.
# Use its adjacent local sample-ID FASTA, without editing that provenance.
if [[ -z "${ANNOTATION:-}" && -z "${FASTA:-}" ]]; then
  source_directory="${RESULTS%/}"
  if [[ -f "$source_directory" ]]; then
    source_directory="$(dirname -- "$source_directory")"
  elif [[ -d "$source_directory/results" ]]; then
    source_directory="$source_directory/results"
  fi
  fasta_candidates=()
  for directory in "$source_directory/../prepared_inputs" "$source_directory/prepared_inputs"; do
    for candidate in "$directory/"*sample_ids.fasta "$directory/"*sample_ids.fa; do
      [[ -f "$candidate" ]] && fasta_candidates+=("$candidate")
    done
  done
  if (( ${#fasta_candidates[@]} == 1 )); then
    FASTA="${fasta_candidates[0]}"
  elif (( ${#fasta_candidates[@]} > 1 )); then
    echo 'Multiple local sample-ID FASTAs found; set FASTA to the correct file explicitly.' >&2
    printf '%s\n' "${fasta_candidates[@]}" >&2
    exit 1
  fi
fi
# Prefer original Bakta outputs. Retained Panaroo GFFs with embedded DNA can
# recover missing inputs without rerunning Bakta or changing CDS IDs.
if [[ -n "${BAKTA:-}" && -n "${GFF_DIR:-}" ]]; then
  echo 'Choose BAKTA or GFF_DIR, not both.' >&2; exit 1
fi
if [[ -z "${BAKTA:-}" && -z "${GFF_DIR:-}" ]]; then
  if [[ -d "$ROOT/panaroo_${SPECIES}_output/bakta" ]]; then
    BAKTA="$ROOT/panaroo_${SPECIES}_output/bakta"
  elif [[ -d "$ROOT/panaroo_${SPECIES}_output/panaroo_gff" ]]; then
    GFF_DIR="$ROOT/panaroo_${SPECIES}_output/panaroo_gff"
  fi
fi
if [[ -z "${PANAROO:-}" ]]; then
  for candidate in "$ROOT/panaroo_${SPECIES}_output/$SPECIES/gene_presence_absence_roary.csv" \
                   "$ROOT/panaroo_${SPECIES}_output/gene_presence_absence_roary.csv" \
                   "$ROOT/panaroo_${SPECIES}_output/$SPECIES/gene_presence_absence.csv" \
                   "$ROOT/panaroo_${SPECIES}_output/gene_presence_absence.csv"; do
    if [[ -s "$candidate" ]]; then PANAROO="$candidate"; break; fi
  done
fi
echo "Species: $SPECIES; results: $RESULTS; map output: $MAP_OUT"
echo "Panaroo: ${PANAROO:-not found}; Bakta: ${BAKTA:-not used}; recovery GFFs: ${GFF_DIR:-not used}"
echo "Sample-ID FASTA: ${FASTA:-from discovery run identity}"
# Report missing source paths instead of silently dropping required arguments.
# Existing annotation plus existing functions (or SKIP_FUNCTION) needs neither.
if [[ -z "${ANNOTATION:-}" || ( -z "${EGGNOG_ANNOTATIONS:-}" && "${SKIP_FUNCTION:-0}" != 1 ) ]]; then
  if [[ -n "${GFF_DIR:-}" ]]; then
    [[ -d "$GFF_DIR" ]] || { echo "GFF directory not found: $GFF_DIR" >&2; exit 1; }
  else
    [[ -d "${BAKTA:-}" ]] || {
      echo "Bakta directory not found: ${BAKTA:-not set}" >&2
      echo 'Set BAKTA to SAMPLE/SAMPLE.{gff3,fna,faa} outputs, or GFF_DIR to Panaroo GFFs with embedded DNA.' >&2
      exit 1
    }
  fi
  [[ -n "${PANAROO:-}" && -s "$PANAROO" ]] || {
    echo 'Panaroo membership CSV not found. Set PANAROO=/actual/path/gene_presence_absence_roary.csv' >&2
    exit 1
  }
fi
args=(--results "$RESULTS" --out "$MAP_OUT" --threads "${THREADS:-4}"
      --distal-bp "${DISTAL_BP:-10000}" --nearby-bp "${NEARBY_BP:-500}"
      --chunk-rows "${CHUNK_ROWS:-50000}" --seed "${SEED:-42}" --dpi "${DPI:-300}")
if [[ -n "${ANNOTATION:-}" ]]; then
  args+=(--annotation "$ANNOTATION")
else
  UNITIGS="${UNITIGS:-$ROOT/pangwes_${SPECIES}_output/$SPECIES/${SPECIES}_k61.unitigs}"
  [[ -s "$UNITIGS" ]] || { echo "Original DNA unitig table not found: $UNITIGS" >&2; exit 1; }
  args+=(--unitigs "$UNITIGS")
fi
[[ -n "${FASTA:-}" ]] && args+=(--fasta "$FASTA")
[[ -n "${GENE_CATALOG:-}" ]] && args+=(--gene-catalog "$GENE_CATALOG")
[[ -n "${PANAROO:-}" ]] && args+=(--panaroo "$PANAROO")
if [[ -n "${GFF_DIR:-}" ]]; then
  args+=(--gff-dir "$GFF_DIR" --translation-table "${TRANSLATION_TABLE:-11}")
elif [[ -n "${BAKTA:-}" ]]; then
  args+=(--bakta "$BAKTA")
fi
[[ -n "${PYSEER_PYTHON:-}" ]] && args+=(--pyseer-python "$PYSEER_PYTHON")
if [[ -n "${EGGNOG_ANNOTATIONS:-}" ]]; then
  args+=(--eggnog "$EGGNOG_ANNOTATIONS")
elif [[ "${SKIP_FUNCTION:-0}" == 1 ]]; then
  args+=(--allow-unassigned)
else
  args+=(--run-eggnog --emapper "${EGGNOG_EXEC:-emapper.py}" --eggnog-data-dir "${EGGNOG_DATA_DIR:-}")
fi
# The Python workflow validates dependencies and resumes completed stages.
exec "$KOVAR_PYTHON" -m ko_variation.functional_map "${args[@]}"
