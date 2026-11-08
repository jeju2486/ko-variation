#!/usr/bin/env bash
# Portable functional map runner; reuses completed associations/effects.
set -euo pipefail
SPECIES="${1:-saureus}"
case "$SPECIES" in saureus|neisseria|klebsiella) ;; *) echo 'Species: saureus, neisseria, klebsiella' >&2; exit 2;; esac
ROOT="${ROOT:?Set ROOT to the project directory containing the upstream outputs}"
REPO="${REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_NAME="${RUN_NAME:-local}"
KOVAR_PYTHON="$(command -v -- "${KOVAR_PYTHON:-python}" || true)"
[[ -n "$KOVAR_PYTHON" ]] || { echo 'Activate your Python environment or set KOVAR_PYTHON' >&2; exit 1; }
# Keep Pyseer's older numerical dependencies separate from the KOVAR runtime.
if [[ -z "${PYSEER_PYTHON:-}" && -x "$ROOT/pyseer_annotation_env/bin/python" ]]; then
  PYSEER_PYTHON="$ROOT/pyseer_annotation_env/bin/python"
fi
PYSEER_PYTHON="${PYSEER_PYTHON:-$KOVAR_PYTHON}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
RESULTS="${RESULTS:-$ROOT/kovar_output/$SPECIES/$RUN_NAME/discovery/results}"
MAP_OUT="${MAP_OUT:-$ROOT/kovar_output/$SPECIES/$RUN_NAME/plots/functional_map}"
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
# Check the whole execution environment before recovery/reference preparation.
# CHECK_ONLY=1 reports it without running the map. SMOKE_TEST=1 additionally
# exercises real Pyseer/native mapping and PNG/SVG export on a tiny fixture.
export MAP_OUT PYSEER_PYTHON
"$KOVAR_PYTHON" - <<'PY'
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from ko_variation.annotation_inputs import annotation_environment
from ko_variation.functional_annotation import read_eggnog

env = os.environ
smoke = env.get('SMOKE_TEST','0') == '1'
force = smoke or env.get('CHECK_ONLY','0') == '1'
report = dict(python=sys.executable,packages={})
errors = []
for name in ('numpy','pandas','scipy','matplotlib','networkx','Bio'):
    try:
        module = importlib.import_module(name)
        report['packages'][name] = getattr(module,'__version__','unknown')
    except Exception as error:
        errors.append(f'{name}: {error}')
cached = Path(env['MAP_OUT'])/'annotation/annotation.cache.json'
if not env.get('ANNOTATION') and (force or not cached.is_file()):
    try:
        _, report['mapping'] = annotation_environment(env['PYSEER_PYTHON'],smoke_test=smoke)
    except Exception as error:
        errors.append(str(error))
else:
    report['mapping'] = 'existing annotation supplied or completed stage present'
if env.get('EGGNOG_ANNOTATIONS'):
    try:
        report['existing_eggnog_queries'] = len(read_eggnog(env['EGGNOG_ANNOTATIONS']))
    except Exception as error:
        errors.append(f'Existing eggNOG table: {error}')
elif env.get('SKIP_FUNCTION','0') != '1':
    try:
        executable = shutil.which(env.get('EGGNOG_EXEC','emapper.py'))
        if not executable: raise ValueError('eggNOG executable not found; set EGGNOG_EXEC')
        egg_env = dict(env,PATH=str(Path(executable).absolute().parent)+os.pathsep+env.get('PATH',''))
        egg_env.pop('PYTHONPATH',None)
        version = subprocess.run([executable,'--version'],env=egg_env,capture_output=True,text=True,check=True,timeout=60)
        diamond = shutil.which('diamond',path=egg_env['PATH'])
        if not diamond: raise ValueError('DIAMOND not found in the eggNOG environment')
        diamond_version = subprocess.run([diamond,'version'],env=egg_env,capture_output=True,text=True,check=True,timeout=30)
        database = Path(env.get('EGGNOG_DATA_DIR',''))
        if not env.get('EGGNOG_DATA_DIR') or not database.is_dir():
            raise ValueError('Set EGGNOG_DATA_DIR to the installed database directory')
        tool_version = (version.stdout+version.stderr).strip()
        report['eggnog'] = dict(executable=executable,version=tool_version,
            diamond=diamond,diamond_version=(diamond_version.stdout+diamond_version.stderr).strip(),database=str(database))
        if re.search(r'emapper[-\s]+v?2\.',tool_version,re.I):
            missing = [name for name in ('eggnog.db','eggnog.taxa.db','eggnog_proteins.dmnd')
                       if not (database/name).is_file() or not (database/name).stat().st_size]
            if missing: errors.append('Missing/empty eggNOG v2 database files: '+', '.join(missing))
    except Exception as error:
        details = getattr(error,'stderr','') or getattr(error,'stdout','') or ''
        errors.append(f'eggNOG environment: {error}'+('\n'+details[-3000:] if details else ''))
if smoke and 'matplotlib' in report['packages']:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory(prefix='kovar-plot-check-') as folder:
            fig,ax = plt.subplots()
            ax.plot([0,1],[0,1],color='black')
            for suffix in ('png','svg'):
                path = Path(folder)/('check.'+suffix)
                fig.savefig(path)
                if not path.stat().st_size: raise ValueError(f'Empty {suffix} export')
            plt.close(fig)
        report['png_svg_smoke_test'] = 'PASSED'
    except Exception as error:
        errors.append(f'PNG/SVG export: {error}')
report['errors'] = errors
print('[preflight] '+json.dumps(report,indent=2),flush=True)
out = Path(env['MAP_OUT'])
out.mkdir(parents=True,exist_ok=True)
(out/'map_environment.json').write_text(json.dumps(report,indent=2))
if errors: raise SystemExit(2)
PY
if [[ "${CHECK_ONLY:-0}" == 1 ]]; then exit 0; fi
exec "$KOVAR_PYTHON" "$REPO/benchmarks/functional_map.py" "${args[@]}"
