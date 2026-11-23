#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec "${KOVAR_PYTHON:-python3}" -m ko_variation.discovery --resume-if-present "$@"
