#!/usr/bin/env bash
# Portable wrapper: pass the same options as `ko-variation discover`.
# Example: bash examples/run_discovery.sh --fasta sample_ids.fasta \
#   --tree core.treefile --out results --threads 4 --memory-gb 4
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec ko-variation discover "$@"
