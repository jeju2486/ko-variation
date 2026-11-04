# Portable runner

`examples/run_discovery.sh` works in Bash on a normal workstation or Linux
server. It requires Python 3.10+ with KOVAR installed and does not use Slurm,
conda activation commands, ARC paths or a previous KOVAR run.

## Install

```bash
git clone --branch codex/downstream-resume-progress https://github.com/jeju2486/ko-variation.git
cd ko-variation
python3 -m pip install .
```

Use `python3 -m pip install '.[network,phenotype]'` if the optional downstream
network/phenotype helpers are also needed. This runner only starts discovery.
`unitig_distance` is an external PAN-GWES executable and is not installed by pip.

## Run

```bash
bash examples/run_discovery.sh \
  --fasta /data/project/graph.sample_ids.fasta \
  --tree /data/project/core.treefile \
  --unitigs /data/project/graph.unitigs \
  --paths /data/project/graph.paths \
  --unitig-distance /tools/PAN-GWES/unitig_distance \
  --kmer-length 61 \
  --out /data/project/kovar_output/saureus/discovery \
  --threads 4
```

To select a particular Python environment, prefix the command with
`KOVAR_PYTHON=/path/to/env/bin/python`. Use `--memory-gb 8` to impose an optional
GiB cap; otherwise KOVAR uses its allocation-aware memory plan. The runner keeps
native BLAS libraries to one thread per worker; `--threads` controls the total
worker CPU budget. No scientific defaults are changed by the runner.

The FASTA must already have unique isolate IDs: A=absence, C=presence, with
original unitig columns preserved. Raw pseudo-FASTA records all named
`contig00001` must be assigned their verified isolate IDs before using this
runner. Tree tips must match those IDs; input preparation inside KOVAR then
builds its own unique-pattern table, covariance and null caches. The runner does
not infer sample order or alter/root the supplied tree. Use `--exclude-samples`
for an explicit exclusion list.

If `.paths` entries are relative, launch from the directory where those entries
resolve, or supply a paths file with absolute graph paths. Older distance-tool
versions also require `--edges /data/project/graph.edges`.

Omit `--unitig-distance` and the graph-distance inputs to defer distance and beta
calculation. They can be added on a compatible restart. The primary defaults
remain 40 PCs, 10% coarse retention, Bonferroni alpha 0.05 and distal distance
greater than 10,000 bp. Candidate screening is heuristic.

## Restart

Repeat the same command. The runner appends `--resume` if it finds
`run_identity.json` in the output or an explicitly supplied persistent
`--tmp-dir`. The main pipeline checks scientific inputs/settings and skips
committed work, including beta fits. Change CPU/memory resources as needed;
retain the same scientific settings and original work location.

The initial `--out` must be empty or absent. The runner does not place logs,
configuration files or prepared inputs there before KOVAR starts. Do not redirect
the shell log into a newly created output folder; use a sibling log instead.
Deleting a run's output and persistent work discards its restart state; use an
empty new `--out` for an independent analysis. There is no separate beta-only
mode.

Outputs are `unitigs.tsv.gz`, `associations.tsv.gz`, `run_summary.json`,
`run_identity.json`, beta-stage `beta_progress.json`, and the persistent
`.kovar_work` directory (or specified `--tmp-dir`). Beta progress counts saved
attempts separately from OK/non-OK fitting statuses.
