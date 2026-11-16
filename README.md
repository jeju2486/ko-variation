# KOVAR

KOVAR tests covariation between binary pangenome features while adjusting for a
phylogeny-derived covariance. Covariation is a candidate association, not proof
of causality or biological epistasis.

This is the minimal public release, `main` (v0.9.4), also maintained on
`codex/publication-clean`.
The development branch `codex/downstream-resume-progress` retains supplementary
benchmarks and diagnostic tools. No research script is needed for a normal run.

## Install

```bash
git clone --branch main https://github.com/jeju2486/ko-variation.git
cd ko-variation
python -m pip install '.[tree,network,phenotype]'
ko-variation --version
```

Python 3.10+ is required. `tree`, `network` and `phenotype` are optional extras;
`pip install .` supplies dense discovery alone. PAN-GWES `unitig_distance` and
annotation tools/databases are installed separately.

## Run

```bash
bash examples/run_discovery.sh \
  --fasta graph.sample_ids.fasta --tree rooted_tree.nwk \
  --unitigs graph.unitigs --paths graph.paths \
  --unitig-distance /path/to/unitig_distance --kmer-length 61 \
  --out results --threads 4
```

The Bash runner uses the installed environment, has no scheduler/account paths
and resumes matching checkpoints automatically. `KOVAR_PYTHON=/path/to/python`
selects another interpreter. The same arguments work with `ko-variation discover`
or `python -m ko_variation.discovery`.

The sample-ID FASTA encodes A=absence and C=presence, preserving original unitig
columns. The rooted Newick tree needs branch lengths and matching sample IDs.
[Input, resource and restart guide](docs/portable_runner.md).

Dense null fitting remains the default while larger validation of the new
optimizer is pending. Add `--null-backend tree` for tree nulls, compact O(n)
per-pattern caches and automatic dense fallback on fitting failure. Tree scoring
uses exact contrasts; selected beta effects retain full dense refits.
[Tree-null details and limitations](docs/tree_null.md).

## Interpretation and outputs

Defaults are MAF >=0.05, minimum final joint-cell count 20, 40 tree PCs and 10%
coarse-pair retention. PC/representative screening is approximate. Bonferroni alpha
is 0.05 over all unique-pattern pairs, including pairs not selected by screening.
Use `--coarse-screen none` or `--search exhaustive` for the corresponding references.

Significant close pairs remain in the output; beta is fitted only for pairs
beyond `--distal-bp` (10,000 bp by default). Distances come from the supplied
PAN-GWES tool. Exact P-values are unnecessary when intervals certify a queried
score decision. [Model](docs/model.md), [discovery](docs/discovery.md),
[interpretation](docs/interpretation.md).

- `unitigs.tsv.gz`: original unitig IDs and exact-pattern assignments.
- `associations.tsv.gz`: significant candidates, P-value fields, distance, beta and status.
- `run_summary.json`: settings, hypothesis family and provenance.
- `.kovar_work`: persistent nulls and search/distance/beta checkpoints.

Repeat a current-version command with `--resume` to reuse committed work. Keep
its working directory. Backend/code changes require a new output; earlier
exports remain readable for plotting. Changing only `--distal-bp` reuses saved
distances and beta attempts and fits newly eligible missing effects.

## Annotation, maps and reports

| Purpose | Installed command |
|---|---|
| Unitig-to-gene annotation | ko-variation-map-loci |
| Functional maps | ko-variation-functional-map |
| Publication PNG/SVG figures | ko-variation-paper-plots |
| Gene networks | ko-variation-network |
| Protein preparation | ko-variation-prepare-proteins |
| AMR phenotype/determinant report | ko-variation-amr-report |
| MIC comparisons | ko-variation-phenotype |
| Saved-table selection and refitting | ko-variation-select, ko-variation-annotation |
| Run progress | ko-variation-progress |

For conventional species folders: `ROOT=/project bash examples/run_functional_map.sh saureus`.
Set `RESULTS` explicitly for another discovery location. See
[functional maps and paper figures](docs/map_and_paper.md),
[AMR reports](docs/amr_report.md) and [phenotype analysis](docs/phenotype.md).

[Current project decisions](PROJECT_STATE.md) identify remaining validation work.
[Development and supplementary programs](https://github.com/jeju2486/ko-variation/tree/codex/downstream-resume-progress/benchmarks)
remain on the separate development branch.
