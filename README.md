# KOVAR

KOVAR identifies associations between binary pangenome features after adjustment
for a phylogeny-derived sample covariance. Direct unitig discovery combines
unique-pattern reuse, a global tree-PC coarse screen, hierarchical search and
adaptive significance bounds. The output describes covariation; it does not
establish causality or biological epistasis.

## Install

From this release checkout, using Python 3.10 or newer:

```bash
python -m pip install -e .
# Optional network/figure and quantitative-phenotype dependencies:
python -m pip install -e '.[network,phenotype]'
ko-variation --version
```

The package version is 0.9.2; the numerical execution revision remains
`pc-screen-1`. The publication cleanup changes interfaces and optional helpers,
while keeping the discovery, null-fitting and score kernels unchanged.

## Discover covariation

```bash
ko-variation discover \
  --fasta graph.sample_ids.fasta --tree rooted_tree.nwk \
  --unitigs graph.unitigs --paths graph.paths \
  --unitig-distance /path/to/unitig_distance --kmer-length 61 \
  --out results --threads 4
```

`ko-variation-discover` and `python -m ko_variation.discovery` run the same
pipeline. The sample-labelled A/C FASTA must preserve original graph columns:
A=absence, C=presence. The tree needs branch lengths and matching sample IDs.
The external PAN-GWES graph-distance executable is supplied separately.

The [portable Bash runner](examples/run_discovery.sh) takes the same options
and automatically resumes an existing checkpoint. It uses the installed Python
environment, accepts `KOVAR_PYTHON=/path/to/python`, and has no scheduler or
account-specific paths. See [portable running instructions](docs/portable_runner.md).
That guide includes isolated-environment installation, input requirements, full
run/restart commands, logging, output descriptions and troubleshooting.

Defaults are MAF >=0.05, minimum final joint-cell count 20, 40 tree PCs and
10% coarse-pair retention. PC and representative screening are heuristics;
use `--coarse-screen none` or `--search exhaustive` for the corresponding
references. Bonferroni alpha is 0.05 across all unique-pattern pairs, including
pairs not selected by the heuristic screen. The exploration gate is separate.

Significance selects candidate edges; post-hoc beta describes their magnitude.
Significant pairs <=10 kb remain in the table with beta skipped. Distal pairs
are >10 kb by default. Exact P-values are optional when bounds certify the
significance decision. See [discovery](docs/discovery.md),
[model](docs/model.md) and [interpretation](docs/interpretation.md).

## Restart and resources

Repeat the same command with `--resume`. Input cohort and scientific settings
must match. `--threads` is the total CPU budget. Memory follows the allocation;
`--memory-gb` optionally caps it. `--tmp-dir` must persist across jobs.
Borrowed null directories must remain accessible until retained runs finish.

The same `--resume` command skips committed beta fits. Its beta-stage log and
`beta_progress.json` report cumulative saved/remaining pattern pairs, OK/non-OK
statuses and new fits in the current invocation. There is no separate beta
restart command. See [beta checkpoints](docs/discovery.md#beta-checkpoints).

The first discovery output must be empty. For a wrapper that prepares inputs
itself, use sibling directories `run/prepared_inputs` and `run/results`, and
pass only `run/results` to `--out`. [Full run/cache instructions](docs/discovery.md).

## Results

- `unitigs.tsv.gz`: original unitig IDs and their unique-pattern assignments.
- `associations.tsv.gz`: significant candidates, exact/bounded P-value fields,
  distance classification, beta and fitting status.
- `run_summary.json`: hypothesis family, decisions, resources and provenance.
- `.kovar_work`: durable preparation, null models and search/distance/beta state.

## Downstream commands

| Purpose | Command | Guide |
|---|---|---|
| Prepare/resolve annotations | `ko-variation-map-loci` | [annotation](docs/downstream.md) |
| Select/refit existing tables | `ko-variation-select`, `ko-variation-annotation` | [downstream](docs/downstream.md) |
| Draw covariation maps | `ko-variation-network` | [networks](docs/functional_map.md) |
| Functional annotation | `ko-variation-prepare-proteins` | [functional map](docs/functional_map.md) |
| Match AMR phenotypes/determinants | `ko-variation-amr-report` | [AMR report](docs/amr_report.md) |
| Quantitative MIC comparison | `ko-variation-phenotype` | [phenotype](docs/phenotype.md) |
| Optional distance/significance plot | `ko-variation-plot` | [downstream](docs/downstream.md) |

Historical result tables remain readable by the downstream helpers. The old
`--pairs` scanner, standalone D audit and fastBAPS wrapper were retired from
this package. Use the historical v0.8.3 tag in a separate environment for old
candidate-table scans; supply externally generated population labels to MIC
analysis. [Package structure](docs/architecture.md).

## Benchmarks

Benchmarks are optional source-checkout programs outside the installed package.
They do not run during discovery. [Benchmark index](benchmarks/README.md) covers
PC-count/retention recovery, exact-versus-bounded significance, beta refits and
eigensolvers. Existing raw benchmark tables/figures are retained separately from
software source. [Validation scope](docs/validation.md).
