# KOVAR

KOVAR (KO-Variation) screens candidate pairs of binary pangenome loci for
covariation after adjustment for shared ancestry. It uses a phylogeny-derived
sample covariance, a binary logistic mixed-model null fit, and a score test.
The output is evidence against no adjusted association; it is not an effect
size, a causal direction, or proof of biological epistasis.

Version 0.9.2 provides direct unitig discovery without SpydrPick, disk-backed
hierarchical search, adaptive significance bounds and persistent null reuse.
It retains significant local pairs and estimates beta only for confirmed distal
pairs. See the [v0.9 discovery guide](docs/discovery.md) for inputs, CPU/memory
controls, restart, and the heuristic limits of representative pruning.
The performance update adds allocation-aware memory/cache budgets, response
affinity, bulk database operations and verified release-checkpoint migration.
Necessary joint-cell bounds skip ineligible hierarchy blocks while preserving
their external associations. Coarse comparisons and queued refinements interleave.

The current **0.9.2 execution revision `streamlined-1`** pre-fits the unique
response pool, stores compact numerical caches, and groups coarse comparisons
by response for reuse. The original `v0.9.2` Git tag is unchanged; use the
`v0.9.2-streamlined1` tag or `codex/downstream-resume-progress` for this update.

The legacy candidate-table scanner reports one result for each unordered pair. The separate
`ko-variation-annotation` helper selects distal significant signals, calculates
phylogeny-adjusted alternative-model odds ratios, and exports annotated gene networks.
See [downstream usage and interpretation](docs/downstream.md).

## Installation

```bash
git clone --branch v0.9.2-streamlined1 https://github.com/jeju2486/Pair-logistic-mds.git
cd Pair-logistic-mds
pip install -e .
ko-variation --version
```

Python 3.10 or newer is required. KOVAR depends on NumPy, pandas, SciPy, and
threadpoolctl.

## Quick start

Direct discovery from PAN-GWES parser outputs:

```bash
ko-variation discover \
  --fasta graph.sample_ids.fasta --tree rooted_tree.nwk \
  --unitigs graph.unitigs --paths graph.paths \
  --unitig-distance /path/to/unitig_distance --kmer-length 61 \
  --out kovar_results --threads 4 --memory-gb 4
```

Final selection uses Bonferroni-adjusted P <=0.05. Exact P-values are optional:
certified significance decisions select edges. Local (<=10kb) significant pairs
remain in the table without beta. Repeat with `--resume` to continue.
Representative screening can miss associations; `--search exhaustive` removes
that heuristic while retaining adaptive bounds. No dense all-pairs matrix is
created. [Full instructions and cache reuse](docs/discovery.md).

Existing candidate-table scans remain available:

```bash
ko-variation \
  --fasta binary.fa \
  --pairs candidates.tsv \
  --tree rooted_tree.nwk \
  --out kovar_results \
  --threads 32
```

SPA is `auto` by default. To resume an interrupted run, repeat the identical
command and add `--resume`. To start a new run in a non-empty output directory,
use `--overwrite`.

## Inputs

`--fasta` is an aligned fake FASTA with identical locus order and length in all
samples. `A` means absence (0) and `C` means presence (1).

`--pairs` is a whitespace-, comma-, or tab-delimited file containing zero-based
`u` and `v` locus columns. A headerless nine-column PAN-GWES file is interpreted
as:

```text
u v distance ARACNE MI count M2 min_distance max_distance
```

Pairs are canonicalized to `u < v`. Duplicate unordered pairs, self-pairs, and
out-of-range loci stop the run. Candidate rows are not filtered by MI inside
KOVAR.

`--tree` must be a rooted Newick tree with branch lengths. Tip labels must match
FASTA sample names exactly. KOVAR does not build a fallback GRM.

By default, a FASTA sample absent from the tree stops the run before candidate
pairs are loaded. To deliberately analyze the intersection, add
`--tree-missing-samples drop`. KOVAR retains matched isolates in FASTA order,
requires at least two, and recalculates locus frequencies, joint tables and
sample-count-dependent thresholds using those isolates. Locus IDs and candidate
rows retain their original meanings; input graph `count` metadata is preserved
and is not recomputed from the subset. Tree tips absent from FASTA are ignored.
This is an analysis of the retained cohort, not a reconstruction of missing
phylogenetic placements. Check for naming mismatches before choosing exclusion.

`sample_inclusion.tsv` lists every input isolate and its inclusion status;
`run_summary.txt` records input, retained and excluded counts and the policy.
The policy is included in checkpoint identity for subset runs. Repeat the same
policy on resume and when running `ko-variation-annotation` with the original
FASTA/tree so that selected effects use the same isolates as screening.

## How the calculation works

For the deterministic computational orientation `u -> v`, KOVAR fits the null
model

```text
logit P(v_i = 1) = alpha_v + b_iv
b_v ~ N(0, tau_v K)
```

where `K` is the root-to-MRCA covariance derived once from the tree. PQL updates
the fitted probabilities, working weights, and response-specific variance
component. Identical or complementary binary response patterns reuse one null
fit. Candidate predictors sharing a response are then scored in matrix blocks.

The normal score statistic tests `H0: beta_uv = 0`. With `--spa-mode auto`, an
experimental saddlepoint approximation is attempted only for a score-tail
result in sparse context; otherwise the normal score p-value remains primary.
This is motivated by SPAtest and SAIGE but is not a reproduction of SAIGE.

Because the reported hypothesis is unordered, reversing an input pair produces
the same canonical pair and p-value. The computational response orientation is
an implementation device, not a claim about causal or evolutionary direction.

## Filters

Filters are applied before expensive response-model scheduling:

- If a recognized `count` column is present, `count >= ceil(0.05 * N)` is
  required. If the column is absent, this route filter is not applicable.
- Both loci require minor-state frequency `>= 0.05` by default.
- All four joint cells require at least one observation by default.

Every input pair remains in `ko_variation.tsv`; filtered rows have an explicit
status and missing p-values. Use `--min-maf` or `--min-cell-count` to change the
last two thresholds. The graph-count fraction is fixed at 0.05 in this release.

## Outputs

- `ko_variation.tsv`: one row per unordered input pair, input metadata, filter
  status, joint counts, score statistic, normal/SPA p-values, primary p-value,
  BH q-value, and Bonferroni flag.
- `response_models.tsv`: response-null convergence and exact-pattern reuse
  diagnostics.
- `run_summary.txt`: short scientific run contract.
- `execution_metadata.tsv`: timing, memory, runtime, kinship, cache, and
  checkpoint diagnostics.
- `sample_inclusion.tsv`: input isolate names with `included` or
  `excluded_missing_tree` status.

`p_primary` is a significance measure, not the magnitude of covariation.
`primary_method` says whether it came from the normal score approximation or a
successful SPA calculation. `neglog10_p` is intended for plotting.

## Downstream helpers

```bash
pip install -e '.[network]'
ko-variation-annotation --results kovar_results/ko_variation.tsv \
  --fasta binary.fa --tree rooted_tree.nwk \
  --annotation loci.tsv --out downstream/hits --network \
  --ld-distance 0 --significance-threshold 0.05
```

Annotation requires zero-based `locus` and stable `gene` columns. The default
selects `p_primary <= 0.05 / n_tests` and physical `distance > 0` in matching units.
Both thresholds are configurable. Only selected pairs are refitted, using the
original FASTA and tree. Outputs include adjusted ORs and approximate PQL Wald
CIs, fit diagnostics, a 300-DPI PNG, editable SVG, offline D3 HTML, saved layout coordinates, and node/edge tables;
gene edges summarize covariation candidates, not proven functional interactions.
Run `python examples/downstream/reproduce.py` for a synthetic reproducible example.
Portable commands replace the former ARC submission wrappers:

- `ko-variation-select`: select a large full scan in chunks without changing n_tests.
- `ko-variation-map-loci`: map selected DNA unitigs to Bakta/Panaroo annotations.
- `ko-variation-network`: redraw completed effects without refitting.
- `ko-variation-prepare-proteins`: prepare mapped representative proteins for
  eggNOG functional annotation and optional COG node colours.
- `ko-variation-amr-report`: link species-specific MIC availability, established
  presence/absence determinant labels, and representation in a selected distal map.
- `ko-variation-unitig-audit`: annotate the entire supplied DNA unitig catalogue
  and report AMR-associated patterns' MAF, entropy and phylogenetic D, before filtering.

The documented [portable pipeline](docs/downstream.md#portable-workflow-for-large-pan-gwes-scans)
uses explicit paths and retains selection, annotation and refit recovery.
See [AMR reporting](docs/amr_report.md) for the curated catalogue, annotation-name
matching, and direct/second-order neighbour report. The tet-specific tracer has
been removed; its old commands are no longer supported.
See the separate [whole-catalogue unitig and D audit](docs/unitig_audit.md) for
installation (`.[audit]` plus R ape/caper), checkpointing and input contracts.
After updating an existing checkout, rerun `pip install -e '.[network]'` to
register the new helper commands. Existing scanner commands are unchanged.

Full options, failed-fit handling, coordinate fallback, and aggregation rules are
documented in [docs/downstream.md](docs/downstream.md).
See [annotation names and functional colours](docs/functional_map.md) for
alias-aware search, gene-name labels, and eggNOG/GO metadata on existing maps.

## Checkpoint and resume

For v0.9 discovery, see [disk-backed restart and saved null reuse](docs/discovery.md).
One focused production-path benchmark replaces the historical standalone
screening prototypes: `benchmarks/discovery_benchmark.py`.

Checkpointing is automatic. Atomic score-stage shards are written to
`OUT/.kovar_checkpoint` unless `--checkpoint-dir` is supplied. A resume is
accepted only when input fingerprints, version, numerical environment, and
scientific settings match. The checkpoint is removed after final outputs are
successfully written.

## Interpretation and limitations

A small p-value identifies a phylogeny-adjusted covariation candidate. It does
not distinguish functional epistasis from physical linkage, co-transfer,
shared ecological selection, annotation artifacts, or residual population
structure. The legacy candidate-table mode cannot recover pairs absent from its
input. Direct discovery bypasses that shortlist, but its default representative
hierarchy can miss significant member pairs. Adaptive bounds certify only
queried ordinary-score decisions, conditional on the fitted null.

The PQL and SPA implementations remain experimental. Calibration should be
checked by simulation over the intended sample sizes, frequencies, tree shapes,
and population structures before confirmatory use.

## Research foundations

- Breslow and Clayton (1993), approximate inference in generalized linear mixed
  models, motivates the PQL framework.
- Chen et al. (2016), GMMAT, motivates fitting a logistic mixed-model null once
  per response and testing predictors with score statistics.
- Dey et al. (2017), SPAtest, and Zhou et al. (2018), SAIGE, motivate
  saddlepoint calibration for binary score-test tails.
- Lees et al. (2018), pyseer, motivates bacterial population-structure-aware
  association analysis and scalable null-model reuse. KOVAR differs by using a
  binary logistic mixed model for locus-pair screening.

See `docs/model.md`, `docs/interpretation.md`, and `docs/research_basis.md` for
more detail.

## Version history

### 0.9.2 execution revision streamlined-1

- consolidate null prefit, numerical storage and inventory in `discovery_nulls.py`;
- consolidate release identity checks and optional old-cache import in existing modules;
- keep legacy diagnostics off the discovery import path;
- defer duplicate prefit array checks to first numerical access;
- retain verified perf1/perf2 checkpoints and the unchanged statistical model.

### 0.9.2 execution revision prefit-compact-2

- skip a completed null stage on resume; reuse inventory rows from interrupted stages;
- retain individual atomic manifests for recovery after interrupted model writes;
- record status/timing summaries in SQLite instead of rescanning model files on every export;
- refresh cache sizes only after reconstruction, and sample worker snapshots every five seconds;
- batch imported-cache records and reuse the startup input identity;
- load legacy scanner/import helpers only when requested;
- migrate verified `v0.9.2-perf1` checkpoints while preserving saved models and decisions.

### 0.9.2 execution revision prefit-compact-1

- fit missing canonical response nulls in a separate, parallel resumable stage;
- preserve native saved fits and reuse successful complementary-pattern fits;
- omit zero-variance/shared eigenvector copies and save bounded spectral bases
  for general caches, reconstructing omitted directions only when needed;
- recover cache capacity from actual headroom instead of geometric shrinkage;
- traverse coarse comparisons by response while retaining legacy queued work;
- expose live cache capacity, refusals, resident models and separate prefit timing;
- provide `benchmarks/null_fit_benchmark.py`; the production fitter is unchanged.

### 0.9.2

- prune final-ineligible hierarchy blocks using necessary marginal counts and
  actual anchor-radius Hamming bounds, before representative null fitting;
- retain all categories and members for possible external associations;
- interleave new coarse comparisons with queued hierarchy refinements;
- report proven block pruning separately from heuristic representative pruning
  and adaptive/exact scoring;
- resume verified v0.9.0/v0.9.1 work with the same hypotheses and thresholds.

### 0.9.1

- adapt memory targets to scheduler/cgroup/host limits, with an optional cap;
- replace fixed-count model caches with byte-limited LRU caches and pressure checks;
- pin responses to workers, coalesce leaf work, and consume completed score jobs promptly;
- bulk-read decisions, batch durable writes, and cache traversal metadata;
- retain verified v0.9.0 checkpoints and fitted models without changing the test;
- report fitting, cache, numerical-kernel and coordinator timings separately.

### 0.9.0

- direct PAN-GWES unitig input without MI or ARACNE;
- exact-pattern compression with original-unitig mappings;
- global representative hierarchy and adaptive ordinary-score bounds;
- optimized exact fallback, response-major parallel workers and cached nulls;
- mapped arrays, bounded queues and disk-backed stage checkpoints;
- all significant candidates retained, with post-hoc beta restricted to >10kb;
- compact compressed results and explicit significance/distance/effect statuses;
- consolidated production benchmark and removal of superseded prototypes.

### 0.8.3

- changed primary reporting from two directional rows to one canonical
  unordered pair;
- changed the default minimum joint-cell count from 5 to 1;
- made SPA `auto` the default;
- applies the PAN-GWES `count >= ceil(0.05*N)` route before GLMM scheduling when
  a `count` column is available;
- removed the built-in alternative full-refit/effect-size stage, directional
  options, GRM fallback, and low-level numerical tuning from the public CLI;
- reduced checkpointing to the only expensive production stage: score testing;
  and
- moved release notes after overview, installation, use, method, and output
  documentation.

### 0.8.2

Added atomic checkpoint/resume support, detailed execution metadata, and
response-pattern reuse diagnostics.

### 0.8.1

Added exact identical/complement response-pattern reuse and shared first-PQL
kinship eigensystem preparation.

### 0.8.0

Introduced the experimental logistic mixed-model score scanner.


Completed downstream results can be redrawn without annotation or refitting:

```bash
ko-variation-network --effects downstream/hits.distal.tsv --out downstream/hits
```

The compact map uses capped node radii, packed connected components, representative
locus-pair |beta| for edge width/spacing and association direction for colour. D3
is bundled for offline search, zoom, dragging, optional reflow and SVG/PNG export.
See [downstream display options](docs/downstream.md#compact-display-semantics) for
coefficient boundaries and fixed cross-species display settings.
