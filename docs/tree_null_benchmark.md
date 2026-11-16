# Dense versus tree-based null fitting

`benchmarks/tree_null_benchmark.py` is an experimental benchmark, separate from
production KOVAR. It compares the existing dense PQL fitter with the same outer
PQL loop using tree-based Gaussian contrasts, solves and covariance products.
It does not change production source files, fit pair-specific beta, replace the
sigmoid, reduce floating-point precision or relax convergence tolerances.

The motivation is the tree covariance structure described by
[Ho and Ané (2014)](https://doi.org/10.1093/sysbio/syu005). The benchmark's
tree implementation is new code to evaluate, not a call to their R package or
a claim of a validated production speedup. Its Gaussian working calculations
are algebraically equivalent to `diag(1/W) + tau*K`; the overall logistic model
still uses production PQL, which is itself an approximation.

## Install and run

Use the KOVAR environment, with `numba` and `matplotlib` available:

```bash
python -m pip install numba matplotlib
```

From the checkout, reuse the immutable inputs prepared by the standalone check:

```bash
PROJECT=/data/kell7366/kovar
export PYTHONPATH="$PROJECT/ko-variation${PYTHONPATH:+:$PYTHONPATH}"

python benchmarks/tree_null_benchmark.py \
  --work "$PROJECT/amr_checks_output/saureus/amr_targets/.inputs" \
  --out "$PROJECT/benchmarks/tree_null_saureus_p50_s42" \
  --patterns 50 --probes 8 --repeats 1 --threads 4 --stress-cases
```

This reads `prepared.json`, `patterns.bin` and `K.npy`; it does not open source
SQLite databases or reuse fitted nulls. `analysis.treefile` is inferred from the
work directory when present. For an original discovery `.kovar_work` directory,
also provide `--tree /path/to/its/original/rooted.treefile`. The tree covariance
must match the frozen K and sample order; changing rooting is not allowed.

For independent sample-ID FASTA and tree inputs:

```bash
python benchmarks/tree_null_benchmark.py \
  --fasta /data/sample_ids.fasta --tree /data/rooted.treefile \
  --tree-missing-samples drop \
  --out /data/benchmarks/tree_null_fasta \
  --patterns 50 --probes 8 --repeats 1 --threads 4 --stress-cases
```

FASTA must contain unique sample IDs and equal-length A/C rows (A=0, C=1).
This benchmark does not recover sample IDs from repeated `contig00001` headers;
use the already recovered `sample_ids.fasta` or the prepared-input mode above.
The FASTA mode samples up to ten times the requested locus count, applies MAF
0.05 and deduplicates the sampled columns. It may return fewer unique patterns
than requested. Prepared-input mode samples uniformly from the frozen unique
patterns and keeps their original filtering.

With prepared inputs, the default Bonferroni family is all frozen unique-pattern
pairs. With FASTA alone, the default is all raw locus pairs, an explicit
conservative family because the full unique-pattern count is not calculated.
`--bonferroni-tests M --alpha 0.05` sets an explicit family when required.
The benchmark does not infer a smaller family from its response sample.

## What is compared

For every sampled response and repeat, both methods are run in the same worker,
with alternating execution order and one native BLAS CPU each. `--threads`
controls the number of independent workers. Each method starts from the same
production cold start. The variance grid, search expansion, bounded optimizer,
PQL damping, iteration limit and convergence criteria are retained. The tree
backend is specific to the intercept-only null model.

The Numba kernels use float64 and `fastmath=False`. Compilation/cache loading is
performed before fitting timers and reported separately in `runtime.json`.
`--engine python` is available for correctness investigation without Numba; its
timing is not representative of the compiled tree implementation. Compilation
is not silently included in one method's per-fit runtime.

The benchmark reports:

- Null-fit time, iteration count, status, tau, intercept and objective.
- Differences in fitted probabilities, intercept, tau and objective.
- Exact ordinary-score U and variance for the same probe predictors, score
  runtime, stable log-P differences, and nominal/Bonferroni decision differences.
- Serialized per-model cache bytes for full dense, current compact dense
  rank-128/zero/shared representations, and the tree representation.

Probes are sampled from the frozen response selection; the first two are replaced
by the response and its complement when possible. `--stress-cases` adds labelled
synthetic rare/complement, clade/complement and monomorphic responses. These are
numerical stress cases, not additional biological observations. Nothing is
selected by AMR gene name, association significance or phenotype.

Score P-values are ordinary-score P-values without SPA, not LRT P-values.
Cache sizes use in-memory NPZ serialization. They exclude the shared tree/K,
global eigenvectors, source patterns, JSON metadata and filesystem overhead.
They do not measure disk throughput or peak RAM. The tree caches require the
benchmark's tree-based scorer; they are not drop-in production null files.

Default agreement thresholds are probability absolute error 1e-7, intercept
absolute error 1e-6, tau relative error 1e-5, working-objective relative error
1e-7 and score-variance relative error 1e-6. Tau/objective relative errors use a
denominator of at least one. Decision and variance-status mismatches also fail
the combined criterion. Review failures as well as successful fits. A status
match alone is not evidence of numerical agreement.

## Outputs, restart and interpretation

The output includes `benchmark_identity.json`, frozen selected arrays/tree,
`checkpoint.jsonl`, `null_results.tsv`, `comparisons.tsv`, `summary.json`,
`accuracy_outliers.tsv`, `runtime.json`, and editable
`tree_null_benchmark.png`/`.svg` plots. No full set
of fitted null caches is retained. Complete response/repeat comparisons are
committed by the coordinator, so a repeated command resumes completed work.
An interrupted final JSONL line is discarded; uncommitted in-flight jobs repeat.
Changed inputs, model code, engine or scientific settings require another output.
Worker count can change on restart. Avoid simultaneous runs into the same output.

To redraw without fitting:

```bash
python benchmarks/tree_null_benchmark.py \
  --out /data/benchmarks/tree_null_saureus_p50_s42 --plot-only
```

Inspect `status_mismatches`, `within_tolerances`, probability/tau/variance
differences and decision mismatches before interpreting speedup. A fit can
converge to a slightly different tau on a flat profile despite similar
probabilities; such differences are reported rather than hidden.
`probes_near_bonferroni_boundary` records coverage within 10% of the critical
score statistic. If zero, decision agreement does not establish accuracy near
the threshold. Timing repeats of one pattern are not independent samples, and
small synthetic test success does not validate all real-data cases.

Run a small first benchmark (50 responses, one repeat), then increase response
count/repeats if the speed and agreement justify it. This benchmark makes no
automatic production-backend or cache-format change.

## Larger accuracy check

Keep the same input cohort, tree and default numerical tolerances. Increase to
1,000 sampled responses, three timing repeats and 32 probes, using a **new output**:

```bash
python benchmarks/tree_null_benchmark.py \
  --work /project/kovar_output/saureus/discovery/results/.kovar_work \
  --tree /project/kovar_output/saureus/discovery/prepared_inputs/saureus_core.rooted.treefile \
  --include-from /project/benchmarks/tree_null_saureus_p50_s42 \
  --out /project/benchmarks/tree_null_saureus_p1000_r3_s42 \
  --patterns 1000 --repeats 3 --probes 32 --threads 4 --stress-cases
```

`--include-from` keeps the earlier run's complete selected response matrix as
the beginning of the new selection, then adds unique sampled patterns until
1,000 real sampled responses are present. Existing synthetic controls are
retained rather than regenerated. The earlier probe vectors are retained as
the first probes for each earlier response; 32 must be at least the original
probe count. Input hashes, sample order, covariance and hypothesis family must
match. Old timings/fits are **not** copied: every retained response is fitted
again using both current methods. This preserves known difficult cases and
threshold probes while expanding coverage.

The six synthetic controls are additional to the sampled responses. Three
repeats measure timing/numerical repeatability; they do not triple the number of
independent patterns or probe vectors. Changing `--patterns` changes the frozen
selection, so a new output is required. The original run is not overwritten.
Without `--include-from`, a new random draw need not contain the old responses.

`summary.json` now includes:

- Counts of distinct sampled and synthetic responses with failures, separate
  from the counts of repeated fit jobs.
- Median, 95th/99th percentile and maximum errors in probability, intercept,
  tau, objective, score variance, test statistic and log10 P.
- Probability-tolerance sensitivity from 1e-8 to 1e-6, while holding every
  other acceptance criterion fixed. Fitting, weights, optimizer and convergence
  tolerances are never changed by this reporting sweep.
- Probe counts within both 10% and 0.01% of the Bonferroni critical statistic.
  The tighter count is unavailable for old checkpoints that did not record it.

`accuracy_outliers.tsv` joins each exceptional comparison to its response ID,
sampled/synthetic label, exact failed criteria and both methods' convergence
diagnostics. Matched monomorphic controls are excluded from the exception list;
matched failed fits are not treated as successful validation.

The configured strict probability tolerance remains 1e-7. A small excess alone
does not establish unacceptable score error, but a looser reporting threshold
cannot repair nonconvergence or decision disagreement. Review fitted-objective
agreement and score/log-P errors, especially near the decision boundary, before
choosing any revised acceptance criterion.

You can produce the new diagnostic reports from the existing 50-response
checkpoint without refitting, using `--plot-only` with its original output.
Per-fit fields not recorded by the old program remain unavailable. Full fitting
with the changed benchmark source still requires a new output, preserving the
original experiment's code identity.

## Investigate the remaining exceptions

Use the saved larger benchmark directly; no original discovery folder is needed:

```bash
python benchmarks/investigate_tree_null.py \
  --benchmark /project/benchmarks/tree_null_saureus_p1000_r3_s42 \
  --out /project/benchmarks/tree_null_saureus_investigation \
  --threads 4
```

Selection includes nonconvergence and numerical failures remaining at the earlier
`5e-7` probability sensitivity threshold (nine plus two responses in the reported
run). This threshold selects cases only: acceptance retains all original strict
tolerances. Set `--selection-probability-tolerance 1e-7` to investigate every
original strict outlier instead.

Each selected response is fitted once with five settings: dense/100 iterations,
tree/100, tree/300, tree/300 with inner log-tau optimization tolerance `1e-8`, and
tree/300 with damping `0.5`. Outer convergence tolerance remains `1e-7` throughout.
The script calls the production fitting loop and observes its working solutions;
it does not replace the loop or modify production files. Scores use the original
probe vectors and ordinary score test, with SPA disabled as in the benchmark.

`fits.tsv` reports convergence, final gaps and numerical/decision differences
against a freshly fitted dense reference. `trace.tsv` records every iteration's
eta/tau gaps, passing conditions and consecutive-pass count. `summary.json`
counts accepted fits per setting. These are targeted diagnostics, not an
independent validation sample. The same command resumes completed responses from
the investigation's own checkpoint; original benchmark products remain unchanged.
