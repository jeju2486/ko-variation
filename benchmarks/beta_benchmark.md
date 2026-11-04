# Benchmark selected-pair beta estimation

This is an optional benchmark, separate from the production runner. It reads an
existing discovery output and its saved null models. It never changes the main
runner, `glmm.py`, the production database or saved null files. No phenotype or
MIC data are used: beta describes the directional unitig association.

From an environment with this checkout importable, run:

```bash
python benchmarks/beta_benchmark.py \
  --run /path/to/kovar_pc40_output/saureus \
  --out /path/to/beta_benchmark \
  --pairs 200 --threads 4
```

The optional plotting dependency is Matplotlib (`pip install matplotlib`). Use
`--no-plots` for numerical results alone. The same command can be run under Bash
or a scheduler. `--threads` is the total CPU worker budget; each worker uses one
native numerical thread. Memory use is bounded by the sampled pair count plus
at most eight saved null decompositions per worker, rather than the full pool.

## Comparisons

- **cold_refit:** the current production PQL fitter, initialized from prevalence
  with tau = 0; the production beta acceptance checks are retained.
- **warm_refit:** that same function initialized from the saved response null's
  linear predictor and tau. An isolated AST adapter changes only those initial
  assignments, without editing production source or changing cache hashes.
- **score_approx:** U/V under the saved response null, using exact score
  calculations with SPA off. If the null contains a compact eigenvector sketch,
  the full weighted decomposition is reconstructed in benchmark worker memory.
  This coefficient is an approximation, not a full alternative-model refit.

Each method receives the same ordered predictor/response pair. Method order is
randomized per pair and repeat. Null access and reconstruction are reported
separately; reconstruction is also included in score elapsed time. Both methods
that use nulls share the worker cache. Loading time is charged to both as an
indicative access cost, so those fields must not be summed across methods to
estimate actual wall time or treated as independent cold-I/O measurements.
Missing or unsuccessful saved nulls are reported, never silently fitted.

## Sampling and restarting

The default source is the distinct significant pattern pairs that have at least
one completed distal unitig distance. A seeded uniform reservoir samples from
that pool without loading all pairs into memory. Distances must already have
started. When distances are still pending, the pool is provisional; the selected
sample is frozen in `selection.json` and will not grow on resume.

For a fitting-only benchmark before distances are available, pass
`--pair-source significant`. That sample can contain local/unknown-distance pairs
and must not be described as a distal-only benchmark. The script streams the
existing `distance_pattern` index and deduplicates adjacent pattern IDs in
Python. This avoids the old SQL DISTINCT query's temporary sorting B-tree.
The scan still reads the source pool and can add I/O load during production.

For an active WAL database, prepare the sample on the **same compute node** as
the production writer, rather than accessing live SQLite WAL state from another
node over a shared filesystem:

```bash
python benchmarks/beta_benchmark.py --run /path/to/kovar_pc40_output/saureus \
  --out /path/to/beta_benchmark --pairs 200 --threads 4 --prepare-only
```

Then run the same command without `--prepare-only` in the benchmark allocation.
Preparation freezes both the sample and the required null-manifest paths.
Subsequent fitting reads those paths and numerical files without opening the
production database, including in workers. No database copy or production pause
is needed for same-node preparation. Do not remove live journals or bypass WAL
using `immutable=1` on a database that is still changing. If same-node preparation
fails, inspect the printed extended SQLite error, storage, quotas and access;
the generic disk I/O error alone does not identify the cause.

Rerun the identical command to reuse completed pair jobs. The final partial
checkpoint line, if any, is discarded; interrupted jobs are repeated. Changing
the pair count, repeats, seed, source, numerical versions or benchmark/fitter
source requires a new output directory. Worker count can change on resume.
Do not run two benchmark processes against the same benchmark output.

Start with 50--200 pairs. Increase the sample or use `--repeats 3` in a new output
after seeing the measured cost. A small sample does not establish agreement for
all selected pairs. Close-to-separation fits and strong effects need particular
attention. Retain the original run and borrowed null directories during testing.

## Outputs and interpretation

- `selection.json`: frozen sample and source/code provenance.
- `checkpoint.jsonl`: independently resumable complete pair jobs.
- `beta_results.tsv`: per-method beta, status, elapsed time, iteration count,
  tau, four joint-state counts, null access/reconstruction and numerical timings.
- `beta_summary.json`: status counts, paired speedup, coefficient errors and sign
  agreement against cold refits; default absolute beta agreement tolerance 1e-4.
- `beta_benchmark.png` and `.svg`: successful-fit runtime and coefficient
  comparisons, using black/white styling and editable SVG text.

Speedup and coefficient errors use pairs successful in both compared methods.
Inspect failure counts as well; a method can appear faster by failing early.
The score coefficient can differ systematically for strong effects. Warm-start
agreement is measured, not assumed; PQL fits can converge differently.

The invocation wall time includes all three methods and worker startup. It is
not a production throughput estimate, especially after a resume. Individual
pair timings remain in the checkpoint. Scheduler and database-write changes
are not compared by this benchmark. No optimization is selected automatically.
