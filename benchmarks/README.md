# Manuscript reproducibility and development

These tools are optional and are not imported by a normal KOVAR run. Keep outputs
outside the checkout, retaining identities, sample selections, checkpoints,
comparison tables and figure source data. Numerical and candidate universes must
match before comparing recovery percentages.

| Priority | Purpose | Programs |
|---|---|---|
| Essential | Dense versus tree nulls: convergence, score agreement, runtime and cache size | tree_null_benchmark.py |
| Essential | MI versus tree-PC recovery, PC counts and retention | screen_benchmark.py, screen_benchmark_extend.py, plot_screen_benchmark.py |
| Essential | Exact versus adaptive decisions and hierarchical/PC screening recovery | discovery_benchmark.py, plot_discovery_benchmark.py |
| Development | Saved PC40 candidate replay with batched tree scores and pair/cluster bounds | cluster_score_benchmark.py |
| Development | End-to-end PC candidate interval replay with worker dispatch and SQLite | production_replay_benchmark.py |
| Useful | Full beta refitting versus warm fitting and score approximation | beta_benchmark.py |
| Conditional | Consolidated KOVAR/pyseer/AMR checks supporting biological claims | amr_checks.py, run_amr_checks.sh |
| Conditional | S. aureus target-unitig allele audit, censor-aware MIC prediction and random partners | saureus_mic.py, run_saureus_mic.sh |
| Temporary | Trace unresolved tree-null cases | investigate_tree_null.py |

[Tree validation and investigation](../docs/tree_null_benchmark.md),
[beta comparison](beta_benchmark.md), [AMR checks](../docs/amr_checks.md).
[S. aureus MIC workflow](../docs/saureus_mic.md) uses the completed functional map;
it does not rerun discovery or reuse a binary KOVAR beta as a MIC coefficient.
The superseded alternative-eigensolver benchmark was retired; historical source
remains available in Git. Preserve any result used by the manuscript externally.

Functional maps, paper figures and progress inspection are installed tools:
`ko-variation-functional-map`, `ko-variation-paper-plots`, `ko-variation-progress`.
They remain available on the minimal publication branch.

PC screening and representative hierarchy are approximate candidate searches.
Their missed-hit rates require measurement; an adaptive score interval certifies
only the queried pair, conditional on its fitted null. The Bonferroni family stays
fixed across benchmark arms. The historical PC benchmark may require an isolated
v0.8.3 reference installation. Production significance comparisons use the current
package. Never reuse a reference generated under different numerical settings as
if it were a matched comparison.

## Saved-candidate score benchmark

Run after the source PC screen and null prefit have finished. The benchmark
opens the source database read-only, maps the saved candidate list and reuses
verified fitted nulls. It samples response columns and retained candidates
within each column using a fixed seed. It neither refits nulls nor runs
distance or beta. Install the tree extra for Numba.

```bash
python benchmarks/cluster_score_benchmark.py \
  --run /path/to/discovery/results --out /path/to/cluster-score-benchmark \
  --responses 64 --clusters-per-response 512 --reference-block 256 --batch-size 128 \
  --repeats 3 --early-check-interval 16
```

The coarse workload replays representative pairs. Member expansion tests the
same response representatives against predictor-cluster members with lower
pattern IDs, following KOVAR's score orientation. These probes do not replay
the production hierarchy's routing. Sampling by response column also means
the measured prune fraction is specific to this slice.

- `current_block` uses the existing exact scorer. `wide_block` changes only
  the block width.
- `fused_tree` precomputes response-specific tree coefficients and accumulates
  contrast norms without allocating a whitened matrix. It classifies score
  statistics directly and skips conversion to individual p-values.
- `fused_rebuild` uses the same fused kernel and block width, with coefficients
  rebuilt for every block. Compare `coefficient_setups` and timing against
  `fused_tree` to isolate reuse across batches.
- `zero_tau_shortcut` uses the closed-form variance for binary predictors with
  an intercept and exactly zero fitted tau. It uses the fused tree scorer when
  tau is positive. `zero_tau_responses` and `zero_tau_pairs` count the sampled
  work eligible for this shortcut.
- `early_stop_tree` drops a predictor column once the squared contrasts already
  accumulated put its largest possible statistic below both cutoffs. It checks
  at bounded intervals and reports the fraction of contrast terms avoided.
- `weighted_bound` uses one matrix product for the numerator and weighted
  prevalence, then applies a variance lower bound before exact tree scoring.
- `cluster_bound` scores anchors in a batch and bounds changes to the numerator
  and variance over each cluster's variable positions and Hamming radius.
- `--scalar` additionally times width-one calls with both tree scorers.

`selection.json` can be replayed with `--selection` and a new output directory.
`response_results.tsv` records timings, pruned counts and discrepancies with the
exact reference. `summary.json` aggregates matched methods by workload and null
mode. Timings use the median across repeats for each response and method.
Common input loading, eligibility and cluster preparation time are
reported separately. The reported speedups are for scoring and bounds on a
single worker, excluding production queueing and SQLite writes. Results are
written before a comparison mismatch returns a nonzero exit status.
`speedup_vs_fused` compares each variant with the fused scorer on the same
workload and saved tree nulls. For `fused_rebuild`, use its reciprocal or
compare the two times to read the speedup from caching.

## Production interval replay

Run against a discovery checkpoint whose PC screen and null prefit are
complete. Use a separate output on the filesystem you want to benchmark.

```bash
python benchmarks/production_replay_benchmark.py \
  --run /path/to/discovery/results --out /path/to/production-replay \
  --start 0 --pairs 100000 --threads 24 --memory-gb 8 --batch-size 256
```

The replay takes a contiguous interval of the existing PC candidate list. It
copies the hierarchy for the selected root blocks into one scratch seed
database. Every mode starts from that seed, reads saved nulls without refitting
them, and follows the usual child traversal and leaf evaluation. The replay
does not include the original run's prior pair decisions, within-root search,
distance calculation or beta fitting. It includes process startup, scheduling,
scoring and durable writes. Seed creation and result comparison are outside the
timed replay.

| Mode | Addition |
|---|---|
| `current` | Historical response scheduler and scorer, before v0.9.5 |
| `fused` | Fused tree scoring with cached response coefficients |
| `early` | Partial-variance termination |
| `counts` | Shared predictor counts and cached decoded responses, with no second decoding of the eligible subset |
| `tile_rows` | Score response-aligned coarse windows before creating tasks, keeping decision rows |
| `tile_compact` | Store routine coarse rejections as pair keys and final states. Compute p-values for tree-scored significant hits |

`--zero-tau` adds the exact shortcut to the optimized modes. Compare a
separate replay on the same interval without the option to isolate its effect.
The summary reports total throughput, database growth, the singleton split
for the input interval, worker counters and timing. It checks every mode
against `current` for pair decisions, child tasks and significant exact
p-values, when available. A discrepancy returns a nonzero exit status.

### Two-sided bounds versus v0.9.5

Use a saved non-exact tree-backed v0.9.5 run whose PC shortlist and null prefit
are complete. Replay its **results** directory and write to a separate folder.

```bash
python benchmarks/production_replay_benchmark.py \
  --run /path/to/discovery/results --out /path/to/two-sided-replay \
  --start 0 --pairs 100000 --threads 36 --memory-gb 40 --batch-size 256 \
  --modes v095 v095_bounds v095_bounds_decision
```

| Mode | Scoring |
|---|---|
| `v095` | Actual v0.9.5 search and early-stopping scorer |
| `v095_bounds` | Test both variance bounds first and keep the baseline's exact significant P-values |
| `v095_bounds_decision` | Same bounds, with exact P-value calculation deferred for certified significant hits |

The added pass calculates the numerator and weighted presence counts together
for sparse, response-aligned batches. It hands undecided pairs and their
already-computed numerators to the v0.9.5 tree scorer. For a second pass, use
`--bound-rounds 2` with a different output folder. It applies the tighter
per-isolate weights (w_i/(1+tau L w_i)) only where round one leaves a decision
unresolved. (L) is a conservative maximum absolute row-sum bound for the
largest eigenvalue of the shared kinship matrix.

`summary.json` compares end-to-end replay wall time, tree work, per-round
decisions, child tasks and exact P-values. The decision-only mode reports
explicitly deferred P-values separately. A disagreement still returns a
nonzero status. The source checkpoint is opened read-only. A repeat in a new
output folder with reversed mode order can help assess startup and warm-cache
effects.

### Decision reuse and checkpoint writes

Compare the exact-P bounds mode with the same scorer and two coordinator
changes:

```bash
python benchmarks/production_replay_benchmark.py \
  --run /path/to/discovery/results --out /path/to/checkpoint-replay \
  --start 400000000 --pairs 100000 --threads 36 --memory-gb 40 --batch-size 256 \
  --modes v095_bounds v095_bounds_reuse v095_bounds_io
```

- `v095_bounds_reuse` uses freshly scored coarse decisions directly in the
  current traversal. Older and resumed tasks still look up saved decisions
  in SQLite.
- `v095_bounds_io` adds durable write batching. The replay commits at
  a frontier boundary or after 16,384 row changes at the end of a complete
  write section. Cursor updates, decisions and tasks stay atomic. SQLite's
  synchronous setting is the same for all modes.

The output reports wall time versus `v095_bounds`, reused gate decisions,
commit counts and time inside commit calls. Commit time can overlap the
existing database-write timer. Both variants compare saved pair decisions,
child task state and significant exact P-values against `v095_bounds`.

To test restart, run with `--modes tile_rows --interrupt before` or
`--modes tile_compact --interrupt after`. This intentionally returns status
75 at the first tile commit boundary. Repeat with the same interval and
output directory, omitting `--interrupt`. Completed modes are skipped.
For combined writes, use `--modes v095_bounds_io --interrupt before` or
`--interrupt after` to stop at the first combined checkpoint commit.
Use independent output directories for throughput measurements so timing
does not include a deliberate interruption.
