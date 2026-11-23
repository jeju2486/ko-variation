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
  --responses 64 --clusters-per-response 512 --reference-block 256 --batch-size 1024
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
- `weighted_bound` uses one matrix product for the numerator and weighted
  prevalence, then applies a variance lower bound before exact tree scoring.
- `cluster_bound` scores anchors in a batch and bounds changes to the numerator
  and variance over each cluster's variable positions and Hamming radius.
- `--scalar` additionally times width-one calls with both tree scorers.

`selection.json` can be replayed with `--selection` and a new output directory.
`response_results.tsv` records timings, pruned counts and discrepancies with the
exact reference. `summary.json` aggregates matched methods by workload and null
mode. Common input loading, eligibility and cluster preparation time are
reported separately. The reported speedups are for scoring and bounds on a
single worker, excluding production queueing and SQLite writes. Results are
written before a comparison mismatch returns a nonzero exit status.
For the bound arms, `speedup_vs_fused` isolates the time saved over the fused
scorer on the same workload and saved tree nulls.
