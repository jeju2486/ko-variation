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
| Supplementary figure | Wall time and result agreement for decision reuse and two-sided bounds | production_replay_benchmark.py |
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

## Search optimization supplementary figure

Use a saved tree-backed discovery run whose PC candidate selection and null
prefit are complete. This replay uses the same candidate interval in four modes:
the v0.9.5 scoring and traversal baseline, decision reuse alone, two-sided
bounds alone, and both changes together.

```bash
python benchmarks/production_replay_benchmark.py \
  --run /path/to/discovery/results --out /path/to/search-optimization \
  --start 400000000 --pairs 1000000 --threads 36 --memory-gb 40 --batch-size 256 \
  --repeats 4 --plot
```

Four repeats rotate the mode order once each. Timing includes worker startup,
traversal, scoring and synchronous SQLite writes. Seed copying and concordance
checks are outside the timer. Saved nulls, candidates and hierarchy are read
from the source checkpoint. All decisions go into a separate scratch SQLite
database. A discrepancy in pair decisions, task states or significant exact
P-values returns a nonzero status.

The output includes `summary.json` with per-repeat timings, decision checks,
tree contrasts and avoided lookups. With `--plot`, the same script renders
`search_optimization.svg` and `search_optimization.png` with individual repeats
and medians. On a server without Matplotlib, omit `--plot`, then run the script
with `--plot-summary /path/to/copied/summary.json` to render locally. Resumed arms
are excluded from figure timings.

Earlier tree-score and checkpoint prototypes are available in Git history.
