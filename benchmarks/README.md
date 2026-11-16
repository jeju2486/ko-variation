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
| Useful | Full beta refitting versus warm fitting and score approximation | beta_benchmark.py |
| Conditional | Consolidated KOVAR/pyseer/AMR checks supporting biological claims | amr_checks.py, run_amr_checks.sh |
| Temporary | Trace unresolved tree-null cases | investigate_tree_null.py |

[Tree validation and investigation](../docs/tree_null_benchmark.md),
[beta comparison](beta_benchmark.md), [AMR checks](../docs/amr_checks.md).
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
