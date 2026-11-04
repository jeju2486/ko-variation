# Benchmarks retained for publication

These optional programs are outside the installed statistical package. Run from
the repository root with KOVAR importable. Store results outside this source tree.

| Question | Program | Preserved outputs |
|---|---|---|
| MI vs tree-PC ranking, PC count/retention | `python -m benchmarks.screen_benchmark` | recovery.tsv, ranks, timings, frozen panel/reference |
| Extend/redraw completed PC results | `benchmarks.screen_benchmark_extend`, `python -m benchmarks.plot_screen_benchmark --out DIR` | extended_pcs and PNG/SVG |
| Exact vs adaptive significance, hierarchy and PC40 recovery | `python benchmarks/discovery_benchmark.py` | comparison.tsv and per-arm summaries/checkpoints |
| Full vs warm/score beta | `python benchmarks/beta_benchmark.py` | beta_results.tsv, beta_summary.json, plots |
| Full-refit eigensolvers | `python benchmarks/beta_eigensolver_benchmark.py` | eigensolver_results.tsv, eigensolver_summary.json, plots |
| Progress of a discovery run | `python benchmarks/inspect_discovery.py` | read-only snapshots and ETA comparisons |

The PC benchmark was moved from the research modules injected into ko_variation
into this separate directory. Existing completed PC results can still be plotted
or extended. A fresh historical reference needs `--reference-python` pointing to
an independent, non-editable v0.8.3 environment; the subprocess removes PYTHONPATH
so this release's direct-discovery CLI cannot override that installation.
Recomputing a reference after source/settings changes needs a new output.

The current significance benchmark uses the production exact/bounded paths and
caps input at 2,000 loci by default. It keeps the hypothesis family fixed when
specified and reuses compatible nulls. It reports final-hit and coarse-gate
recovery separately. The historical PC rank benchmark and production coarse PC
screen are different candidate universes; do not transfer recall percentages
between them without testing.

The beta/eigensolver benchmarks read prepared arrays and a frozen selection.
Their scripts/outputs and checkpoints are independent of the main search. See
[beta details](beta_benchmark.md) and [eigensolver details](beta_eigensolver_benchmark.md).
No benchmark is a default dependency of discovery.
