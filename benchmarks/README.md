# Benchmarks and standalone publication figures

These optional programs are outside the installed statistical package. Run from
the repository root with KOVAR importable. Store results outside this source tree.

| Question | Program | Preserved outputs |
|---|---|---|
| MI vs tree-PC ranking, PC count/retention | `python -m benchmarks.screen_benchmark` | recovery.tsv, ranks, timings, frozen panel/reference |
| Extend/redraw completed PC results | `benchmarks.screen_benchmark_extend`, `python -m benchmarks.plot_screen_benchmark --out DIR` | extended_pcs and PNG/SVG |
| Exact vs adaptive significance, hierarchy and PC40 recovery | `python -m benchmarks.discovery_benchmark` | comparison.tsv and per-arm summaries/checkpoints |
| Redraw saved significance benchmark | `python -m benchmarks.plot_discovery_benchmark --out DIR` | recovery, wall-time and query-count PNG/SVG |
| Full vs warm/score beta | `python benchmarks/beta_benchmark.py` | beta_results.tsv, beta_summary.json, plots |
| Full-refit eigensolvers | `python benchmarks/beta_eigensolver_benchmark.py` | eigensolver_results.tsv, eigensolver_summary.json, plots |
| Progress of a discovery run | `python benchmarks/inspect_discovery.py` | read-only snapshots and ETA comparisons |
| Matched AMR target pairs: pyseer LMM versus saved KOVAR scores | `python benchmarks/diagnose_amr_pyseer.py` | [comparison, thresholds and resume](../docs/amr_pyseer_comparison.md) |
| Assembly-level SCCmec carriage versus graR/sen unitigs | `python benchmarks/sccmec_pyseer.py` | [typing, fixed-response LMM and restart](../docs/sccmec_pyseer.md) |
| Completed discovery to functional covariation map | `python benchmarks/functional_map.py` | annotated HTML/PNG/SVG, gene tables and unitig-support audit |
| Check mapping/eggNOG/plot environment and run the map | `ROOT=/project bash benchmarks/run_functional_map.sh SPECIES` | map_environment.json; CHECK_ONLY/SMOKE_TEST controls |
| Publication overview and focal neighbourhoods | `python benchmarks/paper_plots.py` | editable SVG/PNG, component/function summaries and captions |

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

The two standalone map/figure programs reuse saved association and beta results;
they do not fit statistical models or generate QQ plots. See the
[functional map and publication guide](../docs/map_and_paper.md). Server wrappers
remain separate from this public source tree.

For a fresh production significance benchmark, `--source-run DIR` samples a
frozen random panel from an existing run's prepared **unique patterns**. Add
`--patterns 1000 --seed 42 --threads 4 --out NEW_DIR`. The full source-pattern
Bonferroni family is retained by default. Selection uses no phenotype or
association results. Existing `--fasta PANEL --tree TREE` inputs remain supported.
The benchmark never sends the full genome to exhaustive comparison by default.
Plotting a saved `comparison.tsv` performs no fitting and preserves its timings.

The PC-count benchmark retains its historical design and requires an independent
installed v0.8.3 reference Python for fresh reference fits. Its reference family
is recorded separately from the production significance benchmark. These two
recovery percentages describe their respective candidate/reference universes.
