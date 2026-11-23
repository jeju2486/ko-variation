# Package architecture

One discovery pipeline plus optional downstream workflows. Main runs do not
import benchmarks or diagnostic programs.

| Stage | Modules |
|---|---|
| Discovery stages and restart | `discovery.py`, `null_identity.py` |
| Input preparation, unique patterns and SQLite state | `discovery_store.py`, `kinship.py` |
| Hierarchy, eligibility and pair traversal | `discovery_search.py`, `coarse_screen.py` |
| Worker scheduling and resource limits | `discovery_execution.py`, `memory_budget.py` |
| Score workers, null cache and numerical models | `discovery_score.py`, `discovery_nulls.py`, `glmm.py`, `tree_null.py` |
| Annotation and reports | annotation_inputs, locus_annotation, functional_annotation, amr_report, phenotype, phenotype_lmm |
| Maps, figures, progress | functional_map, paper_plots, network, network_display, progress |

Tree-null caches store per-response vectors and small fixed-effect matrices;
the tree and covariance are shared. Exact tree contrasts replace per-response
spectral work. Dense nulls retain their spectral bounds. Selected distal beta
fits retain the full dense alternative model.

The coordinator schedules both prefit and search. Score workers use the null
cache directly. The cache does not start workers or import the search.
The scheduler receives response affinity explicitly.

`examples/` contains portable Bash runners. `benchmarks/` exists on the development
branch only and contains manuscript reproducibility and temporary investigation.
Server account paths and scheduler settings remain outside the package.
