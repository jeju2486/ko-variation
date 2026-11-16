# Package architecture

One discovery pipeline plus optional downstream workflows. Main runs do not
import benchmarks or diagnostic programs.

| Stage | Modules |
|---|---|
| Inputs, unique patterns, durable state | discovery_store, kinship, checkpoint |
| Search and resource scheduling | discovery, coarse_screen, discovery_eligibility, discovery_execution, memory_budget |
| Null fitting and scores | glmm, tree_null, discovery_nulls, discovery_score, null_identity, spa |
| Annotation and reports | annotation_inputs, locus_annotation, functional_annotation, amr_report, phenotype, phenotype_lmm |
| Maps, figures, progress | functional_map, paper_plots, network, network_display, progress |

Tree-null caches store per-response vectors and small fixed-effect matrices;
the tree and covariance are shared. Exact tree contrasts replace per-response
spectral work. Dense nulls retain their spectral bounds. Selected distal beta
fits retain the full dense alternative model.

`examples/` contains portable Bash runners. `benchmarks/` exists on the development
branch only and contains manuscript reproducibility and temporary investigation.
Server account paths and scheduler settings remain outside the package.
