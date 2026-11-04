# Package architecture

The installed package has one discovery pipeline and optional downstream stages.
Benchmarks, ARC runners and research outputs are outside its runtime modules.

| Stage | Modules | Role |
|---|---|---|
| Public command | cli | Thin discovery entry point and version |
| Preparation/storage | discovery_store, kinship | Streamed A/C inputs, unique patterns, tree covariance, SQLite |
| Candidate search | discovery, coarse_screen, discovery_eligibility | PC shortlist, hierarchical tasks, joint-cell eligibility |
| Numerical workers | discovery_score, glmm | Adaptive score decisions and selected-pair beta fits |
| Null models | discovery_nulls, null_identity | Fitting, bounded caches, immutable imports and compatible restarts |
| CPU/memory | discovery_execution, memory_budget | Bounded scheduling and allocation-aware budgets |
| Annotation | annotation_inputs, locus_annotation | Unitig-to-gene assignment and ambiguity |
| Effects/selection | selection_cli, annotation_cli, postprocess, effect_checkpoint | Existing-table selection and resumable refits |
| Networks | network, network_display, network_cli, functional_annotation | Gene aggregation and PNG/SVG/HTML display |
| Biology reports | amr_report, phenotype, phenotype_lmm | Determinant representation and held-out MIC comparisons |
| Shared utilities | checkpoint, workflow_cache, io_utils, spa | Fingerprints, atomic outputs and numerical/input helpers |

`checkpoint.py` now contains only fingerprint and atomic-write utilities used
by downstream stages. The obsolete candidate-scan shard store, scanner and
scanner-only diagnostics were removed. Existing downstream cache serialization
is preserved. The numerical fitting/score kernels are unchanged. The normal
discovery pipeline reports cumulative beta checkpoints and accepts the verified
earlier PC40 checkpoint after matching inputs, settings and numerical code.

The `benchmarks/` directory contains PC-count recovery and frozen significance
benchmarks plus beta/eigensolver comparisons. It is excluded from package
installation and the main run. Historical reference generation can call an
independently installed v0.8.3 scanner; its code is not duplicated here.

The `examples/` directory contains portable commands and synthetic downstream
inputs. ARC-specific runners and workspace cleanup scripts are distributed
separately. Paths, allocations and user accounts are not part of the package.
