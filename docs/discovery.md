# Unitig discovery in KOVAR 0.9.2

Current execution revision: **`streamlined-1`**, tagged `v0.9.2-streamlined1`.
The original `v0.9.2` tag remains unchanged. `--version` still reports 0.9.2;
the startup banner, identities and summaries identify the execution revision.

`ko-variation-discover` (also `ko-variation discover`) accepts PAN-GWES unitig
presence/absence directly. SpydrPick, MI and ARACNE are not required. The old
`ko-variation --pairs ...` scanner remains available for existing workflows.

## Install and run

```bash
python -m pip install -e .
ko-variation --version

ko-variation-discover \
  --fasta graph.sample_ids.fasta --tree core.treefile \
  --unitigs graph.unitigs --paths graph.paths \
  --unitig-distance /path/to/unitig_distance --kmer-length 61 \
  --out kovar_v090 --threads 4 --memory-gb 4
```

For older distance binaries that require global edges, also pass
`--edges graph.edges`. The adapter examines the installed binary's help and
passes the older `--edges-file`/`--run-sggs-only` options only when supported.
Relative entries in `.paths` are interpreted from the launch working directory;
absolute paths are preferable. The graph and distance tools are external
dependencies, not installed by KOVAR. The adapter expects PAN-GWES distance
columns including count and minimum distance (columns 6 and 8, one-based).

To defer graph-distance calculation and beta fitting, omit `--unitig-distance`.
Add it on a later `--resume`; no significance scan is repeated. Use `--no-beta`
for significance plus distance only. A subsequent resume can add beta.

## Null stage and compact numerical caches

The default `--prefit-nulls all` fits the full unique-response pool before pair
testing. Existing local/imported valid fits are reused; only missing models
are fitted. Each finished model or failed-fit status is checkpointed. The
stage reports checked canonical patterns, status counts and complement aliases.
Completed stages resume without launching fitting workers or revisiting every
model. Interrupted stages check only inventory entries not yet committed for
the requested cache format/rank. The first migration from perf1 inventories
existing manifests once; it does not refit valid models. Model manifests remain
recovery records if a worker finishes before its inventory transaction commits.
Scoring still verifies each numerical cache on its first worker access.
Failed fits stay explicitly unresolved. A failed canonical complement fit is
not blindly copied: its missing native complement is attempted independently.

Use `--prefit-only` to finish this stage and stop. Repeat the same command with
`--resume` and without `--prefit-only` to test pairs using cached models only.
`--require-cached-nulls` prohibits missing fitting in either stage. Its first
use therefore needs a completed compatible null pool. `--prefit-nulls needed`
retains on-demand fitting for comparisons/diagnostics. All-pool fitting can fit
models the hierarchy never needs; it is not an automatic total-runtime saving.

Complement aliases share a fit without merging original patterns, graph nodes,
joint-cell eligibility or the Bonferroni family. Existing native saved models
take precedence, including their statuses. For an aliased complement, y and mu
are transformed, retaining the appropriate signed score. Positive/negative
association directions are not collapsed.

`--null-cache-format compact` is the default. Successful caches store fitted
probabilities, weights, tau and nuisance-projection quantities. Exactly zero
tau omits eigenvectors. Exactly constant weights reference the already-saved
global tree eigenvectors. General caches retain the complete eigenvalue
spectrum plus up to `--null-cache-rank 128` eigenvectors. Omitted directions
are covered by the existing outward variance bounds. If a later rank or exact
fallback requires them, the weighted decomposition is reconstructed once for
that resident response; the iterative null fit is not repeated. Float64 and
the existing statistical criterion are retained. Reconstruction has a cost;
measure it alongside the reduction in loading. `--null-cache-format full`
retains complete general scoring matrices for a controlled comparison.

Legacy local full caches convert without re-fitting. Data are published before
their new manifests, and the superseded local array is removed only after the
new manifest is committed. Imported source caches are not modified. A worker
interrupted before publication can repeat conversion; completed fitting is
retained. Conversion is one-time numerical I/O, not free work. During migration
the old released script must not run against the changing cache format.

Coarse representative comparisons now follow response-major triangular order,
with several response columns advanced concurrently across the worker pool,
so many predictors share one loaded response. The old scheduled row-prefix and
all its pending/completed tasks remain recorded; a new cursor skips that prefix
exactly. No internal-ineligibility decision deletes a node's external search.
Coverage representatives still use the existing any-passing-proxy rule.

Memory-pressure control uses resident bytes and available headroom; repeated
alerts no longer geometrically reduce an empty cache forever. Cgroup-v1
hierarchical usage uses matching hierarchical inactive-file counters. On glibc,
large evictions can request allocator page release. Actual live cache limits,
resident bytes/models and refused insertions are reported by worker. A truly
exhausted budget can still disable admission safely and recover with headroom;
the configured maximum is not a guarantee of live capacity.

Whole-null-fit eigensolver profiling is available without changing production
fitting or saved models:

```bash
python benchmarks/null_fit_benchmark.py --work kovar_v090/.kovar_work \
  --out null_driver_comparison --patterns 12 --threads 4
```

The TSV/JSON compare NumPy and SciPy evd/evr on the same frozen response sample,
with full-fit timings, decomposition/profiling subtimes, convergence statuses
and fitted-probability/tau differences. The benchmark writes reports only.
The production solver remains NumPy: a local three-pattern 613-isolate probe
gave only about a 1% median improvement for evd, insufficient to justify a
default change. This is not an ARC benchmark or a speedup guarantee.

## Inputs and identity

- Pseudo-FASTA has unique sample IDs and A=absence/C=presence. Repeated
  `contig00001` headers are rejected. Sequence columns are zero-based graph
  unitig IDs; filtering never renumbers them.
- `.unitigs` contains consecutive zero-based IDs followed by sequence, with
  exactly one row per original pseudo-FASTA column.
- Tree covariance is computed in retained FASTA sample order. Missing tree
  samples are an error unless `--tree-missing-samples drop` is explicit.
- `--exclude-samples ids.txt` removes specified isolates before filtering or
  tree covariance; excluded IDs are recorded in the run summary.
- MAF >=0.05 is the default. Exact pattern deduplication preserves all original
  mappings. Complementary patterns remain distinct. Within-identical-pattern
  pairs are deterministic redundancy and are not assigned score-test P-values.

## Search and statistical decisions

The default global hierarchy uses positive binary phi-squared >=0.8, followed
by nested 0.9, 0.95, 0.98 and 0.99 partitions and terminal blocks of 32.
Membership must satisfy the threshold directly against an observed anchor;
there is no transitive single-linkage grouping. Necessary presence-count bounds
eliminate impossible memberships before packed-bit parallel comparisons.

The default `single` policy uses a central observed pattern (maximum mean
correlation to members). This is computed from summed standardized vectors
in linear work, without a dense within-category matrix. `--rep-policy coverage`
adds farthest-first representatives, with ceil(1% of node size) representatives,
capped by `--max-reps 10`. Global membership still uses fixed anchors.

Representative comparisons use a relaxed `--screen-p 0.001` exploration gate
with no joint-cell restriction. Internal categories remain searchable even
when they have no external connections. Unresolved gates expand conservatively.
Negative representative gates can prune blocks: **this is heuristic and can
miss significant members**. Neither similarity nor representative significance
is copied to member pairs. `--search exhaustive` removes representative pruning
but still enumerates all unordered unique-pattern pairs.

Final eligibility defaults to `--min-cell-count 20` in each 2x2 cell, matching
the recent benchmark target. This intentionally excludes sparse patterns/pairs;
it is configurable and is not an intrinsic KOVAR requirement. It differs from
the legacy scanner's default of 1. The score response is the higher pattern ID
and the predictor the lower ID. IDs follow first occurrence by original locus.

Before representative fitting, v0.9.2 checks whether a whole hierarchy task can
contain a final-eligible pair. Every node has a saved anchor, its actual maximum
Hamming distance to all members, and the number of members with at least 2c
present and 2c absent observations, where c is the final minimum cell count.
For two nodes, anchor distance d and radii r and s bound every member-pair
Hamming distance between max(0,d-r-s) and min(n,d+r+s). Four cells >=c require
2c <= Hamming distance <= n-2c. A block outside this interval is skipped;
insufficient qualifying members also proves ineligibility. With c=0 this
pruning is disabled. Packed-bit summary work uses the worker CPU budget,
bounded batches and disk checkpoints; it fits no null models.

This skips comparison tasks, **not nodes, members or external associations**.
An internally ineligible category is still compared externally and can be
refined against another category. These are proven eligibility exclusions
under the specified cell rule; they do not establish biological independence.
Representative pruning remains a separate heuristic. Neither pruning changes
the Bonferroni family. Unqueried pairs are not assigned fabricated P-values.

Queued refinements receive at least three quarters of a full scheduling
frontier, while fresh coarse comparisons use up to one quarter when the queue
is full. Spare frontier space also accepts coarse comparisons. External
exploration therefore starts before all internal work finishes. Restart keeps
the same saved traversal tasks and coarse cursor; unfinished tasks can repeat.

Final significance is **Bonferroni adjusted P <=0.05**, not raw P <=0.05 or
P <=0.001. The family is all unordered unique-pattern pairs before hierarchy,
joint-cell or distance filtering. `--bonferroni-tests` can freeze a larger family
but cannot shrink it below that count. The family is at the pattern-test level;
expanded duplicate unitigs do not constitute additional independent tests.
For tiny panels, provide a screening gate at least as permissive as alpha/M.

The adaptive kernel bounds T=U^2/V after the fitted response null. Partial
weighted covariance projections bound the remaining variance; ranks
0/8/16/32/64/128 are added only where decisions are unresolved. Roundoff guards
widen intervals. Singular cases and intervals straddling either required
cutoff fall back to the optimized exact score core. `--exact` disables bounds
for a reference comparison. SPA is **off** in discovery; these intervals do not
certify SPA-adjusted tails. PQL/ordinary-score calibration retains the existing
model limitations; threshold certification is not experimental epistasis.

## Distance and post-hoc beta

Every significant pattern pair is expanded to its original unitig pairs.
Those IDs are supplied to `unitig_distance` in bounded zero-based query files;
unused MI/ARACNE fields are placeholders, not association measurements.

All significant candidates are retained:

| Minimum graph distance | Status | Post-hoc beta |
|---|---|---|
| >10,000 bp | DISTAL | fitted unless `--no-beta` |
| <=10,000 bp | LOCAL | skipped |
| Missing, negative sentinel, or no connected genomes | UNKNOWN | skipped |

Distance means **minimum shortest-path distance across connected single-genome
graphs**, not a reference chromosome coordinate. Unknown/disconnected pairs are
not presumed distal. The cutoff can be set with `--distal-bp` before the first
distance stage, then is frozen. Beta is an alternative phylogenetic logistic
mixed-model coefficient, fitted once per selected distal pattern pair and
reported only on distal original-unitig rows. Separation, nonconvergence,
clipped weights and invalid variance remain explicit effect statuses.

## Memory, CPUs, scratch and restart

`--threads` is a total CPU budget. Independent numerical tasks run in process
workers with one BLAS thread each; preparation's covariance decomposition and
the external distance executable use the same CPU budget in separate phases.
Parallel work submission is bounded. A response is assigned consistently to
one worker; completed score jobs are consumed without waiting for earlier,
slower submissions. Terminal comparisons are coalesced across hierarchy blocks
before response-wise batching. Numerical category reductions retain input order.

Memory is adaptive in KOVAR itself. It reads Slurm per-node/per-CPU allocations,
Linux cgroup v1/v2 limits (including parent limits), and physical memory, taking
the smallest applicable constraint. The working target leaves 20% headroom.
`--memory-gb` is an optional additional cap and cannot override a smaller detected
allocation. Without a job limit or explicit cap, the fallback is conservatively
4 GiB even on a large shared host. Slurm per-CPU memory uses the allocated CPU
count, rather than assuming the requested worker count is the whole allocation.

The planner reserves numerical working sets and coordinator/queue space before
assigning at most 60% of the working target to null caches. Cache entries are
charged by array bytes plus overhead, not a fixed model count. Each worker
evicts least-recently-used saved nulls when its byte budget is reached. A process
or cgroup working-set pressure check can reduce cache capacity; it grows back
gradually when pressure falls. Inactive file cache is excluded where available.
An active response stays reusable for the current scoring job even if the LRU
budget is zero. The budget is an estimate, not an OS-enforced RSS limit; keep
headroom for native workspaces, allocator behaviour and file cache.

Saved arrays are checksummed on first worker access. Eviction/reload checks the
file size, modification time and expected digest; changed files are rehashed.
This retains integrity checks without repeatedly hashing unchanged models.
Decision lookups use bounded bulk SQL, completed rows commit in blocks, and a
byte-bounded node cache avoids repeated traversal metadata reads. No unbounded
pair list or initial full category-pair graph is created.

Pseudo-FASTA is decoded one sample at a time to a temporary mapped matrix.
Locus blocks are packed/deduplicated through a disk index. Patterns and shared
covariance eigenvectors are mapped read-only. Category memberships, traversal
tasks, queried-pair decisions, distance results and beta statuses live in
SQLite, not full in-memory pair matrices. Initial category-pair queries are
generated incrementally; a billion-row initial task list is not preallocated.

Default work is `OUT/.kovar_work`; `--tmp-dir` can select persistent scratch.
**Keep it for restart and null reuse**. Do not use job-local temporary storage
that is deleted when Slurm ends. Startup estimates scratch preparation and
possible null storage. Full-pool nulls can require hundreds of GB or more;
disk-backed computation does not make storage free. SQLite transactions need
a filesystem with reliable file locking. An OS-released coordinator lock
prevents simultaneous writes to one work directory.

Repeat the same command with `--resume`. Committed sample rows, deduplication
blocks, category assignments, pair decisions, traversal batches, distance
batches and beta fits are reused. An unfinished worker job may repeat.
Input contents, fitting implementation, numerical environment and scientific
settings are checked. CPU count, memory budget and numerical batch size may be
adjusted on resume; they are execution settings.
Changing scientific settings requires a new output. Distances/beta can be added
later, but changed distance inputs/cutoffs cannot overwrite earlier results.

### Resume a released v0.9.0, v0.9.1 or v0.9.2 run

Stop the running job **before updating its shared source checkout**. Then run
v0.9.2 against the same OUT and original persistent work directory with `--resume`.
The released source hashes (LF/CRLF), inputs, scientific settings, null-fit
source and numerical environment are verified. A compatible manifest is backed
up as `run_identity.v090.json`, `run_identity.v091.json` or `run_identity.v092.json`; saved patterns, hierarchy, decisions, queued tasks
and null models are retained. Unknown edited implementations or changed science
are rejected. The score test, Bonferroni family/alpha, gate and joint-cell rules
are unchanged. Execution order changes do not provide a new recovery guarantee.
Verified `v0.9.2-perf1` and `v0.9.2-perf2` work also resumes. Their identity
backups include the previous execution revision, retaining earlier release backups.

### Reuse existing nulls

```bash
ko-variation discover ... --reuse-nulls-from /path/previous_output
```

Supply a v0.9 output or the former combined-benchmark output. Benchmark reuse
requires its original `settings.benchmark_dir/preparation.npz` and referenced
prior null directories to remain accessible. Both v1 and v2 provenance are
supported; v2's borrowed v1 cache is followed automatically. Sample order,
exact K, binary response patterns, fitting-source fingerprint, fitting settings,
array checksums and NumPy/SciPy versions must match. Patterns are matched by
bits, not by assuming old pattern IDs are unchanged. Arrays are read in place;
they are not copied or silently refitted. Only missing nulls are fitted.
`--require-cached-nulls` prohibits any missing-model fitting. Cached failed
fits stay unresolved. Old pair-decision checkpoints are not imported into the
new search implementation.

## Final files

- `unitigs.tsv.gz`: original ID, exact pattern ID, presence count, sample count
  and sequence-source path. Blank pattern IDs mark filtered loci.
- `associations.tsv.gz`: significant original-unitig pairs, canonical pattern
  orientation, optional exact P, labelled P bounds, minimum graph distance,
  connected-genome count, distance status and optional post-hoc beta.
- `run_summary.json`: family/cutoff, sample exclusions, decision counts,
  unqueried count, search interpretation, distance/effect statuses, resource
  plan and runtime counters.
- `run_identity.json`: provenance for restart and cache compatibility.
- `null_stage_summary.json`: canonical fitting statuses, cache modes and aliases.

No exact P is manufactured for a bound-certified edge. A small P remains genomic
association, not proof of a phenotype interaction, causality or mechanism.
The work database retains queried nonsignificant/ineligible/unresolved decisions
for restart, without exporting billions of unnecessary text rows. Final exports
are streamed and atomically replaced; an interrupted export can be regenerated.

During a run, `meta.runtime_totals` in `work.sqlite` records new null fits,
RAM hits, disk loads, evictions, bound/exact pair counts, worker-time totals for
fitting/loading/saving/bounds/exact scoring, and coordinator database timings.
Worker times are summed across processes and are not wall-clock durations.
Cache-load time includes hashing; do not add the hash subtotal again. Timing
counters describe completed evaluation batches; interruption can leave a last
batch's already-committed decisions without its final timing update. Legacy
v0.9.0 work lacks these counters, so historical time cannot be reconstructed
from the new counters alone. Progress prints query counts plus fit/cache counts;
read-only inspection uses `benchmarks/inspect_discovery.py`.

`hierarchy_metrics` separately counts proof-pruned internal/cross blocks and
heuristically pruned representative blocks. Progress also prints bound-certified
pair decisions and exact evaluations. Block counts and pair counts have
different denominators. New counters cover work completed by this version;
they cannot reconstruct historical pruning or certified-decision counts.

`prefit_runtime_totals` is separate from pair-stage `runtime_totals` so fitting
is not hidden when reporting pair throughput. `null_stage_progress` and
`active_stage` identify current work. `worker_state` contains live per-worker
cache limits, resident bytes/model counts, admission refusals and observed
memory usage/target. The inspector prefers `coarse_scheduled_count` over the
legacy row cursor; it is still not an overall-completion percentage.

An annotated distal map can use the existing network helper directly:

```bash
ko-variation-network --effects kovar_v090/associations.tsv.gz \
  --annotation locus_to_gene.tsv --gene-catalog gene_catalog.tsv \
  --out maps/covariation
```

Keep `run_summary.json` beside the association file. The helper validates its
Bonferroni cutoff, selects certified DISTAL rows and uses the explicitly labelled
`p_upper` as descriptive significance, without requiring exact P-values or
refitting beta. Gene names and functional colours retain the existing display
options. This optional visualization helper loads the significant edge table;
it is separate from the memory-bounded all-pattern discovery scan.

## Focused benchmark

`benchmarks/discovery_benchmark.py` runs exact, bounded exhaustive, and hierarchical
production paths on a supplied **small frozen pseudo-FASTA panel**. It reuses
compatible fitted nulls and reports query counts, recovery and invocation time.
The default cap is 2,000 original loci; it does not silently launch a full-pool
exhaustive scan. `--screen-p` matches production by default and
`--bonferroni-tests` can freeze the larger full-study hypothesis family for a
sampled panel. The comparison table includes fitting/loading/bound/exact worker
times and coordinator database times. Exact invocation time includes fitting;
the next arms reuse models, so these are not independent cold-run timings.
There are no full ARC runtime or whole-pool recovery guarantees in this release.
