# KOVAR: ranked summary of the whole chat

Updated 2026-11-23. Read this instead of replaying the conversation. This covers
the scientific, computational and operational discussions, not only the solver.
Historical results apply to their tested cohorts. Verify the current run identity
before reusing numbers or caches; proposals are distinguished from conclusions.

## P0 -- release, accepted calculation and immediate next step

- Development v0.9.5 adds bounded tiled admission for tree-backed PC
  hierarchy search, cached batched tree scores with early rejection, and
  compact coarse misses. PC selection and the testing family are unchanged.
  The saved Klebsiella replay measured 1.29x faster search on 100,000 coarse
  candidates with matching decisions. Full-run speed remains to be measured.
  Compatible v0.9.4 runs keep their fitted nulls and position when resumed.

- Development revision tiled-tree-2 reuses freshly scored coarse decisions
  and runs the two-sided variance prepass. Significant tree-score P-values
  remain exact. Keep one four-mode replay for the supplementary optimization
  figure and compare all arms on the same late Klebsiella interval. The
  earlier sibling-tip, checkpoint batching and scoring prototypes stay in
  Git history. Full-data speed for the combined changes is still unmeasured.

- Development discovery layout: `discovery.py` owns stage order and prefit,
  `discovery_search.py` owns hierarchy and eligibility, and
  `discovery_store.py` owns input preparation and SQLite metadata.
  `discovery_nulls.py` stores models without scheduling workers. Both Bash
  launchers call discovery once. Verified v0.9.4 checkpoints have a specific
  migration to this layout with unchanged statistical settings.

- MIC development revision: primary censored ridge models now use no tree PCs;
  retain lineage-separated inner/outer CV. Compare blaZ carriage and unitig
  baselines separately with N1, N1+N2 and budget/MAF-matched random additions.
  Verify gyrA/parC alleles with installed mutation-catalogue reference offsets;
  recover missing calls conservatively from unique TBLASTN reference hits, saving
  provenance and leaving ambiguous calls missing. Compare separate/joint alleles
  and neighbour additions. Preserve raw AMRFinder/prepared caches; revised
  predictions use fingerprinted subfolders, retry failures and expose inner-fit
  errors. The uploaded audit had no callable gyrA/parC sites; previous fitting
  failures were incomplete numerical results, not evidence of no MIC effects.
  Real-data allele correctness and prediction completion remain unverified.

- Public cleanup is deferred until development is finished (user instruction,
  2026-11-17). Keep main unchanged for now. Afterwards remove research-specific
  paper/MIC/AMR helpers and catalogue, project-folder/species assumptions, chat
  history and personal workflow directions; review legacy selection/R plotting
  and unused duplicate cache/SPA code. Retain general discovery, annotation,
  network exploration, progress, resource/checkpoint machinery and focused tests.
- Developing map views now default to a connectivity-community overview with
  drill-down, focal gene neighbourhoods and an ordered adjacency matrix; the
  full graph and original statistics remain accessible. Grouping uses existing
  NetworkX Louvain plus disconnected-group splitting, not Leiden. Saved HTML/
  layout redraw reuses annotation and effects without inference or eggNOG.

- Public release: **main / v0.9.4**. The temporary publication branch was
  deleted after its commits were merged; use main for the minimal public tool.
  Keep discovery, annotation, maps, PNG/SVG figures, AMR/MIC helpers, progress
  and portable runners. Development: **codex/downstream-resume-progress**;
  supplementary benchmarks and temporary diagnostics stay there.
- Revision tree-null-1: dense remains default pending full larger validation.
  Opt in with --null-backend tree; install .[tree].
- Tree nulls: float64 contrasts/solves, same rooted covariance and exact sigmoid;
  inner log-tau tolerance 1e-8, outer PQL tolerance 1e-7, cap 100, damping 0.8.
  Failed tree fits fall back to dense. Full dense beta refits are retained.
- Tree per-response storage is O(n); shared K/eigensystem remain O(n squared).
  Exact tree score variances avoid response-specific spectra. Dense caches keep
  spectral bounds. Serialized cache size is not peak RAM or whole-run time.
- Practical probability agreement criterion: 5e-7, with other accuracy/decision
  checks retained. Convergence does not certify accuracy; fallback detects failed
  fits, not all inaccurate converged fits. Guaranteed agreement needs a justified
  error bound or reference calculation, not merely a status flag.
- NEXT: full 1,000-pattern tighter-optimizer benchmark with frozen responses and
  probes, then a separate validation sample. Do not claim this has already passed.
- Synchronize public changes into development while preserving its research files.
  Metadata-only release bumps may reuse state only with identical scientific
  inputs/settings, numerical code/environment and execution revision. Earlier
  backend/numerical changes need new statistical state.

## P1 -- scientific model and phenotype conclusions

- Goal: find distal genomic associations beyond ancestry, then test whether they
  improve phenotype explanation. Covariation, prediction, statistical interaction
  and functional/fitness epistasis are different claims.
- KOVAR models binary genomic B from binary genomic A:
  logit P(B=1)=alpha+beta*A+b, b~N(0,tau*K). K is normalized shared root-to-MRCA
  covariance. Nulls are response-specific PQL; production discovery uses the
  ordinary score U^2/V with SPA off, not an LRT or exact GMMAT reproduction.
  Output pairs are unordered; internal response orientation is not causality.
- High A-B covariation can be biologically interesting yet predictively redundant
  after B is known: I(Y;A|B)<=H(A|B). Perfect dependence prevents separate A, B,
  A*B effects being identified. Linkage/cassettes, ecology, ancestry and artifacts
  are alternative association causes.
- Test ab/Ab/aB/AB support. Interaction on a chosen MIC scale is
  mu11-mu10-mu01+mu00, not simply AB versus Ab. Sparse discordant states weaken
  inference. MIC interaction is not automatically fitness epistasis.
- Tested mprF/neighbour additive and interaction models did not improve held-out
  MIC prediction over simple benchmarks; predictions were nearly constant.
  This limits those features/models/cohort, not all epistasis or every mprF allele.
- “mprF-only model” differs from genotype Ab (mprF feature present, partner absent).
  A raw 0.3-log2 MIC difference alone establishes neither significance nor
  interaction. Phylogenetic adjustment may increase or decrease it.
- Unitig 0/1 means a sequence feature, not necessarily gene carriage or a known
  resistance mutation. Core mprF/rpoC mechanisms need defined alleles/SNPs.
  Compare models on identical phylogeny-aware held-out folds, training-only
  preprocessing and mean/phylogeny baselines; preserve MIC censoring.
- Current S. aureus follow-up: blaZ carriage and gyrA/parC allele states. The
  development-only `benchmarks/saureus_mic.py` audits target-unitig overlap with
  AMRFinder curated sites, then compares known markers/target unitigs/neighbours
  with matched random partners. Censored Gaussian ridge uses tree-PC covariates,
  not the binary KOVAR model or a full phylogenetic MIC random effect. Actual
  server allele calls and phenotype results remain unverified. mecA remains a
  valid MIC baseline despite absent map partners; no guarantee of improvement.

## P1 -- current pipeline, data and restart contract

- Inputs: PAN-GWES/gfa1_parser unitigs, sample-labelled A/C pseudo-FASTA and rooted
  tree. A=absence, C=presence; preserve original unitig columns. Repeated contig
  headers require authoritative sample order; never guess.
- MAF filtering -> exact unique patterns and original-unitig mapping -> reusable
  canonical/complement nulls -> PC/hierarchy search -> score decisions -> expanded
  unitig pairs -> graph distance -> distal-only beta.
- Defaults: MAF >=0.05; each final 2x2 cell >=20; PC40; 10% coarse retention.
  max(2% of n,10) was proposed as sensitivity analysis, not silently made default.
  Bonferroni alpha=0.05 uses all unordered unique-pattern pairs, not retained pairs.
- Significant close candidates remain exported. Beta is skipped at/below the
  configured cutoff. PAN-GWES unitig_distance supplies minimum graph-derived
  distance over connected genomes; this is not LD or plotted spacing.
  Default distal >10 kb; Neisseria 3 kb is a chosen analysis cutoff, not universal.
- Cutoff-only resume reclassifies saved distances and fits newly eligible missing
  effects. No separate beta-only mode: commit every beta attempt, including
  non-OK statuses; repeats skip committed work. Expanded unitig pairs can share
  a pattern fit. exact_pairs/queried counts are not the queued distal beta count.
- --threads is total CPU budget, one BLAS thread per worker. Use bounded tasks,
  adaptive memory/cache budgets, durable temporary state and atomic checkpoints.
  Null fitting is once per needed response, but many responses remain expensive.
  Prefitting changes scheduling; PC40 does not replace the null model.
- Never merge cohort counts: some runs had 613 raw/612 filtered samples; another
  standalone input had 821. GOLD-GWAS used the paper's 806 isolates. Pattern counts
  such as ~487k, 537,873 or 970,383 belong to particular filtering/canonical pools.

## P2 -- screening, categories, bounds: conclusions and rejected shortcuts

- Exhaustive pairs are quadratic. Exact-pattern compression is useful. MAF,
  entropy and phylogenetic D describe individual loci; none safely excludes all
  significant partners. Low D does not imply no adjusted association. Exploratory
  D/fastBAPS helpers were retired from the public tool.
- Tree-PC residual correlation adjusts each locus once, then ranks pairs cheaply;
  search both signs. Benchmarks favoured 40 PCs over 10/20/80/120 for significant
  reference recovery; more PCs were not monotonically better. Weak overall rank
  correlation can coexist with useful extreme-tail recall. Panel/coarse-pool
  recall percentages are not interchangeable or exhaustive guarantees.
- MI/SpydrPick is no longer a required production candidate input. Low marginal
  MI can coexist with strong adjusted evidence. Earlier simulated rank agreement
  does not establish a universal MI/p-value relationship.
- Global r-squared categories compress similar patterns, not physical locations.
  Representatives are proxies. Significance is not transitive; representative
  nonsignificance does not certify all members, nor within-category tests all
  cross-category associations. Singleton/category discard remains heuristic.
- Independent cluster-only screening misses cross-cluster pairs. ARACNE=1 does
  not guarantee direct KOVAR association or causality. Fixed 10 kb coordinate
  boxes are awkward for mobile/rearranged pangenomes. Group globally and measure
  physical distances after discovery.
- Hierarchy refines categories, representatives and leaf blocks; depths interleave.
  Top-level coverage is not time completion. Tiny panel categories do not describe
  full-pool category sizes. Blocked enumeration reduces memory, not quadratic work.
- Bounds certify only queried score decisions conditional on a fitted null.
  Loose bounds alone did not solve 100-billion-pair enumeration; PC/hierarchy
  reduce queries with measurable missed-hit risk. Keep exact reference panels.

## P2 -- AMR mapping, GOLD/pyseer and figure design

- Diagnose missing targets by tracing original unitigs -> annotation -> variable/
  unique patterns -> eligibility -> screen/hierarchy -> significance -> distance/
  beta -> aggregation/plot. Gene-catalogue presence alone does not imply a map edge.
  Thousands of unitig edges can collapse to hundreds of genes.
- Display annotation names, not group_####; retain internal IDs, ambiguous mappings
  and all unitig support. Multiple clusters may share tetM or another label.
  Distant genes assigned to one unitig require per-isolate full-length sequence/
  coordinate checks for repeated/shared sequences or annotation ambiguity.
- Prefer established presence/absence AMR determinants matched to each species'
  available metadata. Mutation mechanisms are a separate task. mecA/tet/graR/sen
  and mprF/rpoC were diagnostic examples, not universal validated interactions.
  The tet-specific tracer was retired in favour of general reports.
- Match phenotype definition (unitig, gene, mecA or SCCmec), cohort, relatedness,
  filtering and hypothesis family before comparing pyseer LMM with logistic scores.
  SCCmec is not identical to a single mecA unitig. LMM h2 near 1 calls for
  calibration/identifiability checks; it does not by itself prove GOLD-GWAS wrong.
- Functional maps: eggNOG role colours, annotation-name search, small grey unknowns,
  preserved supporting loci. Edge distance, annotation proximity and graphical
  layout are distinct. Gene-edge minima are summaries, not new gene-level tests.
- Paper figures: components >=3 plus a count of 1-to-1 components; large labelled
  named nodes, restrained hypothetical labels, collision-aware layout, focal mprF
  panels, SVG/PNG. Tree panels: midpoint root, consistent contaminant exclusion,
  remove group track/borders, white missing MIC, neighbour-section whitespace.
  Prediction panels mainly black. QQ plotting was explicitly omitted.
- Keep the consolidated AMR/pyseer/SCCmec diagnostic only on development. Preserve
  completed fits when source SQLite comparison fails; avoid blind full refits.

## P2 -- numerical evidence and supplementary material

- Earlier 1,000-pattern test (+6 controls, 3 repeats): median paired null-fit
  speedup ~26.5x, compact-cache reduction ~96%; 9 real tree convergence failures.
  No significance mismatches among jointly successful tested probes, but none
  within 0.01% of the Bonferroni critical statistic. Repeats are not independent.
- Targeted 11-case investigation: baseline tree now converged for all cases, so
  prior failures were not reproduced. Traces showed small eta oscillations near
  tolerance. More iterations/lower damping did not improve acceptance counts.
  Tighter optimization passed 10/11 at the old strict criteria and ran faster.
- Remaining pattern 288718: probability error 1.066e-7 versus strict 1e-7; relative
  P difference ~0.000186%; no tested significance disagreement. Accepted as
  practically negligible, not an error bound for unseen patterns. Environment/
  execution differences remain plausible; exact source of changed convergence
  remains unresolved without the matching earlier identity.
- Score-based beta approximation materially disagreed with full refits and was
  rejected despite speed. Warm refitting gave little gain. Alternative eigensolver
  benchmarks did not justify production complexity; their scripts were retired.
- ESSENTIAL supplementary: final dense/tree accuracy, convergence, decisions,
  fitting/scoring times and storage; MI/PC-count/budget recovery; exact/bounded/
  hierarchy recall, query counts and runtime. USEFUL: beta comparison.
  CONDITIONAL: AMR/MIC figures actually supporting manuscript claims.
  TEMPORARY: current null investigation. DISCARD: duplicate runners, tet tracer,
  legacy scanner, exploratory D/BAPS tools and superseded benchmark code.

## P3 -- operational lessons and evidence handling

- Public runners are host-independent. ARC scheduler/account wrappers remain
  outside Git; zoo-ardbeg uses Bash. ROOT/RESULTS override paths independently
  of hostname. Functional maps/paper plots/progress are installed runtime tools.
- Copied ARC identities may retain inaccessible FASTA paths: supply local prepared
  sample-ID FASTA. A folder containing only identities is not a completed result.
  First-run --out must be empty; prepare inputs in a sibling directory.
- Recover missing Bakta-style inputs from retained Panaroo GFFs with embedded DNA,
  preserving CDS IDs and flagging uncertain protein translations. Pyseer mapping
  needs BWA, BEDTools and BEDOPS/gff2bed in an appropriate separate environment.
  eggNOG requires emapper and nonempty matching DB files; a downloader finishing
  after DNS failure is not evidence of success.
- SQLite WAL/SHM and shared-filesystem behaviour matter. Snapshot on the writer
  node when needed; never delete live WAL/SHM. Deleting borrowed caches can break
  dependent runs. Fresh discovery needs upstream files, not old benchmarks.
- Use isolated noneditable installs/checkouts/PYTHONPATH for v0.8.3 versus current
  KOVAR. git pull updates the current branch; fetched changes on another branch
  do not imply checkout changes.
- Keep identities, seeds, selections, tables and figures outside source. Uploaded
  investigation evidence was copied to workspace outputs/supplementary_evidence/
  tree_null_investigation. Some Downloads filenames have since been overwritten:
  retrieve historical values from their actual frozen runs, not guessed files.
- User preferences: concise updates, focused checks, useful comments, genuinely
  fewer/clearer functions, commit/push authorized work; respect “do not change
  scripts yet” and distinguish local tests from external server execution.

## Prioritized to-do list

1. [ ] Validate the tighter tree optimizer on the frozen 1,000-pattern selection;
   retain dense comparison, original probe provenance, all fitting failures and
   significance decisions. Use the agreed probability criterion of 5e-7.
2. [ ] Validate on an independent selection and improve coverage immediately
   around the Bonferroni decision threshold. Resolve the earlier execution/
   environment discrepancy using the matching original benchmark identity.
3. [ ] Measure integrated runtime, peak memory, cache I/O and storage rather than
   transferring per-fit speedups to the whole pipeline. Keep dense default until
   the larger evidence supports changing it.
4. [ ] Recheck PC40/retention/hierarchy recovery against exact small-panel
   references using the same cohort, eligibility rules and full hypothesis family;
   quantify missed hits before tightening query budgets. For the three-day
   Klebsiella goal, profile a fixed slice, compare 0.5–1% PC40 budgets against
   10%, then benchmark batching by response, safe block pruning and writes
   separately. Measure total runtime including distance and beta.
5. [ ] Run final species-specific configurations with frozen cohort/sample order,
   documented cell and distance cutoffs, durable state and restart checks.
6. [ ] Revisit missing AMR targets and any GOLD/pyseer differences only from
   matched feature definitions, cohorts, relatedness and test families; preserve
   unresolved findings rather than infer absence or epistasis from a map.
7. [ ] Generate final functional maps and publication SVG/PNG panels. Select
   biological interpretation/phenotype analyses according to observed support.
8. [ ] Assemble supplementary figures: null accuracy/runtime/storage; PC-budget
   recovery; exact/bounded/hierarchy recovery. Add beta and AMR/MIC evidence only
   where useful to manuscript claims, with frozen tables/settings and provenance.
9. [ ] Retire temporary investigation code after validation; keep only main and
   the development branch for this workflow, preserving reproducibility sources
   on development and the concise handoff here.
10. [ ] After development is finished, remove paper plotting, MIC comparison and
    their research-only helpers from the publishing tool/main, including
    `ko-variation-paper-plots`, `ko-variation-phenotype`, `paper_plots.py`,
    `phenotype.py` and `phenotype_lmm.py`, plus their study runners, documentation
    and exclusive dependencies. Retain these capabilities, source, tests and
    reproducibility materials on `codex/downstream-resume-progress`. Preserve
    shared utilities needed by general annotation/network tools. This is a
    deferred to-do only: no script edits or branch merges are authorized now.
11. [ ] Consider AlphaFold 3 for candidate pairs with a plausible direct protein
    contact. Resolve unitigs to full annotated coding sequences from carrier
    assemblies, check mapping ambiguity, compare predicted interfaces with
    controls and report structural confidence separately from MIC interaction.
12. [ ] Make a supplementary SVG/PNG figure from the existing blaZ four-cell
    analysis of first neighbours. Show counts of exact and censored MICs and
    lineage support in all four cells, highlighting discordant 01 and 10
    (blaZ state first). For representative unitig 449486, plot held-out
    censored NLL gains over blaZ alone by fold alongside matched random
    controls and annotate sparse cells; retain the 50-unitig overview for
    context. Use the frozen `observed_cells.tsv`, `fold_scores.tsv`,
    `random_comparison.tsv` and `random_matches.tsv`. Label the fixed-penalty
    comparison and keep predictive gains distinct from interaction evidence.
