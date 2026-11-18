# S. aureus marker, neighbour and MIC comparison

Development-only research workflow. It uses an existing functional map and
upstream sequences; it does not rerun discovery, fit KOVAR nulls or run eggNOG.
The purpose is to test incremental phenotype prediction, not prove epistasis.

## Run or resume

Use `codex/downstream-resume-progress`; main remains unchanged. From the project
directory containing the checkout and existing species outputs:

```bash
git -C ko-variation pull --ff-only origin codex/downstream-resume-progress
THREADS=24 PCS=0 bash ko-variation/benchmarks/run_saureus_mic.sh
```

For a runner copied outside the checkout, refresh that copy first:

```bash
cp ko-variation/benchmarks/run_saureus_mic.sh scripts/zoo-ardbeg/run_saureus_mic.sh
THREADS=24 bash scripts/zoo-ardbeg/run_saureus_mic.sh
```

The default `all` stage upgrades existing neighbourhood metadata, reuses the
selected-column matrix, reuses AMRFinder reports when assembly/database identity
matches, rebuilds the derived allele audit, then runs the revised predictions.
There is no need to delete the old MIC output or rerun KOVAR discovery.

Required Python dependencies are available through the development checkout's
`[phenotype]` extra. External annotation tools:

```bash
conda install -c conda-forge -c bioconda ncbi-amrfinderplus blast
amrfinder -u
amrfinder -V
```

The workflow never downloads or updates a database automatically. `amrfinder -V`
locates the installed database, including quoted paths. It must contain
`AMRProt.fa` and `AMRProt-mutation.tsv`. Set `AMRFINDER_EXEC`, `AMRFINDER_DB`,
`TBLASTN_EXEC`, `REFERENCE_PROTEINS` or `MUTATION_CATALOG` for nondefault paths.
BLAST+ is needed only when target calls require recovery. To preserve raw-report
reuse, use the same installed database version as the original run.

Stages: `map`, `prepare`, `audit`, `predict`, `pairs`, `all`. After an initial `all` with
the new script, `predict` alone reuses the updated allele and preparation files.
`audit` alone rebuilds derived calls when the audit revision changes, without
invalidating unchanged raw AMRFinder reports.

## Four-cell blaZ analysis

Run this separately after the completed preparation and allele audit:

```bash
THREADS=4 RANDOM_SETS=100 bash ko-variation/benchmarks/run_saureus_mic.sh pairs
```

This uses the benzylpenicillin cohort and lineage splits from the existing MIC
workflow. A is blaZ gene carriage and B is one coding unitig from a first neighbour
of blaZ. The two characters in each cell give A then B. For each pair it fits an
A-only baseline, an additive A+B model and an A+B+A×B model. The fixed ridge penalty
is 10 on training-standardized features for all three models (`--pair-alpha` to
override). The fits use the censored likelihood with an unpenalized intercept.
Additive fits require both blaZ states and enough isolates on each side of at least
one within-blaZ comparison. Interaction fits require at least five training
isolates in each cell by default (`--min-count`). The cell counts also show how many
lineages contain both neighbour states within the same blaZ state.

For each first neighbour and fold, choose distinct random unitig patterns from
the existing control pool with the same training-cell eligibility. Rank by the
average absolute difference in unitig prevalence across training
(blaZ state, lineage) strata. Lineages are weighted by size within a blaZ state,
and both blaZ states receive equal weight. A distance of 0.05 means an average
difference of five percentage points. Control identities can vary by fold.
Their relative performance is a descriptive reference.

Find the output path in `OUT/latest_pair_analysis.json`. Start with
`observed_cells.tsv` (counts, censoring and lineage overlap) and
`random_comparison.tsv` (overall comparison against blaZ and the matched
controls). `fold_cells.tsv`, `pair_fit_status.tsv` and `random_matches.tsv` show
fold-specific support, fitted effects and control match distances.
`pair_scores.tsv` pools each complete model over the held-out isolates, including
separate scores for 00, 01, 10 and 11. Positive `delta_nll` favors the pair
over blaZ alone. `fold_scores.tsv` retains diagnostics when only some folds are
estimable. Random cell-specific scores use the random unitig's own cell
assignments. Overall neighbour and random scores use the same held-out cohort.
Rerunning with the same settings resumes from the saved folds.

## Inputs and configuration

The Bash runner accepts these environment overrides:

| Variable | Default |
|---|---|
| `ROOT` | Two directories above the runner |
| `REPO` | Checkout containing `benchmarks/saureus_mic.py` |
| `OUT` | `$ROOT/kovar_output/saureus/diagnostics/saureus_mic` |
| `DISCOVERY` | `$ROOT/kovar_output/saureus/discovery`; fallback `saureus/zoo-ardbeg/discovery` |
| `FASTA` | `$DISCOVERY/prepared_inputs/saureus_k61.sample_ids.fasta` |
| `TREE` | `$DISCOVERY/prepared_inputs/saureus_core.rooted.treefile` |
| `MAP_DIR` | Existing species `plots/functional_map`, plain or `zoo-ardbeg` layout |
| `NETWORK` | `map.html` or `map_saureus.html` under `MAP_DIR` |
| `ANNOTATION` | `$MAP_DIR/annotation/locus_to_gene.tsv` |
| `METADATA` | `$ROOT/assemblies/saureus_collection_metadatatxt.txt` |
| `UNITIGS` | `$ROOT/pangwes_saureus_output/saureus/saureus_k61.unitigs` |
| `ASSEMBLIES` | `$ROOT/assemblies`, exact-name recursive lookup |
| `GFF_DIR` | `$ROOT/panaroo_saureus_output/panaroo_gff`, embedded-FASTA fallback |
| `CLUSTERS` | Optional `sample,cluster` TSV, e.g. existing fastBAPS labels |
| `QUINOLONE` | `levofloxacin`; also accepts `ciprofloxacin`, `moxifloxacin` |
| `THREADS`, `RANDOM_SETS`, `PCS`, `FOLDS` | `4`, `100`, **`0`**, `5` |
| `GROUPS_COUNT`, `SEED` | `10`, `42` |

Metadata needs `id`, `isolate`, `species`, `benzylpenicillin_mic`,
`benzylpenicillin_mic_sign`, and the selected quinolone's `_mic`/`_mic_sign`.
Default joining uses exact `id_isolate` assembly basenames; `SAMPLE_MAP`
(`metadata_id,sample`) overrides it. Non-species records and samples absent from
the tree are excluded and recorded. Original zero-based unitig IDs are retained.
The Python interface also accepts `--locus-map` (`fasta_column,locus`).

Set `ROOT`, `METADATA`, `NETWORK`, etc. for another filesystem; examples do not
require the zoo-ardbeg execution host. A tree is still used to define held-out
lineage groups even when no tree PCs enter the regression. Without `CLUSTERS`,
the disclosed complete-linkage patristic partition is used; this is not fastBAPS.

## Allele evidence

1. Read target coding unitigs and full assembly DNA. `parC`/`grlA` are aliases.
2. Retain raw nucleotide AMRFinder reports and `--mutation_all` results, including
   wild-type calls. Accept both `Gene symbol` / `Accession of closest sequence`
   and `Element symbol` / `Closest reference accession` headers. Carriage reports
   accept `% Coverage of reference sequence` or `% Coverage of reference`.
   Normalise gene aliases and collapse repeated annotations of the same physical
   call. Distinct copies stay ambiguous.
3. Read the installed **S. aureus** mutation catalogue. Its reference-protein
   offset can differ from the traditional residue number in a mutation symbol.
4. If AMRFinder yields no parsed calls for a target gene, search the catalogue's
   reference proteins with TBLASTN. Require at least 90% reference coverage and
   90% identity in one alignment, and exactly one qualifying genomic locus per
   gene. Multiple qualifying loci or unresolved residues remain missing.
   Recovery is an alignment-based inference and retains a distinct source label.
5. Translate the reported/recovered interval, align it to the matching reference
   protein, and verify reference and observed residues. Ambiguous mappings stay
   unresolved. Search each complete unitig on both strands and check whether it
   covers the entire recognised codon.

`unitig_allele_audit.tsv` reports site evidence and a marker interpretation:
verified mutant, verified wild type, mixed/incomplete overlap, gene-associated
without confirmed site overlap, or no callable site. Verification applies to
callable audited observations; it does not prove all absent unitigs encode the
opposite allele. Counts are evidence rows, potentially multiple per isolate/site
when multiple gene copies are present. Inspect the detailed sample-level table.

`NO_CALLABLE_KNOWN_SITE` means unassessed, not mutation-free. Catalogue-defined
mutations and callable wild-type residues are distinguished from other alleles
and missing calls. Marker columns include catalogue sites even if fixed in this
cohort. Multiple physical calls at one site are missing for prediction.

blaZ/mecA/mecC carriage is separately derived from high-coverage AMRFinder hits
(90% default). Partial/disrupted hits are missing; no hit in a successful report
is detection-based absence. `GENE_CARRIAGE_MARKER` labels a blaZ-associated unitig;
it is not proof of whole-gene detection. `blaz_carriage_agreement.tsv` directly
tabulates gene/unitig agreement. Expression and MIC effects are not established.

Source definitions: [NCBI database formats](https://github.com/ncbi/amr/wiki/AMRFinderPlus-database)
and [AMRFinderPlus results](https://github.com/ncbi/amr/wiki/Interpreting-results).

## Prediction models

Primary models use **censored Gaussian ridge regression on log2 MIC, without
tree PCs**. Exact MICs contribute Gaussian densities; `<`, `<=`, `>`, `>=` limits
contribute one-sided probabilities. Limits are never imputed as exact values.
Unknown signs are excluded. Confirm common MIC units and assay comparability.
This is a quantitative phenotype model, not KOVAR's binary-response GLMM.

Penicillin has two parallel families:

| Family | Baseline | Extensions |
|---|---|---|
| `blaZ_gene` | Gene detection presence/absence | `_N1`, `_N1_N2`, `_Random_N1`, `_Random_N1_N2` |
| `blaZ_unitigs` | Target unitig presence/absence | Same four extensions |

`blaZ_gene_unitigs` additionally tests target-unitig information beyond carriage.
The primary models do not silently add mecA/mecC to these baselines; their calls
are retained in the genotype audit for possible separate background analyses.

Fluoroquinolone models compare `gyrA_alleles`, `parC_alleles`, and
`gyrA_parC_alleles`. The joint baseline receives the same four neighbour/random
extensions, plus a separate `_unitigs` arm testing gene-associated unitigs beyond
known allele states. Already included target genes are excluded from neighbours.
`N1_N2` means first plus second neighbours, not second neighbours alone.

All comparisons within an endpoint use the same callable target-marker/MIC
cohort and folds, including the mean-only benchmark. The separate allele models
use the joint callable cohort. Fixed/rare markers are reported and filtered within
training partitions, so a baseline may legitimately reduce to an intercept.

Random partners come from coding loci in the frozen map outside both targets'
two-hop neighbourhoods. Each baseline/budget has its own controls, matching the
number of independent added binary patterns and nearest outer-training-fold MAF.
Duplicate/complement patterns do not inflate budgets. MAF mismatch and selected
features are saved. This is a map-conditioned pool, not a genome-wide sample or
a guarantee of physical distance for every control.

Outer validation holds out lineage groups. Ridge tuning uses grouped inner CV;
scaling and redundancy filters use each corresponding training partition. The
default penalty grid is `.1,1,10,100`. Optional `PCS>0` is a separate ancestry
sensitivity analysis with training-only PC projection, not the primary run.

Primary accuracy is held-out censored negative log likelihood (lower is better).
Exact-MIC RMSE, R2 and within-one-dilution accuracy apply only to exact records.
Do not compare raw NLL between antibiotics. Improvements are paired by held-out
fold. Random-control fractions are descriptive, not epistasis P-values. The map
was frozen using cohort genotypes; this is not independent end-to-end discovery.

## Outputs, failures and restart

Preparation and raw calls remain at `OUT`:

- `prepared.{json,npz}`, `features.tsv`, `sample_groups.tsv`, `excluded_samples.tsv`.
- `neighbours.tsv`, `map_summary.json`, `focal_neighbours.{png,svg}`.
- `amrfinder/SAMPLE/`: original reports, logs and restart identities.
- `allele_recovery/SAMPLE/`: TBLASTN hits/log, cached derived calls and diagnostics;
  temporary assembly/reference FASTAs are removed after use.
- `allele_call_diagnostics.tsv`, `unitig_allele_audit.tsv`,
  `unitig_site_evidence.tsv.gz`, `known_genotypes.tsv`, `genotypes.npz`,
  `marker_summary.tsv`, `blaz_carriage_agreement.tsv`, `audit_identity.json`.

Each revised prediction has a deterministic directory `OUT/predictions/RUN_ID/`.
The path is printed and recorded in `OUT/latest_prediction.json`. Existing
earlier prediction files remain untouched. Changed model/input/dependency/code
identities select a different prediction directory while retaining upstream
preparation/call caches. Increasing `RANDOM_SETS` appends deterministic controls.

Prediction outputs:

- `genotype_combinations.tsv`, `training_feature_support.tsv`,
  `*_phenotypes.tsv`, `*_not_run.json`: genotype/MIC availability and filtering.
- `performance.tsv`, `predictions.tsv`, `model_features.tsv`,
  `neighbour_improvement.tsv`, `paired_fold_improvement.tsv`.
- `fit_status.tsv`, `fit_failures.tsv`, `mic_performance.{png,svg}`.
- `checkpoints/ENDPOINT/`: atomic per-fold/model/random-set saves.

Only successful checkpoints are skipped; failed checkpoints retry. The optimiser
allows up to 2,000 iterations per attempt with one numerical restart. Every
failed inner penalty/fold records the actual reason, feature count and exact-MIC
count. A failed fit is not evidence of no effect. Scale-boundary or nonconverged
fits are rejected. Overall performance requires **every outer fold**; successful
folds alone are diagnostics and must not be interpreted as the full comparison.

`THREADS` sets process workers; numerical libraries and BLAST/AMRFinder use one
thread per worker. Only the selected-column matrix is loaded, not the complete
million-column unitig matrix. Real-data allele correctness and convergence still
need inspection after running; synthetic tests do not establish calibration.
