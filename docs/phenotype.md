# Phylogenetic MIC prediction and identifiable pair interactions

This optional downstream helper reads a completed covariation map. It does not
change the main KOVAR runner, network selection or effect-estimation workflow.

```bash
python -m pip install -e '.[phenotype]'
python -m ko_variation.phenotype \
  --fasta sample_ids.fasta --fasta-columns-are-locus-ids \
  --tree core.treefile --metadata collection.tsv \
  --species 'Staphylococcus aureus' \
  --network gene_covariation_functional.html \
  --annotation annotation/locus_to_gene.tsv --target mprF \
  --clusters population_groups/clusters.tsv --out mprF_mic
```

## Input and cohort contracts

FASTA records require unique sample labels and equal-length A/C sequences;
C/c means exact unitig presence and A/a absence. Use the sample-labelled FASTA
prepared for KOVAR, not repeated artificial contig headers. Declare either
`--fasta-columns-are-locus-ids` (unfiltered column i = original zero-based unitig i)
or `--locus-map FILE` supplying every `fasta_column,locus` pair explicitly.

Metadata requires `id,isolate,species` and MIC/sign columns (defaults:
`daptomycin_mic,daptomycin_mic_sign`). IDs join as
`id + '_' + isolate.replace('.', '_')`. Override with `--sample-map`, a one-to-one
`metadata_id,sample` TSV. `--species` excludes explicit non-target species;
`--exclude-sample` accepts additional exact IDs. Branch length does not determine
exclusion. Unmatched records are audited and cannot contribute an exact MIC.
Original inputs remain intact. The tree must match the FASTA after recorded
exclusions; other tips are never silently dropped. Pruning precedes midpoint
rooting, grouping and covariance calculation.

The network is completed HTML containing embedded `data` JSON, or that JSON,
with `gene_a,gene_b,locus_pairs`. Annotation requires `locus,gene,annotation_class`;
only coding-overlap mappings enter this analysis. Annotation names label plots;
cluster IDs join records. Duplicate locus mappings are errors. Locus annotations,
not sorted gene endpoint order, assign each unitig. An ambiguous target name
requires an explicit cluster ID. Direct and second-order neighbours are required.

Features are mapped unitigs supporting selected network edges, not every gene
variant or named resistance substitution. M0 is a **target-unitig baseline**, not
a complete curated determinant set. Exclusion here does not rescan/refit the
frozen network; cohort-clean network claims require separate reconstruction.

## Models and evaluation

Only positive MICs with an explicit equality sign enter fitting; other records
are audited. The response is log2(MIC). The earlier additive ridge helper is
replaced with Gaussian phylogenetic mixed models fitted by maximum likelihood:

```text
y = F beta + u + error
u ~ N(0, tau_squared K); error ~ N(0, sigma_squared I)
V = s_squared [(1-h)I + hK], 0 <= h <= 0.999999
```

K contains shared root-to-MRCA branch lengths from the pruned midpoint-rooted
tree, normalized by its mean diagonal. Fixed effects are unpenalized GLS.
Held-out predictions include conditional random-effect means from training MICs
and test-to-training covariance. This is a Gaussian MIC model, not the logistic
KOVAR genomic association model.

- M0: mapped target unitigs; M1 adds direct neighbours; M2 adds second-order neighbours.
- M1_pair: M1 plus raw binary products for selected direct target-unitig pairs.
- Mean_only and Phylogeny_only: intercept benchmarks without and with K.
- Random1, Random2 and RandomPair: matched added main effects and, where estimable,
  independent product terms.

Default grouping uses an external fastBAPS partition (`--clusters`, TSV columns
`sample,cluster`) inferred from genuine aligned core-genome DNA after cohort
exclusion. MIC values and target unitigs do not choose the partition. The final
cluster count is inferred, not fixed at ten. Labels join by sample ID; missing,
duplicate or unexpected extra samples are errors. Up to five GroupKFold
partitions (`--folds`) keep each cluster together. At least three clusters must
contain exact MICs. All models share test isolates. Cluster count depends on
alignment, prior and algorithm settings; it is not an established biological truth
or a guarantee that all related lineages are separated. Cluster grouping does
not replace the continuous tree covariance K in the MIC models.

Population labels are supplied as a `sample,cluster` TSV from an external
clustering analysis, such as the official fastBAPS R package. The clustering
wrapper was retired from KOVAR; existing cluster tables remain valid inputs.
The table must cover the retained cohort exactly. Cluster choice must be fixed
without using MIC values. The explicit complete-linkage option remains for
reproducing earlier group partitions.

Training-only filtering requires five present and five absent observations
(`--min-count`). Baseline-first rank filtering removes linear dependencies;
standardization uses training rows. A product requires five observations in
each of four states (`--interaction-min-cell`) and an independent design column.
Missing/sparse states are reported; a fold with no estimable product can give
M1_pair identical to M1.

Default controls are coding unitigs in other nodes of the frozen map, excluding
the target neighbourhood. Nearest training minor-frequency matching is approximate
and audited. Counts match independent retained columns, not redundant unitig IDs.
RandomPair also matches retained product count where possible; unmatched controls
are flagged and excluded from control boxplots. Random choices can differ across
folds. The default pool does not verify distance to the target; provide
`--random-pool` (a `locus` TSV) for an independently justified distal background.

Primary performance is pooled out-of-fold RMSE. Tables also report training RMSE,
within-one-dilution accuracy and reduction relative to M0. Fold dots are dependent
diagnostics and control boxes are distributions, not confidence intervals.
The genotype-defined network is frozen across folds: this tests held-out phenotype
relevance, not independent end-to-end network discovery. Prediction does not by
itself establish mechanistic epistasis.

## Pair-level inference

Every direct target-unitig pair receives four-state counts and raw mean MICs.
Estimable pairs compare target baseline plus both main effects against the same
model plus the product, refitting both by ML. The statistic is
`2(logLik_alternative - logLik_null)`, not a KOVAR score-test P value.

Default null parametric bootstrap uses 200 draws (`--interaction-bootstrap`),
refitting both models per draw; P = `(1 + exceedances)/(1 + draws)`.
Increase draws for precise or smaller P values: 200 cannot resolve P below 1/201.
Setting draws to 0 uses asymptotic chi-squared P. BH correction includes all
prespecified direct pairs, treating untestable pairs as P=1. Equivalent genotype
patterns are not independent biological replications. Coefficients use raw
binary-product/log2-MIC units; nominal intervals use GLS standard errors with
estimated variance components, not bootstrap intervals. Missing, sparse or
linearly dependent states get explicit untestable statuses. Phenotype
nonadditivity under this model does not identify mechanism or fitness epistasis.

## Outputs, figures and restart

- `excluded_samples.tsv,metadata_filtered.tsv`: exclusion audit and cleaned metadata.
- `analysis_matrix.npz,unitig_presence_absence.tsv.gz,column_mapping.tsv`: cleaned
  binary matrix, full TSV and original locus IDs.
- `matrix.npz,matrix.json`: **raw** cache, possibly still including excluded samples.
  `--matrix-cache DIR` reuses a compatible old cache; it is not a cleaned output.
- `features.tsv,selected_pairs.tsv,neighbourhood_presence_absence.tsv,neighbourhood.json`.
- `phenotypes.tsv,unmatched_metadata.tsv`: MIC and join audit.
- `midpoint.treefile,tree_order.tsv,phylogenetic_covariance.npz`.
- `performance.tsv,fold_performance.tsv,training_performance.tsv,model_features.tsv,
  predictions.tsv,interaction_report.tsv`.
- `tree_presence_absence,mic_performance,observed_predicted,interaction_states`:
  PNG and editable-text SVG.
- `prediction_checkpoints/,interaction_checkpoints/,run.json,figure_captions.txt`.

The tree heatmap omits groups and borders, separates neighbourhood blocks with
whitespace and shows unavailable exact MICs as white. Prediction/state plots
use black. `--stage tree` skips fitting; `--no-full-tsv` skips the large text
export while retaining the cleaned binary matrix.

Repeat the command to reuse completed folds and pair tests. Identity includes
source hashes, numerical helper code, settings and dependency versions; changes
recompute fits, except display DPI. Interrupted folds or pair bootstraps restart
that unit. Progress logs remain visible; numerical BLAS uses one thread.
ARC-specific wrappers stay outside this public repository.
