# S. aureus blaZ and gyrA/parC MIC comparison

This development-only workflow uses the completed map and upstream sequences.
It does not run discovery, fit KOVAR nulls, call eggNOG, or modify any source run.
`parC` and `grlA` are treated as nomenclature aliases in S. aureus. An annotation
name identifies a gene region; it does not establish a resistance allele.

## Installation and execution

Use the Codex development checkout; this helper is not in the minimal v0.9.4
release. Install the Python plotting/phenotype extras in the active environment:

```bash
python -m pip install -e '/data/kell7366/kovar/ko-variation[phenotype]'
conda install -c conda-forge -c bioconda ncbi-amrfinderplus
amrfinder -u
amrfinder -V
```

AMRFinderPlus can instead live in another environment: set `AMRFINDER_EXEC` to
its executable. Set `AMRFINDER_DB` to the database directory printed by `-V`,
containing `AMRProt.fa`. Both caller output and the reference database version are
recorded. No database is downloaded or updated by this workflow.
The default database is also read automatically from `amrfinder -V`; quoted
paths and terminal colour codes are supported. If detection fails, inspect
`$OUT/amrfinder_version.txt` and set `AMRFINDER_DB` explicitly. The initial
helper's prepared matrix remains reusable after this database-path-only fix;
changed biological inputs or cohort settings still require a new output.

From the project root on zoo-ardbeg:

```bash
ROOT=/data/kell7366/kovar \
AMRFINDER_DB=/actual/amrfinder/database/directory \
THREADS=4 bash ko-variation/benchmarks/run_saureus_mic.sh
```

The same Bash runner can be copied unchanged to
`scripts/zoo-ardbeg/run_saureus_mic.sh`. It finds the checkout at `$ROOT/ko-variation`.
There is no `sbatch` dependency. An optional first argument selects `map`,
`prepare`, `audit`, `predict`, or `all` (default).

Default layout and overrides:

| Variable | Default |
|---|---|
| `ROOT` | Two directories above the runner |
| `REPO` | Checkout containing `benchmarks/saureus_mic.py` |
| `DISCOVERY` | `$ROOT/kovar_output/saureus/discovery`; falls back to the `zoo-ardbeg/discovery` subdirectory |
| `FASTA` | `$DISCOVERY/prepared_inputs/saureus_k61.sample_ids.fasta` |
| `TREE` | `$DISCOVERY/prepared_inputs/saureus_core.rooted.treefile` |
| `MAP_DIR` | Existing `plots/functional_map` under plain species or `zoo-ardbeg` layout |
| `NETWORK` | `map.html` or `map_saureus.html` under `MAP_DIR` |
| `ANNOTATION` | `$MAP_DIR/annotation/locus_to_gene.tsv` |
| `METADATA` | `$ROOT/assemblies/saureus_collection_metadatatxt.txt` |
| `UNITIGS` | `$ROOT/pangwes_saureus_output/saureus/saureus_k61.unitigs` |
| `ASSEMBLIES` | `$ROOT/assemblies`, recursively searched by exact sample filename |
| `GFF_DIR` | `$ROOT/panaroo_saureus_output/panaroo_gff`; embedded-FASTA fallback |
| `OUT` | `$ROOT/kovar_output/saureus/diagnostics/saureus_mic` |
| `CLUSTERS` | Optional existing `sample,cluster` TSV, e.g. fastBAPS |
| `QUINOLONE` | `levofloxacin`; change to `ciprofloxacin` or `moxifloxacin` in a new output |
| `THREADS`, `RANDOM_SETS`, `PCS`, `FOLDS` | `4`, `100`, `10`, `5` |

Set `METADATA` explicitly if the table is elsewhere. It must contain `id`,
`isolate`, `species`, `benzylpenicillin_mic`, `benzylpenicillin_mic_sign`, and the
chosen fluoroquinolone's `_mic` and `_mic_sign` fields. Default sample joining
uses the established `id_isolate` assembly basename, with exact matching; an
explicit `SAMPLE_MAP` TSV (`metadata_id,sample`) can override it. Non-S. aureus
metadata records and FASTA samples absent from the supplied tree are excluded
and recorded. Original unitig column numbering is preserved.

Example with explicit inputs:

```bash
ROOT=/data/kell7366/kovar \
METADATA=/actual/path/saureus_collection_metadatatxt.txt \
NETWORK=/actual/path/map_saureus.html \
ANNOTATION=/actual/path/annotation/locus_to_gene.tsv \
AMRFINDER_DB=/actual/amrfinder/database/directory \
bash scripts/zoo-ardbeg/run_saureus_mic.sh
```

The direct Python interface supports `--locus-map` (`fasta_column,locus`) for a
FASTA whose columns are not original zero-based unitig IDs. The Bash runner
explicitly declares the upstream gfa1_parser column contract.

## What the allele audit establishes

1. Extract the map's coding unitigs for blaZ, gyrA and parC, retaining internal
   cluster IDs, all directly adjacent genes and representative edge statistics.
2. Run nucleotide AMRFinderPlus per assembly with `--organism
   Staphylococcus_aureus --mutation_all`. The latter distinguishes callable
   wild-type sites from missing gene/site calls. A missing mutation report is
   never silently encoded as wild type.
3. For each gyrA/parC call, translate the reported genomic region and align to
   its **matching database reference protein**. Map the curated residue position
   through that alignment, checking reference and observed residues. Ambiguous
   or unresolved mappings remain unconfirmed.
4. Search full target-unitig DNA on both strands around the called gene. Report
   whether it spans the complete recognised codon, plus its co-occurrence with
   the called allele. Matching genotype distributions alone is not direct site
   evidence. Multiple-copy site calls are missing for prediction.

`DIRECT_KNOWN_SITE_OVERLAP` says the unitig covers a recognised site; it does not
say unitig absence uniquely identifies its alternative allele. Review the
present/absent versus mutant/wild-type counts. No overlap can mean a linked
marker elsewhere in the gene rather than a recognised resistance variant.

blaZ, mecA and mecC are separately encoded as high-coverage AMRFinder detections (90% default).
Partial/disrupted detections are missing rather than absent; a complete report
without a blaZ hit is a detection-based absence call. This is not proof of gene
expression or a functional MIC effect.

References: [AMRFinderPlus usage](https://github.com/ncbi/amr/wiki/Running-AMRFinderPlus)
and [interpretation](https://github.com/ncbi/amr/wiki/Interpreting-results).

## Prediction and controls

This is a quantitative MIC experiment, **not KOVAR's binary-response test**.
The model is a Gaussian location-scale model on log2 MIC with ridge-penalised
genomic features and training-derived tree-PC covariates. It uses exact Gaussian
densities for `=` MICs and Gaussian tail probabilities for `<`, `<=`, `>`, `>=`.
Strict/non-strict limits are equivalent for the continuous latent model. Unknown
or missing signs are excluded. Confirm that source measurements really are MICs
in consistent units, rather than inhibition-zone diameters.

This tree-PC adjustment approximates ancestry; it is **not** a full phylogenetic
random-effect covariance or a newly validated mixed-model implementation. The
normal latent distribution and assay comparability are assumptions. Inspect
scale-boundary/optimisation failures; they are not treated as successful fits.

Models share the same callable genotype/MIC cohort and outer held-out groups:

- **Mean**: training mean/location and scale under the censoring model.
- **Tree**: tree PCs only.
- **Background** (penicillin): tree PCs plus mecA/mecC carriage, even when these
  genes have no displayed map edges. This avoids attributing their omitted
  predictive information to blaZ neighbours.
- **Known**: curated target markers plus tree PCs; fluoroquinolones also have
  separate gyrA-only and parC-only marker comparisons. Penicillin retains mecA/mecC.
- **Target_unitigs**: known markers plus all map coding target unitigs.
- **Neighbours**: Target_unitigs plus direct-neighbour coding unitigs.
- **Random_partners**: the same target baseline plus matched random partners,
  repeated 100 times. These are anchored target/random-partner comparisons, not
  unrelated replacement of both members of the pair.

Random partners come from other coding loci in the **frozen map**, outside both
targets' two-hop neighbourhoods. Their counts match independent added binary
patterns, with nearest training-fold minor-frequency matching; duplicate and
complement patterns do not inflate counts. MAF mismatch is reported. This is a
map-conditioned control pool, not a genome-wide sample or a guarantee that every
control is physically distal to the target. The map and features are frozen
using genotype data from this cohort, so this evaluates held-out phenotype
prediction, not independent end-to-end map discovery.

Use existing fastBAPS labels via `CLUSTERS` when available. Otherwise the runner
uses a disclosed complete-linkage tree partition (10 groups); it does not run or
claim fastBAPS. Tree PCs, scaling, feature redundancy filtering and ridge tuning
are fitted on training rows. Inner and outer folds both keep groups together.
Groups do not guarantee that all between-fold relatedness has been eliminated.

Primary performance is held-out **censored negative log likelihood** (lower is
better). Exact-MIC RMSE, R2 and within-one-dilution agreement are secondary and
apply only to exact observations, whose distribution may be selective. Do not
compare raw NLL across different antibiotic datasets. Neighbour improvement is
paired against Target_unitigs on the same held-out isolates. The fraction of
random controls doing at least as well is a descriptive benchmark, not a
calibrated epistasis P-value. There is no functional interaction claim.

## Outputs and restart

- `neighbours.tsv`, `map_summary.json`, `focal_neighbours.{png,svg}`: actual map
  inventory and readable target-centred view; no whole-network edge removal.
- `prepared.{json,npz}`, `features.tsv`, `sample_groups.tsv`, `excluded_samples.tsv`: selected
  coding columns, frozen cohort and ancestry inputs. Full million-column FASTA
  matrices are not retained in memory or copied into an enormous text table.
- `amrfinder/SAMPLE/`: raw reports, log and per-sample completion identity; temporary
  extracted assembly DNA is removed after calling.
- `unitig_allele_audit.tsv`, `unitig_site_evidence.tsv.gz`, `known_genotypes.tsv`:
  site overlap, detailed evidence and missingness-aware marker calls.
- `*_phenotypes.tsv`, `fit_status.tsv`, `performance.tsv`, `predictions.tsv`,
  `model_features.tsv`, `neighbour_improvement.tsv`, `mic_performance.{png,svg}`: cohort, failure audit,
  held-out comparisons and editable-text figures.
- `prediction_checkpoints/`: each endpoint/fold/model/random-set fit is committed
  separately. Rerun the same command to skip completed work. Increasing the
  random-set count appends reproducible controls. Changed input/model identities
  require a new output directory; source discovery files are never rewritten.

CPU budget is `THREADS` processes with one numerical/AMRFinder thread per worker.
Purely visual changes do not change the source map. A focal panel is preferable
to interpreting the full force-directed hairball; the full map remains available
as supplementary exploratory material.
