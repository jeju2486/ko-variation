# SCCmec carriage versus graR/sen unitigs

This workflow calls SCCmec from the original assemblies and tests whether coding
unitigs assigned to `graR` or `sen` associate with **carriage as the response**.
It uses the actual pyseer LMM API and the frozen KOVAR sample order and covariance.
Whole-gene `mecA` carriage is reported as a separate sensitivity response. No
KOVAR mixed models, annotations or source SQLite tables are recalculated/read.

## Environment and inputs

A single Conda environment is the simplest arrangement:

```bash
conda create -n sccmec_pyseer -c conda-forge -c bioconda pyseer sccmec
conda activate sccmec_pyseer
```

The [sccmec assembly classifier](https://github.com/rpetit3/sccmec) includes its
reference panel. If it is in a different environment, pass the absolute
`--sccmec /path/to/sccmec_env/bin/sccmec`. Its bin directory is prepended to the
classifier subprocess PATH so its Python, `sccmec-main` and BLAST dependencies
come from that environment. The coordinator still runs in the pyseer environment.

Required existing files:

- Completed target diagnosis: `diagnosis_identity.json` and `target_unitigs.tsv`.
- Frozen discovery work: `patterns.bin` and `K.npy`. The damaged `work.sqlite`
  is not required or opened.
- Assembly FASTA files whose basenames match the frozen isolate IDs. Supported
  extensions are `.fa`, `.fas`, `.fna`, `.fasta`, optionally followed by `.gz`.
- Alternatively, matching `.gff`/`.gff3` files with embedded `##FASTA` sequences.
  These original genome sequences are extracted temporarily, without annotation.

## Run

```bash
python /path/to/ko-variation/benchmarks/sccmec_pyseer.py \
  --diagnosis /project/kovar_output/saureus/diagnostics/amr_targets \
  --work /project/kovar_output/saureus/discovery/results/.kovar_work \
  --assemblies /project/assemblies/saureus \
  --gff-dir /project/panaroo_saureus_output/panaroo_gff \
  --out /project/kovar_output/saureus/diagnostics/sccmec_pyseer \
  --threads 4
```

Only one genome source is needed. Assemblies take precedence over GFF FASTA
fallbacks. Both directories are searched recursively. Multiple matching assembly
files for one sample are rejected rather than selected arbitrarily. Missing
genomes and tool failures have unknown carriage calls with named reasons.

`--work` is optional when the products remain in the standard species directory
or their recorded path. Source paths and genome checksums are retained in outputs.
The exact tool/schema versions and thresholds are recorded in `run_identity.json`.

## Carriage calls

The classifier runs with explicit target identity/coverage thresholds of 90/80%
and cassette-region identity/coverage thresholds of 85/83%, following its
documented defaults. The workflow interprets its completed reports as follows:

| Evidence | SCCmec response | mecA response |
|---|---|---|
| mecA detected with supported target-based type or cassette-region call | 1 | 1 |
| mecA detected but cassette untyped/unsupported | Unknown | 1 |
| mecA not detected, with conflicting cassette or partial mec/ccr evidence | Unknown | 0 |
| No mecA, cassette call or supporting mec/ccr target evidence | 0: panel-negative | 0 |
| Missing assembly or failed typing | Unknown | Unknown |

Multiple supported cassette types do not prevent a positive carriage call;
the uncertain type remains visible. Unknowns are never converted automatically
to zero. A negative call means **not detected by the installed reference panel
at the recorded thresholds**, not proven complete biological absence. Divergent
resistance genes or assembly gaps can escape detection. The sensitivity response
uses the tool's explicitly named `mecA` field; it is not a separate comprehensive
screen for every mec homologue.

## Association model and significance

For each response, unknown calls are excluded. The predictor vectors and both
axes of the covariance matrix are subset to those exact samples in their frozen
order. These exclusions are reported; a cohort can therefore differ between the
primary and sensitivity analyses. No tree is recalculated. Pyseer applies its
usual mean-diagonal covariance normalization after subsetting.

Each model is `carriage ~ unitig + phylogenetic random effect`, with an intercept
and no additional fixed covariates. One null is fitted per carriage response,
not per graR/sen unitig. All valid coding target patterns from the completed
diagnosis are considered, including patterns absent from the significant map.
Multiple cluster/gene assignments are preserved. They are sequence-variant
tests, not claims that core `graR` or `sen` genes are absent when a unitig is zero.

Frequency bounds default to `--min-af 0.05 --max-af 0.99`, matching the public
GOLD-GWAS runner and pyseer's default maximum AF. They are applied on each actual
analysis cohort. No MI/PC, marginal-P or four-cell-count gate is applied. Cell
counts remain in the table. Monomorphic/filtered/invalid tests have blank P-values
and significance flags, rather than false evidence of nonsignificance.

Every valid test reports:

- Raw `p_lmm`, beta/SE, response-null h2 and a flag for the usual h2 upper bound.
- `nominal`: P < 0.05, descriptive only.
- `significant_focal_genomewide`: alpha divided by the number of **all frozen
  unique patterns**, suitable for a focal GWAS comparison. This conservative
  family is fixed rather than reduced to the selected graR/sen panel.
- `significant_all_pairs`: the original discovery's full all-pairs family.

Alpha defaults to 0.05. SCCmec is the primary analysis; mecA is a sensitivity
analysis, not independent confirmatory evidence. This remains a targeted
current-cohort/k=61 test. It does not reproduce the original 806-isolate/k=31
GOLD-GWAS feature pool, tree or classifier, and it does not establish epistasis.

## Restart, CPU use and outputs

Repeat the same command to resume. Completed isolate calls are reused when their
genome checksum matches. Interrupted/failed isolates are retried, while completed
ones stay saved. Completed LMM results are reused when carriage inputs are
unchanged. If new typing calls become available, only the affected downstream
analysis is recalculated. Changed tool versions, frozen inputs or model settings
require a separate output. Run one coordinator per output directory.

`--threads` controls concurrent external typing jobs; each uses the classifier's
single-process default, with BLAS limited to one thread. Only selected patterns
are extracted from the memory-mapped full pool. Embedded genome FASTAs are
temporary; they are removed after each typing attempt. SCCmec and mecA each have
one LMM fit, so their numerical work does not require many worker processes.

Main outputs:

- `carriage.tsv`: every frozen isolate, assembly source, typing result and exclusions.
- `meca_unitig_agreement.tsv`: whole-gene mecA carriers versus each old mecA unitig,
  including gene-positive/unitig-negative isolates.
- `summary.tsv`, `summary.json`: carrier counts and graR/sen results for both responses.
- `SCCmec/associations.tsv`, `mecA/associations.tsv`: individual unitig P-values,
  coefficients, cell counts, frequency and significance/status flags.
- Each response folder also has `phenotype.tsv`, `sample_order.tsv`,
  `excluded_samples.tsv`, fitted-cohort `covariance.tsv` and `completed.json`.
- `isolates/`: native typing reports/logs and atomic per-isolate checkpoints.

Inspect `carriage.tsv` and the response carrier counts before interpreting the
association results. A response with too few samples or only one state is
reported as not testable rather than fitted to an invalid binary contrast.
