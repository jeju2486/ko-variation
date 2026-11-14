# KOVAR and pyseer checks

Use **one Python script**, `benchmarks/amr_checks.py`, and optionally the single
Bash runner, `benchmarks/run_amr_checks.sh`. The previous
`diagnose_amr_targets.py`, `diagnose_amr_pyseer.py` and `sccmec_pyseer.py` have been
consolidated here. Numerical methods, target definitions, response orientation
and output columns are unchanged. Production discovery and plotting are unchanged.

## Run on the existing project

From the project root (the directory containing `assemblies`, `kovar_output`
and the checkout):

```bash
# Run KOVAR target diagnosis, then the matched pyseer comparison.
KOVAR_PYTHON=/path/to/kmer_gwes_env/bin/python \
PYSEER_PYTHON=/path/to/pyseer_env/bin/python \
THREADS=4 bash ko-variation/benchmarks/run_amr_checks.sh saureus
```

The two Python paths may be the same if one environment contains both tools.
KOVAR diagnosis requires KOVAR, NumPy, SciPy, Biopython and threadpoolctl.
The pyseer commands require pyseer, pandas, NumPy and SciPy; **they do not import
KOVAR or require Biopython**. Set `REPO` if using a differently named checkout.
Set `ROOT` when running outside the project root. No scheduler is required.

For existing results, select just the check needed:

```bash
# No KOVAR rerun: read the completed target diagnosis.
PYSEER_PYTHON=/path/to/pyseer_env/bin/python \
bash ko-variation/benchmarks/run_amr_checks.sh saureus pyseer

# Optional assembly-carriage check; pyseer environment must also contain sccmec/BLAST.
PYSEER_PYTHON=/path/to/sccmec_pyseer_env/bin/python \
bash ko-variation/benchmarks/run_amr_checks.sh saureus sccmec

# Recover reports from committed KOVAR fits without fitting anything.
bash ko-variation/benchmarks/run_amr_checks.sh saureus report

# Run only KOVAR, requiring the original null cache even across library versions.
ORIGINAL_NULLS=require bash ko-variation/benchmarks/run_amr_checks.sh saureus kovar
```

Stages: `both` (default), `kovar`, `pyseer`, `sccmec`, `report`, or `all`
(both plus optional carriage). Additional CLI flags can follow a **single**
stage, for example `saureus kovar --report-only`.

The runner selects a completed discovery with `associations.tsv.gz`, checking
`kovar_output/saureus/discovery/results` first, then
`kovar_output/saureus/zoo-ardbeg/discovery/results`. Set `RESULTS` to select an
explicit run, including an ARC run copied to another machine. An unfinished
directory containing only JSON files is not selected automatically.

Overrides, when needed:

| Variable | Meaning |
|---|---|
| `RESULTS` | Completed discovery `results` directory |
| `SOURCE_WORK` | Relocated original `.kovar_work` directory |
| `DIAGNOSIS` | Existing target diagnosis, including an older run elsewhere |
| `FASTA`, `UNITIGS`, `PANAROO` | Sample-ID FASTA, unitig DNA, Panaroo membership CSV |
| `GFF_DIR` or `BAKTA` | Existing per-sample annotations; no new annotation required |
| `MAP_DIR` | Functional-map output directory |
| `PYSEER_OUT`, `SCCMEC_OUT` | Existing comparison directories to resume |
| `ASSEMBLIES`, `SCCMEC_EXEC` | Assembly source and optional classifier executable |

The defaults keep the existing sibling output directories:
`diagnostics/amr_targets`, `diagnostics/amr_pyseer_lmm`, and
`diagnostics/sccmec_pyseer`. Reuse those directories rather than moving fits.

## Direct Python commands

For arbitrary directory layouts, run the same file with explicit inputs:

```bash
python /repo/benchmarks/amr_checks.py kovar \
  --results /project/discovery/results \
  --fasta /project/discovery/prepared_inputs/saureus_k61.sample_ids.fasta \
  --unitigs /project/pangwes/saureus_k61.unitigs \
  --panaroo /project/panaroo/gene_presence_absence_roary.csv \
  --gff-dir /project/panaroo/panaroo_gff \
  --map-dir /project/plots/functional_map \
  --out /project/diagnostics/amr_targets --threads 4

python /repo/benchmarks/amr_checks.py pyseer \
  --diagnosis /project/diagnostics/amr_targets \
  --work /project/discovery/results/.kovar_work --threads 4

python /repo/benchmarks/amr_checks.py sccmec \
  --diagnosis /project/diagnostics/amr_targets \
  --work /project/discovery/results/.kovar_work \
  --assemblies /project/assemblies --gff-dir /project/panaroo/panaroo_gff --threads 4
```

Use `python benchmarks/amr_checks.py COMMAND --help` for stage-specific options.
The KOVAR command needs the installed package or the checkout on `PYTHONPATH`.
The Bash runner supplies the latter automatically.

## What each command checks

**`kovar`** retains the existing mecA/graR/sen/tetM/tetK annotation audit and the
existing mecA–graR/mecA–sen pair tests. It scans all original unitig DNA for
full-length exact matches at target CDSs or within a bounded flank (500 bp by
default). Multiple cluster matches remain explicit; the unrelated MecA adaptor
is not assigned to resistance-associated mecA. Exact matches must agree with
binary presence in the same retained sample. Missing or divergent matches are
not evidence of gene absence. Coding matches enter the direct tests; labelled
nearby candidates require `--include-nearby` and a separate output.

The frozen sample order, patterns, covariance, testing family and source inputs
are checked. Lower pattern ID remains the predictor. KOVAR score P-values are
not likelihood-ratio P-values; beta is independently refitted. The original
four-cell gate is compared with `max(10, ceil(0.02*N))`, without changing the
underlying P-values. Ineligible pairs, identical patterns, separation and failed
fits retain explicit statuses. Physical distance is read from existing outputs;
the diagnostic does not calculate new distances or add candidates to the map.

**`pyseer`** uses the same frozen cohort, covariance, coding pattern pairs and
response orientation as the saved KOVAR diagnosis. It calls the installed
`pyseer.lmm.initialise_lmm` and `fit_lmm_block`, fits one null per response and
scores predictors in blocks. There is no additional MI/cell/AF gate. Existing
KOVAR columns stay intact beside the LMM results. Nominal and original
Bonferroni comparisons remain separate. This is a matched model comparison,
not a reproduction of another cohort or sequencing/annotation pipeline.

**`sccmec`** optionally calls assembly-level carriage with the installed `sccmec`
reference panel, using exact sample-ID assemblies or embedded GFF FASTA. Its
existing classification thresholds are recorded in the identity. Supported
cassette carriage, uncertain carriage, failed typing and panel-negative calls
remain distinct; panel-negative does not prove absence. SCCmec and mecA carriage
are separate fixed-response LMMs against the saved coding graR/sen variants.
Unknown responses are excluded with named reasons and a matching covariance
subset. These variants do not represent whole-gene graR/sen absence. Focal
genome-wide and all-pair Bonferroni thresholds are both reported. This optional
command needs `sccmec` and BLAST; it is not run by default.

## Resume and saved results

Repeat the **same command with the same output**. KOVAR retains annotation
checkpoints, nulls and committed pair fits. Pyseer retains completed response
models. Carriage retains per-isolate calls and completed LMM results. Changed
cohorts, model code, inputs or statistical settings remain incompatible.
Changing only the script organization accepts the previous KOVAR helper's known
fingerprint and keeps an unchanged diagnosis identity byte-for-byte, preserving
downstream pyseer checkpoint references.

`--original-nulls require` permits NumPy/SciPy version differences only when the
null-fitting source matches and cache contents validate. It fails instead of
silently refitting missing nulls. Independent completed beta fits remain reusable.
Library differences can still cause small numerical differences.

If the original SQLite checkpoint is damaged after all target fits were saved:

```bash
python /repo/benchmarks/amr_checks.py kovar \
  --results /project/discovery/results \
  --out /project/diagnostics/amr_targets --report-only
```

This exports committed diagnostic fits and tries the original significant-pair
TSV for historical comparisons. An unavailable historical comparison remains
unavailable, not nonsignificant. It never repairs source SQLite or fits a null.

Main reports remain `amr_targets/pair_comparison.tsv`,
`amr_pyseer_lmm/pyseer_comparison.tsv`, and `sccmec_pyseer/summary.tsv`.
Supporting mapping, sample-order, carriage, agreement, status and progress files
remain alongside them. The source discovery and map are read-only.
