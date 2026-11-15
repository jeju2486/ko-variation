# KOVAR and pyseer checks

Use **one Python script**, `benchmarks/amr_checks.py`, and optionally the single
Bash runner, `benchmarks/run_amr_checks.sh`. The previous
`diagnose_amr_targets.py`, `diagnose_amr_pyseer.py` and `sccmec_pyseer.py` have been
consolidated here. Numerical methods, target definitions, response orientation
and output columns are unchanged. Production discovery and plotting are unchanged.

## Standalone run from upstream files

The default now starts from PAN-GWES outputs and **does not read `kovar_output`**.
From inside the checkout:

```bash
# Run KOVAR target diagnosis, then the matched pyseer comparison.
KOVAR_PYTHON=/path/to/kmer_gwes_env/bin/python \
PYSEER_PYTHON=/path/to/pyseer_env/bin/python \
ROOT=/project THREADS=4 bash benchmarks/run_amr_checks.sh saureus
```

The two Python paths may be the same if one environment contains both tools.
KOVAR diagnosis requires KOVAR, NumPy, SciPy, Biopython and threadpoolctl.
The pyseer commands require pyseer, pandas, NumPy and SciPy; **they do not import
KOVAR or require Biopython**. Set `REPO` if using a differently named checkout.
`ROOT` is the project containing the upstream data, not the checkout. The runner
can infer it when run from the project root or its checkout. No scheduler is required.

With `ROOT=/project`, the standalone defaults are:

| Input | Default |
|---|---|
| Binary FASTA (`C=1`, `A=0`) | `pangwes_saureus_output/saureus/saureus_k61.fasta` |
| Unitig DNA with zero-based IDs | `pangwes_saureus_output/saureus/saureus_k61.unitigs` |
| Phylogenetic tree | `iqtree_saureus_output/saureus/saureus_core.treefile`, then the same filename directly under `iqtree_saureus_output` |
| Per-isolate annotations/genomes | `panaroo_saureus_output/panaroo_gff/SAMPLE.gff` with embedded FASTA, or explicit Bakta GFF/FNA inputs |
| Optional Panaroo membership CSV | `panaroo_saureus_output/saureus/gene_presence_absence_roary.csv`, then `gene_presence_absence.csv` |

**Two sequence files alone cannot supply gene names or phylogenetic adjustment.**
The GFFs supply CDS annotations and reference DNA; the tree supplies covariance.
Panaroo membership is optional: named/product-annotated GFF CDSs work directly.
If CDSs lack informative names/products and there is no matching membership CSV,
the script cannot assign them to a target gene by name alone.

Raw gfa1_parser FASTA often repeats `contig00001` rather than sample IDs. The
runner uses the original `assemblies_list.txt`, `saureus_k61.gfa1` and
`saureus_k61.json` in the PAN-GWES folder to recover row names. It verifies the
assembly order against Cuttlefish metadata, GFA reference order and FASTA record
count. It never guesses a sample from a repeated header. Alternatively set
`SAMPLE_IDS=/path/to/ids.txt` (one sample ID per FASTA record, in the original row
order), or pass a FASTA whose headers already contain unique sample IDs.

The standalone sequence is:

1. Recover sample IDs and save this run's own sample mapping and FASTA.
2. Match samples to the tree; prune excluded/unmatched tips and midpoint-root a
   local copy. `TREE_ROOTING=as-is` preserves supplied rooting; missing FASTA
   samples cause an error unless `TREE_MISSING_SAMPLES=drop` is explicit.
   `EXCLUDE_SAMPLES` supplies a named exclusion list when needed.
3. Prepare exact unique binary patterns and covariance using KOVAR's existing
   disk-backed preparation. MAF defaults to 0.05. The original unitig IDs remain
   unchanged. The default Bonferroni family is all retained unique-pattern pairs,
   **not only the selected target comparisons**.
4. Assign unitigs to the existing target annotations, then directly calculate
   their KOVAR scores and independently refitted beta. Nulls are fitted only for
   response patterns actually needed here, then checkpointed and reused.
5. Run the matched pyseer tests on those same patterns, samples and covariance.

This does not run the genome-wide hierarchy/PC search, fit every locus null,
calculate new physical distances, or require a previous covariation map.

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

# Run only the standalone KOVAR checks.
bash ko-variation/benchmarks/run_amr_checks.sh saureus kovar
```

Stages: `both` (default), `kovar`, `pyseer`, `sccmec`, `report`, or `all`
(both plus optional carriage). Additional CLI flags can follow a **single**
stage, for example `saureus kovar --report-only`.

Set `RESULTS` **explicitly** only to retain the older completed-discovery
comparison mode. That mode can still reuse compatible original nulls with
`ORIGINAL_NULLS=require`. Standalone mode has no original nulls and rejects that
option; it reuses its own saved nulls on rerun. Unset an exported `RESULTS` to
start a standalone run.

Overrides, when needed:

| Variable | Meaning |
|---|---|
| `RESULTS` | Optional older discovery directory; unset for standalone |
| `SOURCE_WORK` | Optional relocated original `.kovar_work`, for existing-run mode |
| `DIAGNOSIS` | Existing target diagnosis, including an older run elsewhere |
| `FASTA`, `UNITIGS`, `TREE`, `PANAROO` | Upstream inputs; Panaroo CSV is optional |
| `SAMPLE_IDS` | Explicit row IDs when FASTA headers are not sample names |
| `ASSEMBLIES_LIST`, `GFA`, `CUTTLEFISH_JSON` | Original metadata for automatic row-ID recovery |
| `EXCLUDE_SAMPLES`, `TREE_ROOTING`, `TREE_MISSING_SAMPLES` | Explicit cohort/rooting choices |
| `MIN_MAF`, `ALPHA`, `MEMORY_GB` | Preparation/filter/resource overrides |
| `GFF_DIR` or `BAKTA` | Existing per-sample annotations; no new annotation required |
| `MAP_DIR` | Optional old map, only for historical annotation comparison |
| `OUT` | Standalone output parent; default `$ROOT/amr_checks_output/saureus` |
| `PYSEER_OUT`, `SCCMEC_OUT` | Existing comparison directories to resume |
| `ASSEMBLIES`, `SCCMEC_EXEC` | Assembly source and optional classifier executable |

New standalone outputs are `amr_checks_output/saureus/amr_targets` and
`amr_checks_output/saureus/amr_pyseer_lmm`. This run's prepared arrays/tree/FASTA
are inside `amr_targets/.inputs`; nulls and pair fits are also local to this run.
The optional carriage output is the sibling `sccmec_pyseer`. Existing-run mode
keeps the earlier sibling `diagnostics` outputs. Repeat the same command and
output to resume; changing the input cohort or model requires a separate output.

## Direct Python commands

For arbitrary directory layouts, run the same file with explicit inputs:

```bash
python /repo/benchmarks/amr_checks.py kovar \
  --fasta /project/pangwes/saureus_k61.sample_ids.fasta \
  --unitigs /project/pangwes/saureus_k61.unitigs \
  --tree /project/iqtree/saureus_core.treefile \
  --gff-dir /project/panaroo/panaroo_gff \
  --out /project/amr_checks_output/saureus/amr_targets --threads 4

python /repo/benchmarks/amr_checks.py pyseer \
  --diagnosis /project/amr_checks_output/saureus/amr_targets --threads 4

python /repo/benchmarks/amr_checks.py sccmec \
  --diagnosis /project/amr_checks_output/saureus/amr_targets \
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
