# S. aureus AMR target diagnosis

`benchmarks/diagnose_amr_targets.py` audits mecA (the resistance-associated PBP2a,
not the unrelated adaptor), graR, sen, tetM and tetK. It directly tests all
distinct retained coding-overlap pattern combinations for **mecA–graR** and
**mecA–sen**. This is a separate genomic-association diagnostic, not an MIC
model, gene-level significance test, or automatic addition to the network.
The discovery code, original SQLite database and map are not edited.

```bash
PYTHONPATH=/path/to/ko-variation python /path/to/ko-variation/benchmarks/diagnose_amr_targets.py \
  --results /data/saureus/discovery/results \
  --fasta /data/saureus/discovery/prepared_inputs/saureus_k61.sample_ids.fasta \
  --unitigs /data/pangwes/saureus_k61.unitigs \
  --panaroo /data/panaroo/gene_presence_absence_roary.csv \
  --gff-dir /data/panaroo/panaroo_gff \
  --map-dir /data/saureus/plots/functional_map \
  --out /data/saureus/diagnostics/amr_targets --threads 4
```

Use the original `kmer_gwes_env` (NumPy, SciPy, threadpoolctl and Biopython).
No Pyseer, BWA, eggNOG, R or D calculation is required. Alternatively pass
`--bakta` with `SAMPLE/SAMPLE.gff3` and `.fna` files, including recovered inputs.
The source `.kovar_work` must be available. An adjacent local checkpoint takes
precedence over the saved server path; `--work` specifies another location.

## What is checked

1. Freeze the original cohort/order, pattern IDs, covariance and Bonferroni
   family. Verify the sample-ID FASTA and prepared products; verify the DNA
   unitig file against its saved distance-stage hash when present.
   A full DNA match in a reference must also have binary presence in that same
   retained sample. Such discrepancies are counted per target/unitig and those
   candidates are withheld from direct tests, not silently treated as absent.
2. Read target CDS memberships and products from Panaroo/GFFs across the
   retained cohort. Multiple clusters with the same target name are retained.
   Missing reference samples are reported, not silently treated as gene absence.
3. Scan **all original unitigs**, including MAF-filtered unitigs. Sparse exact
   seeds nominate matches, then the **entire unitig**, in either orientation,
   is compared with its original assembly at the predicted coordinate. This
   handles long unitigs extending beyond a target window. No nearest-name
   assignment or selection of one best cluster occurs.
4. Report coding matches and nearby matches within 500 bp separately. These
   are exact overlaps at target references, not a genome-wide uniqueness test:
   homologous sequences may also occur elsewhere. Existing annotation ambiguity
   and multiple target matches stay explicit. A missing exact match does not
   exclude a divergent biological homologue. Seeds default to 31 bp and require
   at least 61 bp of exact window overlap; this matches the current k=61 pool.
5. Read original filtering/pattern/query decisions. Test retained coding
   patterns directly, bypassing PC/hierarchy screening for this diagnostic.
   MAF-filtered patterns are reported but not silently admitted to the test pool.
   `--include-nearby` adds labelled nearby candidates as a separate sensitivity
   run; choose a separate diagnostic output when changing this setting.
6. Compare the original four-cell gate with `max(10, ceil(0.02*N))`, where N is
   the frozen genomic cohort. P-values and beta do not depend on this gate.
   Diagnostic P-values for ineligible pairs are retained, clearly labelled;
   they are not automatically accepted discoveries. Identical patterns are
   reported as deterministic redundancy without an ordinary pairwise P-value.

The lower pattern ID is always the predictor, as in production. Exact score
P-values use the same ordinary score statistic as production, without SPA;
they are **not likelihood-ratio P-values**. The beta is independently refitted
with the production phylogenetic logistic model, not approximated from the
score. Zero cells, unstable fits and nonconvergence remain explicit statuses.
Original exact P-values/bounds and beta are shown side by side. Unknown physical
distances remain unknown: this script does not run `unitig_distance`, and does
not claim an unmeasured pair is distal.

## Reuse and outputs

Compatible original nulls are loaded read-only; missing nulls are fitted in the
diagnostic output by default (`--original-nulls auto`). If the fitting-source
fingerprint or NumPy/SciPy versions differ, this default disables original
null reuse and explicitly states the refit reason.

To require original fitted nulls despite **library version differences**, add:

```bash
--original-nulls require
```

The fitting-code fingerprint, frozen cohort, covariance, file checksums,
known cache format, response vector and array shapes/finite values must still
match. This is not an override for a different scientific model. Library
differences are logged and retained in `summary.json`. Numerical roundoff may
differ; byte-identical test results across library/BLAS versions are not promised.
The cache stores numerical arrays in NumPy's portable NPY/NPZ format; its file
format versioning is independent of NumPy package versioning (see
[NumPy format documentation](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html)).
Missing original nulls stop with the attempted cache paths; they are never
silently replaced by a new fit in this mode. A recorded original fit failure
remains a failed null rather than triggering refitting. Original-derived
diagnostic cache copies can be reused; fresh diagnostic copies are ignored.

Stop the active diagnostic before upgrading/restarting it. The preceding
released helper's checkpoint is migrated in place when all scientific inputs,
environment and mapping settings match. The same output can switch between
`auto` and `require`. Region/sequence mapping checkpoints stay intact. On
switching to `require`, only completed scores that used fresh diagnostic nulls
are recalculated; their independent beta fits are retained. New, uncompleted
pairs still need their alternative beta fits. The previous diagnostic identity
is saved as `diagnosis_identity.before_null_policy.json`.
Each response has a single worker owner and BLAS uses one thread per worker.
`--threads` controls simultaneous numerical workers; unitig discovery is one
streaming pass with bounded checkpoint batches. Reference preparation,
unitig-scan offsets, exact-match validation and completed pair fits resume by
repeating the same command. The original run must be stopped/complete; do not
modify its input files while diagnosing it. Changed diagnostic inputs/settings
require a separate diagnostic output. A default cap of 10,000 pattern pairs
prevents accidentally launching an unexpectedly large refit job; increase
`--max-pairs` explicitly after checking `target_unitigs.tsv`.

Outputs are:

- `targets.tsv`: source carriers/clusters and coding/nearby unitig coverage.
- `target_unitigs.tsv`: every target/unitig candidate, pattern/MAF status,
  mapping ambiguity, selected-annotation status and map representation.
- `pair_comparison.tsv`: cell counts, both eligibility gates, exact score P,
  independently fitted beta/SE/status, original values/bounds and differences.
- `unitig_pair_evidence.tsv.gz`: links pattern tests back to original unitigs,
  annotation classes and saved distances/statuses.
- `summary.json`, `progress.log`, `diagnosis.sqlite`, `diagnosis_identity.json`
  and diagnostic `null_models/`: interpretation, progress and resume state.

### Recover reports after the original checkpoint becomes unreadable

If all target fits were saved before a historical-comparison error, regenerate
the reports from the existing diagnostic output:

```bash
python benchmarks/diagnose_amr_targets.py \
  --results /path/to/discovery/results \
  --out /path/to/diagnostics/amr_targets --report-only
```

The host runner also accepts `--report-only` after the species name. This mode
reads the saved diagnostic identity, target mapping and committed fits. It does
not remap sequences, load genotype arrays or fit null/alternative models. It
stops if any required diagnostic fit is missing. Keep `diagnosis.sqlite`,
`diagnosis_identity.json` and `target_unitigs.tsv` together.

Reports normally compare against the original checkpoint read-only. If any
targeted SQLite read fails, all partial checkpoint comparisons are discarded;
the helper streams `associations.tsv.gz` instead. Exported significant pairs
retain their recorded values and distances. A pair absent from this export is
labelled **unavailable**, never unqueried or nonsignificant. Missing distances
remain unknown. `old_comparison_source` and `summary.json` record the source
and read errors. Missing or incompatible original provenance disables historical
comparison while preserving the saved diagnostic results. This fallback does
not repair the original checkpoint or establish that its other data are intact.

The original all-pattern Bonferroni family stays fixed. Do not correct only for
these two chosen gene comparisons or select a favourable unitig post hoc and
call it a gene-level result. If the cell-count gate is later changed in a full
search, keep that search separate while reusing compatible preparation/nulls.
