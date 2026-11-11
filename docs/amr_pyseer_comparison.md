# Matched AMR target comparison with pyseer LMM

This diagnostic asks whether pyseer's linear mixed model detects associations
among the **same binary vectors** used in the completed KOVAR AMR diagnosis.
It does not recreate GOLD-GWAS's 806-isolate SCCmec-carriage analysis.

To test an assembly-level SCCmec carriage response against graR/sen predictors,
use the [SCCmec fixed-response workflow](sccmec_pyseer.md).

## Run

Use a Python environment containing pyseer, NumPy, pandas, SciPy and pyseer's
dependencies (Python 3.8 or newer). KOVAR does not need to be installed in that
environment. Run the program directly from a source checkout:

```bash
python /path/to/ko-variation/benchmarks/diagnose_amr_pyseer.py \
  --diagnosis /project/kovar_output/saureus/diagnostics/amr_targets \
  --work /project/kovar_output/saureus/discovery/results/.kovar_work \
  --out /project/kovar_output/saureus/diagnostics/amr_pyseer_lmm \
  --threads 4
```

`--work` can be omitted when the original products remain in the standard
species directory or at their recorded path. With copied runs or externally
located work files, set it explicitly. It needs only `patterns.bin` and `K.npy`;
the original `work.sqlite` is **not opened**, so a damaged decisions database
does not block this comparison. Source files remain unchanged. The completed
diagnostic must contain `diagnosis_identity.json` and `pair_comparison.tsv`.

## What stays matched

- The frozen sample cohort and order, with no additional exclusions or dropping.
- The original binary pattern IDs and presence/absence vectors, including their
  original orientation: `predictor_pattern` predicts `response_pattern`.
- The complete frozen covariance matrix, not 40 tree PCs. The script passes it
  to pyseer, which uses its usual mean-diagonal normalization; the factor is
  recorded in the comparison identity. No new tree is calculated.
- Intercept-only fixed covariates, as in the original KOVAR null model.
- All saved diagnostic pairs, including those below either four-cell threshold.
  Every reconstructed four-cell table must agree with the saved diagnostic.
- The original genome-wide testing family and alpha, rather than a new
  Bonferroni correction over just the selected diagnostic pairs.

The helper calls the installed `pyseer.lmm.initialise_lmm` and
`pyseer.lmm.fit_lmm_block` functions. This is the actual pyseer FaST-LMM model,
not an independent linear-regression approximation. Calling the fitting API
directly avoids another marginal-association or allele-frequency prefilter.
See [pyseer LMM source](https://pyseer.readthedocs.io/en/master/_modules/pyseer/lmm.html).
The installed pyseer version and LMM source fingerprints are recorded.

## Fitting, memory and restart

Pyseer needs its **own linear-model null** per response pattern; a fitted KOVAR
logistic null cannot replace it. Each response null is fitted once and all its
predictors are scored in blocks. Completed response results are saved atomically
under `responses/`. Repeating the same command skips those completed responses;
an interrupted response restarts while earlier responses remain saved.

`--threads` sets independent response processes. Each process uses one BLAS
thread. Only the target vectors are extracted from the memory-mapped complete
pattern pool. Memory for numerical fitting scales approximately with the
number of workers times the square of the cohort size, so use modest worker
counts. Do not start two coordinators on the same comparison output.

Changing the vectors, source report, or pyseer numerical environment requires
a separate output directory. No KOVAR null, score or beta is recalculated.

## Outputs and interpretation

- `pyseer_comparison.tsv`: original diagnostic columns, followed by `lmm_p`,
  `lmm_beta`, `lmm_se`, `lmm_h2`, fit status and significance flags.
- `summary.tsv`: per-target pair counts, lowest LMM P-values, nominal discoveries,
  Bonferroni discoveries and discoveries meeting each four-cell eligibility gate.
- `summary.json`, `run_identity.json`, `progress.log`, `responses/`: interpretation,
  environment, provenance and restart records.
- `selected_patterns.npy`, `sample_order.tsv`, `covariance.tsv`: the small input
  subset and exact sample labels supplied to pyseer.

`lmm_nominal` means P < 0.05 and is a descriptive diagnostic, not a genome-wide
discovery criterion. `lmm_bonferroni` applies the original full-family cutoff;
the eligibility flags additionally apply the original or relaxed cell gate.
Invalid fits have blank significance flags, not evidence of nonsignificance.
Shared identical patterns have no ordinary association P-value.

The original `p_exact`, `beta` and status columns still describe KOVAR. The
`lmm_*` columns describe pyseer. Their coefficients use different scales.
An LMM-only association identifies a model disagreement; it does not establish
which model is calibrated or prove biological epistasis. Gene-level and
SCCmec-carriage definitions remain separate follow-up analyses.
