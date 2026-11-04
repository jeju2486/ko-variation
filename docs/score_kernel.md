# Exact cached-null scoring

`score_predictor_block` reuses the fitted response-null cache. This optimization
changes neither null fitting nor the statistical test, thresholds, result fields,
pair selection, or SPA correction. It requires no new command-line option.

Let `S = W^-1 + tau*K`, `M = C.T @ S^-1 @ C`, and
`P = S^-1 - S^-1 @ C @ M^-1 @ C.T @ S^-1`. The score remains
`U = X.T @ (y - mu)` and its variance remains `diag(X.T @ P @ X)`.

The scorer now computes `R = X - C @ M^-1 @ (S^-1 @ C).T @ X`, using the
cached `S^-1 @ C`. If `Q` and `lambda` are the eigensystem of
`sqrt(W) @ K @ sqrt(W)`, the same variance is
`sum((Q.T @ sqrt(W) @ R)**2 / (1 + tau*lambda), axis=0)`.
This removes the inverse eigen-transform. When `tau == 0`, the variance is
simply `sum((sqrt(W) @ R)**2, axis=0)`, without either eigen-transform.
P-values are evaluated in one vectorized chi-square survival-function call per
block. SPA predictor residualization continues to use the original working
weights. Ordinary floating-point differences are expected.

Focused regression tests compare the kernel with an independent dense-precision
calculation, including heterogeneous weights, multiple nuisance covariates,
zero and large `tau`, invalid/singular predictors, and SPA-adjusted predictors:

```sh
python -m unittest discover -s tests -p test_glmm.py -q
python -m unittest discover -s tests -p test_scan.py -q
```

A local synthetic benchmark used 612 samples, 8,192 binary predictors, blocks of
256, one BLAS thread, and median times from three repeats. At `tau=2.5`, scoring
fell from 1.119 to 0.325 seconds without SPA predictor adjustment (3.44x), and
from 1.164 to 0.411 seconds with adjustment (2.83x). At `tau=0`, speedups were
5.29x and 4.21x, respectively. All score fields agreed at `rtol=1e-9`,
`atol=1e-13`. These are synthetic scoring timings, not ARC collection timings;
they exclude null preparation, null fitting, SPA tail evaluation, and I/O.
Null fitting is an amortized cost only when the corresponding fitted cache is
actually reused. End-to-end speedup depends on the remaining workload.
