# Optional tree-null backend

Install with `python -m pip install '.[tree]'` and add `--null-backend tree` to
the normal discovery command. Dense fitting remains the default while the
larger real-data validation of the tighter optimizer is pending.

The same outer logistic PQL loop uses tree contrasts, covariance products and
solves for `S = diag(1/W) + tau*K`. Rooting, normalization, retained sample order,
float64 arithmetic, sigmoid and score definition are unchanged. The tree-null
inner log-tau tolerance is `1e-8`; dense reference and beta fits retain `1e-6`.
Outer tolerance is `1e-7`, damping 0.8 and the iteration cap 100.

Failed tree nulls are refitted with dense algebra and recorded with a
`tree_fallback` status in the null manifest and aggregate runtime counters.
Successful tree caches save response vectors and small fixed-effect matrices,
with the tree/K shared across patterns. They omit per-pattern eigenvectors.
The scorer obtains exact variances through tree contrasts. Dense caches retain
the spectral interval path. This certifies neither candidate-screen recovery
nor the numerical error of an already-converged null.

Per-pattern storage is O(n) for intercept-only tree nulls; shared K and its
eigensystem are still O(n squared). Peak worker memory and total pipeline time
must be measured separately from serialized cache size and isolated fit time.
Selected distal beta fits still use the full alternative model.

Use a new output when changing backend or upgrading from earlier fitting code.
Repeating the same current-version command with `--resume` reuses committed
nulls, decisions, distances and beta fits. Existing result exports remain
readable by plotting and reporting tools. Current checkpoint compatibility
checks remain in place; old automatic release-hash migration was retired.

The proposed validation probability tolerance is `5e-7`, alongside the existing
intercept/objective/tau/variance checks and significance decisions. The large
tighter-optimizer benchmark remains pending. See `PROJECT_STATE.md` for the
current evidence and the development branch for reproducibility programs.
