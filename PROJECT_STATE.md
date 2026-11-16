# Current project state

Read this file before consulting historical discussion.

- **Publication:** `codex/publication-clean`. Installed discovery, annotation,
  maps, paper figures, AMR/MIC helpers and portable runners; no research scripts.
- **Development:** `codex/downstream-resume-progress`. Same runtime plus essential
  supplementary benchmarks and temporary investigation. Do development here.
- **Version:** 0.9.3, execution revision `tree-null-1`.

## Numerical decisions

- Add tree working-covariance algebra for intercept-only nulls using float64,
  exact sigmoid and log-tau optimizer tolerance `1e-8`.
- Keep outer PQL tolerance `1e-7`, 100 iterations and damping 0.8. Dense fallback
  handles failed tree fitting. Convergence alone is not an accuracy certificate.
- Dense remains the default pending full larger validation. Opt in with
  `--null-backend tree` and install `.[tree]`.
- Tree caches store O(n) response vectors, sharing tree/K; exact tree score
  variances avoid per-response spectra. Dense caches retain adaptive spectral
  bounds. Beta refits remain full dense fits.
- Validation probability agreement criterion: `5e-7`. Preserve other accuracy
  checks and nominal/Bonferroni decision comparisons.
- PC40/10% and representative hierarchy are candidate heuristics. Keep the
  full unique-pattern Bonferroni family at alpha 0.05.
- Keep significant close pairs; skip beta below the configured distance cutoff.
- Resume within the same numerical release/backend. New solver runs need a new
  output; do not migrate old statistical decisions automatically.

## Evidence and remaining work

- Earlier 1,000-pattern comparison: median paired null-fit speedup about 26.5x,
  compact cache reduction about 96%; nine tree convergence exceptions.
- Targeted tighter-optimizer check: 10/11 pass the old strict criteria; remaining
  probability difference 1.066e-7 was accepted as negligible on tested probes.
- **Pending:** full 1,000-pattern run with tighter optimizer and practical
  probability criterion. Baseline convergence changed between executions;
  its exact cause remains unresolved. Do not claim universal agreement.
- Supplementary priorities: null accuracy/runtime/storage; PC recovery/budget;
  exact/bounded/hierarchy recovery. Beta comparison useful; AMR checks conditional.
- Keep data, logs and figure tables outside the source tree with manifests.
  Do not retain troubleshooting transcripts or duplicate legacy runners here.
