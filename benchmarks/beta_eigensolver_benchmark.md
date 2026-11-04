# Full beta-refit eigensolver benchmark

This compares the current NumPy eigensolver with SciPy `eigh(driver="evd")` and
`eigh(driver="evr")` in complete PQL beta refits. It changes only the eigensolver
call in an isolated worker process and restores that call after each fit. It
does not edit the fitter, main runner or saved models.

The full double-precision spectrum and all eigenvectors are computed. The
model, cold initialization, variance-component profiling, convergence tolerance,
iteration limit and production beta acceptance checks remain unchanged. SciPy
uses a fresh Fortran-order matrix with finite checking enabled; its copying and
solver overhead are included in measured fit time. The immutable input K is
never overwritten. No score-based approximation is used.

NumPy already uses the divide-and-conquer LAPACK `syevd` family. SciPy EVD is a
comparison of the wrapper/backend path, not a new statistical model. SciPy EVR
uses a different LAPACK driver. Hardware, matrix size and linked numerical
libraries determine which is faster. Numerical rounding can affect iterative
fits, so coefficient agreement and convergence are measured rather than assumed.

Primary references:
- https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html

## Run

The only benchmark input is the existing `selection.json` from the earlier beta
benchmark. Its sample and prepared cohort are reused; no database or saved null
model is opened. The script needs the source work directory's `prepared.json`,
`patterns.bin`, `K.npy`, `K_values.npy` and `K_vectors.npy`. It checks their
identity once. The main search database can be large or on a different node;
this benchmark does not access it.

```bash
python benchmarks/beta_eigensolver_benchmark.py \
  --selection /path/to/beta_benchmark/selection.json \
  --out /path/to/beta_eigensolver_benchmark \
  --threads 4 --repeats 3
```

No companion benchmark Python script is required. The new file can also run
standalone with installed KOVAR importable. If the numerical work directory was
relocated, supply `--work /new/path/.kovar_work`; its recorded cohort and product
hashes must still match. It is unnecessary to copy the SQLite database.

For your previous 200-pair sample and three repeats, this schedules 600
pair-repeat jobs, each fitting all three solvers: 1,800 complete fits. The NumPy
reference is recalculated on the same allocation, rather than using historical
wall times from the old run. Method order and job order are randomized with a
fixed seed. Each worker uses one native numerical CPU. The tiny backend warm-up
is untimed; matrix copying, patch setup and complete fitting are timed.

## Outputs

- `benchmark_identity.json`: sample, numerical arrays, code hashes and settings.
- `checkpoint.jsonl`: completed pair-repeat jobs; restart by repeating the command.
- `eigensolver_results.tsv`: beta, standard error, tau, iterations, status,
  complete fit time, decomposition/profiling/copy timings and numerical differences.
- `eigensolver_summary.json`: per-solver speedup, convergence and numerical agreement.
- `eigensolver_benchmark.png` and `.svg`: runtime distributions and signed beta
  differences against the NumPy reference, using black/white and editable SVG text.

The runtime plot summarizes the median over repeats within each pair. The
coefficient-difference plots show signed differences, rather than concealing
small differences beneath an identity line. Matplotlib is optional if
`--no-plots` is used; rerunning without that flag generates plots from saved jobs.

Default comparison tolerances are absolute beta difference <= 1e-6 and maximum
absolute fitted-probability difference <= 1e-7. These are reporting thresholds,
not relaxed fitting tolerances. The summary also reports standard-error and tau
relative differences, all failures/status changes, and NumPy's coefficient span
over repeated fits. Eigenvector signs or bases in degenerate eigenspaces need
not match, so downstream fit quantities are compared instead.

Speedup uses successful matched fits: first calculate each pair's median
repeat-specific speedup, then the median across pairs. Failure counts must be
read alongside speed; failing early is not an optimization. Repeats are timing
replicates of the same pairs, not independent biological observations. No
production driver is automatically selected.

## Restart

Rerun the same command to skip complete jobs. Only an incomplete final checkpoint
line is discarded; interrupted jobs are repeated. Threads may change on resume.
Changing repeats, seed, code, selected pairs or numerical libraries requires a
new output. Reporting tolerances can change without refitting. Do not run two
processes writing to the same benchmark output. Invocation wall time refers to
the latest invocation; all per-fit times remain in the checkpoint after resume.

ARC-specific submission is supplied separately and stays outside GitHub.
