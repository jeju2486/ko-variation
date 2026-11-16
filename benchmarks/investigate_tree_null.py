#!/usr/bin/env python3
"""Investigate saved tree-null exceptions; retain the production fitting loop."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing
import os
from pathlib import Path
import time
from unittest.mock import patch

try:  # Support both direct execution and imports by focused checks.
    from benchmarks import tree_null_benchmark as b
except ImportError:
    import tree_null_benchmark as b

np, glmm = b.np, b.glmm
VARIANTS = (
    ('dense_100', False, 100, .8, 1e-6),
    ('tree_100', True, 100, .8, 1e-6),
    ('tree_300', True, 300, .8, 1e-6),
    ('tree_300_tight', True, 300, .8, 1e-8),
    ('tree_300_damped', True, 300, .5, 1e-6),
)


def targets(payloads, identity, probability_limit):
    # Selection only: keep failures remaining after the earlier sensitivity sweep.
    limits = dict(identity, probability_tolerance=probability_limit)
    result = set()
    for payload in payloads:
        rows = {r['method']: r for r in payload['rows']}
        c = dict(payload['comparison'], dense_status=rows['dense']['status'])
        failures = b.accuracy_failures(c, limits)
        if failures and failures != ['BOTH_MONOMORPHIC']:
            result.add(c['index'])
    return sorted(result)


def traced_fit(index, y, variant):
    name, tree, iterations, damping, xatol = variant
    prevalence = (y.sum() + .5) / (len(y) + 1.)
    eta = np.full(len(y), math.log(prevalence / (1 - prevalence)))
    previous_tau, consecutive, trace = 0., 0, []
    profile = b.profile_tree if tree else glmm._profile_tau
    minimize = b.optimize.minimize_scalar

    def optimizer(*args, **kwargs):
        kwargs['options'] = dict(kwargs.get('options', {}), xatol=xatol)
        return minimize(*args, **kwargs)

    def observe(*args, **kwargs):
        nonlocal eta, previous_tau, consecutive
        tau, boundary, solution = profile(*args, **kwargs)
        objective, beta, random_effect, _, _ = solution
        delta = args[2] @ beta + random_effect - eta
        gap = float(np.max(np.abs(delta)))
        tau_change = abs(math.log1p(tau) - math.log1p(previous_tau))
        passed = gap <= 1e-7 and tau_change <= math.sqrt(1e-7)
        consecutive = consecutive + 1 if passed else 0
        mu = np.clip(b.special.expit(eta), 1e-12, 1 - 1e-12)
        trace.append(dict(iteration=len(trace) + 1, tau=float(tau),
            objective=float(objective), max_eta_change=gap, tau_change=tau_change,
            eta_pass=gap <= 1e-7, tau_pass=tau_change <= math.sqrt(1e-7),
            consecutive_passes=consecutive, tau_boundary=bool(boundary),
            weights_clipped=int(np.count_nonzero(mu * (1 - mu) < 1e-8))))
        eta += damping * np.clip(delta, -5., 5.)
        previous_tau = tau
        return tau, boundary, solution

    weighted = b.weighted_tree if tree else glmm._weighted_eigen
    start = time.perf_counter()
    with patch.object(glmm, '_weighted_eigen', weighted), \
         patch.object(glmm, '_profile_tau', observe), \
         patch.object(b.optimize, 'minimize_scalar', optimizer):
        fit, cache = glmm.fit_null_glmm(index, y, b.STATE['K'],
            kinship_eigensystem=b.STATE['eigen'], max_iter=iterations, damping=damping)
    row = dict(variant=name, status=fit.status, iterations=fit.iterations,
        seconds=time.perf_counter() - start, tau=b.finite(fit.tau),
        intercept=b.finite(fit.beta[0]), objective=b.finite(fit.objective),
        max_eta_change=b.finite(fit.max_eta_change),
        weights_clipped=fit.n_weights_clipped, tau_boundary=fit.tau_boundary,
        message=fit.message)
    if trace:
        row.update(final_tau_change=trace[-1]['tau_change'],
            final_consecutive_passes=trace[-1]['consecutive_passes'])
    return fit, cache, row, trace


def scores(X, cache, tree):
    if tree:
        return b.tree_scores(X, cache, b.STATE['tree'])
    results = glmm.score_predictor_block(X, cache, compute_spa_adjustment=False)
    return np.array([r.score_u for r in results]), np.array([r.score_variance for r in results])


def compare(reference, fit, reference_score, trial_score, identity):
    c = dict(status_match=reference.status == fit.status,
        jointly_successful=reference.status == fit.status == 'OK', dense_status=reference.status)
    if c['jointly_successful']:
        c.update(max_probability_difference=float(np.max(np.abs(reference.fitted_probability - fit.fitted_probability))),
            intercept_abs_difference=abs(float(reference.beta[0] - fit.beta[0])),
            tau_relative_difference=abs(reference.tau - fit.tau) / max(1., abs(reference.tau)),
            objective_relative_difference=abs(reference.objective - fit.objective) / max(1., abs(reference.objective)))
        u0, v0 = reference_score; u1, v1 = trial_score
        valid0 = np.isfinite(u0) & np.isfinite(v0) & (v0 > 1e-12)
        valid1 = np.isfinite(u1) & np.isfinite(v1) & (v1 > 1e-12)
        both = valid0 & valid1
        finite = np.isfinite(v0) & np.isfinite(v1)
        c.update(variance_status_mismatches=int(np.count_nonzero(valid0 != valid1)),
            jointly_valid_probes=int(both.sum()),
            max_variance_relative_difference=float(np.max(np.abs(v0[finite] - v1[finite]) /
                np.maximum(np.abs(v0[finite]), 1e-12))) if finite.any() else None)
        if both.any():
            t0, t1 = u0[both] ** 2 / v0[both], u1[both] ** 2 / v1[both]
            c.update(max_log10p_difference=float(np.max(np.abs(2 * b.special.log_ndtr(-np.sqrt(t0)) -
                2 * b.special.log_ndtr(-np.sqrt(t1)))) / math.log(10)),
                max_statistic_difference=float(np.max(np.abs(t0 - t1))))
            for label, critical in (('nominal', identity['nominal_critical']), ('bonferroni', identity['critical'])):
                c[label + '_decision_mismatches'] = int(np.count_nonzero((t0 >= critical) != (t1 >= critical)))
        else:
            c['no_valid_probes'] = True
    failures = b.accuracy_failures(c, identity)
    if c.get('no_valid_probes'): failures.append('NO_VALID_PROBES')
    return dict(c, within_original_tolerances=not failures, failure_reasons=';'.join(failures))


def investigate(index):
    identity = b.STATE['identity']
    y = np.array(b.STATE['patterns'][index], dtype=float)
    ids = b.probe_indices(identity, index, len(b.STATE['patterns']))
    X = np.ascontiguousarray(b.STATE['patterns'][ids].T, dtype=float)
    X[:, 0] = y
    if X.shape[1] > 1: X[:, 1] = 1 - y
    rows, traces = [], []
    reference = reference_score = None
    for variant in VARIANTS:
        fit, cache, row, trace = traced_fit(index, y, variant)
        score = scores(X, cache, variant[1]) if fit.status == 'OK' else None
        if reference is None:
            reference, reference_score = fit, score
        row.update(index=index, response=identity['responses'][index], n=len(y), n_present=int(y.sum()))
        row.update(compare(reference, fit, reference_score, score, identity))
        rows.append(row)
        traces.extend(dict(index=index, response=row['response'], variant=variant[0], **r) for r in trace)
    return dict(index=index, rows=rows, trace=traces)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--selection-probability-tolerance', type=float, default=5e-7,
        help='Select remaining exceptions using this sensitivity threshold; acceptance limits stay unchanged')
    args = p.parse_args(argv)
    args.benchmark = args.benchmark.resolve(); args.out = args.out.resolve()
    if args.out == args.benchmark or args.out in args.benchmark.parents or args.benchmark in args.out.parents:
        p.error('--out must be separate from the original benchmark')
    if args.threads < 1 or not math.isfinite(args.selection_probability_tolerance) or args.selection_probability_tolerance <= 0:
        p.error('Threads and selection tolerance must be positive')
    identity = json.loads((args.benchmark / 'benchmark_identity.json').read_text())
    products = json.loads((args.benchmark / 'products.json').read_text())
    for name in ('selected.npy', 'K.npy', 'tree.npz'):
        if b.digest(args.benchmark / name) != products[name]:
            raise ValueError('Frozen input changed: ' + name)
    if b.digest(Path(glmm.__file__)) != identity['sources'][Path(glmm.__file__).name]:
        raise ValueError('Production fitting source changed since benchmark; use the matching checkout')
    payloads = b.checkpoint(args.benchmark / 'checkpoint.jsonl', repair=False)
    indices = targets(payloads, identity, args.selection_probability_tolerance)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = dict(benchmark_identity=b.digest(args.benchmark / 'benchmark_identity.json'),
        products=products, indices=indices, variants=VARIANTS,
        selection_probability_tolerance=args.selection_probability_tolerance,
        numpy=np.__version__, scipy=b.scipy.__version__, engine=identity['engine'],
        sources={path.name: b.digest(path) for path in (Path(__file__), Path(b.__file__), Path(glmm.__file__))})
    manifest = json.loads(json.dumps(manifest))
    path = args.out / 'identity.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Investigation settings changed; choose a new --out')
    b.atomic_json(path, manifest)
    saved = b.checkpoint(args.out / 'checkpoint.jsonl')
    done = {r['index'] for r in saved}
    pending = [index for index in indices if index not in done]
    print(f'[investigation] {len(indices)} responses; {len(pending)} pending; five fits per response', flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=args.threads, mp_context=multiprocessing.get_context('spawn'),
                initializer=b.initialize, initargs=(str(args.benchmark), identity['engine'])) as pool:
            with (args.out / 'checkpoint.jsonl').open('ab') as stream:
                for result in b.bounded_map(pool, investigate, pending, args.threads, ordered=False):
                    stream.write((json.dumps(result, allow_nan=False) + '\n').encode())
                    stream.flush(); os.fsync(stream.fileno()); saved.append(result)
                    print(f'[investigation] saved {len(saved)}/{len(indices)} responses', flush=True)
    b.table(args.out / 'fits.tsv', [row for r in saved for row in r['rows']])
    b.table(args.out / 'trace.tsv', [row for r in saved for row in r['trace']])
    summary = {}
    for variant in VARIANTS:
        rows = [row for r in saved for row in r['rows'] if row['variant'] == variant[0]]
        summary[variant[0]] = dict(responses=len(rows), converged=sum(r['status'] == 'OK' for r in rows),
            within_original_tolerances=sum(r['within_original_tolerances'] for r in rows))
    b.atomic_json(args.out / 'summary.json', dict(variants=summary,
        note='Targeted exception investigation only; no full-pool accuracy or timing claim. '
             'Acceptance uses the original benchmark tolerances. Source results and production code are unchanged.'))
    print(f'[investigation] complete: {args.out}', flush=True)


if __name__ == '__main__':
    main()
