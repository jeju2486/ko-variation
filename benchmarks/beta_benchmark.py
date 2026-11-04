#!/usr/bin/env python3
"""Read-only, resumable beta benchmark on an existing discovery checkpoint.

Compare the production cold PQL refit, the same fitter initialized from a saved
response null, and U/V from the exact null score. Never refit/save a missing null
or write to the production database. Run from a checkout with KOVAR importable.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, OrderedDict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import csv
import inspect
import json
import multiprocessing
from pathlib import Path
import platform
import sqlite3
import sys
import time

import numpy as np
import scipy
from scipy.special import logit
from threadpoolctl import threadpool_info, threadpool_limits

from ko_variation import glmm
from ko_variation import discovery_nulls
from ko_variation.discovery_store import atomic_json, digest, open_patterns

METHODS = ('cold_refit', 'warm_refit', 'score_approx')
STATE = {}


def readonly(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True, timeout=30)


def warm_fitter():
    """Change only two initial values in an isolated copy of the live fitter.

    AST editing avoids maintaining a second numerical implementation or changing
    glmm.py (whose hash identifies the running production null caches).
    Fail explicitly if the production initialization has changed.
    """
    tree = ast.parse(inspect.getsource(glmm.fit_logistic_mixed))
    function = tree.body[0]
    function.name = '_benchmark_warm_fit'
    function.args.kwonlyargs += [ast.arg(arg='initial_eta'), ast.arg(arg='initial_tau')]
    function.args.kw_defaults += [None, None]
    replacements = {'eta': 'eta = np.array(initial_eta, dtype=np.float64, copy=True)',
                    'tau_previous': 'tau_previous = float(initial_tau)'}
    found = set()
    for index, node in enumerate(function.body):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in replacements:
                found.add(target.id)
                function.body[index] = ast.parse(replacements[target.id]).body[0]
    if found != set(replacements):
        raise RuntimeError('Fitter initialization changed; review the benchmark warm-start adapter')
    namespace = dict(vars(glmm))
    exec(compile(ast.fix_missing_locations(tree), '<beta-benchmark-warm-start>', 'exec'), namespace)
    return namespace[function.name]


def initialize(work, info, null_sources):
    threadpool_limits(limits=1)
    work = Path(work)
    STATE.update(work=work, info=info, packed=open_patterns(work, info),
                 null_sources=null_sources, cache=OrderedDict(),
                 K=np.load(work/'K.npy', mmap_mode='r'),
                 eigen=glmm.KinshipEigensystem(np.load(work/'K_values.npy', mmap_mode='r'),
                    np.load(work/'K_vectors.npy', mmap_mode='r'), info['shared_covariance']),
                 warm=warm_fitter())
    alias = work/'null_canonical.npy'
    STATE['aliases'] = np.load(alias, mmap_mode='r') if alias.exists() else None


def bits(pattern):
    return np.unpackbits(STATE['packed'][pattern], bitorder='little')[:STATE['info']['n']].astype(float)


def saved_null(response):
    """Bounded worker-local read cache; complements retain original orientation."""
    started = time.perf_counter()
    aliases = STATE['aliases']
    canonical = int(aliases[response]) if aliases is not None else response
    cache = STATE['cache']
    if canonical not in cache:
        supplied = STATE['null_sources'].get(str(canonical))
        source = Path(supplied) if supplied else None
        value, status = None, 'CACHED_NULL_MISSING'
        if source is not None and source.exists():
            record = json.loads(source.read_text())
            status = record['status']
            if status == 'OK':
                def verify(path, expected):
                    if digest(path) != expected:
                        raise ValueError(f'Corrupted cached null: {path}')
                value = discovery_nulls.load(record, source, STATE['eigen'], verify)
                if not np.array_equal(value['y'], bits(canonical)):
                    raise ValueError('Saved null response does not match the pattern pool')
        cache[canonical] = (value, status)
        # Eight n-by-n decompositions at most, independent of pool size.
        while len(cache) > 8:
            cache.popitem(last=False)
    cache.move_to_end(canonical)
    value, status = cache[canonical]
    if value is not None and canonical != response:
        value = dict(value, y=bits(response), mu=1-value['mu'])
    return value, status, time.perf_counter()-started


def fitted_result(fit, cache):
    status = fit.status
    beta = variance = None
    if fit.converged and status == 'OK':
        beta, variance = float(fit.beta[1]), float(cache['M_inv'][1, 1])
        if fit.n_weights_clipped or not np.isfinite([beta, variance]).all() or variance <= 0:
            status, beta = 'UNSTABLE_EFFECT', None
    return dict(status=status, beta=beta, iterations=fit.iterations,
                tau=float(fit.tau) if np.isfinite(fit.tau) else None)


def compare(job):
    u, v, repeat, seed = job
    started = time.perf_counter()
    x, y = bits(u), bits(v)
    overlap = int(x@y)
    cells = [int(len(x)-x.sum()-y.sum()+overlap), int(x.sum()-overlap),
             int(y.sum()-overlap), overlap]
    common = dict(u=u, v=v, repeat=repeat, n00=cells[0], n10=cells[1], n01=cells[2],
                  n11=cells[3], min_cell=min(cells))
    if min(cells) == 0:
        return [dict(common, method=m, status='SEPARATION_OR_MONOMORPHIC', beta=None,
                     seconds=0., null_load_seconds=0., reconstruction_seconds=0.,
                     iterations=0, tau=None, eigendecomposition_seconds=0., profiling_seconds=0.)
                for m in METHODS]
    # Randomize order to limit systematic CPU/cache timing advantages.
    order = np.random.default_rng(np.random.SeedSequence([seed, u, v, repeat])).permutation(METHODS)
    rows, null, null_status, load_seconds = [], None, None, 0.
    for method in order:
        reconstruction = 0.
        # Loading is reported separately; it is never hidden in claimed fit speedups.
        if method != 'cold_refit' and null_status is None:
            null, null_status, load_seconds = saved_null(v)
        timing = {}
        tick = time.perf_counter()
        if method != 'cold_refit' and null is None:
            result = dict(status='NULL_UNAVAILABLE_'+str(null_status), beta=None, iterations=0, tau=None)
        elif method == 'score_approx':
            before = time.perf_counter()
            discovery_nulls.complete(null, STATE['K'], STATE['eigen'])
            reconstruction = time.perf_counter()-before
            score = glmm.score_predictor_block(x[:, None], null, compute_spa_adjustment=False)[0]
            valid = score.status == 'OK' and np.isfinite(score.beta_score)
            result = dict(status=score.status if valid else 'UNSTABLE_SCORE_'+score.status,
                          beta=float(score.beta_score) if valid else None, iterations=0,
                          tau=float(null['tau']))
        else:
            kwargs = dict(covariates=x[:, None], kinship_eigensystem=STATE['eigen'], timing=timing)
            fitter = glmm.fit_logistic_mixed
            if method == 'warm_refit':
                fitter = STATE['warm']
                kwargs.update(initial_eta=logit(np.clip(null['mu'], 1e-12, 1-1e-12)),
                              initial_tau=null['tau'])
            fit, fitted_cache = fitter(y, STATE['K'], **kwargs)
            result = fitted_result(fit, fitted_cache)
        rows.append(dict(common, method=str(method), **result, seconds=time.perf_counter()-tick,
                    null_load_seconds=load_seconds if method != 'cold_refit' else 0.,
                    reconstruction_seconds=reconstruction,
                    eigendecomposition_seconds=timing.get('weighted_eigendecomposition', 0.),
                    profiling_seconds=timing.get('tau_profiling', 0.)))
    elapsed = time.perf_counter()-started
    for row in rows:
        row['pair_job_seconds'] = elapsed
    return rows


def select_pairs(db, source, count, seed):
    # The old DISTINCT query used distance_pending followed by a temporary B-tree
    # sort. Force the already-existing pattern index and deduplicate adjacent
    # rows in Python: constant extra memory, no SQL temporary sorting file.
    sql = ('SELECT pu,pv FROM distance INDEXED BY distance_pattern '
           'WHERE status=\'DISTAL\' ORDER BY pu,pv'
           if source == 'distal' else 'SELECT u,v FROM decisions WHERE final=1 ORDER BY u,v')
    rng, selected, seen = np.random.default_rng(seed), [], 0
    last, previous, scanned = time.monotonic(), None, 0
    # Uniform reservoir: no all-pair list or dense similarity matrix in memory.
    for u, v in db.execute(sql):
        scanned += 1
        if scanned % 100000 == 0 and time.monotonic()-last > 10:
            print(f'[beta benchmark] sampling: {scanned:,} rows; {seen:,} unique pattern pairs', flush=True)
            last = time.monotonic()
        if (u,v) == previous:
            continue
        previous = (u,v)
        seen += 1
        pair = [int(u), int(v)]
        if len(selected) < count:
            selected.append(pair)
        else:
            index = int(rng.integers(seen))
            if index < count:
                selected[index] = pair
    return sorted(selected), seen


def null_sources(db, work, selected):
    """Resolve the small sample once, so fitting workers never open SQLite."""
    alias = work/'null_canonical.npy'
    aliases = np.load(alias,mmap_mode='r') if alias.exists() else None
    sources = {}
    for response in sorted({v for _,v in selected}):
        canonical = int(aliases[response]) if aliases is not None else response
        key = str(canonical)
        if key in sources:
            continue
        local = work/'null_models'/f'{canonical//1000:06d}'/f'null_{canonical:09d}.json'
        if local.exists():
            sources[key] = str(local)
        else:
            row = db.execute('SELECT path FROM external_nulls WHERE pid=?',(canonical,)).fetchone()
            sources[key] = row[0] if row else None
    return sources


def load_checkpoint(path):
    rows = []
    if path.exists():
        with path.open('rb+') as stream:
            # A job is one line. Only an incomplete final line is discarded.
            while line := stream.readline():
                if not line.endswith(b'\n'):
                    stream.truncate(stream.tell()-len(line))
                    break
                rows.extend(json.loads(line))
    return rows


def summarize(rows, wall, threads, tolerance):
    reference = {(r['u'], r['v'], r['repeat']): r for r in rows if r['method'] == 'cold_refit'}
    summary = {}
    for method in METHODS:
        subset = [r for r in rows if r['method'] == method]
        matches = [(r, reference[(r['u'], r['v'], r['repeat'])]) for r in subset]
        valid = [(r, ref) for r, ref in matches if r['status'] == ref['status'] == 'OK'
                 and r['beta'] is not None and ref['beta'] is not None]
        errors = [abs(r['beta']-ref['beta']) for r, ref in valid]
        ratios = [ref['seconds']/r['seconds'] for r, ref in valid if r['seconds'] > 0]
        summary[method] = dict(n=len(subset), statuses=dict(Counter(r['status'] for r in subset)),
            median_seconds=float(np.median([r['seconds'] for r in subset])),
            mean_seconds=float(np.mean([r['seconds'] for r in subset])),
            median_paired_fit_speedup=float(np.median(ratios)) if ratios else None,
            median_null_access_seconds=float(np.median([r['null_load_seconds'] for r in subset])),
            median_seconds_plus_null_access=float(np.median([r['seconds']+r['null_load_seconds'] for r in subset])),
            jointly_successful=len(valid), status_agreement=sum(r['status']==ref['status'] for r,ref in matches),
            beta_mae=float(np.mean(errors)) if errors else None,
            beta_abs_error_p95=float(np.quantile(errors, .95)) if errors else None,
            beta_abs_error_max=max(errors) if errors else None,
            beta_within_tolerance=sum(e <= tolerance for e in errors),
            sign_agreement=int(sum(np.sign(r['beta'])==np.sign(ref['beta']) for r,ref in valid)))
    return dict(methods=summary, invocation_wall_seconds=wall, threads=threads,
        beta_absolute_tolerance=tolerance, production_modified=False,
        notes=[
            'Cold refits use the production fitter and production beta acceptance rules.',
            'Warm fits change only initial eta and tau; agreement is measured, not assumed.',
            'Score beta is U/V under the null, not a refitted logistic coefficient.',
            'Null access includes file loading/hash verification on cache misses and is reported separately.',
            'Exact sketch completion is included in score seconds; it may be reused by later pairs.',
            'Methods share a bounded null cache; seconds_plus_null_access is not an independent cold-I/O benchmark.',
            'Job order is randomized and all methods use the same sampled pairs.',
            'Invocation wall time includes all three methods, worker startup and benchmark checkpoint writes.',
            'This benchmarks fitting, not production scheduler or transaction speedups.',
            'Failed beta fits and status differences must be reviewed before adopting an optimization.'])


def plot(rows, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.fonttype'] = 'none'
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), layout='constrained')
    times = [[max(r['seconds'], 1e-6) for r in rows if r['method']==m and r['status']=='OK'] for m in METHODS]
    positions = [i+1 for i, values in enumerate(times) if values]
    if positions:
        axes[0].boxplot([values for values in times if values], positions=positions, patch_artist=True,
                        boxprops=dict(facecolor='white', edgecolor='black'),
                        medianprops=dict(color='black'), flierprops=dict(marker='.', markeredgecolor='black'))
    axes[0].set_xticks([1, 2, 3], ['Cold refit', 'Warm refit', 'Score'])
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Seconds per successful fit')
    reference = {(r['u'],r['v'],r['repeat']): r for r in rows if r['method']=='cold_refit'}
    for ax, method, label in zip(axes[1:], METHODS[1:], ['Warm-start beta', 'Score-based beta (approximate)']):
        pairs = [(reference[(r['u'],r['v'],r['repeat'])], r) for r in rows if r['method']==method]
        points = [(ref['beta'],r['beta']) for ref,r in pairs if ref['status']==r['status']=='OK'
                  and ref['beta'] is not None and r['beta'] is not None]
        if points:
            x, y = np.array(points).T
            ax.scatter(x, y, s=9, color='black', alpha=.45)
            low, high = min(x.min(),y.min()), max(x.max(),y.max())
            padding = max(.1, (high-low)*.05)
            ax.plot([low-padding,high+padding], [low-padding,high+padding], 'k--', linewidth=.8)
            ax.set_xlim(low-padding,high+padding); ax.set_ylim(low-padding,high+padding)
        ax.set_xlabel('Cold-refit beta'); ax.set_ylabel(label)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
    for extension in ('png', 'svg'):
        fig.savefig(out/f'beta_benchmark.{extension}', dpi=300)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Existing KOVAR output with run_identity.json')
    parser.add_argument('--out', type=Path, required=True, help='Separate benchmark output directory')
    parser.add_argument('--pairs', type=int, default=200)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pair-source', choices=['distal','significant'], default='distal')
    parser.add_argument('--beta-tolerance', type=float, default=1e-4)
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--prepare-only', action='store_true',
                        help='Freeze sample/null paths on the production database host; fit later without SQLite')
    args = parser.parse_args(argv)
    if min(args.pairs,args.threads,args.repeats) < 1 or args.beta_tolerance <= 0:
        parser.error('Counts and beta tolerance must be positive')
    args.run, args.out = args.run.resolve(), args.out.resolve()
    identity = json.loads((args.run/'run_identity.json').read_text())
    work = Path(identity.get('work_directory', args.run/'.kovar_work')).resolve()
    if args.out in (args.run, work) or work in args.out.parents:
        parser.error('Use a separate output directory outside the production checkpoint')
    info = json.loads((work/'prepared.json').read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    provenance = dict(run=str(args.run), work=str(work), pairs=args.pairs, repeats=args.repeats,
        pair_source=args.pair_source, seed=args.seed, numpy=np.__version__, scipy=scipy.__version__,
        python=platform.python_version(), fitter_sha256=digest(inspect.getfile(glmm)),
        benchmark_sha256=digest(__file__), prepared_sha256=digest(work/'prepared.json'),
        run_identity_sha256=digest(args.run/'run_identity.json'))
    manifest = args.out/'selection.json'
    if manifest.exists():
        selection = json.loads(manifest.read_text())
        if selection['identity'] != provenance:
            parser.error('Benchmark inputs/settings/code changed; choose a new --out')
    else:
        with readonly(work/'work.sqlite') as db:
            selected, available = select_pairs(db,args.pair_source,args.pairs,args.seed)
            pending = db.execute("SELECT count(*) FROM distance WHERE status='PENDING'").fetchone()[0]
            sources = null_sources(db,work,selected)
        if not selected:
            parser.error('No eligible pairs yet; finish some distances or use --pair-source significant')
        selection = dict(identity=provenance, selected_pairs=selected, available_pairs=available,
                         pending_distance_unitig_pairs=pending, provisional_distal_pool=bool(pending),
                         null_sources=sources)
        atomic_json(manifest, selection)
    print(f"[beta benchmark] frozen sample {len(selection['selected_pairs']):,}/{selection['available_pairs']:,} "
          f"{args.pair_source} pattern pairs; {args.threads} CPUs", flush=True)
    if selection['provisional_distal_pool']:
        print('[beta benchmark] Distance processing was incomplete at sampling; this frozen sample is provisional.', flush=True)
    if args.prepare_only:
        print(f'[beta benchmark] Prepared {manifest}; rerun without --prepare-only to fit. '
              'Subsequent fitting does not open the production database.',flush=True)
        return
    checkpoint = args.out/'checkpoint.jsonl'
    rows = load_checkpoint(checkpoint)
    done = {(r['u'],r['v'],r['repeat']) for r in rows}
    jobs = [(u,v,repeat,args.seed) for repeat in range(args.repeats) for u,v in selection['selected_pairs']
            if (u,v,repeat) not in done]
    # Interleave responses before dynamic dispatch; avoid one response dominating early timings.
    np.random.default_rng(args.seed).shuffle(jobs)
    started = time.perf_counter()
    with checkpoint.open('a', encoding='utf-8') as stream:
        with ProcessPoolExecutor(max_workers=args.threads, mp_context=multiprocessing.get_context('spawn'),
                initializer=initialize, initargs=(str(work),info,selection['null_sources'])) as pool:
            source, pending = iter(jobs), set()
            def submit():
                job = next(source, None)
                if job is not None:
                    pending.add(pool.submit(compare,job))
            for _ in range(min(len(jobs),args.threads*2)):
                submit()
            while pending:
                finished, pending = wait(pending,return_when=FIRST_COMPLETED)
                for future in finished:
                    result = future.result()
                    stream.write(json.dumps(result,allow_nan=False)+'\n'); stream.flush()
                    rows.extend(result); done.add((result[0]['u'],result[0]['v'],result[0]['repeat']))
                    print(f'[beta benchmark] {len(done):,}/{len(selection["selected_pairs"])*args.repeats:,} pair jobs saved',flush=True)
                    submit()
    rows.sort(key=lambda r:(r['u'],r['v'],r['repeat'],r['method']))
    with (args.out/'beta_results.tsv').open('w',newline='',encoding='utf-8') as stream:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(stream,fieldnames=fields,delimiter='\t',lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    summary = summarize(rows,time.perf_counter()-started,args.threads,args.beta_tolerance)
    summary.update(selection=selection, coordinator_numerical_libraries=threadpool_info(),
                   native_threads_per_worker=1,
                   invocation_pair_jobs=len(jobs), checkpoint_pair_jobs=len(done))
    atomic_json(args.out/'beta_summary.json',summary)
    if not args.no_plots:
        plot(rows,args.out)
    print(json.dumps(summary['methods'],indent=2),flush=True)
    print(f'[beta benchmark] Reports saved to {args.out}',flush=True)


if __name__ == '__main__':
    try:
        main()
    except sqlite3.Error as error:
        print(f'[beta benchmark] SQLite {getattr(error,"sqlite_errorname","UNKNOWN")} '
              f'(code {getattr(error,"sqlite_errorcode","unknown")}): {error}',file=sys.stderr)
        print('Sampling needs readable work.sqlite and any live WAL/shared-memory files. '
              'Run --prepare-only on the same compute node as the active KOVAR writer; '
              'then run fitting from the saved selection on another node. '
              'If preparation also fails on that node, check filesystem space/quota/permissions '
              'and the extended SQLite error above. Do not delete live WAL/SHM files.',file=sys.stderr)
        raise SystemExit(2) from error
