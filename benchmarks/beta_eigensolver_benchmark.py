#!/usr/bin/env python3
"""Compare full beta refits with NumPy, SciPy EVD and SciPy EVR.

Read an existing beta-benchmark selection and immutable prepared arrays only.
No SQLite, saved null loading, approximated effects or production code changes.
Each isolated worker switches its eigensolver temporarily and uses one BLAS CPU.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import csv
import inspect
import json
import multiprocessing
import os
from pathlib import Path
import platform
import time
from unittest.mock import patch

import numpy as np
import scipy
from scipy import linalg
from threadpoolctl import threadpool_info, threadpool_limits

from ko_variation import glmm
from ko_variation.discovery_store import atomic_json, digest, open_patterns

DRIVERS = ('numpy', 'scipy_evd', 'scipy_evr')
STATE = {}


def initialize(work, info):
    threadpool_limits(limits=1)
    work = Path(work)
    STATE.update(info=info, patterns=open_patterns(work,info), K=np.load(work/'K.npy',mmap_mode='r'),
        eigen=glmm.KinshipEigensystem(np.load(work/'K_values.npy',mmap_mode='r'),
            np.load(work/'K_vectors.npy',mmap_mode='r'),info['shared_covariance']))
    # Tiny unmeasured backend initialization, not a model fit or timed matrix.
    tiny = np.array([[1.,.1],[.1,1.]])
    np.linalg.eigh(tiny)
    for driver in ('evd','evr'):
        linalg.eigh(tiny,driver=driver)
    STATE['libraries'] = threadpool_info()


def finite(value):
    return float(value) if np.isfinite(value) else None


def compare(job):
    u,v,repeat,seed = job
    data = np.unpackbits(STATE['patterns'][[u,v]],axis=1,bitorder='little')[:,:STATE['info']['n']].astype(float)
    x,y = data
    overlap = int(x@y)
    cells = [int(len(x)-x.sum()-y.sum()+overlap),int(y.sum()-overlap),int(x.sum()-overlap),overlap]
    common = dict(u=u,v=v,repeat=repeat,n00=cells[0],n01=cells[1],n10=cells[2],n11=cells[3],min_cell=min(cells))
    records, fits = [], {}
    order = np.random.default_rng(np.random.SeedSequence([seed,u,v,repeat])).permutation(DRIVERS)
    for order_index, driver in enumerate(order):
        row = dict(common,driver=str(driver),method_order=order_index,beta=None,se=None,tau=None,
            status='SEPARATION_OR_MONOMORPHIC',iterations=0,seconds=0.,weighted_eigen_seconds=0.,
            profiling_seconds=0.,solver_seconds=0.,explicit_copy_seconds=0.,eigen_calls=0)
        if min(cells):
            timing, solver_timing = {}, dict(seconds=0.,copy=0.,calls=0)
            original = np.linalg.eigh
            def solver(matrix, UPLO='L'):
                start = time.perf_counter()
                solver_timing['calls'] += 1
                try:
                    if driver == 'numpy':
                        return original(matrix,UPLO=UPLO)
                    # Only a fresh matrix is overwritten. Input K and prepared
                    # arrays remain read-only; full float64 eigenpairs are kept.
                    before = time.perf_counter()
                    owned = np.array(matrix,dtype=np.float64,order='F',copy=True)
                    solver_timing['copy'] += time.perf_counter()-before
                    return linalg.eigh(owned,driver=str(driver).removeprefix('scipy_'),
                        lower=UPLO=='L',overwrite_a=True,check_finite=True)
                finally:
                    solver_timing['seconds'] += time.perf_counter()-start
            start = time.perf_counter()
            # Isolated process, one Python fit at a time. The original function
            # is restored even if fitting raises; source/cache hashes never change.
            with patch.object(glmm.np.linalg,'eigh',new=solver):
                fit, cache = glmm.fit_logistic_mixed(y,STATE['K'],covariates=x[:,None],
                    kinship_eigensystem=STATE['eigen'],timing=timing)
            row.update(seconds=time.perf_counter()-start,status=fit.status,iterations=fit.iterations,
                tau=finite(fit.tau),weighted_eigen_seconds=timing.get('weighted_eigendecomposition',0.),
                profiling_seconds=timing.get('tau_profiling',0.),solver_seconds=solver_timing['seconds'],
                explicit_copy_seconds=solver_timing['copy'],eigen_calls=solver_timing['calls'])
            if fit.converged and fit.status == 'OK':
                variance, value = float(cache['M_inv'][1,1]),float(fit.beta[1])
                if fit.n_weights_clipped or not np.isfinite([value,variance]).all() or variance<=0:
                    row['status'] = 'UNSTABLE_EFFECT'
                else:
                    row.update(beta=value,se=float(np.sqrt(variance)))
            fits[str(driver)] = fit
            del cache
        records.append(row)
    reference = next(r for r in records if r['driver']=='numpy')
    for row in records:
        row.update(status_matches_numpy=row['status']==reference['status'],beta_abs_difference=None,
            se_relative_difference=None,max_probability_difference=None,tau_relative_difference=None)
        if row['status'] == reference['status'] == 'OK':
            other, base = fits[row['driver']], fits['numpy']
            row.update(beta_abs_difference=abs(row['beta']-reference['beta']),
                se_relative_difference=abs(row['se']-reference['se'])/reference['se'],
                max_probability_difference=finite(np.max(np.abs(other.fitted_probability-base.fitted_probability))),
                tau_relative_difference=abs(row['tau']-reference['tau'])/max(1.,abs(reference['tau'])))
    return dict(rows=records,libraries=STATE['libraries'])


def checkpoint_rows(path):
    rows, libraries = [], []
    if path.exists():
        with path.open('rb+') as stream:
            while line := stream.readline():
                if not line.endswith(b'\n'):
                    stream.truncate(stream.tell()-len(line))
                    break
                payload = json.loads(line)
                rows.extend(payload['rows'])
                libraries = payload['libraries']
    return rows, libraries


def summarize(rows, args):
    baseline = {(r['u'],r['v'],r['repeat']):r for r in rows if r['driver']=='numpy'}
    output = {}
    for driver in DRIVERS:
        subset = [r for r in rows if r['driver']==driver]
        good = [r for r in subset if r['status']=='OK' and r['beta_abs_difference'] is not None]
        ratios = defaultdict(list)
        for row in good:
            ref = baseline[(row['u'],row['v'],row['repeat'])]
            if row['seconds']>0:
                ratios[(row['u'],row['v'])].append(ref['seconds']/row['seconds'])
        times = [r['seconds'] for r in subset]
        total = sum(times)
        errors = [r['beta_abs_difference'] for r in good]
        probabilities = [r['max_probability_difference'] for r in good]
        within = [r for r in good if r['beta_abs_difference']<=args.beta_tolerance
                  and r['max_probability_difference']<=args.probability_tolerance]
        output[driver] = dict(fit_count=len(subset),unique_pairs=len({(r['u'],r['v']) for r in subset}),
            statuses=dict(Counter(r['status'] for r in subset)),median_seconds=float(np.median(times)),
            mean_seconds=float(np.mean(times)),total_worker_fit_seconds=total,
            median_paired_speedup=float(np.median([np.median(v) for v in ratios.values()])) if ratios else None,
            status_matches_numpy=sum(r['status_matches_numpy'] for r in subset),
            jointly_successful_fits=len(good),jointly_successful_pairs=len(ratios),
            beta_abs_difference_max=max(errors,default=None),
            beta_abs_difference_p95=float(np.quantile(errors,.95)) if errors else None,
            max_probability_difference=max(probabilities,default=None),
            max_se_relative_difference=max((r['se_relative_difference'] for r in good),default=None),
            max_tau_relative_difference=max((r['tau_relative_difference'] for r in good),default=None),
            fits_within_beta_and_probability_tolerances=len(within),
            median_iterations=float(np.median([r['iterations'] for r in subset])),
            weighted_eigen_fraction=sum(r['weighted_eigen_seconds'] for r in subset)/total if total else None,
            profiling_fraction=sum(r['profiling_seconds'] for r in subset)/total if total else None,
            explicit_copy_fraction=sum(r['explicit_copy_seconds'] for r in subset)/total if total else None)
    repeats = defaultdict(list)
    for row in baseline.values():
        if row['beta'] is not None:
            repeats[(row['u'],row['v'])].append(row['beta'])
    return dict(drivers=output,beta_absolute_tolerance=args.beta_tolerance,
        probability_absolute_tolerance=args.probability_tolerance,
        numpy_beta_max_span_across_repeats=max((max(v)-min(v) for v in repeats.values()),default=None),
        production_modified=False,score_approximation_used=False,
        notes=[
            'All methods use the production cold-start PQL fitter and beta acceptance rules.',
            'Only the eigensolver call changes; float64, full eigenpairs, tolerances and iteration limits are unchanged.',
            'SciPy includes an owned Fortran-order copy and finite checks in measured time.',
            'Both NumPy and SciPy EVD use divide-and-conquer LAPACK families; a speedup is not guaranteed.',
            'Paired speedup is the median across pairs of their median repeat-specific speedups.',
            'Timing comparisons use jointly successful fits; always review failure/status changes too.',
            'Repeat fits are repeated timings on the same pairs, not independent biological samples.',
            'No eigenvector-by-eigenvector comparison is used because degenerate eigenspaces may rotate.',
            'Fitted-probability, coefficient, standard-error and tau differences assess downstream numerical agreement.',
            'No production solver is selected or changed automatically.'])


def plot(rows,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.fonttype']='none'
    grouped = {driver:defaultdict(list) for driver in DRIVERS}
    for row in rows:
        grouped[row['driver']][(row['u'],row['v'])].append(row)
    times = [[float(np.median([r['seconds'] for r in group if r['status']=='OK']))
              for group in grouped[d].values() if any(r['status']=='OK' for r in group)] for d in DRIVERS]
    fig,axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
    positions=[i+1 for i,t in enumerate(times) if t]
    if positions:
        axes[0].boxplot([t for t in times if t],positions=positions,patch_artist=True,
            boxprops=dict(facecolor='white',edgecolor='black'),medianprops=dict(color='black'),
            flierprops=dict(marker='.',markeredgecolor='black'))
    axes[0].set_xticks([1,2,3],['NumPy','SciPy EVD','SciPy EVR'])
    axes[0].set_yscale('log');axes[0].set_ylabel('Median seconds per successful fit')
    reference={(r['u'],r['v'],r['repeat']):r for r in rows if r['driver']=='numpy'}
    for ax,driver in zip(axes[1:],DRIVERS[1:]):
        points=[]
        for group in grouped[driver].values():
            paired=[(reference[(r['u'],r['v'],r['repeat'])],r) for r in group]
            paired=[(ref,r) for ref,r in paired if ref['status']==r['status']=='OK']
            if paired:
                points.append((np.median([ref['beta'] for ref,r in paired]),
                               np.median([r['beta']-ref['beta'] for ref,r in paired])))
        if points:
            x,y=np.array(points).T
            ax.scatter(x,y,color='black',s=10,alpha=.5)
        ax.axhline(0,color='black',linestyle='--',linewidth=.8)
        ax.set_xlabel('NumPy-refit beta')
        ax.set_ylabel(f'{driver.replace("scipy_","SciPy ").upper().replace("SCIPY","SciPy")} beta − NumPy beta')
        ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
    for suffix in ('png','svg'):
        fig.savefig(out/f'eigensolver_benchmark.{suffix}',dpi=300)
    plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',required=True,type=Path,help='Existing beta benchmark selection.json')
    p.add_argument('--out',required=True,type=Path,help='Separate eigensolver benchmark output')
    p.add_argument('--work',type=Path,help='Prepared work directory override if the original was relocated')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--beta-tolerance',type=float,default=1e-6)
    p.add_argument('--probability-tolerance',type=float,default=1e-7)
    p.add_argument('--no-plots',action='store_true')
    args=p.parse_args(argv)
    if min(args.threads,args.repeats)<1 or min(args.beta_tolerance,args.probability_tolerance)<=0:
        p.error('Counts and numerical comparison tolerances must be positive')
    args.selection,args.out=args.selection.resolve(),args.out.resolve()
    selection=json.loads(args.selection.read_text())
    work=(args.work or Path(selection['identity']['work'])).resolve()
    if args.out==work or work in args.out.parents or args.out==args.selection.parent:
        p.error('Use a separate --out, outside the production work and original benchmark output')
    info=json.loads((work/'prepared.json').read_text())
    for key,path in [('fitter_sha256',Path(inspect.getfile(glmm))),('prepared_sha256',work/'prepared.json')]:
        expected=selection['identity'].get(key)
        if expected is not None and digest(path)!=expected:
            p.error(f'{key} differs from the selected cohort/fitter; do not mix benchmark inputs')
    pairs=[(int(u),int(v)) for u,v in selection['selected_pairs']]
    if not pairs or len(pairs)!=len(set(pairs)) or any(not 0<=u<v<info['patterns'] for u,v in pairs):
        p.error('Selection must contain unique, valid ordered pattern pairs')
    # Hash only the small prepared numerical pool, never the large search DB.
    products={name:digest(work/name) for name in ('patterns.bin','K.npy','K_values.npy','K_vectors.npy')}
    if any(products[name]!=info['products'][name] for name in products):
        p.error('Prepared numerical product differs from its recorded cohort')
    identity=dict(selection_sha256=digest(args.selection),work=str(work),products=products,
        fitter_sha256=digest(inspect.getfile(glmm)),benchmark_sha256=digest(__file__),
        numpy=np.__version__,scipy=scipy.__version__,python=platform.python_version(),
        repeats=args.repeats,seed=args.seed,drivers=list(DRIVERS))
    args.out.mkdir(parents=True,exist_ok=True)
    manifest=args.out/'benchmark_identity.json'
    if manifest.exists():
        if json.loads(manifest.read_text())!=identity:
            p.error('Inputs/code/settings changed; use a new --out')
    else:
        atomic_json(manifest,identity)
    checkpoint=args.out/'checkpoint.jsonl'
    rows,libraries=checkpoint_rows(checkpoint)
    completed={(r['u'],r['v'],r['repeat']) for r in rows}
    jobs=[(u,v,repeat,args.seed) for repeat in range(args.repeats) for u,v in pairs if (u,v,repeat) not in completed]
    np.random.default_rng(args.seed).shuffle(jobs)
    print(f'[eigensolver benchmark] {len(pairs):,} frozen pairs x {args.repeats} repeats x 3 solvers; '
          f'{args.threads} CPUs; {len(jobs):,} pair-repeat jobs remaining; SQLite not used',flush=True)
    started=time.perf_counter()
    with checkpoint.open('a',encoding='utf-8') as stream:
        if jobs:
            with ProcessPoolExecutor(max_workers=args.threads,mp_context=multiprocessing.get_context('spawn'),
                    initializer=initialize,initargs=(str(work),info)) as pool:
                source,pending=iter(jobs),set()
                def submit():
                    job=next(source,None)
                    if job is not None:
                        pending.add(pool.submit(compare,job))
                for _ in range(min(len(jobs),args.threads*2)):
                    submit()
                while pending:
                    finished,pending=wait(pending,return_when=FIRST_COMPLETED)
                    for future in finished:
                        payload=future.result()
                        stream.write(json.dumps(payload,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
                        rows.extend(payload['rows']);libraries=payload['libraries']
                        first=payload['rows'][0];completed.add((first['u'],first['v'],first['repeat']))
                        print(f'[eigensolver benchmark] {len(completed):,}/{len(pairs)*args.repeats:,} pair-repeat jobs saved',flush=True)
                        submit()
    rows.sort(key=lambda r:(r['u'],r['v'],r['repeat'],r['driver']))
    with (args.out/'eigensolver_results.tsv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=sorted(rows[0]),delimiter='\t',lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    summary=summarize(rows,args)
    summary.update(pairs=len(pairs),samples=info['n'],repeats=args.repeats,threads=args.threads,
        native_threads_per_worker=1,worker_numerical_libraries=libraries,
        invocation_wall_seconds=time.perf_counter()-started,invocation_pair_repeat_jobs=len(jobs),
        completed_pair_repeat_jobs=len(completed),selection_path=str(args.selection),identity=identity)
    atomic_json(args.out/'eigensolver_summary.json',summary)
    if not args.no_plots:
        plot(rows,args.out)
    print(json.dumps(summary['drivers'],indent=2),flush=True)
    print(f'[eigensolver benchmark] Reports: {args.out}',flush=True)


if __name__=='__main__':
    main()
