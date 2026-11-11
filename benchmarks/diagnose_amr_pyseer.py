#!/usr/bin/env python3
"""Compare saved AMR target tests with pyseer's actual FaST-LMM implementation.

Run in an environment containing pyseer. KOVAR need not be installed there.
The source diagnostic and discovery inputs are read-only; no source SQLite
database is opened. One response-level checkpoint is saved per completed fit.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import time

# Parallelism is across response models, not nested BLAS threads.
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np

_WORKER = {}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def table(path, rows, fields):
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fields, delimiter='\t', extrasaction='ignore')
        writer.writeheader()
        for r in rows:
            writer.writerow({k: '' if r.get(k) is None else r.get(k, '') for k in fields})
    temporary.replace(path)


def read_table(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def log(out, message):
    text = '[AMR pyseer] ' + message
    print(text, flush=True)
    with (out/'progress.log').open('a', encoding='utf-8') as stream:
        stream.write(time.strftime('%Y-%m-%d %H:%M:%S ') + text + '\n')


def pyseer_environment():
    try:
        import pyseer
        import pyseer.lmm
        import pyseer.fastlmm.lmm_cov
        import pandas
        import scipy
    except ImportError as error:
        raise ValueError('Run with the Python environment containing pyseer, pandas and scipy. '
                         'KOVAR itself is not required in this environment: ' + str(error)) from error
    return dict(pyseer=getattr(pyseer, '__version__', 'unknown'), numpy=np.__version__,
                pandas=pandas.__version__, scipy=scipy.__version__,
                lmm_source_sha256=digest(inspect.getfile(pyseer.lmm)),
                fastlmm_source_sha256=digest(inspect.getfile(pyseer.fastlmm.lmm_cov)))


def prepare(args, environment):
    diagnosis = json.loads((args.diagnosis/'diagnosis_identity.json').read_text())
    info, source = diagnosis['prepared'], diagnosis['source_identity']
    rows = read_table(args.diagnosis/'pair_comparison.tsv')
    if not rows:
        raise ValueError('pair_comparison.tsv has no saved diagnostic tests')
    if args.work is None:
        candidates = [args.diagnosis.parent.parent/'discovery/results/.kovar_work',
                      args.diagnosis.parent.parent/'discovery/.kovar_work',
                      Path(source.get('work_directory', '/not-found'))]
        args.work = next((p for p in candidates if (p/'patterns.bin').is_file() and (p/'K.npy').is_file()), None)
    if args.work is None:
        raise ValueError('Provide --work /path/to/the/original/.kovar_work (patterns.bin and K.npy required)')
    args.work = args.work.resolve()
    for path in (args.diagnosis, args.work):
        # Paths are resolved above; parents membership also works on Python 3.8.
        if args.out == path or args.out in path.parents or path in args.out.parents:
            raise ValueError('--out must be separate from the source diagnosis and work directories')
    # These are the only full frozen-product reads. No decisions/beta DB is needed.
    for name in ('patterns.bin', 'K.npy'):
        if digest(args.work/name) != info['products'][name]:
            raise ValueError(f'{name} differs from the frozen diagnostic input')
    ids = sorted({int(r[k]) for r in rows for k in ('predictor_pattern', 'response_pattern')})
    if ids[0] < 0 or ids[-1] >= info['patterns']:
        raise ValueError('Pattern ID is outside the frozen pattern pool')
    packed = np.memmap(args.work/'patterns.bin', mode='r', dtype='u1',
                       shape=(info['patterns'], info['packed_width']))
    selected = np.unpackbits(np.asarray(packed[ids]), axis=1, bitorder='little')[:, :info['n']]
    lookup = {pid: i for i, pid in enumerate(ids)}
    family = int(source['settings'].get('bonferroni_tests') or math.comb(info['patterns'], 2))
    cutoff = float(source['settings']['alpha']) / family
    seen = set()
    for r in rows:
        u, v = int(r['predictor_pattern']), int(r['response_pattern'])
        if (u, v) in seen:
            raise ValueError(f'Duplicate diagnostic pair {u},{v}')
        seen.add((u, v))
        x, y = selected[lookup[u]].astype(np.int64), selected[lookup[v]].astype(np.int64)
        n11 = int(x @ y)
        cells = [info['n'] - int(x.sum()) - int(y.sum()) + n11, int(x.sum()) - n11, int(y.sum()) - n11, n11]
        if cells != [int(r[k]) for k in ('n00', 'n10', 'n01', 'n11')]:
            raise ValueError(f'Frozen vectors do not reproduce the saved cells for {u},{v}')
        if int(r['bonferroni_tests']) != family or not math.isclose(float(r['p_cutoff']), cutoff, rel_tol=1e-12):
            raise ValueError('Saved diagnostic has a different testing family or threshold')
    K = np.load(args.work/'K.npy', allow_pickle=False)
    if K.shape != (info['n'], info['n']) or not np.isfinite(K).all():
        raise ValueError('Invalid frozen covariance dimensions/values')
    # Match pyseer initialise_lmm's mean-diagonal normalization, not a new kernel.
    factor = info['n'] / float(np.diag(K).sum())
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError('Frozen K must have a positive mean diagonal')
    identity = dict(revision=1, environment=environment,
                    diagnostic_identity_sha256=digest(args.diagnosis/'diagnosis_identity.json'),
                    comparison_sha256=digest(args.diagnosis/'pair_comparison.tsv'),
                    products={k: info['products'][k] for k in ('patterns.bin', 'K.npy')},
                    samples=info['samples'], pattern_ids=ids, bonferroni_tests=family,
                    p_cutoff=cutoff, pyseer_K_normalization_factor=factor,
                    orientation='saved predictor_pattern -> response_pattern',
                    model='pyseer initialise_lmm + fit_lmm_block; intercept only; no candidate/cell/prefilter gate')
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out/'run_identity.json'
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError('Comparison inputs or pyseer environment changed; use a separate --out')
    if not manifest.exists() and any(args.out.iterdir()):
        raise ValueError('Nonempty output has no comparison identity; choose an empty --out')
    atomic_json(manifest, identity)
    np.save(args.out/'selected_patterns.npy', selected, allow_pickle=False)
    table(args.out/'sample_order.tsv', ({'row': i, 'sample': s} for i, s in enumerate(info['samples'])), ['row', 'sample'])
    import pandas as pd
    pd.DataFrame(K, index=info['samples'], columns=info['samples']).to_csv(
        args.out/'covariance.tsv', sep='\t', float_format='%.17g')
    (args.out/'responses').mkdir(exist_ok=True)
    return rows, identity


def initialize(out, identity):
    # Reuse the installed implementation, not a reimplementation of FaST-LMM.
    from pyseer.lmm import initialise_lmm, fit_lmm_block
    _WORKER.update(out=Path(out), identity=identity,
                   matrix=np.load(Path(out)/'selected_patterns.npy', mmap_mode='r'),
                   lookup={pid: i for i, pid in enumerate(identity['pattern_ids'])},
                   initialise=initialise_lmm, score=fit_lmm_block)


def fit_response(job):
    import pandas as pd
    response, predictors = job
    s = _WORKER
    started = time.monotonic()
    y = np.array(s['matrix'][s['lookup'][response]], dtype=float)
    phenotype = pd.Series(y, index=s['identity']['samples'])
    fitted_p, lmm, h2 = s['initialise'](phenotype, pd.DataFrame(), str(s['out']/'covariance.tsv'))
    if list(fitted_p.index) != s['identity']['samples'] or not np.array_equal(fitted_p.to_numpy(), y):
        raise ValueError('Pyseer changed sample order or dropped samples')
    if not math.isfinite(float(h2)) or not 0 <= h2 <= 1:
        raise ValueError(f'Invalid LMM null h2 for response {response}: {h2}')
    output = []
    # Each block shares a single fitted response null. No four-cell/MAF/MI filter.
    for first in range(0, len(predictors), 256):
        pids = predictors[first:first+256]
        X = np.array(s['matrix'][[s['lookup'][p] for p in pids]].T, dtype=float)
        result = s['score'](lmm, h2, X)
        if any(len(result[k]) != len(pids) for k in ('p_values', 'beta', 'bse')):
            raise ValueError('Unexpected pyseer LMM block dimensions')
        for i, predictor in enumerate(pids):
            values = [float(result[k][i]) for k in ('p_values', 'beta', 'bse')]
            good = all(math.isfinite(x) for x in values) and 0 <= values[0] <= 1 and values[2] > 0
            output.append(dict(predictor_pattern=predictor, response_pattern=response,
                               lmm_p=values[0] if good else None, lmm_beta=values[1] if math.isfinite(values[1]) else None,
                               lmm_se=values[2] if math.isfinite(values[2]) else None,
                               lmm_status='OK' if good else 'NONFINITE_OR_INVALID_LMM_RESULT', lmm_h2=float(h2)))
    record = dict(response=response, predictors=predictors, results=output, elapsed_seconds=time.monotonic()-started)
    atomic_json(s['out']/'responses'/f'{response:09d}.json', record)
    return response


def export(args, rows, identity, resumed):
    results = {}
    for path in (args.out/'responses').glob('*.json'):
        record = json.loads(path.read_text())
        for r in record['results']:
            results[r['predictor_pattern'], r['response_pattern']] = r
    output = []
    for r in rows:
        u, v = int(r['predictor_pattern']), int(r['response_pattern'])
        value = results[u, v] if u != v else dict(lmm_p=None, lmm_beta=None, lmm_se=None,
                                                   lmm_h2=None, lmm_status='SHARED_EXACT_PATTERN')
        pv = value['lmm_p']
        kp = float(r['p_exact']) if r['p_exact'] and r['score_status'] == 'OK' else None
        cut = identity['p_cutoff']
        output.append(dict(r, **value, lmm_nominal=int(pv < .05) if pv is not None else None,
                           lmm_bonferroni=int(pv <= cut) if pv is not None else None,
                           lmm_bonferroni_eligible_original=int(pv <= cut and r['eligible_original'] == '1') if pv is not None else None,
                           lmm_bonferroni_eligible_new=int(pv <= cut and r['eligible_new'] == '1') if pv is not None else None,
                           nominal_comparison='UNAVAILABLE' if pv is None or kp is None else
                           'BOTH' if pv < .05 and kp < .05 else 'PYSEER_ONLY' if pv < .05 else
                           'KOVAR_ONLY' if kp < .05 else 'NEITHER'))
    extra = ['lmm_p', 'lmm_beta', 'lmm_se', 'lmm_h2', 'lmm_status', 'lmm_nominal', 'lmm_bonferroni',
             'lmm_bonferroni_eligible_original', 'lmm_bonferroni_eligible_new', 'nominal_comparison']
    table(args.out/'pyseer_comparison.tsv', output, list(rows[0]) + extra)
    groups = []
    for name in sorted({r['target_pairs'] for r in output}):
        rs = [r for r in output if r['target_pairs'] == name]
        ps = [r['lmm_p'] for r in rs if r['lmm_p'] is not None]
        eligible = [r['lmm_p'] for r in rs if r['lmm_p'] is not None and r['eligible_new'] == '1']
        groups.append(dict(target_pairs=name, pairs=len(rs), lmm_ok=len(ps),
                           lmm_nominal=sum(r['lmm_nominal'] or 0 for r in rs),
                           lmm_bonferroni=sum(r['lmm_bonferroni'] or 0 for r in rs),
                           lmm_bonferroni_eligible_original=sum(r['lmm_bonferroni_eligible_original'] or 0 for r in rs),
                           lmm_bonferroni_eligible_new=sum(r['lmm_bonferroni_eligible_new'] or 0 for r in rs),
                           lowest_lmm_p=min(ps) if ps else None,
                           lowest_lmm_p_eligible_new=min(eligible) if eligible else None,
                           pyseer_only_nominal=sum(r['nominal_comparison'] == 'PYSEER_ONLY' for r in rs)))
    table(args.out/'summary.tsv', groups, list(groups[0]))
    summary = dict(samples=len(identity['samples']), pairs=len(rows), resumed_response_models=resumed,
                   bonferroni_tests=identity['bonferroni_tests'], p_cutoff=identity['p_cutoff'], groups=groups,
                   environment=identity['environment'], source_modified=False,
                   interpretation='Matched unitig-pattern association, not a recreation of the 806-isolate SCCmec-carriage study. '
                   'p_exact and beta remain the saved KOVAR results; lmm_* come from pyseer. '
                   'No source SQL, KOVAR null fitting or beta fitting. Pyseer LMM nulls are newly fitted per response. '
                   'Unfiltered significance is distinguished from the original/new cell eligibility gates.')
    atomic_json(args.out/'summary.json', summary)
    for g in groups:
        log(args.out, f"{g['target_pairs']}: {g['lmm_nominal']}/{g['pairs']} nominal; "
                      f"{g['lmm_bonferroni']} Bonferroni; {g['lmm_bonferroni_eligible_new']} Bonferroni + new cell gate")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnosis', type=Path, required=True, help='Completed AMR diagnostic output directory')
    parser.add_argument('--work', type=Path, help='Original .kovar_work; only patterns.bin and K.npy are read')
    parser.add_argument('--out', type=Path, help='Default: sibling amr_pyseer_lmm directory')
    parser.add_argument('--threads', type=int, default=4, help='Independent pyseer response workers; BLAS=1 each')
    args = parser.parse_args(argv)
    args.diagnosis = args.diagnosis.resolve()
    args.out = args.out.resolve() if args.out else args.diagnosis.parent/'amr_pyseer_lmm'
    if args.threads < 1:
        parser.error('--threads must be positive')
    environment = pyseer_environment()
    rows, identity = prepare(args, environment)
    grouped = defaultdict(list)
    for r in rows:
        u, v = int(r['predictor_pattern']), int(r['response_pattern'])
        if u != v:
            grouped[v].append(u)
    pending, resumed = [], 0
    for response, predictors in sorted(grouped.items()):
        predictors.sort()
        path = args.out/'responses'/f'{response:09d}.json'
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['response'] != response or saved['predictors'] != predictors or len(saved['results']) != len(predictors):
                raise ValueError(f'Unexpected response checkpoint: {path}')
            resumed += 1
        else:
            pending.append((response, predictors))
    log(args.out, f"{len(rows):,} same-orientation pairs; {len(identity['samples'])} isolates; "
                  f"{resumed}/{len(grouped)} response models reused; {len(pending)} to fit; workers={args.threads}")
    if pending:
        if args.threads == 1:
            initialize(str(args.out), identity)
            for i, job in enumerate(pending, 1):
                fit_response(job)
                log(args.out, f'response models saved: {resumed+i}/{len(grouped)}')
        else:
            with ProcessPoolExecutor(max_workers=min(args.threads, len(pending)), initializer=initialize,
                                     initargs=(str(args.out), identity)) as pool:
                futures = [pool.submit(fit_response, job) for job in pending]
                for i, future in enumerate(as_completed(futures), 1):
                    future.result()
                    log(args.out, f'response models saved: {resumed+i}/{len(grouped)}')
    export(args, rows, identity, resumed)
    log(args.out, 'complete: ' + str(args.out/'pyseer_comparison.tsv'))


if __name__ == '__main__':
    main()
