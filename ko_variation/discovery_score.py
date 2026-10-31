"""Response-major worker kernels: saved nulls, spectral bounds and distal beta.

Workers never write the search database. The coordinator commits complete
batches, so a killed worker cannot turn a partial calculation into a result.
"""
from __future__ import annotations

from collections import OrderedDict
import json
import os
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .discovery_store import digest, open_patterns
from . import discovery_null_cache as null_cache
from .glmm import (KinshipEigensystem, _WeightedEigen, fit_logistic_mixed,
                   fit_null_glmm, score_predictor_block)
from .memory_budget import ByteLRU, working_usage, process_rss, MIB

_STATE = {}
_POPCOUNT = np.array([int(i).bit_count() for i in range(256)], dtype="u1")


def initialize(work, info, settings):
    # One native numerical thread per worker: --threads is a TOTAL CPU budget.
    threadpool_limits(limits=1)
    work = Path(work)
    _STATE.clear()
    cache_limit=settings.get("cache_bytes_per_worker",0)
    _STATE.update(work=work, info=info, settings=settings,
                  packed=open_patterns(work, info), nulls=ByteLRU(cache_limit),
                  cache_limit=cache_limit,last_pressure=0.,verified=OrderedDict(),metrics={},last_log=0.,
                  K=np.load(work/"K.npy", mmap_mode="r"),
                  eigen=KinshipEigensystem(np.load(work/"K_values.npy", mmap_mode="r"),
                      np.load(work/"K_vectors.npy", mmap_mode="r"), info["shared_covariance"]))
    alias=work/"null_canonical.npy"
    _STATE["canonical"]=np.load(alias,mmap_mode="r") if alias.exists() else None


def metric(key,value=1):
    values=_STATE.get("metrics")
    if values is not None:
        values[key]=values.get(key,0)+value


def pressure_guard():
    s=_STATE
    now=time.monotonic()
    if now-s["last_pressure"]<2:
        return
    s["last_pressure"]=now
    settings=s["settings"]
    usage=working_usage(settings.get("monitor"))
    target=settings.get("total_working_bytes")
    if usage is None:
        usage=process_rss()
        target=settings.get("worker_target_bytes")
    if usage is None or not target:
        return
    cache=s["nulls"]
    s['observed_usage']=usage
    s['observed_target']=target
    if usage>=target:
        # Reduce RESIDENT bytes by our share of the excess, not the previous
        # capacity geometrically. A repeated alarm with an empty cache must not
        # permanently ratchet the admission budget to zero.
        workers=max(1,settings.get('workers',1)) if settings.get('monitor') else 1
        excess=max(1,int((usage-target)/workers))
        before=cache.bytes
        cache.trim(max(0,cache.bytes-max(excess,int(.1*cache.bytes))))
        if before-cache.bytes>=16*MIB and now-s.get('last_allocator_trim',0)>30:
            # Evicted NumPy allocations can remain in glibc arenas. Returning
            # those free pages avoids mistaking allocator retention for live
            # cache pressure. Unsupported platforms simply skip this hint.
            s['last_allocator_trim']=now
            try:
                import ctypes
                trim=ctypes.CDLL(None).malloc_trim
                trim.argtypes=[ctypes.c_size_t]
                trim.restype=ctypes.c_int
                trim(0)
                metric('allocator_trim_calls')
            except (AttributeError,OSError):
                pass
        metric("memory_pressure_events")
    elif cache.limit<s["cache_limit"]:
        workers=max(1,settings.get('workers',1)) if settings.get('monitor') else 1
        headroom=max(0,int((target-usage)/workers))
        cache.limit=min(s["cache_limit"],cache.bytes+headroom)


def worker_state():
    s=_STATE
    cache=s['nulls']
    return dict(pid=os.getpid(),cache_limit=cache.limit,cache_bytes=cache.bytes,
        cache_models=len(cache.items),cache_refused=cache.refused,
        observed_usage=s.get('observed_usage'),observed_target=s.get('observed_target'))


def sampled_worker_state():
    now=time.monotonic()
    if now-_STATE.get('last_snapshot',0)<5:
        return None
    _STATE['last_snapshot']=now
    return worker_state()


def cache_charge(cache):
    if cache is None:
        return 1024
    seen=set()
    size=4096
    arrays=[v for k,v in cache.items() if k!="K" and isinstance(v,np.ndarray)]
    arrays.extend(v for v in vars(cache["decomp"]).values() if isinstance(v,np.ndarray))
    shared=[a for a in (_STATE.get('K'),getattr(_STATE.get('eigen'),'eigenvectors',None))
            if a is not None]
    for array in arrays:
        key=(array.__array_interface__["data"][0],array.nbytes)
        if any(np.shares_memory(array,other) for other in shared):
            continue
        if key not in seen:
            seen.add(key)
            size+=array.nbytes
    return size


def verify_product(product,expected):
    # Rehash only when the immutable file's size/mtime or expected digest changed.
    # Validation is never skipped on a worker's first access to a saved model.
    stamp=product.stat()
    signature=(stamp.st_size,stamp.st_mtime_ns,expected)
    verified=_STATE["verified"]
    key=str(product)
    if verified.get(key)!=signature:
        started=time.perf_counter()
        if digest(product)!=expected:
            raise ValueError(f"Corrupted null cache: {product}")
        metric("cache_hash_worker_seconds",time.perf_counter()-started)
        verified[key]=signature
        # Small verification metadata has a separate bounded allowance.
        while len(verified)*512>max(MIB,int(.01*_STATE["cache_limit"])):
            verified.popitem(last=False)
    verified.move_to_end(key)


def bits(ids):
    return np.unpackbits(_STATE["packed"][np.asarray(ids, dtype=int)], axis=1,
                         bitorder="little")[:, :_STATE["info"]["n"]]


def eligibility_summary(job):
    """Packed Hamming radius and necessary marginal eligibility; no null fit."""
    node,center,ids,minimum = job
    packed = _STATE['packed'][np.asarray(ids,dtype=int)]
    radius = int(_POPCOUNT[np.bitwise_xor(packed,_STATE['packed'][center])].sum(axis=1).max())
    counts = _POPCOUNT[packed].sum(axis=1)
    n = _STATE['info']['n']
    eligible = int(np.count_nonzero((counts>=2*minimum) & (counts<=n-2*minimum)))
    return node,radius,eligible


def similarity(job):
    """Bounded binary-phi comparisons; no dense all-pattern matrix."""
    representative, ids, threshold = job
    x = _STATE["packed"][representative]
    X = _STATE["packed"][ids]
    n, k = _STATE["info"]["n"], float(_POPCOUNT[x].sum())
    counts = _POPCOUNT[X].sum(axis=1).astype(float)
    overlap = _POPCOUNT[np.bitwise_and(X,x)].sum(axis=1).astype(float)
    rho = (n*overlap-counts*k)/np.sqrt(counts*(n-counts)*k*(n-k))
    return [int(i) for i, r in zip(ids, rho) if r > 0 and r*r+1e-12 >= threshold]


def farthest(job):
    ids, representatives = job
    X, R = bits(ids).astype(float), bits(representatives).astype(float)
    n = X.shape[1]
    k, m = X.sum(axis=1), R.sum(axis=1)
    rho = (n*(X @ R.T)-k[:, None]*m)/np.sqrt((k*(n-k))[:, None]*(m*(n-m)))
    nearest = rho.max(axis=1)
    order = np.lexsort((ids, nearest))
    for index in order:
        if ids[index] not in representatives:
            return float(nearest[index]), int(ids[index])
    return float("inf"), -1


def standardized_sum(ids):
    X=bits(ids).astype(float)
    n=X.shape[1]
    k=X.sum(axis=1)
    return ((X-k[:,None]/n)/np.sqrt(k*(n-k)/n)[:,None]).sum(axis=0)


def central(job):
    ids,total=job
    X=bits(ids).astype(float)
    n=X.shape[1]
    k=X.sum(axis=1)
    values=((X-k[:,None]/n)/np.sqrt(k*(n-k)/n)[:,None]) @ total
    order=np.lexsort((ids,-values))
    return -float(values[order[0]]),int(ids[order[0]])


def null_meta(pattern):
    return _STATE['work']/'null_models'/f'{pattern//1000:06d}'/f'null_{pattern:09d}.json'


def _view(value, pattern, canonical):
    cache,status=value
    if cache is not None and pattern!=canonical:
        cache=dict(cache,y=bits([pattern])[0].astype(float),mu=1-cache['mu'],_base=cache)
    return cache,status


def get_null(pattern, external=None, *, manifest=None):
    s=_STATE
    pressure_guard()
    if s.get('active_null') is not None and s['active_null'][0]==pattern:
        metric('null_job_reuse_hits')
        return s['active_null'][1]
    canonical=int(s['canonical'][pattern]) if s.get('canonical') is not None else pattern
    if canonical in s['nulls']:
        metric('null_cache_hits')
        value=s['nulls'].get(canonical)
    else:
        meta=null_meta(canonical)
        source,record=manifest if manifest is not None else (None,None)
        if manifest is None:
            source=meta if meta.exists() else Path(external) if external else None
        cache=None
        if source is not None:
            started=time.perf_counter()
            metric('null_cache_disk_loads')
            if record is None:
                record=json.loads(source.read_text())
            if record['array'] is not None:
                cache=null_cache.load(record,source,s['K'],s['eigen'],verify_product)
                if not np.array_equal(cache['y'],bits([canonical])[0]):
                    raise ValueError('Cached null response differs')
            if record['status']=='OK' and cache is None:
                raise ValueError('Successful null cache has no arrays')
            metric('cache_load_worker_seconds',time.perf_counter()-started)
        else:
            if s['settings']['require_cached_nulls']:
                raise ValueError(f'No cached null for pattern {canonical}')
            started=time.monotonic()
            if started-s['last_log']>25:
                print(f'[KOVAR worker {os.getpid()}] fitting response pattern {canonical}',flush=True)
                s['last_log']=started
            fit,cache=fit_null_glmm(canonical,bits([canonical])[0].astype(float),s['K'],
                                   kinship_eigensystem=s['eigen'])
            metric('new_null_fits')
            metric('null_fit_worker_seconds',time.monotonic()-started)
            record=dict(pattern=canonical,status=fit.status,fit_seconds=time.monotonic()-started,
                        iterations=fit.iterations,array=None)
            if fit.status!='OK':
                cache=None
        compact=s['settings'].get('null_cache_format','full')=='compact'
        rank=s['settings'].get('null_cache_rank',128)
        compatible=null_cache.compatible(record,compact,rank,s['info']['n'])
        if source is None or not compatible:
            started=time.perf_counter()
            if cache is not None and cache['decomp'].eigenvectors.shape[1]<min(rank,len(cache['y'])):
                null_cache.complete(cache,s['K'],s['eigen'])
            record=null_cache.save(meta.parent,canonical,cache,record,s['K'],s['eigen'],compact=compact,rank=rank,
                                   previous_record=record if source==meta else {})
            source=meta
            s.setdefault('inventory_updates',{})[canonical]=dict(pattern=canonical,record=record,source=str(meta),local=True)
            metric('cache_save_worker_seconds',time.perf_counter()-started)
        s['last_null_manifest']=(source,record)
        if cache is not None:
            cache['_pattern']=canonical
            cache['_inverse_spectrum']=1/(1+cache['tau']*cache['decomp'].eigenvalues)
        value=(cache,record['status'])
        s['nulls'].put(canonical,value,cache_charge(cache))
    value=_view(value,pattern,canonical)
    if 'active_null' in s:
        s['active_null']=(pattern,value)
    return value


def ensure_complete(cache):
    started=time.perf_counter()
    needed=cache['tau']!=0 and cache['decomp'].eigenvectors.shape[1]<len(cache['decomp'].eigenvalues)
    if needed and null_cache.complete(cache,_STATE['K'],_STATE['eigen']):
        metric('cache_reconstruct_worker_seconds',time.perf_counter()-started)
        metric('cache_reconstructions')
        cache.get('_base',cache)['_charge_dirty']=True
    base=cache.get('_base',cache)
    if needed or '_inverse_spectrum' not in base:
        base['_inverse_spectrum']=1/(1+cache['tau']*cache['decomp'].eigenvalues)
    cache['_inverse_spectrum']=base['_inverse_spectrum']


def prefit_profiled(job):
    """Fit/convert one canonical pattern; never keep a full prefit pool in RAM."""
    pattern,external=job
    s=_STATE
    before=dict(s['metrics'])
    meta=null_meta(pattern)
    source=meta if meta.exists() else Path(external) if external else None
    record=json.loads(source.read_text()) if source is not None else None
    compact=s['settings'].get('null_cache_format','full')=='compact'
    rank=s['settings'].get('null_cache_rank',128)
    ready=record is not None and null_cache.compatible(record,compact,rank,s['info']['n'])
    if ready and record.get('array'):
        product=source.parent/record['array']['name']
        try:
            size=product.stat().st_size
            ready=not record['array'].get('bytes') or size==record['array']['bytes']
        except FileNotFoundError:
            ready=False
        if not ready:
            raise ValueError(f'Missing/truncated saved null: {product}')
    if not ready:
        mapping=s.get('canonical')
        s['canonical']=None
        try:
            get_null(pattern,external,manifest=(source,record))
        finally:
            s['canonical']=mapping
        source,record=s.pop('last_null_manifest')
    # All validation/checksums occur on first actual numerical read. Inventory
    # resumes avoid rereading every already-compact array merely to count it.
    s['nulls'].items.clear()
    s['nulls'].bytes=0
    s.pop('inventory_updates',None)
    return dict(pattern=pattern,status=record['status'],mode=record.get('mode','failed'),
                record=record,source=str(source),local=source==meta,
                metrics={k:v-before.get(k,0) for k,v in s['metrics'].items()},worker_state=worker_state())


def adaptive_statistics(X, cache, criticals, ranks=(0, 8, 16, 32, 64, 128)):
    """Bound ordinary score statistics, requiring all requested cutoffs resolved.

    Outward roundoff margins prevent near-boundary decisions from being
    accepted prematurely. Singular/invalid cases use the existing exact core.
    """
    C, decomp = cache["C"], cache["decomp"]
    A = decomp.sqrt_w[:, None]*(X-C @ (cache["M_inv"] @ (cache["Sinv_C"].T @ X)))
    energy = np.sum(A*A, axis=0)
    U = X.T @ (cache["y"]-cache["mu"])
    d = cache.get("_inverse_spectrum")
    if d is None:
        d = 1/(1+cache["tau"]*decomp.eigenvalues)
    if not np.all(np.isfinite(d)) or np.any(d <= 0):
        raise ValueError("Invalid inverse covariance spectrum")
    width = len(U)
    lo, hi = np.zeros(width), np.full(width, np.inf)
    captured, projected = np.zeros(width), np.zeros(width)
    active, previous = np.ones(width, dtype=bool), 0
    ranks = sorted({min(len(d), int(rank)) for rank in ranks})
    for rank in ranks:
        ids = np.flatnonzero(active)
        if not len(ids):
            break
        if rank > previous and cache['tau']!=0:
            if rank>decomp.eigenvectors.shape[1]:
                ensure_complete(cache)
                d=cache['_inverse_spectrum']
            z = decomp.eigenvectors[:, previous:rank].T @ A[:, ids]
            captured[ids] += np.sum(d[previous:rank, None]*z*z, axis=0)
            projected[ids] += np.sum(z*z, axis=0)
        guard = 1e-12 + 1e-10*np.maximum(energy[ids], 1)
        remaining = energy[ids]-projected[ids]
        if np.any(remaining < -10*guard):
            raise ArithmeticError("Spectral energy discrepancy exceeds tolerance")
        remaining = np.maximum(remaining, 0)
        lower = captured[ids].copy()
        upper = lower.copy()
        if rank < len(d):
            lower += np.min(d[rank:])*remaining
            upper += np.max(d[rank:])*remaining
        lower = np.maximum(0, lower-guard)
        upper = np.maximum(0, upper+guard)
        lo[ids] = np.divide(U[ids]**2, upper, out=np.zeros(len(ids)), where=upper > 0)
        hi[ids] = np.divide(U[ids]**2, lower, out=np.full(len(ids), np.inf), where=lower > 0)
        resolved = lower > 1e-12
        for cutoff in criticals:
            target = np.asarray(cutoff)[ids] if np.ndim(cutoff) else cutoff
            resolved &= (hi[ids] < target) | (lo[ids] >= target)
        active[ids[resolved]] = False
        previous = rank
    exact_p = [None]*width
    statuses = ["BOUNDED"]*width
    pending = np.flatnonzero(active)
    if len(pending):
        ensure_complete(cache)
        started=time.perf_counter()
        for index, result in zip(pending, score_predictor_block(X[:, pending], cache, compute_spa_adjustment=False)):
            statuses[index] = result.status
            if result.status == "OK" and np.isfinite(result.score_z):
                lo[index] = hi[index] = result.score_z**2
                exact_p[index] = float(result.p_score)
        metric("exact_worker_seconds",time.perf_counter()-started)
        metric("exact_pairs",len(pending))
    return lo, hi, exact_p, statuses


def _score_chunk(job):
    response, predictors, external, screening = job
    s, n = _STATE, _STATE["info"]["n"]
    X, y = bits(predictors).T.astype(float), bits([response])[0].astype(float)
    overlap, counts = X.T @ y, X.sum(axis=0)
    cell = np.minimum.reduce([overlap, counts-overlap, y.sum()-overlap, n-counts-y.sum()+overlap])
    eligible = cell >= s["settings"]["min_cell_count"]
    if not screening and not eligible.all():
        # Final-only leaf work cannot produce a discovery from ineligible
        # pairs. Save the eligibility decision without fitting a useless null.
        result=[(int(u),response,0,-2,None,None,None,"INELIGIBLE_NOT_SCREENED")
                for u,ok in zip(predictors,eligible) if not ok]
        selected=[u for u,ok in zip(predictors,eligible) if ok]
        if selected:
            result.extend(_score_chunk((response,selected,external,False)))
        return result
    cache, status = get_null(response, external)
    if cache is None:
        return [(int(u), response, 0, 0 if ok else -2, None, None, None, status)
                for u, ok in zip(predictors, eligible)]
    gate, final = s["settings"]["gate_critical"], s["settings"]["final_critical"]
    # Ineligible pairs route the hierarchy, but can never become discoveries.
    cutoffs = [gate, np.where(eligible, final, gate)]
    if s["settings"]["exact"]:
        ensure_complete(cache)
        started=time.perf_counter()
        lo, hi, p, statuses = np.zeros(len(predictors)), np.full(len(predictors), np.inf), [], []
        for i, result in enumerate(score_predictor_block(X, cache, compute_spa_adjustment=False)):
            statuses.append(result.status)
            p.append(float(result.p_score) if result.status == "OK" else None)
            if result.status == "OK":
                lo[i] = hi[i] = result.score_z**2
        metric("exact_worker_seconds",time.perf_counter()-started)
        metric("exact_pairs",len(predictors))
    else:
        started=time.perf_counter()
        before=s["metrics"].get("exact_worker_seconds",0.)
        lo, hi, p, statuses = adaptive_statistics(X, cache, cutoffs)
        metric("bounds_worker_seconds",max(0.,time.perf_counter()-started-(s["metrics"].get("exact_worker_seconds",0.)-before)))
        metric("bound_pairs",len(predictors))
        metric("bound_decided_pairs",sum(status=='BOUNDED' for status in statuses))
    def decision(low, high, cutoff):
        return 1 if low >= cutoff else -1 if high < cutoff else 0
    return [(int(u), response, decision(l, h, gate), decision(l, h, final) if ok else -2,
             float(l), float(h) if np.isfinite(h) else None, value, status)
            for u, ok, l, h, value, status in zip(predictors, eligible, lo, hi, p, statuses)]


def score(job):
    response, predictors, external, screening = job
    width = _STATE["settings"]["batch_size"]
    result = []
    _STATE['active_null']=None
    try:
        for start in range(0,len(predictors),width):
            result.extend(_score_chunk((response,predictors[start:start+width],external,screening)))
    finally:
        active=_STATE.pop('active_null',None)
        if active and active[1][0] is not None:
            value,status=active[1]
            base=value.get('_base',value)
            key=base.get('_pattern')
            if base.pop('_charge_dirty',False) and key in _STATE['nulls']:
                _STATE['nulls'].put(key,(base,status),cache_charge(base))
    return result


def score_profiled(job):
    started=time.perf_counter()
    before=dict(_STATE.get("metrics",{}))
    old_evictions=_STATE["nulls"].evictions if "nulls" in _STATE else 0
    rows=score(job)
    metrics={key:value-before.get(key,0) for key,value in _STATE.get("metrics",{}).items()}
    metrics.update(worker_job_seconds=time.perf_counter()-started,jobs=1,pairs=len(rows))
    if "nulls" in _STATE:
        metrics["null_cache_evictions"]=_STATE["nulls"].evictions-old_evictions
        metrics["cache_peak_bytes_worker"]=_STATE["nulls"].peak_bytes
    return dict(rows=rows,metrics=metrics,worker_state=sampled_worker_state(),
                inventory=list(_STATE.pop('inventory_updates',{}).values()))


def beta(job):
    u, v = job
    if time.monotonic()-_STATE["last_log"]>25:
        print(f"[KOVAR worker {os.getpid()}] beta for pattern pair {u},{v}",flush=True)
        _STATE["last_log"]=time.monotonic()
    X = bits([u, v]).T.astype(float)
    overlap = np.sum(X[:, 0]*X[:, 1])
    a, b = X.sum(axis=0)
    if min(overlap, a-overlap, b-overlap, len(X)-a-b+overlap) == 0:
        return u, v, None, "SEPARATION_OR_MONOMORPHIC"
    fit, cache = fit_logistic_mixed(X[:, 1], _STATE["K"], covariates=X[:, :1],
                                  kinship_eigensystem=_STATE["eigen"])
    if not fit.converged or fit.status != "OK":
        return u, v, None, fit.status
    variance = cache["M_inv"][1, 1]
    value = float(fit.beta[1])
    if fit.n_weights_clipped or not np.isfinite([value, variance]).all() or variance <= 0:
        return u, v, None, "UNSTABLE_EFFECT"
    return u, v, value, "OK"
