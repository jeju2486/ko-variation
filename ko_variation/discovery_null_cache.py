"""Compact numerical null caches; omitted directions are reconstructed on demand.

No fitted parameter is approximated. Spectral sketches retain the complete
eigenvalue spectrum, so existing outward variance bounds remain applicable.
"""
import os
from pathlib import Path

import numpy as np

from .discovery_store import atomic_json, digest
from .glmm import _WeightedEigen, _weighted_eigen

FIELDS = ('y','mu','weights','C','Sinv_C','M_inv')


def compatible(record, compact, rank, n):
    if record['status']!='OK': return True
    if record.get('format')!=2: return False
    mode=record.get('mode')
    if not compact: return mode=='full'
    return mode in ('zero','shared') or (mode=='full' and n<=rank) or (
        mode=='sketch' and record.get('rank',0)>=min(rank,n))


def complete(cache, K, eigen):
    """Materialize missing eigenvectors only for a genuine exact fallback."""
    d=cache['decomp']
    if cache['tau']==0 or d.eigenvectors.shape[1]==len(d.eigenvalues):
        return False
    full=_weighted_eigen(K,cache['weights'],eigen)
    order=np.argsort(full.eigenvalues)[::-1]
    # Mutate the shared decomposition: complement views reuse this result too.
    d.eigenvectors=np.ascontiguousarray(full.eigenvectors[:,order])
    d.eigenvalues=full.eigenvalues[order]
    return True


def load(record, meta, K, eigen, verify):
    product=Path(meta).parent/record['array']['name']
    verify(product,record['array']['sha256'])
    with np.load(product,allow_pickle=False) as saved:
        cache={key:saved[key] for key in FIELDS}
        cache['tau']=float(saved['tau'])
        n=len(cache['y'])
        mode=record.get('mode','full')
        sqrt_w=saved['sqrt_w']
        if mode=='zero':
            values,vectors=np.zeros(n),np.empty((n,0))
        elif mode=='shared':
            values=cache['weights'][0]*eigen.eigenvalues[::-1]
            vectors=eigen.eigenvectors[:,::-1]
        else:
            values,vectors=saved['eigenvalues'],saved['eigenvectors']
        cache['decomp']=_WeightedEigen(sqrt_w,values,vectors,float(saved['logdet_w_inverse']))
    return cache


def save(folder, pattern, cache, record, K, eigen, *, compact=True, rank=128, previous_record=None):
    """Publish data before its manifest; release superseded LOCAL arrays last."""
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    meta=folder/f'null_{pattern:09d}.json'
    previous=None
    if previous_record is not None:
        if previous_record.get('array'):
            previous=folder/previous_record['array']['name']
    elif meta.exists():
        import json
        old=json.loads(meta.read_text())
        if old.get('array'):
            previous=folder/old['array']['name']
    record=dict(record,pattern=pattern,format=2,array=None)
    if cache is not None:
        d=cache['decomp']
        if compact and cache['tau']==0:
            mode='zero'
        elif compact and np.all(cache['weights']==cache['weights'][0]):
            mode='shared'
        else:
            mode='sketch' if compact and rank<len(cache['y']) else 'full'
        if mode=='full':
            complete(cache,K,eigen)
        data={key:cache[key] for key in FIELDS}
        data.update(tau=cache['tau'],sqrt_w=d.sqrt_w,logdet_w_inverse=d.logdet_w_inverse)
        if mode not in ('zero','shared'):
            order=np.argsort(d.eigenvalues)[::-1]
            # Fits/legacy caches may use ascending eigenvectors. A loaded sketch
            # already has descending values; never index beyond its stored rank.
            if d.eigenvectors.shape[1]==len(order):
                vectors=d.eigenvectors[:,order]
            else:
                vectors=d.eigenvectors
            width=min(rank,vectors.shape[1]) if mode=='sketch' else len(order)
            data.update(eigenvalues=d.eigenvalues[order],
                        eigenvectors=np.ascontiguousarray(vectors[:,:width]))
        name=f'null_{pattern:09d}.{mode}{rank if mode=="sketch" else ""}.npz'
        target=folder/name
        temporary=target.with_suffix(f'.partial.{os.getpid()}')
        with temporary.open('wb') as stream:
            np.savez(stream,**data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary,target)
        record.update(mode=mode,rank=data.get('eigenvectors',np.empty((0,0))).shape[1],
                      array=dict(name=name,sha256=digest(target),bytes=target.stat().st_size))
    atomic_json(meta,record)
    current=record.get('array') or {}
    if previous is not None and previous!=folder/current.get('name',''):
        # Only the old array named by this local manifest is eligible for cleanup.
        if previous.parent.resolve()==folder.resolve() and previous.exists():
            previous.unlink()
    return record
