"""Null model lifecycle: bounded prefit, numerical storage and restart inventory.

The coordinator owns SQLite inventory/progress. Workers publish numerical files
atomically; first numerical reads verify saved products. Statistical fitting and
score calculations remain in glmm.py and discovery_score.py.
"""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import time

import numpy as np

from .discovery_execution import bounded_map
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


def load(record, meta, eigen, verify):
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


def record_models(db, rows, signature=None):
    """Called inside the same coordinator transaction as progress/decisions."""
    db.executemany('INSERT OR REPLACE INTO null_inventory VALUES (?,?,?,?,?,?,?)',
        ((r['pattern'],r['record']['status'],r['record'].get('mode','failed'),
          r['record'].get('fit_seconds',0.),r['source'],int(r['local']),
          signature) for r in rows))


def local_summary(db, work):
    """Migrate old/on-demand manifests once, then aggregate without filesystem scans."""
    initialized=db.execute("SELECT value FROM meta WHERE key='null_inventory_seeded'").fetchone()
    if not initialized:
        pending=[]
        for source in (work/'null_models').glob('*/null_*.json'):
            record=json.loads(source.read_text())
            pending.append((record['pattern'],record['status'],record.get('mode','failed'),
                record.get('fit_seconds',0.),str(source),1,None))
            if len(pending)>=1000:
                with db:
                    db.executemany('INSERT OR IGNORE INTO null_inventory VALUES (?,?,?,?,?,?,?)',pending)
                pending.clear()
        with db:
            db.executemany('INSERT OR IGNORE INTO null_inventory VALUES (?,?,?,?,?,?,?)',pending)
            db.execute("INSERT OR REPLACE INTO meta VALUES ('null_inventory_seeded','true')")
    counts=dict(db.execute('SELECT status,count(*) FROM null_inventory WHERE local=1 GROUP BY status'))
    seconds=db.execute('SELECT coalesce(sum(fit_seconds),0.) FROM null_inventory WHERE local=1').fetchone()[0]
    return counts,seconds


def publish_aliases(db, work, aliases):
    path=work/'null_canonical.npy'
    temporary=work/'null_canonical.partial.npy'
    np.save(temporary,aliases)
    os.replace(temporary,path)
    stamp=path.stat()
    with db:
        db.execute("INSERT OR REPLACE INTO meta VALUES ('null_aliases_published',?)",
                   (json.dumps([stamp.st_size,stamp.st_mtime_ns]),))


def aliases_for_pool(db, work, info, *, reuse_complements=True):
    """Preserve native saved fits; alias only responses without their own fit."""
    if db.execute('SELECT count(*) FROM null_aliases').fetchone()[0]==info['patterns']:
        path=work/'null_canonical.npy'
        published=db.execute("SELECT value FROM meta WHERE key='null_aliases_published'").fetchone()
        stamp=path.stat() if path.exists() else None
        if stamp and published and json.loads(published[0])==[stamp.st_size,stamp.st_mtime_ns]:
            return np.load(path,mmap_mode='r')
        aliases=np.fromiter((r[0] for r in db.execute('SELECT canonical FROM null_aliases ORDER BY pid')),
                            dtype=np.int64,count=info['patterns'])
        publish_aliases(db,work,aliases)
        return aliases
    existing=np.zeros(info['patterns'],dtype=bool)
    for path in (work/'null_models').glob('*/null_*.json'):
        pid=int(path.stem.split('_')[1])
        if 0<=pid<len(existing): existing[pid]=True
    for pid,path in db.execute('SELECT pid,path FROM external_nulls'):
        if os.path.isfile(path): existing[pid]=True
    aliases=np.arange(info['patterns'],dtype=np.int64)
    last=time.monotonic()
    if reuse_complements:
        for pid,raw in db.execute('SELECT id,bits FROM patterns ORDER BY id'):
            complement=bytearray(value^255 for value in raw)
            if info['n']%8:
                complement[-1]&=(1<<(info['n']%8))-1
            found=db.execute('SELECT id FROM patterns WHERE bits=?',(bytes(complement),)).fetchone()
            if found and found[0]>pid:
                other=found[0]
                if not existing[pid]: aliases[pid]=other if existing[other] else pid
                if not existing[other]: aliases[other]=pid
            if time.monotonic()-last>25:
                print(f'[KOVAR] null inventory: {pid+1:,}/{len(aliases):,} patterns',flush=True)
                last=time.monotonic()
    with db:
        db.execute('DELETE FROM null_aliases')
        db.execute("DELETE FROM meta WHERE key='null_aliases_published'")
        db.executemany('INSERT INTO null_aliases VALUES (?,?)',enumerate(map(int,aliases)))
    publish_aliases(db,work,aliases)
    return aliases


def prefit(db, work, info, args, settings, get_meta, set_meta, merge_metrics):
    from . import discovery_score as kernels
    # Inputs, cohort, null-fitting source and numerical environment were already
    # verified by the run identity. Only cache representation can change here.
    signature=json.dumps([args.null_cache_format,args.null_cache_rank,info['patterns']],separators=(',',':'))
    aliases=aliases_for_pool(db,work,info,reuse_complements=args.prefit_nulls=='all')
    if args.prefit_nulls!='all':
        return
    if get_meta(db,'null_prefit_complete')==signature:
        if not (args.out/'null_stage_summary.json').exists():
            atomic_json(args.out/'null_stage_summary.json',get_meta(db,'null_stage_summary'))
        print('[KOVAR] reusing completed null stage; numerical caches verified on first use',flush=True)
        return
    started=time.monotonic()
    summary={'stage':'null_fitting','response_patterns':info['patterns'],'completed':0}
    with db:
        set_meta(db,'active_stage','null_fitting')
    def run():
        pending=[]
        totals={}
        states={}
        completed=0
        last=time.monotonic()
        query=('SELECT a.pid,e.path FROM null_aliases a '
               'LEFT JOIN external_nulls e ON e.pid=a.pid '
               'LEFT JOIN null_inventory i ON i.pid=a.pid '
               'WHERE a.pid=a.canonical AND (i.signature IS NULL OR i.signature!=?)')
        count=db.execute('SELECT count(*) FROM ('+query+')',(signature,)).fetchone()[0]
        if not count:
            return
        def flush():
            if not pending: return
            with db:
                record_models(db,pending,signature)
                saved=get_meta(db,'prefit_runtime_totals',{})
                merge_metrics(saved,totals)
                set_meta(db,'prefit_runtime_totals',saved)
                set_meta(db,'worker_state',states)
                set_meta(db,'null_stage_progress',dict(checked=completed,total=count,
                         seconds=time.monotonic()-started))
            pending.clear();totals.clear()
        def jobs():
            # One streaming join replaces one path lookup per pattern. Keep a
            # stable reader while the coordinator commits completed inventory.
            path=db.execute('PRAGMA database_list').fetchone()[2]
            reader=sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True)
            try:
                reader.execute('BEGIN')
                for pid,external in reader.execute(query,(signature,)):
                    yield pid,external
            finally:
                reader.close()
        # Dedicated dynamic process queue for independent fits. Scoring later
        # uses response affinity; no nested BLAS threads exceed --threads.
        with ProcessPoolExecutor(max_workers=args.threads,mp_context=multiprocessing.get_context('spawn'),
                initializer=kernels.initialize,initargs=(str(work),info,settings)) as pool:
            for payload in bounded_map(pool,kernels.prefit_profiled,jobs(),args.threads,ordered=False):
                completed+=1
                pending.append(payload)
                merge_metrics(totals,payload['metrics'])
                states[str(payload['worker_state']['pid'])]=payload['worker_state']
                if len(pending)>=256 or time.monotonic()-last>25:
                    flush()
                    print(f'[KOVAR] null fitting: {completed:,}/{count:,} unfinished canonical patterns checked; '
                          f'{time.monotonic()-started:.0f}s this stage',flush=True)
                    last=time.monotonic()
            flush()
    run()
    # A failed canonical fit must not suppress a potentially successful native
    # complement fit. Restore its own response and fit it independently once.
    failures={pid for pid,status in db.execute('SELECT pid,status FROM null_inventory') if status!='OK'}
    retry=[pid for pid,canonical in enumerate(aliases) if pid!=canonical and int(canonical) in failures]
    if retry:
        aliases=np.array(aliases,copy=True)
        for pid in retry: aliases[pid]=pid
        with db:
            db.execute("DELETE FROM meta WHERE key='null_aliases_published'")
            db.executemany('UPDATE null_aliases SET canonical=pid WHERE pid=?',((p,) for p in retry))
        publish_aliases(db,work,aliases)
        run()
    counts=dict(db.execute('SELECT status,count(*) FROM null_inventory GROUP BY status'))
    modes=dict(db.execute('SELECT mode,count(*) FROM null_inventory GROUP BY mode'))
    summary.update(stage='null_fitting_complete',completed=sum(counts.values()),
        canonical_models=int(np.count_nonzero(aliases==np.arange(len(aliases)))),
        complement_aliases=int(np.count_nonzero(aliases!=np.arange(len(aliases)))),
        statuses=counts,cache_modes=modes,seconds_this_invocation=time.monotonic()-started)
    with db:
        set_meta(db,'null_stage_summary',summary)
        set_meta(db,'null_prefit_complete',signature)
        set_meta(db,'null_inventory_seeded',True)
    atomic_json(args.out/'null_stage_summary.json',summary)
    print(f"[KOVAR] null stage complete: {summary['completed']:,} models; "
          f"{summary['complement_aliases']:,} complement aliases; statuses={counts}",flush=True)
