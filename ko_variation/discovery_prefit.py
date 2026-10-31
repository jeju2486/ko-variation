"""Resumable, bounded pre-fitting of the complete unique response pool."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import time

import numpy as np

from . import discovery_score as kernels
from .discovery_execution import bounded_map
from .discovery_store import atomic_json
from .discovery_inventory import record_models


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
    db.execute('CREATE TABLE IF NOT EXISTS null_aliases(pid INTEGER PRIMARY KEY,canonical INTEGER)')
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
        db.execute('DROP TABLE IF EXISTS null_stage')
    atomic_json(args.out/'null_stage_summary.json',summary)
    print(f"[KOVAR] null stage complete: {summary['completed']:,} models; "
          f"{summary['complement_aliases']:,} complement aliases; statuses={counts}",flush=True)
