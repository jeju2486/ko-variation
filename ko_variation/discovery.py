"""KOVAR v0.9 unitig discovery without a SpydrPick candidate table.

The hierarchy is a heuristic search; spectral bounds certify only queried
ordinary-score decisions. All persistent work is bounded-batch and resumable.
"""
from __future__ import annotations

import argparse
from collections import deque
import itertools
import json
import math
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time

from .memory_budget import guard_blas_environment
guard_blas_environment(default_threads=1)

import numpy as np
import scipy
from scipy.stats import chi2
from threadpoolctl import threadpool_limits

from . import __version__, __discovery_revision__
from . import discovery_score as kernels
from .discovery_store import atomic_json, connect, digest, prepare, write_gzip, import_benchmark_nulls
from .discovery_execution import ResponsePool, bounded_map
from . import discovery_nulls as nulls
from . import coarse_screen
from .memory_budget import ByteLRU, detect_budget, plan_workers, MIB


def batches(values, size):
    source = iter(values)
    while True:
        batch = list(itertools.islice(source, size))
        if not batch:
            break
        yield batch


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fasta", type=Path, required=True, help="Unique sample-ID FASTA; A=0,C=1")
    p.add_argument("--tree", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--unitigs", type=Path, help="PAN-GWES zero-based unitig-ID/sequence file")
    p.add_argument("--paths", type=Path, help="PAN-GWES per-isolate graph path list")
    p.add_argument("--edges", type=Path, help="Global graph edges, for older unitig_distance versions")
    p.add_argument("--unitig-distance", help="unitig_distance executable; omit to defer distances/beta")
    p.add_argument("--kmer-length", type=int, default=61)
    p.add_argument("--exclude-samples", type=Path, help="One sample ID per line")
    p.add_argument("--tree-missing-samples", choices=["error", "drop"], default="error")
    p.add_argument("--min-maf", type=float, default=.05)
    p.add_argument("--min-cell-count", type=int, default=20, help="FINAL eligibility; proxies have no cell-count gate [20]")
    p.add_argument("--alpha", type=float, default=.05, help="Bonferroni family-wise alpha [.05]")
    p.add_argument("--bonferroni-tests", type=int, help="Frozen larger hypothesis family; never smaller than all pattern pairs")
    p.add_argument("--screen-p", type=float, default=.001, help="Hierarchy exploration only, not final alpha")
    p.add_argument("--search", choices=["hierarchy", "exhaustive"], default="hierarchy")
    p.add_argument('--coarse-screen',choices=['pc','none'],default='pc',help='Tree-PC screen before coarse KOVAR tasks [pc]; none retains the full coarse scan')
    p.add_argument('--screen-pcs',type=int,default=40)
    p.add_argument('--coarse-retain',type=float,default=.10,help='Global fraction of coarse pairs retained by absolute residual correlation [.10]')
    p.add_argument('--screen-block-size',type=int,default=256,help='Maximum residual-screen row block [256]; reduced automatically for memory')
    p.add_argument("--exact", action="store_true", help="Disable spectral bounds (benchmark/reference)")
    p.add_argument("--coarse-r2", type=float, default=.8)
    p.add_argument("--refine-r2", type=float, nargs="+", default=[.9, .95, .98, .99])
    p.add_argument("--leaf-size", type=int, default=32)
    p.add_argument("--rep-policy", choices=["single", "coverage"], default="single")
    p.add_argument("--max-reps", type=int, default=10)
    p.add_argument("--threads", type=int, default=1, help="Total CPU budget; process workers use one BLAS thread each")
    p.add_argument("--memory-gb", type=float, default=None, help="Optional GiB cap; auto-detect job allocation with headroom, otherwise conservative 4 GiB")
    p.add_argument("--tmp-dir", type=Path, help="Persistent work directory on scratch; must survive restart")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--distance-batch-size", type=int, default=5000)
    p.add_argument("--distal-bp", type=int, default=10000)
    p.add_argument("--no-beta", action="store_true", help="Keep significance/distance results without post-hoc refits")
    p.add_argument("--reuse-nulls-from", type=Path, action="append", default=[], help="Previous v0.9 output or combined benchmark output")
    p.add_argument('--reuse-prepared-from',type=Path,help='Copy compatible prepared data/hierarchy into this NEW output and borrow its null caches')
    p.add_argument("--require-cached-nulls", action="store_true")
    p.add_argument('--prefit-nulls',choices=['all','needed'],default='all',
                   help='Fit all unique response nulls before scoring [all]; needed retains on-demand fitting')
    p.add_argument('--prefit-only',action='store_true',help='Complete/resume null fitting and stop before pair testing')
    p.add_argument('--null-cache-format',choices=['compact','full'],default='compact')
    p.add_argument('--null-cache-rank',type=int,default=128,
                   help='Retained spectral directions; omitted directions use conservative bounds or exact reconstruction [128]')
    p.add_argument("--resume", action="store_true", help="Reuse compatible completed work")
    p.add_argument("--version", action="version", version=__version__)
    return p


def validate(args):
    for name in ("threads", "batch_size", "distance_batch_size", "leaf_size", "max_reps", "kmer_length",'null_cache_rank','screen_block_size'):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_','-')} must be positive")
    if args.memory_gb is not None and (not np.isfinite(args.memory_gb) or args.memory_gb < .25):
        raise ValueError("--memory-gb must be finite and >=0.25")
    if not 0 <= args.min_maf <= .5 or args.min_cell_count < 0 or args.distal_bp < 0:
        raise ValueError("Invalid frequency/cell-count/distance threshold")
    if not 0 < args.alpha < 1 or not 0 < args.screen_p < 1:
        raise ValueError("Alpha and screening P must lie in (0,1)")
    if not 0 < args.coarse_r2 < 1 or any(not args.coarse_r2 < t <= 1 for t in args.refine_r2):
        raise ValueError("Refinement thresholds must exceed the coarse threshold and be <=1")
    args.refine_r2 = sorted(set(args.refine_r2))
    if args.screen_pcs<0 or not 0<args.coarse_retain<=1:
        raise ValueError('PC count must be nonnegative and --coarse-retain must be in (0,1]')
    if args.search=='exhaustive': args.coarse_screen='none'
    if args.coarse_screen=='pc' and args.rep_policy!='single':
        raise ValueError('PC coarse screening currently requires --rep-policy single; use --coarse-screen none for coverage representatives')
    if args.prefit_only:
        args.prefit_nulls='all'
    if args.unitig_distance and not (args.unitigs and args.paths):
        raise ValueError("Distance calculation needs --unitigs and --paths")


def identity(args):
    # Execution resources can change on resume; statistical settings cannot.
    omitted = {"threads", "memory_gb", "out", "tmp_dir", "resume", "reuse_nulls_from", "require_cached_nulls",
               "unitig_distance", "unitigs", "paths", "edges", "kmer_length", "distal_bp", "no_beta", "distance_batch_size", "batch_size",
               'prefit_nulls','prefit_only','null_cache_format','null_cache_rank','screen_block_size','reuse_prepared_from'}
    if args.coarse_screen=='none' or args.search=='exhaustive': omitted.update(('coarse_screen','screen_pcs','coarse_retain'))
    settings = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
                if k not in omitted and not k.startswith('_')}
    inputs = {k: digest(getattr(args, k)) for k in ("fasta", "tree", "exclude_samples") if getattr(args, k)}
    code = {name: digest(Path(__file__).with_name(name)) for name in
            ("discovery.py", "discovery_store.py", "discovery_score.py", "discovery_execution.py", "memory_budget.py",
             "discovery_eligibility.py", "discovery_nulls.py", "coarse_screen.py", "null_identity.py", "glmm.py", "kinship.py")}
    from .null_identity import null_source_hash
    return dict(version=__version__, execution_revision=__discovery_revision__,inputs=inputs, settings=settings, code=code,
                numpy=np.__version__, scipy=scipy.__version__,null_model_source_sha256=null_source_hash())


def schema(db):
    db.executescript("""
    CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS nodes(id INTEGER PRIMARY KEY,parent INTEGER,level INTEGER,
      anchor INTEGER,size INTEGER DEFAULT 0,ready INTEGER DEFAULT 0);
    CREATE INDEX IF NOT EXISTS node_parent ON nodes(parent,id);
    CREATE TABLE IF NOT EXISTS members(node INTEGER,pid INTEGER,PRIMARY KEY(node,pid)) WITHOUT ROWID;
    CREATE TABLE IF NOT EXISTS assigned(parent INTEGER,pid INTEGER,child INTEGER,PRIMARY KEY(parent,pid)) WITHOUT ROWID;
    CREATE TABLE IF NOT EXISTS reps(node INTEGER,pid INTEGER,PRIMARY KEY(node,pid)) WITHOUT ROWID;
    CREATE TABLE IF NOT EXISTS tasks(a INTEGER,b INTEGER,done INTEGER DEFAULT 0,PRIMARY KEY(a,b)) WITHOUT ROWID;
    CREATE INDEX IF NOT EXISTS pending_tasks ON tasks(done,a,b);
    CREATE TABLE IF NOT EXISTS block_eligibility(node INTEGER,min_cell INTEGER,center INTEGER,
      radius INTEGER,eligible_members INTEGER,PRIMARY KEY(node,min_cell)) WITHOUT ROWID;
    CREATE TABLE IF NOT EXISTS decisions(u INTEGER,v INTEGER,gate INTEGER,final INTEGER,
      t_low REAL,t_high REAL,p REAL,status TEXT,PRIMARY KEY(u,v)) WITHOUT ROWID;
    CREATE INDEX IF NOT EXISTS discovery_hits ON decisions(final,u,v);
    CREATE TABLE IF NOT EXISTS external_nulls(pid INTEGER PRIMARY KEY,path TEXT);
    CREATE TABLE IF NOT EXISTS null_aliases(pid INTEGER PRIMARY KEY,canonical INTEGER);
    CREATE TABLE IF NOT EXISTS null_inventory(pid INTEGER PRIMARY KEY,status TEXT,mode TEXT,
      fit_seconds REAL,source TEXT,local INTEGER,signature TEXT);
    CREATE TABLE IF NOT EXISTS beta(u INTEGER,v INTEGER,value REAL,status TEXT,PRIMARY KEY(u,v)) WITHOUT ROWID;
    CREATE TABLE IF NOT EXISTS distance(u INTEGER,v INTEGER,pu INTEGER,pv INTEGER,
      distance REAL,connected INTEGER,status TEXT DEFAULT 'PENDING',PRIMARY KEY(u,v)) WITHOUT ROWID;
    CREATE INDEX IF NOT EXISTS distance_pending ON distance(status,u,v);
    CREATE INDEX IF NOT EXISTS distance_pattern ON distance(pu,pv,distance);
    """)


def get_meta(db, key, default=None):
    row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else default


def set_meta(db, key, value):
    db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, json.dumps(value, allow_nan=False)))


def partition(db, node, threshold, pool, args, n):
    """Greedy fixed-anchor star partition. Every membership passes directly.

    Assignments are persisted as each bounded batch completes. A representative
    can be resumed without repeating assignments from an interrupted pass.
    """
    last_log=time.monotonic()
    while True:
        active = get_meta(db, f"partition:{node}")
        lower_seed = get_meta(db,f"partition_next:{node}",0)
        row = db.execute("SELECT m.pid FROM members m LEFT JOIN assigned a ON a.parent=m.node AND a.pid=m.pid "
                         "WHERE m.node=? AND m.pid>=? AND a.pid IS NULL ORDER BY m.pid LIMIT 1", (node,lower_seed)).fetchone()
        if row is None and active is None:
            break
        representative = active[0] if active else row[0]
        # Recover an unfinished representative pass after an interruption.
        child = db.execute("SELECT id FROM nodes WHERE parent=? AND anchor=?", (node, representative)).fetchone()
        if child:
            child = child[0]
        else:
            level = db.execute("SELECT level FROM nodes WHERE id=?", (node,)).fetchone()[0]+1
            with db:
                child = db.execute("INSERT INTO nodes(parent,level,anchor) VALUES (?,?,?)", (node, level, representative)).lastrowid
        with db:
            set_meta(db, f"partition:{node}", [representative,child])
        k = db.execute("SELECT present FROM patterns WHERE id=?",(representative,)).fetchone()[0]
        # Necessary positive-phi margin bounds: impossible membership is never
        # sent to a worker. This is exact, unlike representative hit pruning.
        lo = max(1,math.ceil(n*threshold*k/(n-k+threshold*k)-1e-10))
        hi = min(n-1,math.floor(n*k/(threshold*(n-k)+k)+1e-10))
        cursor = db.execute("SELECT m.pid FROM patterns p JOIN members m ON m.pid=p.id "
                            "LEFT JOIN assigned a ON a.parent=m.node AND a.pid=m.pid "
                            "WHERE m.node=? AND a.pid IS NULL AND p.present BETWEEN ? AND ? ORDER BY m.pid", (node,lo,hi))
        jobs = ((representative, [r[0] for r in batch], threshold) for batch in batches(cursor, args.batch_size))
        for matches in bounded_map(pool, kernels.similarity, jobs, args.threads):
            with db:
                db.executemany("INSERT OR IGNORE INTO assigned VALUES (?,?,?)", ((node, p, child) for p in matches))
            if time.monotonic()-last_log>25:
                print(f"[KOVAR] category node={node}; representative={representative}; assigning batches",flush=True)
                last_log=time.monotonic()
        with db:
            db.execute("INSERT OR IGNORE INTO members SELECT child,pid FROM assigned WHERE parent=? AND child=?", (node, child))
            db.execute("UPDATE nodes SET size=(SELECT count(*) FROM members WHERE node=?) WHERE id=?", (child, child))
            db.execute("DELETE FROM meta WHERE key=?", (f"partition:{node}",))
            set_meta(db,f"partition_next:{node}",representative+1)
        # Progress once per representative, also visible for very large pools.
        if child % 100 == 0:
            print(f"[KOVAR] categories: node={node}; latest child={child}", flush=True)
    with db:
        db.execute("UPDATE nodes SET ready=1 WHERE id=?", (node,))


def make_hierarchy(db, info, pool, args):
    if get_meta(db, "hierarchy_complete"):
        return
    with db:
        db.execute("INSERT OR IGNORE INTO nodes(id,parent,level,anchor,size) VALUES (0,NULL,-1,0,?)", (info["patterns"],))
        db.execute("INSERT OR IGNORE INTO members SELECT 0,id FROM patterns")
    thresholds = [args.coarse_r2]+args.refine_r2
    while True:
        row = db.execute("SELECT id,level,size FROM nodes WHERE ready=0 ORDER BY id LIMIT 1").fetchone()
        if row is None:
            break
        node, level, size = row
        if node and size <= args.leaf_size:
            with db:
                db.execute("UPDATE nodes SET ready=1 WHERE id=?", (node,))
        elif level+1 < len(thresholds):
            partition(db, node, thresholds[level+1], pool, args,info["n"])
        else:
            # Final chunks are computational leaves, not biological categories.
            with db:
                for batch in batches(db.execute("SELECT pid FROM members WHERE node=? ORDER BY pid", (node,)), args.leaf_size):
                    child = db.execute("INSERT INTO nodes(parent,level,anchor,size,ready) VALUES (?,?,?,?,1)",
                                       (node, level+1, batch[0][0], len(batch))).lastrowid
                    db.executemany("INSERT INTO members VALUES (?,?)", ((child, p[0]) for p in batch))
                db.execute("UPDATE nodes SET ready=1 WHERE id=?", (node,))
    # Mean correlation to all members equals a dot product with their summed
    # standardized vectors. A central observed representative therefore costs
    # O(n*members), not a dense members-by-members correlation matrix.
    for node, anchor, size in db.execute("SELECT id,anchor,size FROM nodes WHERE id>0 ORDER BY id"):
        width = 1 if args.rep_policy == "single" else max(1, min(math.ceil(size*.01), args.max_reps))
        selected = [r[0] for r in db.execute("SELECT pid FROM reps WHERE node=? ORDER BY pid", (node,))]
        if not selected and size==1:
            with db:
                db.execute("INSERT INTO reps VALUES (?,?)",(node,anchor))
            continue
        if not selected:
            total=np.zeros(info["n"])
            jobs=([r[0] for r in batch] for batch in batches(db.execute("SELECT pid FROM members WHERE node=? ORDER BY pid",(node,)),args.batch_size))
            for vector in bounded_map(pool,kernels.standardized_sum,jobs,args.threads):
                total+=vector
            jobs=(([r[0] for r in batch],total) for batch in batches(db.execute("SELECT pid FROM members WHERE node=? ORDER BY pid",(node,)),args.batch_size))
            _,representative=min(bounded_map(pool,kernels.central,jobs,args.threads))
            selected=[representative]
            with db:
                db.execute("INSERT INTO reps VALUES (?,?)",(node,representative))
        while len(selected) < width:
            cursor = db.execute("SELECT pid FROM members WHERE node=? ORDER BY pid", (node,))
            jobs = (([r[0] for r in batch], selected) for batch in batches(cursor, args.batch_size))
            pick = min(bounded_map(pool, kernels.farthest, jobs, args.threads))
            if pick[1] < 0:
                break
            selected.append(pick[1])
            with db:
                db.execute("INSERT INTO reps VALUES (?,?)", (node, pick[1]))
    with db:
        set_meta(db, "hierarchy_complete", True)


def merge_metrics(target, values):
    for key,value in values.items():
        target[key]=max(target.get(key,0),value) if key=="cache_peak_bytes_worker" else target.get(key,0)+value


def evaluate(db, pairs, pool, args, screening=False, *, return_mapping=False):
    """Bulk-read decisions, coalesce responses, and commit bounded result blocks."""
    started=time.perf_counter()
    requested=sorted({tuple(sorted((int(u),int(v)))) for u,v in pairs if u!=v},key=lambda pair:(pair[1],pair[0]))
    if not requested:
        return {} if return_mapping else []
    known,missing,new_pairs={},[],set()
    read_started=time.perf_counter()
    for block in batches(requested,400):
        values=','.join(['(?,?)']*len(block))
        query=(f'WITH requested(u,v) AS (VALUES {values}) '
               'SELECT r.u,r.v,d.gate,d.final,d.status FROM requested r '
               'LEFT JOIN decisions d ON d.u=r.u AND d.v=r.v')
        rows={(u,v):(gate,final,status) for u,v,gate,final,status in db.execute(query,[x for pair in block for x in pair])}
        for pair in block:
            gate,final,status=rows[pair]
            if status is None or screening and status=="INELIGIBLE_NOT_SCREENED":
                missing.append(pair)
                if status is None:
                    new_pairs.add(pair)
            else:
                known[pair]=(gate,final)
    external={}
    responses=sorted({v for _,v in missing})
    for block in batches(responses if getattr(args,'_has_external_nulls',True) else [],400):
        marks=','.join('?' for _ in block)
        external.update(db.execute(f'SELECT pid,path FROM external_nulls WHERE pid IN ({marks})',block))
        external.update(db.execute(f'SELECT a.pid,e.path FROM null_aliases a JOIN external_nulls e '
                                   f'ON e.pid=a.canonical WHERE a.pid IN ({marks})',block))
    totals=dict(driver_db_read_seconds=time.perf_counter()-read_started)
    pending=[]
    models=[]
    def flush():
        if not pending:
            return
        write_started=time.perf_counter()
        with db:
            count=get_meta(db,"queried_pairs_count")
            if count is None:
                count=db.execute('SELECT count(*) FROM decisions').fetchone()[0]
            db.executemany('INSERT OR REPLACE INTO decisions VALUES (?,?,?,?,?,?,?,?)',pending)
            count+=sum((row[0],row[1]) in new_pairs for row in pending)
            set_meta(db,"queried_pairs_count",count)
            nulls.record_models(db,models)
        models.clear()
        pending.clear()
        merge_metrics(totals,dict(driver_db_write_seconds=time.perf_counter()-write_started))
    def jobs():
        for response,group in itertools.groupby(missing,key=lambda pair:pair[1]):
            yield response,[pair[0] for pair in group],external.get(response),screening
    states={}
    for payload in bounded_map(pool,kernels.score_profiled,jobs(),args.threads,ordered=False):
        result=payload['rows']
        models.extend(payload.get('inventory',[]))
        merge_metrics(totals,payload['metrics'])
        if payload.get('worker_state'):
            states[str(payload['worker_state']['pid'])]=payload['worker_state']
        pending.extend(result)
        known.update(((row[0],row[1]),(row[2],row[3])) for row in result)
        if len(pending)>=4096:
            flush()
    flush()
    merge_metrics(totals,dict(coordinator_batch_wall_seconds=time.perf_counter()-started))
    with db:
        saved=get_meta(db,'runtime_totals',{})
        merge_metrics(saved,totals)
        set_meta(db,'runtime_totals',saved)
        if states:
            current=get_meta(db,'worker_state',{})
            current.update(states)
            set_meta(db,'worker_state',current)
    return known if return_mapping else [known[pair] for pair in requested]


class NodeCache:
    def __init__(self,db,limit):
        self.db,self.cache=db,ByteLRU(limit)

    def get(self,node):
        if node in self.cache:
            return self.cache.get(node)
        size=self.db.execute('SELECT size FROM nodes WHERE id=?',(node,)).fetchone()[0]
        children=tuple(r[0] for r in self.db.execute('SELECT id FROM nodes WHERE parent=? ORDER BY id',(node,)))
        reps=tuple(r[0] for r in self.db.execute('SELECT pid FROM reps WHERE node=? ORDER BY pid',(node,)))
        members=() if children else tuple(r[0] for r in self.db.execute('SELECT pid FROM members WHERE node=? ORDER BY pid',(node,)))
        value=(size,children,reps,members)
        self.cache.put(node,value,256+40*(len(children)+len(reps)+len(members)))
        return value


def coarse_tasks(db, roots, ordered, capacity, old_position=None, workers=1):
    """Bounded concurrent response columns, excluding a frozen legacy prefix.

    Several response columns advance together to occupy the worker pool. Each
    column retains its null across batches; no dense root-pair matrix exists.
    Old queued/completed tasks and their row-prefix remain untouched.
    """
    n=len(roots)
    legacy=get_meta(db,'legacy_root_stop')
    if legacy is None:
        legacy=get_meta(db,'root_cursor',[0,1])
        i,j=legacy
        with db:
            set_meta(db,'legacy_root_stop',legacy)
            set_meta(db,'coarse_scheduled_count',i*(2*n-i-1)//2+j-i-1 if n>1 else 0)
    active=get_meta(db,'response_coarse_columns',[])
    next_right=get_meta(db,'response_coarse_next',1)
    if next_right>=n and not active: return [],True
    if old_position is None: old_position={node:i for i,node in enumerate(roots)}
    fresh=[]
    scanned=0
    quantum=max(1,min(256,capacity//max(1,workers)))
    while len(fresh)<capacity and scanned<max(4096,capacity*8):
        while len(active)<workers and next_right<n:
            active.append([next_right,0]);next_right+=1
        if not active: break
        right,left=active.pop(0)
        accepted=0
        while left<right and accepted<quantum and len(fresh)<capacity and scanned<max(4096,capacity*8):
            a,b=ordered[left],ordered[right]
            i,j=sorted((old_position[a],old_position[b]))
            if not (i<legacy[0] or (i==legacy[0] and j<legacy[1])):
                fresh.append(tuple(sorted((a,b))))
                accepted+=1
            left+=1;scanned+=1
        if left<right: active.append([right,left])
    with db:
        db.executemany('INSERT OR IGNORE INTO tasks(a,b) VALUES (?,?)',fresh)
        set_meta(db,'response_coarse_columns',active)
        set_meta(db,'response_coarse_next',next_right)
        set_meta(db,'coarse_scheduled_count',get_meta(db,'coarse_scheduled_count',0)+len(fresh))
    return fresh,next_right>=n and not active


def search(db, info, pool, args):
    if get_meta(db,'search_complete'):
        return
    started,last=time.monotonic(),0.
    plan=getattr(args,'_plan',{})
    buffer_limit=plan.get('query_buffer_pairs',8192)
    frontier=plan.get('frontier_tasks',max(128,args.threads*64))
    nodes=NodeCache(db,min(64*MIB,max(MIB,int(args.memory_gb*(1<<30)*.03))))
    eligibility=None
    screen_info=get_meta(db,'coarse_screen_summary')
    if args.search=='hierarchy' and getattr(args,'min_cell_count',0)>0:
        from .discovery_eligibility import Eligibility
        eligibility=Eligibility(db,pool,args,info)
    def progress():
        nonlocal last
        if time.monotonic()-last<25:
            return
        count=get_meta(db,'queried_pairs_count',0)
        runtime=get_meta(db,'runtime_totals',{})
        hierarchy=get_meta(db,'hierarchy_metrics',{})
        print(f"[KOVAR] search: {count:,} queried pattern pairs; {time.monotonic()-started:.0f}s this invocation; "
              f"new_nulls={runtime.get('new_null_fits',0):,}; cache_hits={runtime.get('null_cache_hits',0):,}; "
              f"disk_loads={runtime.get('null_cache_disk_loads',0):,}; "
              f"ineligible_blocks={hierarchy.get('eligibility_pruned_internal_blocks',0)+hierarchy.get('eligibility_pruned_cross_blocks',0):,}; "
              f"proxy_pruned_blocks={hierarchy.get('representative_pruned_blocks',0):,}; "
              f"bounded_pairs={runtime.get('bound_decided_pairs',0):,}; exact_pairs={runtime.get('exact_pairs',0):,}"
              +(f"; coarse_opened={get_meta(db,'coarse_scheduled_count',0):,}/{screen_info['selected_pairs']:,}" if screen_info else ''),flush=True)
        last=time.monotonic()
    if args.search=='exhaustive':
        response,offset=get_meta(db,'exhaustive_cursor',[1,0])
        while response<info['patterns']:
            query=[]
            while response<info['patterns'] and len(query)<buffer_limit:
                stop=min(response,offset+buffer_limit-len(query))
                query.extend((u,response) for u in range(offset,stop))
                offset=stop
                if offset>=response:
                    response,offset=response+1,0
            evaluate(db,query,pool,args)
            with db:
                set_meta(db,'exhaustive_cursor',[response,offset])
            progress()
    else:
        if not get_meta(db,'search_initialized'):
            with db:
                # Singleton internal blocks contain no pair to test.
                db.execute('INSERT OR IGNORE INTO tasks SELECT id,id,0 FROM nodes WHERE parent=0 AND size>1')
                set_meta(db,'root_cursor',[0,1])
                set_meta(db,'search_initialized',True)
        roots=[row[0] for row in db.execute('SELECT id FROM nodes WHERE parent=0 ORDER BY id')]
        ordered=[row[0] for row in db.execute('SELECT n.id FROM nodes n JOIN reps r ON r.node=n.id '
                                             'WHERE n.parent=0 GROUP BY n.id ORDER BY max(r.pid),n.id')]
        old_position={node:i for i,node in enumerate(roots)}
        filtered=getattr(args,'coarse_screen','none')=='pc'
        arrays=coarse_screen.open_candidates(Path(args.tmp_dir) if args.tmp_dir else args.out/'.kovar_work',
            get_meta(db,'coarse_screen_summary')['selected_pairs']) if filtered else None
        while True:
            # Reserve at least 3/4 of each frontier for queued refinements and
            # at most 1/4 for fresh coarse comparisons. Neither waits for the
            # other queue to finish. Scheduling/cursor updates are atomic.
            quota=max(1,frontier//4)
            tasks=db.execute('SELECT a,b FROM tasks WHERE done=0 ORDER BY a,b LIMIT ?',
                             (max(1,frontier-quota),)).fetchall()
            if filtered:
                fresh,coarse_done=coarse_screen.candidate_tasks(db,arrays,frontier-len(tasks),get_meta,set_meta,args.threads)
            else:
                fresh,coarse_done=coarse_tasks(db,roots,ordered,frontier-len(tasks),old_position,args.threads)
            tasks.extend(fresh)
            if not tasks:
                if coarse_done: break
                progress()
                continue
            summaries=eligibility.summaries(tasks) if eligibility else {}
            pruned={(a,b) for a,b in tasks if eligibility and eligibility.impossible(a,b,summaries)}
            def gate_pairs():
                for a,b in tasks:
                    if a!=b and (a,b) not in pruned:
                        yield from itertools.product(nodes.get(a)[2],nodes.get(b)[2])
            gates={}
            for block in batches(gate_pairs(),buffer_limit):
                gates.update(evaluate(db,block,pool,args,screening=True,return_mapping=True))
            leaf_queries,leaf_done,transitions=[],[],[]
            metrics={}
            def finish_leaves():
                if leaf_queries:
                    evaluate(db,leaf_queries,pool,args)
                if leaf_done:
                    with db:
                        db.executemany('UPDATE tasks SET done=1 WHERE a=? AND b=?',leaf_done)
                leaf_queries.clear()
                leaf_done.clear()
            for a,b in tasks:
                if (a,b) in pruned:
                    key='eligibility_pruned_internal_blocks' if a==b else 'eligibility_pruned_cross_blocks'
                    metrics[key]=metrics.get(key,0)+1
                    transitions.append((a,b,()))
                    continue
                sa,ca,ra,ma=nodes.get(a)
                sb,cb,rb,mb=nodes.get(b)
                expand=a==b or any(gates[tuple(sorted(pair))][0]!=-1 for pair in itertools.product(ra,rb))
                if not expand:
                    metrics['representative_pruned_blocks']=metrics.get('representative_pruned_blocks',0)+1
                children=()
                if expand and not ca and not cb:
                    # Coalesce terminal work across blocks before scheduling it
                    # by response. The buffer remains bounded independently of
                    # the number of categories or total genomic pair count.
                    pairs=itertools.combinations(ma,2) if a==b else itertools.product(ma,mb)
                    for block in batches(pairs,buffer_limit):
                        if len(leaf_queries)+len(block)>buffer_limit:
                            finish_leaves()
                        leaf_queries.extend(block)
                        if len(leaf_queries)>=buffer_limit:
                            evaluate(db,leaf_queries,pool,args)
                            leaf_queries.clear()
                    leaf_done.append((a,b))
                    continue
                if expand:
                    if a==b:
                        children=itertools.combinations_with_replacement(ca,2)
                    elif ca and (not cb or sa>=sb):
                        children=itertools.product(ca,(b,))
                    else:
                        children=itertools.product((a,),cb)
                transitions.append((a,b,children))
            finish_leaves()
            # Preserve atomic child scheduling/task completion while avoiding
            # one FULL-synchronous transaction for every tiny traversal block.
            with db:
                for a,b,children in transitions:
                    db.executemany('INSERT OR IGNORE INTO tasks(a,b) VALUES (?,?)',(tuple(sorted(pair)) for pair in children))
                    db.execute('UPDATE tasks SET done=1 WHERE a=? AND b=?',(a,b))
                saved=get_meta(db,'hierarchy_metrics',{})
                merge_metrics(saved,metrics)
                set_meta(db,'hierarchy_metrics',saved)
            progress()
    with db:
        set_meta(db,'search_complete',True)


def import_nulls(db, work, info, args, *, current_identity=None):
    """Reuse v0.9 caches, or explicitly verify the former benchmark preparation.

    NPZ preparation components are extracted to disk before mapping, rather
    than loading the entire former unitig pool into RAM.
    """
    if not args.reuse_nulls_from:
        return
    current_identity=current_identity or identity(args)
    sources,seen = deque(args.reuse_nulls_from),set()
    while sources:
        supplied = sources.popleft()
        supplied = supplied.resolve()
        if supplied in seen:
            continue
        seen.add(supplied)
        if get_meta(db, "imported:"+str(supplied)):
            # A prior import may have committed before its borrowed-source
            # imports finished. Follow those links again on resume.
            source_root=supplied.parent if supplied.name=="null_models" else supplied
            if (source_root/"run_identity.json").exists():
                source_identity=json.loads((source_root/"run_identity.json").read_text())
                sources.extend(Path(item["path"]) for item in source_identity.get("reused_null_sources",[]))
            continue
        root = supplied.parent if supplied.name == "null_models" else supplied
        prior_identity=json.loads((root/"run_identity.json").read_text())
        if str(prior_identity.get("version","")).startswith("0.9."):
            if any(prior_identity.get(k)!=current_identity.get(k) for k in ("numpy","scipy","null_model_source_sha256")):
                raise ValueError("Prior v0.9 null-fitting environment/source differs")
            summary = json.loads((root/"run_summary.json").read_text()) if (root/"run_summary.json").exists() else {}
            prior_work = Path(summary.get("work_directory",prior_identity.get("work_directory",root/".kovar_work")))
            prior_info=json.loads((prior_work/"prepared.json").read_text())
            if (prior_info["samples"]!=info["samples"] or
                    any(prior_info["products"][k]!=info["products"][k] for k in ("patterns.bin","K.npy"))):
                raise ValueError("Prior prepared patterns/covariance differ")
            records=((json.loads(meta.read_text())["pattern"],str(meta))
                     for meta in (prior_work/"null_models").glob("*/null_*.json"))
            for batch in batches(records,1000):
                with db:
                    db.executemany("INSERT OR IGNORE INTO external_nulls VALUES (?,?)",batch)
            borrowed=sqlite3.connect((prior_work/"work.sqlite").resolve().as_uri()+"?mode=ro",uri=True)
            try:
                for batch in batches(borrowed.execute("SELECT pid,path FROM external_nulls"),1000):
                    with db:
                        db.executemany("INSERT OR IGNORE INTO external_nulls VALUES (?,?)",batch)
            finally:
                borrowed.close()
        else:
            sources.extend(Path(item['path']) for item in prior_identity.get('reused_null_sources',[]))
            import_benchmark_nulls(db,work,info,root,prior_identity)
        with db:
            set_meta(db,"imported:"+str(supplied),True)


def distance_and_beta(db, work, pool, args):
    if not args.unitig_distance:
        return
    binary = shutil.which(args.unitig_distance)
    if binary is None:
        raise ValueError(f"unitig_distance executable not found: {args.unitig_distance}")
    help_result = subprocess.run([binary,"--help"],capture_output=True,text=True,check=False)
    help_text = (help_result.stdout or "")+(help_result.stderr or "")
    # Freeze distance inputs separately, so distances can be added on a resume.
    graph = {name: digest(getattr(args,name)) for name in ("unitigs","paths","edges") if getattr(args,name)}
    # .paths entries are resolved exactly as the executable will see them.
    graph["path_graphs"] = []
    with args.paths.open() as stream:
        for line in stream:
            if line.strip():
                path = Path(line.strip()).resolve()
                graph["path_graphs"].append([str(path),digest(path)])
    graph.update(binary=digest(binary),k=args.kmer_length,distal_bp=args.distal_bp)
    old = get_meta(db,"distance_identity")
    if old is not None and old != graph:
        raise ValueError("Distance inputs/settings changed; use a new output")
    with db:
        set_meta(db,"distance_identity",graph)
    if not get_meta(db,"distance_expanded"):
        with db:
            db.execute("INSERT OR IGNORE INTO distance(u,v,pu,pv) "
                       "SELECT min(a.id,b.id),max(a.id,b.id),d.u,d.v FROM decisions d "
                       "JOIN unitigs a ON a.pattern=d.u JOIN unitigs b ON b.pattern=d.v WHERE d.final=1")
            set_meta(db,"distance_expanded",True)
    directory = work/"distance"
    directory.mkdir(exist_ok=True)
    while True:
        rows = db.execute("SELECT u,v FROM distance WHERE status='PENDING' ORDER BY u,v LIMIT ?",(args.distance_batch_size,)).fetchall()
        if not rows:
            break
        import uuid
        stem = directory/("batch_"+uuid.uuid4().hex)
        query = stem.with_suffix(".queries")
        with query.open("w") as stream:
            for u,v in rows:
                stream.write(f"{u} {v} -1 0 0\n")
        command = [binary,"--unitigs-file",str(args.unitigs.resolve()),"--sgg-paths-file",str(args.paths.resolve()),
                   "--k-mer-length",str(args.kmer_length),"--queries-file",str(query.resolve()),
                   "--threads",str(args.threads),"--output-stem",str(stem.resolve())]
        if "--edges-file" in help_text:
            if not args.edges:
                raise ValueError("This unitig_distance version requires --edges")
            command += ["--edges-file",str(args.edges.resolve())]
        if "--run-sggs-only" in help_text:
            command += ["--run-sggs-only"]
        if "--memory" in help_text:
            # Keep headroom for the coordinator and idle numerical workers.
            command += ["--memory",f"{max(1,int(args.memory_gb*1024*.4))}M"]
        log_path = stem.with_suffix(".log")
        print(f"[KOVAR] distance batch: {len(rows):,} pairs; log: {log_path}",flush=True)
        with log_path.open("w") as stream:
            subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
        outputs = [p for p in directory.glob(stem.name+".ud*") if p.is_file()]
        if len(outputs) != 1:
            raise ValueError(f"Expected one zero-based distance output, found {outputs}; see {log_path}")
        expected = set(rows)
        found = set()
        with db, outputs[0].open() as stream:
            for line in stream:
                fields = line.split()
                if not fields or fields[0].startswith("#"):
                    continue
                if len(fields) < 9:
                    raise ValueError("Expected PAN-GWES distance output with min_distance in column 8")
                u,v = sorted((int(fields[0]),int(fields[1])))
                if (u,v) not in expected or (u,v) in found:
                    raise ValueError("Unexpected/duplicate distance pair")
                found.add((u,v))
                count, distance = int(fields[5]),float(fields[7])
                valid = count > 0 and np.isfinite(distance) and distance >= 0
                status = "DISTAL" if valid and distance > args.distal_bp else "LOCAL" if valid else "UNKNOWN"
                db.execute("UPDATE distance SET distance=?,connected=?,status=? WHERE u=? AND v=?",
                           (distance if valid else None,count,status,u,v))
            if found != expected:
                raise ValueError("Distance output did not cover the complete query batch")
        for product in (query,outputs[0],log_path):
            product.unlink()
        print(f"[KOVAR] distance: committed {len(rows):,} unitig pairs",flush=True)
    if not args.no_beta:
        cursor = db.execute("SELECT DISTINCT d.pu,d.pv FROM distance d LEFT JOIN beta b ON b.u=d.pu AND b.v=d.pv "
                            "WHERE d.status='DISTAL' AND b.u IS NULL ORDER BY d.pv,d.pu")
        jobs = ((u,v) for u,v in cursor)
        completed,last=0,time.monotonic()
        for row in bounded_map(pool,kernels.beta,jobs,args.threads,ordered=False):
            with db:
                db.execute("INSERT OR IGNORE INTO beta VALUES (?,?,?,?)",row)
            completed+=1
            if time.monotonic()-last>25:
                print(f"[KOVAR] beta: {completed:,} pattern pairs committed this invocation",flush=True)
                last=time.monotonic()


def export(db, info, work, args, total_seconds):
    args.out.mkdir(exist_ok=True,parents=True)
    write_gzip(args.out/"unitigs.tsv.gz", ["unitig_id","pattern_id","present","n_samples","sequence_source"],
               ((u,p,k,info["n"],str(args.unitigs.resolve()) if args.unitigs else "")
                for u,p,k in db.execute("SELECT id,pattern,present FROM unitigs ORDER BY id")))
    def rows():
        query = """SELECT min(a.id,b.id),max(a.id,b.id),d.u,d.v,d.p,d.t_low,d.t_high,
          x.distance,x.connected,coalesce(x.status,'NOT_CALCULATED'),beta.value,beta.status
          FROM decisions d JOIN unitigs a ON a.pattern=d.u JOIN unitigs b ON b.pattern=d.v
          LEFT JOIN distance x ON x.u=min(a.id,b.id) AND x.v=max(a.id,b.id)
          LEFT JOIN beta ON beta.u=d.u AND beta.v=d.v WHERE d.final=1"""
        for u,v,pu,pv,p,lo,hi,distance,count,status,beta,beta_status in db.execute(query):
            # Never present statistic/P-value bounds as an exact P-value.
            yield (u,v,pu,pv,"SIGNIFICANT",p,float(chi2.sf(hi,1)) if hi is not None else None,
                   float(chi2.sf(lo,1)) if lo is not None else None,distance,count,status,
                   beta if status=="DISTAL" else None,
                   beta_status if status=="DISTAL" and beta_status else "NOT_REQUESTED" if status=="DISTAL" else "SKIPPED_"+status)
    write_gzip(args.out/"associations.tsv.gz",
        ["u","v","predictor_pattern","response_pattern","significance","p_exact","p_lower","p_upper",
         "min_distance","connected_genomes","distance_status","beta","beta_status"],rows())
    counts = {str(state):count for state,count in db.execute("SELECT final,count(*) FROM decisions GROUP BY final")}
    family = args.bonferroni_tests or math.comb(info["patterns"],2)
    summary = dict(version=__version__,execution_revision=__discovery_revision__,work_directory=str(work.resolve()),
        samples=info["n"],excluded_samples=info["excluded"],original_unitigs=info["loci"],
        unique_patterns=info["patterns"],bonferroni_tests=family,alpha=args.alpha,
        raw_p_threshold=args.alpha/family,screen_p=args.screen_p,
        final_min_cell_count=args.min_cell_count,spa="off",search=args.search,
        representative_pruning_is_heuristic=args.search=="hierarchy",counts_by_final_decision=counts,
        decisions={"1":"significant","-1":"nonsignificant","0":"unresolved","-2":"ineligible"},
        within_exact_pattern_pairs="deterministic redundancy; not assigned a score-test P-value",
        statistic_orientation="lower pattern ID predictor; higher pattern ID response",
        original_ids="zero-based FASTA columns, never renumbered by filtering",
        threads=args.threads,memory_gb=args.memory_gb,seconds_this_invocation=total_seconds,
        distance_status_counts=dict(db.execute("SELECT status,count(*) FROM distance GROUP BY status")),
        beta_status_counts=dict(db.execute("SELECT status,count(*) FROM beta GROUP BY status")),
        queried_pairs=sum(counts.values()),total_pattern_pairs=math.comb(info["patterns"],2),
        unqueried_pairs=math.comb(info["patterns"],2)-sum(counts.values()))
    null_statuses,null_seconds=nulls.local_summary(db,work)
    summary.update(saved_local_null_statuses=null_statuses,
                   saved_local_null_fit_seconds=null_seconds,
                   available_imported_nulls=db.execute("SELECT count(*) FROM external_nulls").fetchone()[0])
    summary['runtime_totals']=get_meta(db,'runtime_totals',{})
    summary['prefit_runtime_totals']=get_meta(db,'prefit_runtime_totals',{})
    summary['null_stage_summary']=get_meta(db,'null_stage_summary',{})
    summary['worker_state']=get_meta(db,'worker_state',{})
    summary['hierarchy_metrics']=get_meta(db,'hierarchy_metrics',{})
    summary['block_eligibility_pruning']='necessary marginal counts and anchor-radius Hamming bounds; categories retained'
    summary['memory_budget']=getattr(args,'_memory_info',None)
    summary['execution_plan']=getattr(args,'_plan',None)
    summary['coarse_screen']=get_meta(db,'coarse_screen_summary')
    summary['reused_prepared_from']=get_meta(db,'reused_prepared_from')
    atomic_json(args.out/"run_summary.json",summary)


def main(argv=None):
    args = parser().parse_args(argv)
    validate(args)
    started = time.monotonic()
    args._memory_info=detect_budget(args.memory_gb,args.threads)
    args.memory_gb=args._memory_info['working_gb']
    args.out = args.out.resolve()
    args.out.mkdir(parents=True,exist_ok=True)
    work = args.tmp_dir.resolve() if args.tmp_dir else args.out/".kovar_work"
    output_identity=args.out/"run_identity.json"
    if output_identity.exists():
        existing=json.loads(output_identity.read_text())
        if existing.get("work_directory") and Path(existing["work_directory"]).resolve()!=work.resolve():
            raise ValueError("Output belongs to another work directory; restore the original --tmp-dir")
    elif any(args.out.iterdir()):
        raise ValueError("Nonempty output is not an existing discovery run; choose a new --out")
    work.mkdir(parents=True,exist_ok=True)
    # An OS-released SQLite transaction lock prevents concurrent coordinators,
    # including on shared HPC storage. No stale PID lock needs manual removal.
    lock = sqlite3.connect(work/"coordinator.lock.sqlite",timeout=0)
    lock.execute("CREATE TABLE IF NOT EXISTS lock(id INTEGER)")
    try:
        lock.execute("BEGIN EXCLUSIVE")
    except sqlite3.OperationalError as error:
        raise ValueError("Another coordinator is using this work directory") from error
    try:
        expected = identity(args)
        manifest = work/"run_identity.json"
        if manifest.exists():
            if not args.resume:
                raise ValueError("Existing work requires --resume")
            from .null_identity import migrate_identity
            if migrate_identity(json.loads(manifest.read_text()),expected,manifest):
                print('[KOVAR] verified released checkpoint migrated; saved patterns, hierarchy, decisions and nulls retained',flush=True)
        else:
            atomic_json(manifest,expected)
        atomic_json(args.out/"run_identity.json",dict(expected,work_directory=str(work)))
        db = connect(work/"work.sqlite",min(64,args.memory_gb*1024*.05))
        try:
            schema(db)
            if get_meta(db,'queried_pairs_count') is None:
                with db:
                    set_meta(db,'queried_pairs_count',db.execute('SELECT count(*) FROM decisions').fetchone()[0])
            with threadpool_limits(limits=args.threads):
                if args.reuse_prepared_from:
                    from .discovery_store import reuse_prepared
                    reuse_prepared(args,db,work,expected)
                info = prepare(args,db,work)
            if args.reuse_prepared_from and args.reuse_prepared_from not in args.reuse_nulls_from:
                args.reuse_nulls_from.append(args.reuse_prepared_from)
            family = math.comb(info["patterns"],2)
            if args.bonferroni_tests is not None and args.bonferroni_tests < family:
                raise ValueError("Bonferroni family cannot be smaller than all unique-pattern pairs")
            family = args.bonferroni_tests or family
            if args.screen_p < args.alpha/family:
                raise ValueError("Screening gate must be at least as permissive as final significance")
            import_nulls(db,work,info,args,current_identity=expected)
            args._has_external_nulls=db.execute('SELECT 1 FROM external_nulls LIMIT 1').fetchone() is not None
            n=info['n']
            args._plan=plan_workers(args._memory_info,n,args.threads,args.batch_size)
            with db:
                set_meta(db,'memory_budget',args._memory_info)
                set_meta(db,'resource_plan',args._plan)
            settings = dict(min_cell_count=args.min_cell_count,gate_critical=float(chi2.isf(args.screen_p,1)),
                            final_critical=float(chi2.isf(args.alpha/family,1)),exact=args.exact,
                            require_cached_nulls=args.require_cached_nulls,batch_size=args.batch_size,
                            null_cache_format=args.null_cache_format,null_cache_rank=args.null_cache_rank,
                            workers=args.threads,**args._plan)
            print(f'[KOVAR] execution revision: {__discovery_revision__}; '
                  f'null stage={args.prefit_nulls}; cache={args.null_cache_format}',flush=True)
            print(f"[KOVAR {__version__}] {info['patterns']:,} patterns; {family:,} hypotheses; "
                  f"final P <= {args.alpha/family:.6g}; workers={args.threads}; "
                  f"cache/worker={args._plan['cache_bytes_per_worker']/(1<<30):.3f} GiB",flush=True)
            print(f"[KOVAR] memory target={args.memory_gb:.3f} GiB; source={args._memory_info['limiting_source']}; "
                  f"bounded query buffer={args._plan['query_buffer_pairs']:,} pairs",flush=True)
            estimate=8*n*min(args.null_cache_rank,n)+12*n*8 if args.null_cache_format=='compact' else 12*n*n
            print("[KOVAR] numerical null-cache estimate before complement/shared/zero savings: "
                  f"{info['patterns']*estimate/(1<<30):.1f} GiB; "
                  "pair checkpoints and expanded unitig results require additional disk space",flush=True)
            if not get_meta(db,'search_complete') or args.prefit_only:
                nulls.prefit(db,work,info,args,settings,get_meta,set_meta,merge_metrics)
                if args.prefit_only:
                    return 0
            if args.prefit_nulls=='all':
                settings['require_cached_nulls']=True
            with db:
                set_meta(db,'worker_state',{})
                set_meta(db,'active_stage','pair_search')
            with ResponsePool(args.threads,kernels.initialize,(str(work),info,settings)) as pool:
                if args.search=="hierarchy":
                    make_hierarchy(db,info,pool,args)
                    if args.coarse_screen=='pc':
                        coarse_screen.prepare_screen(db,work,info,args,get_meta,set_meta)
                        with db: set_meta(db,'active_stage','pair_search')
                search(db,info,pool,args)
            # Release every score-worker cache before the external graph tool
            # gets its own memory budget. Refit workers start lazily afterwards.
            if args.unitig_distance:
                with db: set_meta(db,'active_stage','distance_and_beta')
                with ResponsePool(args.threads,kernels.initialize,(str(work),info,settings)) as pool:
                    distance_and_beta(db,work,pool,args)
            export(db,info,work,args,time.monotonic()-started)
            with db: set_meta(db,'active_stage','complete')
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            db.close()
    finally:
        lock.rollback()
        lock.close()
    print(f"[KOVAR] complete: {args.out}",flush=True)
    return 0


if __name__ == "__main__":
    main()
