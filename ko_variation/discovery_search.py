"""Category construction, eligibility and resumable pair traversal."""
from collections import deque
import itertools
import math
from pathlib import Path
import time

import numpy as np

from . import coarse_screen
from . import discovery_nulls as nulls
from . import discovery_score as kernels
from .discovery_execution import batches, bounded_map, merge_metrics
from .discovery_store import get_meta, open_patterns, set_meta
from .memory_budget import ByteLRU, MIB


def impossible_block(n, minimum, radius_a, radius_b, distance, count_a, count_b, same=False):
    """Return a proof of ineligibility, not a covariation significance claim."""
    if minimum <= 0:
        return False
    if count_a < (2 if same else 1) or count_b < 1:
        return True
    lower = max(0, distance-radius_a-radius_b)
    upper = min(n, distance+radius_a+radius_b)
    # All four cells >= c requires 2c <= Hamming(A,B) <= n-2c.
    return upper < 2*minimum or lower > n-2*minimum


class Eligibility:
    def __init__(self, db, pool, args, info):
        self.db, self.pool, self.args = db, pool, args
        self.n, self.minimum = info['n'], args.min_cell_count
        self.cache = ByteLRU(min(8*MIB,max(MIB,int(args.memory_gb*(1<<30)*.005))))
        work = Path(args.tmp_dir) if args.tmp_dir else args.out/'.kovar_work'
        self.packed = open_patterns(work, info)

    def summaries(self, tasks):
        """Compute missing summaries in bounded worker batches and save on disk.

        Radius is the actual maximum distance to the fixed node anchor over ALL
        members, including members with insufficient marginal counts. No dense
        pair matrix or per-node member list is held by the coordinator.
        """
        result, missing = {}, []
        for node in sorted({node for task in tasks for node in task}):
            if node in self.cache:
                result[node] = self.cache.get(node)
                continue
            row = self.db.execute('SELECT center,radius,eligible_members FROM block_eligibility '
                                  'WHERE node=? AND min_cell=?', (node,self.minimum)).fetchone()
            if row is not None:
                result[node] = row
            else:
                center = self.db.execute('SELECT anchor FROM nodes WHERE id=?',(node,)).fetchone()[0]
                missing.append((node,center))
                result[node] = [center,0,0]
        def jobs():
            for node,center in missing:
                cursor = self.db.execute('SELECT pid FROM members WHERE node=? ORDER BY pid',(node,))
                while True:
                    rows = cursor.fetchmany(2048)
                    if not rows:
                        break
                    yield node,center,[row[0] for row in rows],self.minimum
        for node,radius,count in bounded_map(self.pool,kernels.eligibility_summary,jobs(),
                                              self.args.threads,ordered=False):
            result[node][1] = max(result[node][1],radius)
            result[node][2] += count
        if missing:
            with self.db:
                self.db.executemany('INSERT OR REPLACE INTO block_eligibility VALUES (?,?,?,?,?)',
                    ((node,self.minimum,*result[node]) for node,_ in missing))
        for node,summary in result.items():
            self.cache.put(node,tuple(summary),192)
        return result

    def impossible(self, a, b, summaries):
        center_a,radius_a,count_a = summaries[a]
        center_b,radius_b,count_b = summaries[b]
        distance = int(kernels._POPCOUNT[np.bitwise_xor(self.packed[center_a],
                                                       self.packed[center_b])].sum())
        return impossible_block(self.n,self.minimum,radius_a,radius_b,distance,
                                count_a,count_b,same=a==b)


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


def score_jobs(db, pairs, args, screening):
    grouped = {}
    for u, v in pairs:
        grouped.setdefault(v, []).append(u)
    external = {}
    if getattr(args, "_has_external_nulls", True):
        for block in batches(sorted(grouped), 400):
            marks = ",".join("?" for _ in block)
            external.update(db.execute(f"SELECT pid,path FROM external_nulls WHERE pid IN ({marks})", block))
            external.update(db.execute("SELECT a.pid,e.path FROM null_aliases a "
                f"JOIN external_nulls e ON a.canonical=e.pid WHERE a.pid IN ({marks})", block))
    return ((v, us, external.get(v), screening) for v, us in grouped.items())


def queried_count(db):
    count = get_meta(db, "queried_pairs_count")
    if count is None:
        count = db.execute("SELECT (SELECT count(*) FROM decisions) + "
                           "(SELECT count(*) FROM compact_rejections)").fetchone()[0]
    return count


def save_worker_profile(db, totals, states):
    saved = get_meta(db, "runtime_totals", {})
    merge_metrics(saved, totals)
    set_meta(db, "runtime_totals", saved)
    if states:
        current = get_meta(db, "worker_state", {})
        current.update(states)
        set_meta(db, "worker_state", current)


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
               'SELECT r.u,r.v,COALESCE(d.gate,CASE WHEN c.u IS NOT NULL THEN -1 END),'
               "COALESCE(d.final,c.final),COALESCE(d.status,CASE WHEN c.u IS NOT NULL THEN 'COMPACT' END) "
               'FROM requested r LEFT JOIN decisions d ON d.u=r.u AND d.v=r.v '
               'LEFT JOIN compact_rejections c ON c.u=r.u AND c.v=r.v')
        rows={(u,v):(gate,final,status) for u,v,gate,final,status in db.execute(query,[x for pair in block for x in pair])}
        for pair in block:
            gate,final,status=rows[pair]
            if status is None or screening and status=="INELIGIBLE_NOT_SCREENED":
                missing.append(pair)
                if status is None:
                    new_pairs.add(pair)
            else:
                known[pair]=(gate,final)
    jobs = score_jobs(db, missing, args, screening)
    totals=dict(driver_db_read_seconds=time.perf_counter()-read_started)
    pending=[]
    models=[]
    def flush():
        if not pending:
            return
        write_started=time.perf_counter()
        with db:
            count=queried_count(db)
            db.executemany('INSERT OR REPLACE INTO decisions VALUES (?,?,?,?,?,?,?,?)',pending)
            count+=sum((row[0],row[1]) in new_pairs for row in pending)
            set_meta(db,"queried_pairs_count",count)
            nulls.record_models(db,models)
        models.clear()
        pending.clear()
        merge_metrics(totals,dict(driver_db_write_seconds=time.perf_counter()-write_started))
    states={}
    for payload in bounded_map(pool,kernels.score_profiled,jobs,args.threads,ordered=False, response_index=0):
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
        save_worker_profile(db, totals, states)
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


def admit_coarse_tile(db, arrays, capacity, reps, pool, args, eligibility, *, fresh_gates=None):
    blocks, offset, active, next_right = coarse_screen.candidate_window(db, arrays, capacity, args.threads)
    done = offset >= len(arrays[1])
    if not blocks:
        if not done:
            raise ValueError("PC candidate cursor ended before the selected pairs were processed")
        return [], True
    started = time.perf_counter()
    summaries = eligibility.summaries(blocks) if eligibility else {}
    tested = {tuple(sorted((reps[a], reps[b]))): (a, b) for a, b in blocks
              if eligibility is None or not eligibility.impossible(a, b, summaries)}
    decisions, negatives, survivors, inventory = [], [], [], []
    totals, states = {}, {}
    jobs = score_jobs(db, sorted(tested, key=lambda pair: (pair[1], pair[0])), args, True)
    for payload in bounded_map(pool, kernels.score_profiled, jobs, args.threads,
                               ordered=False, response_index=0):
        merge_metrics(totals, payload["metrics"])
        inventory.extend(payload["inventory"])
        if payload["worker_state"]:
            states[str(payload["worker_state"]["pid"])] = payload["worker_state"]
        for row in payload["rows"]:
            u, v, gate, final = row[:4]
            if gate == -1:
                negatives.append((u, v, final))
            else:
                decisions.append(row)
                survivors.append(tested[u, v])
    survivors.sort()
    # Commit the tile and its cursor together, including when every block was rejected.
    writing = time.perf_counter()
    with db:
        count = queried_count(db)
        db.executemany("INSERT INTO compact_rejections VALUES (?,?,?)", negatives)
        db.executemany("INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?)", decisions)
        db.executemany("INSERT OR IGNORE INTO tasks(a,b) VALUES (?,?)", survivors)
        nulls.record_models(db, inventory)
        set_meta(db, "queried_pairs_count", count + len(tested))
        coarse_screen.save_candidate_cursor(db, offset, active, next_right)
        hierarchy = get_meta(db, "hierarchy_metrics", {})
        merge_metrics(hierarchy, dict(eligibility_pruned_cross_blocks=len(blocks)-len(tested),
                                      representative_pruned_blocks=len(negatives)))
        set_meta(db, "hierarchy_metrics", hierarchy)
        merge_metrics(totals, dict(coarse_tiles=1, coarse_tile_candidates=len(blocks),
                                  compact_rejections=len(negatives),
                                  fresh_gate_lookups_saved=len(decisions) if fresh_gates is not None else 0,
                                  driver_db_write_seconds=time.perf_counter()-writing,
                                  coordinator_batch_wall_seconds=time.perf_counter()-started))
        save_worker_profile(db, totals, states)
    if fresh_gates is not None:
        fresh_gates.update(((u, v), (gate, final)) for u, v, gate, final, *_ in decisions)
    return survivors, done


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
        filtered=getattr(args,'coarse_screen','none')=='pc'
        tiled=filtered and getattr(args, '_tiled_coarse', False)
        if filtered:
            arrays=coarse_screen.open_candidates(Path(args.tmp_dir) if args.tmp_dir else args.out/'.kovar_work',
                get_meta(db,'coarse_screen_summary')['selected_pairs'])
            if tiled:
                root_reps=dict(db.execute('SELECT r.node,r.pid FROM reps r JOIN nodes n ON n.id=r.node WHERE n.parent=0'))
        else:
            roots=[row[0] for row in db.execute('SELECT id FROM nodes WHERE parent=0 ORDER BY id')]
            ordered=[row[0] for row in db.execute('SELECT n.id FROM nodes n JOIN reps r ON r.node=n.id '
                                                 'WHERE n.parent=0 GROUP BY n.id ORDER BY max(r.pid),n.id')]
            old_position={node:i for i,node in enumerate(roots)}
        while True:
            # Reserve at least 3/4 of each frontier for queued refinements and
            # at most 1/4 for fresh coarse comparisons. Neither waits for the
            # other queue to finish. Scheduling/cursor updates are atomic.
            quota=max(1,frontier//4)
            tasks=db.execute('SELECT a,b FROM tasks WHERE done=0 ORDER BY a,b LIMIT ?',
                             (max(1,frontier-quota),)).fetchall()
            fresh_gates = {} if tiled and getattr(args, "_reuse_fresh_gates", True) else None
            if tiled:
                fresh, coarse_done = admit_coarse_tile(
                    db, arrays, frontier-len(tasks), root_reps, pool, args, eligibility,
                    fresh_gates=fresh_gates)
            elif filtered:
                fresh,coarse_done=coarse_screen.candidate_tasks(db,arrays,frontier-len(tasks),args.threads)
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
                        pairs = itertools.product(nodes.get(a)[2], nodes.get(b)[2])
                        if fresh_gates:
                            yield from ((u, v) for u, v in pairs
                                        if ((u, v) if u <= v else (v, u)) not in fresh_gates)
                        else:
                            yield from pairs
            gates = fresh_gates.copy() if fresh_gates else {}
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
