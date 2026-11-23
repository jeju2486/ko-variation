"""KOVAR v0.9 unitig discovery without a SpydrPick candidate table.

The hierarchy is a heuristic search; spectral bounds certify only queried
ordinary-score decisions. All persistent work is bounded-batch and resumable.
"""
from __future__ import annotations

import argparse
from collections import deque
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
from .discovery_store import (atomic_json, connect, digest, get_meta, import_benchmark_nulls,
                              localize_graph_paths, prepare, prepare_pangwes, schema, set_meta,
                              write_gzip)
from .discovery_execution import ResponsePool, batches, bounded_map, merge_metrics, process_pool
from .discovery_search import make_hierarchy, queried_count, search
from . import discovery_nulls as nulls
from . import coarse_screen
from .memory_budget import detect_budget, plan_workers
from .null_identity import check_identity, null_source_hash


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fasta", type=Path, required=True, help="Unique sample-ID FASTA; A=0,C=1")
    p.add_argument("--tree", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--prepare-pangwes", type=Path, nargs=5,
                   metavar=("ASSEMBLIES", "GFA", "METADATA", "PREPARED_FASTA", "PREPARED_TREE"),
                   help="Use raw --fasta and --tree to produce sample-labelled discovery inputs")
    p.add_argument("--prepare-only", action="store_true", help="Prepare PAN-GWES inputs and stop")
    p.add_argument("--local-graph-paths", type=Path, help="Write an absolute, localized copy of --paths")
    p.add_argument("--graph-path-rebase", nargs=2, metavar=("OLD_PREFIX", "NEW_ROOT"), action="append",
                   default=[], help="Replace an old graph path prefix when writing --local-graph-paths")
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
    p.add_argument('--null-backend',choices=['dense','tree'],default='dense',
                   help='Tree working algebra with dense fallback; opt in pending full real-data validation')
    p.add_argument('--null-cache-rank',type=int,default=128,
                   help='Retained spectral directions; omitted directions use conservative bounds or exact reconstruction [128]')
    p.add_argument("--resume", action="store_true", help="Reuse compatible completed work")
    p.add_argument("--resume-if-present", action="store_true", help="Resume automatically if a checkpoint exists")
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
    if args.prepare_only and (not args.prepare_pangwes or args.prefit_only):
        raise ValueError("--prepare-only requires --prepare-pangwes and cannot be combined with --prefit-only")
    if (args.local_graph_paths and not args.paths) or (args.graph_path_rebase and not args.local_graph_paths):
        raise ValueError("Graph localization requires --paths and --local-graph-paths")


def identity(args):
    # Execution resources can change on resume; statistical settings cannot.
    omitted = {"threads", "memory_gb", "out", "tmp_dir", "resume", "reuse_nulls_from", "require_cached_nulls",
               "resume_if_present", "prepare_pangwes", "prepare_only", "local_graph_paths", "graph_path_rebase",
               "unitig_distance", "unitigs", "paths", "edges", "kmer_length", "distal_bp", "no_beta", "distance_batch_size", "batch_size",
               'prefit_nulls','prefit_only','null_cache_format','null_cache_rank','screen_block_size','reuse_prepared_from'}
    if args.coarse_screen=='none' or args.search=='exhaustive': omitted.update(('coarse_screen','screen_pcs','coarse_retain'))
    settings = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
                if k not in omitted and not k.startswith('_')}
    inputs = {k: digest(getattr(args, k)) for k in ("fasta", "tree", "exclude_samples") if getattr(args, k)}
    code = {name: digest(Path(__file__).with_name(name)) for name in
            ("discovery.py", "discovery_store.py", "discovery_score.py", "discovery_execution.py", "memory_budget.py",
             "discovery_search.py", "discovery_nulls.py", "coarse_screen.py", "null_identity.py", "glmm.py", "kinship.py", "tree_null.py", "tree_score.py")}
    return dict(version=__version__, execution_revision=__discovery_revision__,inputs=inputs, settings=settings, code=code,
                numpy=np.__version__, scipy=scipy.__version__,null_model_source_sha256=null_source_hash())


def prefit(db, work, info, args, settings):
    # Inputs, cohort, null-fitting source and numerical environment were already
    # verified by the run identity. Only cache representation can change here.
    signature=json.dumps([args.null_cache_format,args.null_cache_rank,info['patterns']],separators=(',',':'))
    aliases=nulls.aliases_for_pool(db,work,info,reuse_complements=args.prefit_nulls=='all')
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
                nulls.record_models(db,pending,signature)
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
        with process_pool(args.threads, kernels.initialize, (str(work), info, settings)) as pool:
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
        nulls.publish_aliases(db,work,aliases)
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
            if prior_identity.get('settings',{}).get('null_backend','dense')!=args.null_backend:
                raise ValueError('Prior null backend differs; use matching caches or fit fresh nulls')
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
    reclassify = old is not None and old != graph
    if reclassify and ({k:v for k,v in old.items() if k!='distal_bp'} !=
                       {k:v for k,v in graph.items() if k!='distal_bp'}):
        raise ValueError("Distance inputs changed; use a new output")
    with db:
        if reclassify:
            # The cutoff selects beta eligibility, not the association test or
            # physical distance. Keep pending/unknown rows and all saved fits.
            changed = db.execute("""UPDATE distance SET status=CASE
                WHEN distance>? THEN 'DISTAL' ELSE 'LOCAL' END
                WHERE status IN ('LOCAL','DISTAL') AND status!=CASE
                WHEN distance>? THEN 'DISTAL' ELSE 'LOCAL' END""",
                (args.distal_bp,args.distal_bp)).rowcount
            history = get_meta(db,'distance_threshold_history',[])
            history.append(dict(previous_bp=old['distal_bp'],distal_bp=args.distal_bp,
                                reclassified_unitig_pairs=changed))
            set_meta(db,'distance_threshold_history',history)
        set_meta(db,"distance_identity",graph)
    if reclassify:
        print(f"[KOVAR] distance cutoff: {old['distal_bp']:,} -> {args.distal_bp:,} bp; "
              f"reclassified {changed:,} saved unitig pairs; distances, association tests "
              "and saved beta fits retained",flush=True)
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
        # Freeze a disk-backed queue before writing beta rows. Repeated unitig
        # mappings share one fit; committed attempts never re-enter the queue.
        db.execute('DROP TABLE IF EXISTS temp.beta_queue')
        db.execute('CREATE TEMP TABLE beta_queue(u INTEGER,v INTEGER,PRIMARY KEY(v,u)) WITHOUT ROWID')
        db.execute("INSERT OR IGNORE INTO beta_queue SELECT pu,pv FROM distance WHERE status='DISTAL'")
        total = db.execute('SELECT count(*) FROM beta_queue').fetchone()[0]
        statuses = dict(db.execute('SELECT b.status,count(*) FROM beta b JOIN beta_queue q '
                                   'ON b.u=q.u AND b.v=q.v GROUP BY b.status'))
        saved_before = sum(statuses.values())
        db.execute('DELETE FROM beta_queue WHERE EXISTS '
                   '(SELECT 1 FROM beta b WHERE b.u=beta_queue.u AND b.v=beta_queue.v)')
        db.commit()
        completed = 0
        started = last = time.monotonic()

        def publish_beta_progress(finished=False):
            nonlocal last
            now = time.monotonic()
            saved = saved_before+completed
            remaining = total-saved
            eta = (now-started)/completed*remaining if completed else None
            progress = dict(stage='beta_complete' if finished else 'beta',
                            total_distal_pattern_pairs=total, saved=saved, remaining=remaining,
                            saved_before_invocation=saved_before, committed_this_invocation=completed,
                            status_counts=statuses, seconds_this_invocation=now-started,
                            eta_seconds=eta, work_directory=str(work.resolve()))
            atomic_json(args.out/'beta_progress.json',progress)
            message = f'; rough ETA {eta/3600:.2f} h' if eta is not None else ''
            print(f"[KOVAR] beta: {saved:,}/{total:,} pattern pairs saved; {remaining:,} remaining; "
                  f"{completed:,} new this invocation; {statuses.get('OK',0):,} OK; "
                  f"{saved-statuses.get('OK',0):,} non-OK{message}",flush=True)
            last = now

        publish_beta_progress()
        jobs = db.execute('SELECT u,v FROM beta_queue ORDER BY v,u')
        try:
            for row in bounded_map(pool,kernels.beta,jobs,args.threads,ordered=False):
                # SQLite uses synchronous=FULL; publish progress only after commit.
                # A crash can repeat in-flight work, never an acknowledged fit.
                with db:
                    db.execute("INSERT OR IGNORE INTO beta VALUES (?,?,?,?)",row)
                completed += 1
                statuses[row[3]] = statuses.get(row[3],0)+1
                if time.monotonic()-last>25:
                    publish_beta_progress()
        finally:
            jobs.close()
        publish_beta_progress(finished=True)
        db.execute('DROP TABLE beta_queue')


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
    compact_count, compact_ineligible = db.execute(
        "SELECT count(*),count(CASE WHEN final=-2 THEN 1 END) FROM compact_rejections").fetchone()
    for state, count in (("-2", compact_ineligible), ("-1", compact_count - compact_ineligible)):
        if count:
            counts[state] = counts.get(state, 0) + count
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
    summary['compact_coarse_rejections']=compact_count
    summary['coarse_screen']=get_meta(db,'coarse_screen_summary')
    summary['distance_threshold_history']=get_meta(db,'distance_threshold_history',[])
    summary['reused_prepared_from']=get_meta(db,'reused_prepared_from')
    atomic_json(args.out/"run_summary.json",summary)


def prepare_presence_counts(db, work, info):
    target = work / "pattern_presence.npy"
    expected = get_meta(db, "pattern_presence_sha256")
    if expected and target.is_file() and digest(target) == expected:
        return
    counts = np.full(info["patterns"], -1, dtype=np.int32)
    for pid, count in db.execute("SELECT id,present FROM patterns"):
        counts[pid] = count
    if np.any(counts < 0):
        raise ValueError("Incomplete pattern counts in checkpoint")
    coarse_screen.array_file(target, counts)
    with db:
        set_meta(db, "pattern_presence_sha256", digest(target))


def main(argv=None):
    args = parser().parse_args(argv)
    validate(args)
    args.out = args.out.resolve()
    work = args.tmp_dir.resolve() if args.tmp_dir else args.out / ".kovar_work"
    if args.prepare_pangwes:
        assembly_list, gfa, metadata, fasta, tree = (p.resolve() for p in args.prepare_pangwes)
        prepare_pangwes(args.fasta.resolve(), args.tree.resolve(), assembly_list, gfa, metadata,
                        fasta, tree, args.exclude_samples.resolve() if args.exclude_samples else None,
                        args.tree_missing_samples, work)
        args.fasta, args.tree = fasta, tree
    if args.local_graph_paths:
        paths = args.local_graph_paths.resolve()
        inputs = (args.fasta.resolve(), args.tree.resolve(), args.paths.resolve())
        if paths in inputs:
            raise ValueError("Local graph path list must differ from the input paths")
        localize_graph_paths(args.paths.resolve(), paths, args.graph_path_rebase)
        args.paths = paths
    if args.prepare_only:
        return 0
    started = time.monotonic()
    args._memory_info=detect_budget(args.memory_gb,args.threads)
    args.memory_gb=args._memory_info['working_gb']
    args.out.mkdir(parents=True,exist_ok=True)
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
            if not (args.resume or args.resume_if_present):
                raise ValueError("Existing work requires --resume")
            if check_identity(json.loads(manifest.read_text()), expected):
                atomic_json(manifest, expected)
            if args.resume_if_present and not args.resume:
                print(f"[KOVAR] resuming checkpoint: {work}", flush=True)
        else:
            atomic_json(manifest,expected)
        atomic_json(args.out/"run_identity.json",dict(expected,work_directory=str(work)))
        db = connect(work/"work.sqlite",min(64,args.memory_gb*1024*.05))
        try:
            schema(db)
            if get_meta(db,'queried_pairs_count') is None:
                with db:
                    set_meta(db,'queried_pairs_count',queried_count(db))
            with threadpool_limits(limits=args.threads):
                if args.reuse_prepared_from:
                    from .discovery_store import reuse_prepared
                    reuse_prepared(args,db,work,expected)
                info = prepare(args,db,work)
                if args.null_backend=='tree':
                    from .tree_null import TreeAlgebra, compiled_contrasts
                    if compiled_contrasts is None:
                        raise ValueError('Tree backend requires numba: pip install "ko-variation[tree]"')
                    tree=TreeAlgebra.from_newick(args.tree,info['samples'])
                    target=work/'tree.npz'
                    if not target.exists():
                        np.savez(target,left=tree.left,right=tree.right,tip=tree.tip,length=tree.length)
                    tree_sha=digest(target)
                    if info['products'].get('tree.npz',tree_sha)!=tree_sha:
                        raise ValueError('Shared tree algebra changed')
                    if 'tree.npz' not in info['products']:
                        info['products']['tree.npz']=tree_sha
                        atomic_json(work/'prepared.json',info)
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
                            null_backend=args.null_backend,
                            workers=args.threads,**args._plan)
            print(f'[KOVAR] execution revision: {__discovery_revision__}; '
                  f'null stage={args.prefit_nulls}; cache={args.null_cache_format}',flush=True)
            print(f"[KOVAR {__version__}] {info['patterns']:,} patterns; {family:,} hypotheses; "
                  f"final P <= {args.alpha/family:.6g}; workers={args.threads}; "
                  f"cache/worker={args._plan['cache_bytes_per_worker']/(1<<30):.3f} GiB",flush=True)
            print(f"[KOVAR] memory target={args.memory_gb:.3f} GiB; source={args._memory_info['limiting_source']}; "
                  f"bounded query buffer={args._plan['query_buffer_pairs']:,} pairs",flush=True)
            estimate=(12*n*8 if args.null_backend=='tree' else
                      8*n*min(args.null_cache_rank,n)+12*n*8 if args.null_cache_format=='compact' else 12*n*n)
            print("[KOVAR] numerical null-cache estimate before complement/shared/zero savings: "
                  f"{info['patterns']*estimate/(1<<30):.1f} GiB; "
                  "pair checkpoints and expanded unitig results require additional disk space",flush=True)
            if not get_meta(db,'search_complete') or args.prefit_only:
                prefit(db, work, info, args, settings)
                if args.prefit_only:
                    return 0
            if args.prefit_nulls=='all':
                settings['require_cached_nulls']=True
            args._tiled_coarse = (args.search == "hierarchy" and args.coarse_screen == "pc"
                                  and args.null_backend == "tree" and not args.exact)
            if args._tiled_coarse and not get_meta(db, "search_complete"):
                prepare_presence_counts(db, work, info)
                settings["fast_tree_scores"] = True
                print("[KOVAR] PC hierarchy: tiled coarse admission and early tree-score rejection", flush=True)
            with db:
                set_meta(db,'worker_state',{})
                set_meta(db,'active_stage','pair_search')
            with ResponsePool(args.threads, kernels.initialize, (str(work), info, settings),
                              canonical_responses=nulls.open_aliases(work)) as pool:
                if args.search=="hierarchy":
                    make_hierarchy(db,info,pool,args)
                    if args.coarse_screen=='pc':
                        coarse_screen.prepare_screen(db, work, info, args)
                        with db: set_meta(db,'active_stage','pair_search')
                search(db,info,pool,args)
            # Release every score-worker cache before the external graph tool
            # gets its own memory budget. Refit workers start lazily afterwards.
            settings.pop('fast_tree_scores', None)
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
