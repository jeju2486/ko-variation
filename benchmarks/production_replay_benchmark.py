"""Replay a saved interval of the production PC-screen search in scratch databases."""
from __future__ import annotations

import argparse
from contextlib import closing
import json
import math
from pathlib import Path
import shutil
import sqlite3
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ko_variation.memory_budget import ByteLRU, MIB, detect_budget, guard_blas_environment, plan_workers

guard_blas_environment(default_threads=1)

import numpy as np
from scipy.stats import chi2

from ko_variation.tree_score import TreeScore, ZeroTauScore, classify
from ko_variation import coarse_screen, discovery_nulls, discovery_score as kernels, discovery_search
from ko_variation.discovery_execution import ResponsePool, batches, bounded_map, merge_metrics
from ko_variation.discovery_store import atomic_json, connect, digest, get_meta, schema, set_meta


MODES = ("current", "fused", "early", "counts", "tile_rows", "tile_compact")
ORIGINAL_CHUNK = kernels._score_chunk
ORIGINAL_EVALUATE = discovery_search.evaluate


class InterruptedReplay(Exception):
    pass


def prevent_cache_write(*args, **kwargs):
    raise ValueError("This replay requires compatible saved nulls and never writes to the source cache")


def allow_saved_format(record, compact, rank, n):
    return record["status"] != "OK" or record.get("format") == 2


def initialize_worker(work, info, settings, mode, presence_file, interval, zero_tau):
    kernels.initialize(work, info, settings)
    discovery_nulls.save = prevent_cache_write
    discovery_nulls.compatible = allow_saved_format
    if mode == "current":
        return
    state = kernels._STATE
    limit = settings["cache_bytes_per_worker"] // 4
    state["nulls"].limit -= limit
    state["cache_limit"] -= limit
    state["replay_cache"] = ByteLRU(limit)
    state["replay_mode"] = mode
    state["replay_presence"] = np.load(presence_file, mmap_mode="r")
    state["replay_interval"] = interval
    state["replay_zero_tau"] = zero_tau
    kernels._score_chunk = optimized_chunk


def cache_value(state, key, build, size):
    cache = state["replay_cache"]
    if key in cache:
        return cache.get(key)
    value = build()
    cache.put(key, value, size)
    return value


def response_pattern(state, response):
    def unpack():
        y = kernels.bits([response])[0].astype(float)
        return y, y.sum()

    if state["replay_mode"] in ("fused", "early"):
        return unpack()
    return cache_value(state, ("response", response), unpack, 256 + 8 * state["info"]["n"])


def optimized_chunk(job):
    response, predictors, external, screening = job
    state = kernels._STATE
    settings = state["settings"]
    started = time.perf_counter()
    reuse_counts = state["replay_mode"] not in ("fused", "early")
    X = kernels.bits(predictors).T.astype(float)
    y, present = response_pattern(state, response)
    if reuse_counts:
        counts = state["replay_presence"][predictors].astype(float)
    else:
        counts = X.sum(axis=0)
    overlap = X.T @ y
    eligible = np.minimum.reduce((overlap, counts - overlap, present - overlap,
                                 len(y) - counts - present + overlap)) >= settings["min_cell_count"]
    kernels.metric("decode_and_eligibility_worker_seconds", time.perf_counter() - started)
    rows = []
    if not screening and not eligible.all():
        rows = [(int(pid), response, 0, -2, None, None, None, "INELIGIBLE_NOT_SCREENED")
                for pid, ok in zip(predictors, eligible) if not ok]
        predictors = [pid for pid, ok in zip(predictors, eligible) if ok]
        if not predictors:
            return rows
        if not reuse_counts:
            return rows + optimized_chunk((response, predictors, external, screening))
        X = X[:, eligible]
        eligible = np.ones(len(predictors), dtype=bool)

    cache, status = kernels.get_null(response, external)
    if cache is None:
        return rows + [(int(pid), response, 0, 0 if ok else -2, None, None, None, status)
                       for pid, ok in zip(predictors, eligible)]
    if not hasattr(cache["decomp"], "score_variance"):
        return rows + ORIGINAL_CHUNK((response, predictors, external, screening))

    started = time.perf_counter()
    gate, final = settings["gate_critical"], settings["final_critical"]
    stopped = np.zeros(len(predictors), dtype=bool)
    terms = X.size

    def build():
        kernels.metric("tree_coefficient_setups")
        return TreeScore(cache)

    cutoff = min(gate, final)
    if state["replay_zero_tau"] and cache["tau"] == 0.:
        u, v = ZeroTauScore(cache)(X)
        terms = 0
        kernels.metric("zero_tau_pairs", len(predictors))
    else:
        scorer = cache_value(state, ("scorer", response), build, 512 + 112 * len(y))
        if state["replay_mode"] == "fused":
            u, v = scorer(X)
        else:
            u, v, stopped, terms = scorer.early_stop(X, cutoff, state["replay_interval"])
    routing, decision, statistic = classify(u, v, eligible, gate, final, stopped)
    compute_p = np.isfinite(statistic)
    if state["replay_mode"] == "tile_compact":
        compute_p &= decision == 1
    p_values = np.full(len(predictors), np.nan)
    p_values[compute_p] = chi2.sf(statistic[compute_p], 1)
    for i, pid in enumerate(predictors):
        if stopped[i]:
            low, high, flag = 0., cutoff, "BOUNDED"
        elif math.isfinite(statistic[i]):
            low = high = float(statistic[i])
            flag = "OK"
        else:
            low, high, flag = 0., None, "SINGULAR_PREDICTOR_AFTER_KINSHIP"
        rows.append((int(pid), response, int(routing[i]), int(decision[i]), low, high,
                     float(p_values[i]) if math.isfinite(p_values[i]) else None, flag))
    kernels.metric("tree_terms", terms)
    kernels.metric("possible_tree_terms", X.size)
    kernels.metric("early_stopped_pairs", int(stopped.sum()))
    kernels.metric("exact_pairs", len(predictors) - int(stopped.sum()))
    kernels.metric("exact_worker_seconds", time.perf_counter() - started)
    return rows


def check_for_compact(db, pairs):
    hits = {}
    for block in batches(pairs, 400):
        values = ",".join(["(?,?)"] * len(block))
        sql = (f"WITH requested(u,v) AS (VALUES {values}) "
               "SELECT c.u,c.v,c.final FROM requested r "
               "JOIN compact_rejections c ON (r.u,r.v)=(c.u,c.v)")
        hits.update(((u, v), (-1, final)) for u, v, final in db.execute(
            sql, [i for pair in block for i in pair]))
    return hits


def evaluate_with_compact(db, pairs, pool, args, screening=False, *, return_mapping=False):
    requested = sorted({tuple(sorted((int(u), int(v)))) for u, v in pairs if u != v},
                       key=lambda pair: (pair[1], pair[0]))
    started = time.perf_counter()
    saved = check_for_compact(db, requested)
    args.compact_lookup_seconds += time.perf_counter() - started
    pending = [pair for pair in requested if pair not in saved]
    saved.update(ORIGINAL_EVALUATE(db, pending, pool, args, screening, return_mapping=True))
    return saved if return_mapping else [saved[key] for key in requested]


def tile_scores(db, pairs, pool, args):
    grouped = {}
    for u, v in pairs:
        grouped.setdefault(v, []).append(u)
    external = {}
    if args._has_external_nulls:
        for block in batches(sorted(grouped), 400):
            marks = ",".join("?" for _ in block)
            external.update(db.execute(f"SELECT pid,path FROM external_nulls WHERE pid IN ({marks})", block))
            external.update(db.execute("SELECT a.pid,e.path FROM null_aliases a "
                f"JOIN external_nulls e ON a.canonical=e.pid WHERE a.pid IN ({marks})", block))
    jobs = ((v, sorted(us), external.get(v), True) for v, us in grouped.items())
    rows, metrics = [], {}
    for payload in bounded_map(pool, kernels.score_profiled, jobs, args.threads,
                               ordered=False, response_index=0):
        rows.extend(payload["rows"])
        merge_metrics(metrics, payload["metrics"])
    return rows, metrics


class TileAdmission:
    def __init__(self, db, pool, args, info, mode, interrupt):
        self.pool, self.args, self.mode, self.interrupt = pool, args, mode, interrupt
        self.eligibility = discovery_search.Eligibility(db, pool, args, info) if args.min_cell_count else None
        self.reps = dict(db.execute("SELECT r.node,r.pid FROM reps r JOIN nodes n ON n.id=r.node WHERE n.parent=0"))
        self.seconds = self.write_seconds = 0.

    def __call__(self, db, arrays, capacity, workers=1):
        started = time.perf_counter()
        offset = get_meta(db, "pc_coarse_cursor", 0)
        total = len(arrays[1])
        if offset >= total:
            return [], True
        blocks, offset, active, next_right = coarse_screen.candidate_window(db, arrays, capacity, workers)
        if not blocks:
            raise ValueError("Candidate cursor ended before the selected interval was exhausted")
        pruned = set()
        if self.eligibility:
            summary = self.eligibility.summaries(blocks)
            pruned = {pair for pair in blocks if self.eligibility.impossible(*pair, summary)}
        tested = {tuple(sorted((self.reps[a], self.reps[b]))): (a, b)
                  for a, b in blocks if (a, b) not in pruned}
        rows, metrics = tile_scores(db, tested, self.pool, self.args)
        survivors = [tested[u, v] for u, v, gate, *_ in rows if gate != -1]
        rejected = len(tested) - len(survivors)
        if self.interrupt == "before":
            raise InterruptedReplay("before tile commit")
        writing = time.perf_counter()
        with db:
            if self.mode == "tile_compact":
                db.executemany("INSERT INTO compact_rejections VALUES (?,?,?)",
                               ((u, v, final) for u, v, gate, final, *_ in rows if gate == -1))
                rows = [row for row in rows if row[2] != -1]
            db.executemany("INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?)", rows)
            db.executemany("INSERT OR IGNORE INTO tasks(a,b) VALUES (?,?)", sorted(survivors))
            set_meta(db, "queried_pairs_count", get_meta(db, "queried_pairs_count", 0) + len(tested))
            coarse_screen.save_candidate_cursor(db, offset, active, next_right)
            saved = get_meta(db, "hierarchy_metrics", {})
            merge_metrics(saved, dict(eligibility_pruned_cross_blocks=len(pruned), representative_pruned_blocks=rejected))
            set_meta(db, "hierarchy_metrics", saved)
            saved = get_meta(db, "runtime_totals", {})
            merge_metrics(saved, metrics)
            set_meta(db, "runtime_totals", saved)
        self.write_seconds += time.perf_counter() - writing
        self.seconds += time.perf_counter() - started
        if self.interrupt == "after":
            raise InterruptedReplay("after tile commit")
        return sorted(survivors), offset >= total


def source_inputs(cli):
    run = cli.run.resolve()
    run_manifest = run / "run_identity.json"
    manifest = json.loads(run_manifest.read_text())
    work = Path(manifest.get("work_directory", run / ".kovar_work")).resolve()
    if cli.out.resolve().is_relative_to(run) or cli.out.resolve().is_relative_to(work):
        raise ValueError("Put the benchmark output outside the source results and work directories")
    info = json.loads((work / "prepared.json").read_text())
    settings = manifest["settings"]
    if settings.get("search") != "hierarchy" or settings.get("coarse_screen") != "pc":
        raise ValueError("Replay needs a source hierarchy with a completed PC screen")
    with closing(sqlite3.connect((work / "work.sqlite").as_uri() + "?mode=ro", uri=True)) as source:
        screen = get_meta(source, "coarse_screen_summary")
    if not screen or screen["selected_pairs"] <= cli.start:
        raise ValueError("Start position is outside the completed PC-screen candidate list")
    roots, selected, offsets = coarse_screen.open_candidates(work, screen["selected_pairs"])
    end = min(cli.start + cli.pairs, len(selected))
    array = selected[cli.start:end]
    arrays = roots, array, np.clip(offsets, cli.start, end) - cli.start
    identity = dict(source_run=digest(run_manifest), prepared=digest(work / "prepared.json"),
                    benchmark=digest(Path(__file__)), scorer=digest(Path(kernels.__file__).with_name("tree_score.py")),
                    start=cli.start, end=end, threads=cli.threads, memory_gb=cli.memory_gb,
                    batch_size=cli.batch_size, early_check_interval=cli.early_check_interval, zero_tau=cli.zero_tau)
    return work, info, settings, screen, arrays, identity


def create_seed(out, source_work, data, arrays, screen):
    target = out / "seed.sqlite"
    root_ids = arrays[0][np.unique(arrays[1])]
    presence = np.empty(data["patterns"], dtype=np.int32)
    with closing(sqlite3.connect(target.as_uri(), uri=True, timeout=120)) as db:
        db.execute("PRAGMA temp_store=FILE")
        schema(db)
        source = (source_work / "work.sqlite").as_uri() + "?mode=ro"
        db.execute("ATTACH DATABASE ? AS source", (source,))
        with db:
            db.execute("CREATE TEMP TABLE wanted_roots(id INTEGER PRIMARY KEY)")
            db.executemany("INSERT INTO wanted_roots VALUES (?)", ((int(root),) for root in root_ids))
            db.execute("""WITH RECURSIVE keep(id) AS (
                SELECT id FROM wanted_roots UNION ALL SELECT n.id FROM source.nodes n JOIN keep k ON n.parent=k.id
            ) INSERT INTO nodes SELECT n.* FROM source.nodes n JOIN keep USING (id)""")
            for table in ("members", "reps", "block_eligibility"):
                db.execute(f"INSERT INTO {table} SELECT s.* FROM source.{table} s JOIN nodes n ON n.id=s.node")
            db.execute("INSERT INTO external_nulls SELECT * FROM source.external_nulls")
            db.execute("INSERT INTO null_aliases SELECT * FROM source.null_aliases")
            for pid, present in db.execute("SELECT id,present FROM source.patterns"):
                presence[pid] = present
            set_meta(db, "search_initialized", True)
            set_meta(db, "coarse_screen_summary", dict(screen, selected_pairs=len(arrays[1])))
            set_meta(db, "queried_pairs_count", 0)
        db.execute("DETACH DATABASE source")
    np.save(out / "presence.npy", presence)


def run_mode(cli, work, info, source_settings, arrays, mode):
    destination = cli.out / mode
    destination.mkdir(exist_ok=True)
    database = destination / "work.sqlite"
    if not database.exists():
        shutil.copyfile(cli.out / "seed.sqlite", database)
    started = time.perf_counter()
    incomplete = False
    budget = detect_budget(cli.memory_gb, cli.threads)
    plan = plan_workers(budget, info["n"], cli.threads, cli.batch_size)
    family = source_settings.get("bonferroni_tests") or math.comb(info["patterns"], 2)
    args = SimpleNamespace(**source_settings, threads=cli.threads, batch_size=cli.batch_size, tmp_dir=work,
                           out=destination, memory_gb=budget["working_gb"], _plan=plan,
                           compact_lookup_seconds=0.)
    settings = dict(min_cell_count=args.min_cell_count, gate_critical=float(chi2.isf(args.screen_p, 1)),
                    final_critical=float(chi2.isf(args.alpha / family, 1)), exact=args.exact,
                    require_cached_nulls=True, batch_size=args.batch_size, null_cache_format="compact",
                    null_cache_rank=128, null_backend=getattr(args, "null_backend", "dense"), workers=cli.threads, **plan)
    worker_args = (str(work), info, settings, mode, str(cli.out / "presence.npy"),
                   cli.early_check_interval, cli.zero_tau)
    with closing(connect(database, min(64, budget["working_gb"] * 1024 * .05))) as db:
        if get_meta(db, "search_complete"):
            return False
        args._has_external_nulls = db.execute("SELECT 1 FROM external_nulls LIMIT 1").fetchone() is not None
        with ResponsePool(cli.threads, initialize_worker, worker_args,
                          canonical_responses=discovery_nulls.open_aliases(work)) as pool:
            tiled = mode.startswith("tile_")
            tiles = TileAdmission(db, pool, args, info, mode, cli.interrupt) if tiled else None
            producer = tiles if tiled else coarse_screen.candidate_tasks
            evaluate = evaluate_with_compact if mode == "tile_compact" else ORIGINAL_EVALUATE
            with patch.object(coarse_screen, "open_candidates", return_value=arrays), \
                 patch.object(coarse_screen, "candidate_tasks", producer), \
                 patch.object(discovery_search, "evaluate", evaluate):
                try:
                    discovery_search.search(db, info, pool, args)
                except InterruptedReplay as error:
                    print(f"[{mode}] intentional interruption {error}", flush=True)
                    incomplete = True
        seconds = time.perf_counter() - started
        with db:
            saved = get_meta(db, "replay_timing", {})
            merge_metrics(saved, dict(seconds=seconds, invocations=1, interruptions=int(incomplete),
                                      tile_seconds=tiles.seconds if tiles else 0.,
                                      tile_write_seconds=tiles.write_seconds if tiles else 0.,
                                      compact_lookup_seconds=args.compact_lookup_seconds))
            set_meta(db, "replay_timing", saved)
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return incomplete


def differences(db, reference, variant):
    db.execute("ATTACH DATABASE ? AS baseline", (reference.as_uri() + "?mode=ro",))
    db.execute("ATTACH DATABASE ? AS trial", (variant.as_uri() + "?mode=ro",))
    try:
        queries = {
            "pair_decisions": ("SELECT u,v,gate,final FROM {db}.decisions UNION ALL "
                               "SELECT u,v,-1,final FROM {db}.compact_rejections"),
            "child_tasks": ("SELECT a,b FROM {db}.tasks t JOIN {db}.nodes x ON x.id=t.a "
                            "JOIN {db}.nodes y ON y.id=t.b WHERE x.parent!=0 OR y.parent!=0"),
        }
        counts = {}
        for label, query in queries.items():
            first, second = (query.format(db=s) for s in ("baseline", "trial"))
            for lhs, rhs, suffix in ((first, second, "missing"), (second, first, "extra")):
                counts[f"{label}_{suffix}"] = db.execute(
                    f"SELECT count(*) FROM (SELECT * FROM ({lhs}) EXCEPT SELECT * FROM ({rhs}))").fetchone()[0]
        counts["significant_p_mismatches"] = db.execute("""
            SELECT count(*) FROM baseline.decisions r JOIN trial.decisions t USING(u,v)
            WHERE r.final=1 AND t.final=1 AND (
                (r.p IS NULL)!=(t.p IS NULL) OR (r.p IS NOT NULL AND t.p IS NOT NULL
                AND abs(r.p-t.p)>1e-8*max(abs(r.p),abs(t.p),1e-300))
            )
        """).fetchone()[0]
        return counts
    finally:
        db.execute("DETACH DATABASE baseline")
        db.execute("DETACH DATABASE trial")


def report(out, modes, identity, arrays):
    results = []
    baseline = out / "current/work.sqlite"
    seed_bytes = (out / "seed.sqlite").stat().st_size
    with closing(sqlite3.connect((out / "seed.sqlite").as_uri() + "?mode=ro", uri=True)) as db:
        singleton_nodes = {node for node, in db.execute("SELECT id FROM nodes WHERE parent=0 AND size=1")}
    singleton = np.isin(arrays[0], list(singleton_nodes))
    singleton_pairs = sum(int(np.count_nonzero(singleton[block].all(axis=1)))
                          for block in (arrays[1][start:start + 100_000] for start in range(0, len(arrays[1]), 100_000)))
    breakdown = dict(singleton_singleton=singleton_pairs, with_members=len(arrays[1]) - singleton_pairs)
    for mode in modes:
        path = out / mode / "work.sqlite"
        if not path.exists():
            continue
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
            if not get_meta(db, "search_complete"):
                continue
            timing = get_meta(db, "replay_timing")
            rows = {table: db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                    for table in ("decisions", "tasks", "compact_rejections")}
            significant = db.execute("SELECT count(*) FROM decisions WHERE final=1").fetchone()[0]
            results.append(dict(mode=mode, coarse_per_second=len(arrays[1]) / timing["seconds"],
                                significant=significant, db_bytes=path.stat().st_size,
                                db_growth_bytes=path.stat().st_size - seed_bytes, rows=rows,
                                timing=timing, runtime=get_meta(db, "runtime_totals", {}),
                                hierarchy=get_meta(db, "hierarchy_metrics", {})))
    have_baseline = any(result["mode"] == "current" for result in results)
    with closing(sqlite3.connect(":memory:", uri=True)) as db:
        for result in results:
            mode = result["mode"]
            comparison = (differences(db, baseline, out / mode / "work.sqlite")
                          if have_baseline and mode != "current" else {})
            result["comparison"] = comparison
            check = f"; {sum(comparison.values()):,} discrepancies" if comparison else ""
            print(f"{mode}: {result['coarse_per_second']:,.0f} coarse pairs/s; "
                  f"{result['db_bytes'] / MIB:,.1f} MiB; {result['significant']:,} significant{check}", flush=True)
    atomic_json(out / "summary.json", dict(selection=identity, coarse_pairs=breakdown, modes=results))
    return int(any(count for row in results for count in row["comparison"].values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Source discovery results directory")
    parser.add_argument("--out", type=Path, required=True, help="Separate benchmark directory")
    parser.add_argument("--start", type=int, default=0, help="Zero-based position in the saved PC candidate list")
    parser.add_argument("--pairs", type=int, default=100_000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-gb", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--early-check-interval", type=int, default=16)
    parser.add_argument("--zero-tau", action="store_true", help="Enable the exact zero-tau shortcut in optimized modes")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--interrupt", choices=("before", "after"), help="Stop at the first tile commit boundary")
    cli = parser.parse_args()
    if cli.start < 0 or min(cli.pairs, cli.threads, cli.batch_size, cli.early_check_interval) < 1:
        parser.error("Start must be nonnegative and counts must be positive")
    if cli.interrupt and (len(cli.modes) != 1 or not cli.modes[0].startswith("tile_")):
        parser.error("--interrupt requires a single tiled mode")
    cli.out = cli.out.resolve()
    work, info, settings, screen, arrays, identity = source_inputs(cli)
    cli.out.mkdir(parents=True, exist_ok=True)
    lock_path = cli.out / ".replay.lock.sqlite"
    with closing(sqlite3.connect(lock_path, timeout=0)) as lock:
        lock.execute("BEGIN EXCLUSIVE")
        manifest = cli.out / "replay_identity.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != identity:
                raise ValueError("This benchmark belongs to a different interval, configuration or code version")
        else:
            if any(path.name not in (lock_path.name, lock_path.name + "-journal") for path in cli.out.iterdir()):
                raise ValueError("Choose an empty benchmark directory")
            create_seed(cli.out, work, info, arrays, screen)
            atomic_json(manifest, identity)
        for mode in dict.fromkeys(cli.modes):
            print(f"[{mode}] replaying coarse interval [{identity['start']:,}, {identity['end']:,})", flush=True)
            if run_mode(cli, work, info, settings, arrays, mode):
                return 75
        return report(cli.out, tuple(m for m in MODES if (cli.out / m).is_dir()), identity, arrays)


if __name__ == "__main__":
    raise SystemExit(main())
