"""Ablate decision reuse and two-sided bounds on a saved PC shortlist."""
from __future__ import annotations

import argparse
from contextlib import closing
import json
import math
from pathlib import Path
import shutil
import sqlite3
from statistics import median
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ko_variation.memory_budget import detect_budget, guard_blas_environment, plan_workers

guard_blas_environment(default_threads=1)

import numpy as np
from scipy.stats import chi2

from ko_variation import coarse_screen, discovery_nulls, discovery_score, discovery_search, tree_score
from ko_variation.discovery_execution import ResponsePool
from ko_variation.discovery_store import atomic_json, connect, digest, get_meta, schema, set_meta


MODES = {
    "v095": (False, False),
    "reuse": (True, False),
    "bounds": (False, True),
    "combined": (True, True),
}
LABELS = {"v095": "v0.9.5", "reuse": "Decision reuse", "bounds": "Two-sided bounds", "combined": "Combined"}


def prevent_cache_write(*args, **kwargs):
    raise ValueError("The replay must use saved nulls without writing to the source")


def initialize_worker(work, info, settings):
    discovery_score.initialize(work, info, settings)
    discovery_nulls.save = prevent_cache_write
    discovery_nulls.compatible = lambda record, *_: record["status"] != "OK" or record.get("format") == 2


def source_inputs(args):
    run, out = args.run.resolve(), args.out.resolve()
    manifest_path = run / "run_identity.json"
    manifest = json.loads(manifest_path.read_text())
    work = Path(manifest.get("work_directory", run / ".kovar_work")).resolve()
    if out.is_relative_to(run) or out.is_relative_to(work) or work.is_relative_to(out):
        raise ValueError("Put the benchmark output in a separate directory")
    settings = manifest["settings"]
    if (settings.get("search") != "hierarchy" or settings.get("coarse_screen") != "pc"
            or settings.get("null_backend") != "tree" or settings.get("exact")
            or not (work / "pattern_presence.npy").is_file()):
        raise ValueError("Use a saved non-exact tree discovery run with a completed PC shortlist")
    with closing(sqlite3.connect((work / "work.sqlite").as_uri() + "?mode=ro", uri=True)) as source:
        screen = get_meta(source, "coarse_screen_summary")
    if not screen or args.start >= screen["selected_pairs"]:
        raise ValueError("Start position is outside the completed PC shortlist")
    roots, selected, offsets = coarse_screen.open_candidates(work, screen["selected_pairs"])
    end = min(args.start + args.pairs, len(selected))
    arrays = roots, selected[args.start:end], np.clip(offsets, args.start, end) - args.start
    prepared = work / "prepared.json"
    identity = dict(source=digest(manifest_path), prepared=digest(prepared), script=digest(Path(__file__)),
                    code={Path(module.__file__).name: digest(Path(module.__file__))
                          for module in (discovery_search, discovery_score, coarse_screen, tree_score)},
                    start=args.start, end=end, threads=args.threads, memory_gb=args.memory_gb,
                    batch_size=args.batch_size)
    return work, json.loads(prepared.read_text()), settings, screen, arrays, identity


def create_seed(out, work, arrays, screen):
    pending = out / "seed.partial.sqlite"
    target = out / "seed.sqlite"
    pending.unlink(missing_ok=True)
    root_ids = arrays[0][np.unique(arrays[1])]
    with closing(sqlite3.connect(pending.as_uri(), uri=True, timeout=120)) as db:
        schema(db)
        source = (work / "work.sqlite").as_uri() + "?mode=ro"
        db.execute("ATTACH DATABASE ? AS source", (source,))
        with db:
            db.execute("CREATE TEMP TABLE wanted_roots(id INTEGER PRIMARY KEY)")
            db.executemany("INSERT INTO wanted_roots VALUES (?)", ((int(root),) for root in root_ids))
            db.execute("""WITH RECURSIVE keep(id) AS (
                SELECT id FROM wanted_roots UNION ALL SELECT n.id FROM source.nodes n JOIN keep k ON n.parent=k.id
            ) INSERT INTO nodes SELECT n.* FROM source.nodes n JOIN keep USING (id)""")
            for table in ("members", "reps", "block_eligibility"):
                db.execute(f"INSERT INTO {table} SELECT s.* FROM source.{table} s JOIN nodes n ON n.id=s.node")
            for table in ("external_nulls", "null_aliases"):
                db.execute(f"INSERT INTO {table} SELECT * FROM source.{table}")
            set_meta(db, "search_initialized", True)
            set_meta(db, "coarse_screen_summary", dict(screen, selected_pairs=len(arrays[1])))
            set_meta(db, "queried_pairs_count", 0)
    pending.replace(target)


def replay(path, seed, work, info, source, arrays, cli, mode):
    if not path.exists():
        setup = path.with_suffix(".partial")
        shutil.copyfile(seed, setup)
        setup.replace(path)
    reuse, bounds = MODES[mode]
    started = time.perf_counter()
    budget = detect_budget(cli.memory_gb, cli.threads)
    plan = plan_workers(budget, info["n"], cli.threads, cli.batch_size)
    family = source.get("bonferroni_tests") or math.comb(info["patterns"], 2)
    args = SimpleNamespace(**source, threads=cli.threads, tmp_dir=work, out=path.parent,
                           batch_size=cli.batch_size, memory_gb=budget["working_gb"], _plan=plan,
                           _tiled_coarse=True, _reuse_fresh_gates=reuse)
    settings = dict(min_cell_count=args.min_cell_count, gate_critical=float(chi2.isf(args.screen_p, 1)),
                    final_critical=float(chi2.isf(args.alpha / family, 1)), exact=False,
                    require_cached_nulls=True, batch_size=args.batch_size, null_cache_format="compact",
                    null_cache_rank=128, null_backend="tree", workers=cli.threads, fast_tree_scores=True, **plan)
    if bounds:
        settings["tree_bound_eigenvalue"] = tree_score.kinship_eigenvalue_bound(np.load(work / "K.npy", mmap_mode="r"))
    with closing(connect(path, min(64, budget["working_gb"] * 1024 * .05))) as db:
        if not get_meta(db, "search_complete"):
            args._has_external_nulls = db.execute("SELECT 1 FROM external_nulls LIMIT 1").fetchone() is not None
            with db:
                set_meta(db, "replay_has_resumed", bool(get_meta(db, "pc_coarse_cursor", 0)))
            pool_args = (str(work), info, settings)
            with ResponsePool(cli.threads, initialize_worker, pool_args,
                              canonical_responses=discovery_nulls.open_aliases(work)) as pool:
                with patch.object(coarse_screen, "open_candidates", return_value=arrays):
                    discovery_search.search(db, info, pool, args)
            with db:
                set_meta(db, "replay_wall_seconds", time.perf_counter() - started)
        elapsed = get_meta(db, "replay_wall_seconds")
        resumed = elapsed is None or get_meta(db, "replay_has_resumed", False)
        return dict(mode=mode, wall_seconds=elapsed, resumed=resumed,
                    coarse_per_second=len(arrays[1]) / elapsed if elapsed else None,
                    runtime=get_meta(db, "runtime_totals", {}),
                    significant=db.execute("SELECT count(*) FROM decisions WHERE final=1").fetchone()[0])


def compare(reference, trial):
    with closing(sqlite3.connect(":memory:", uri=True)) as db:
        for name, path in (("reference", reference), ("trial", trial)):
            db.execute(f"ATTACH DATABASE ? AS {name}", (path.as_uri() + "?mode=ro",))
        queries = {
            "pair_decisions": ("SELECT u,v,gate,final FROM {db}.decisions UNION ALL "
                               "SELECT u,v,-1,final FROM {db}.compact_rejections"),
            "tasks": "SELECT a,b,done FROM {db}.tasks",
        }
        counts = {}
        for label, query in queries.items():
            left, right = (query.format(db=name) for name in ("reference", "trial"))
            for first, second, suffix in ((left, right, "missing"), (right, left, "extra")):
                counts[f"{label}_{suffix}"] = db.execute(
                    f"SELECT count(*) FROM (SELECT * FROM ({first}) EXCEPT SELECT * FROM ({second}))").fetchone()[0]
        counts["significant_p_mismatches"] = db.execute("""
            SELECT count(*) FROM reference.decisions r JOIN trial.decisions t USING(u,v)
            WHERE r.final=1 AND t.final=1 AND (
                (r.p IS NULL)!=(t.p IS NULL) OR
                (r.p IS NOT NULL AND t.p IS NOT NULL
                 AND abs(r.p-t.p)>1e-8*max(abs(r.p),abs(t.p),1e-300)))
        """).fetchone()[0]
        return counts


def plot(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["svg.fonttype"] = "none"

    keys = list(MODES)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.15), layout="constrained")
    for ax, panel, ylabel, field in zip(
            axes, ("A", "B"), ("Search wall time (s)", "Recursive tree contrasts (millions)"),
            ("wall_seconds", "tree_terms")):
        for idx, key in enumerate(keys):
            rows = [row for row in summary["runs"] if row["mode"] == key and not row["resumed"]]
            values = ([row[field] for row in rows] if field == "wall_seconds"
                      else [row["runtime"].get(field, 0) / 1e6 for row in rows])
            if not values:
                continue
            ax.bar(idx, median(values), color="#bdd8eb", edgecolor="#185681", width=.68)
            ax.scatter(idx + np.linspace(-.13, .13, len(values)) if len(values) > 1 else [idx],
                       values, color="#18354a", s=15, zorder=3)
        ax.set_xticks(range(len(keys)), [LABELS[key] for key in keys], rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(panel, loc="left", fontweight="bold")
        ax.set_ylim(bottom=0)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "search_optimization.svg")
    fig.savefig(out / "search_optimization.png", dpi=300)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="Source discovery results directory")
    parser.add_argument("--out", type=Path, help="Separate benchmark directory")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--pairs", type=int, default=1_000_000)
    parser.add_argument("--repeats", type=int, default=1, help="Use four repeats to rotate all four mode positions")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-gb", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--plot", action="store_true", help="Write PNG and SVG after replay")
    parser.add_argument("--plot-summary", type=Path, help="Render an existing summary.json beside that file")
    args = parser.parse_args()
    if args.plot_summary:
        plot(json.loads(args.plot_summary.read_text()), args.plot_summary.parent)
        return 0
    if args.run is None or args.out is None:
        parser.error("--run and --out are required to replay")
    if args.start < 0 or min(args.pairs, args.repeats, args.threads, args.batch_size) < 1:
        parser.error("Start must be nonnegative and counts must be positive")
    args.out = args.out.resolve()
    work, info, source, screen, arrays, identity = source_inputs(args)
    args.out.mkdir(parents=True, exist_ok=True)
    lockfile = args.out / ".replay.lock.sqlite"
    with closing(sqlite3.connect(lockfile, timeout=0)) as lock:
        lock.execute("BEGIN EXCLUSIVE")
        manifest = args.out / "replay_identity.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != identity:
                raise ValueError("Choose a new output folder for a different code version, interval or execution settings")
        else:
            if any(path.name not in (lockfile.name, lockfile.name + "-journal", "seed.sqlite", "seed.partial.sqlite")
                   for path in args.out.iterdir()):
                raise ValueError("Choose an empty benchmark output folder")
            create_seed(args.out, work, arrays, screen)
            atomic_json(manifest, identity)
        results = []
        modes = list(MODES)
        for repeat in range(args.repeats):
            destination = args.out / f"repeat_{repeat + 1:02d}"
            destination.mkdir(exist_ok=True)
            offset = repeat % len(modes)
            order = modes[offset:] + modes[:offset]
            for mode in order:
                print(f"[{repeat + 1}/{args.repeats} {mode}] PC candidates [{identity['start']:,}, {identity['end']:,})",
                      flush=True)
                row = replay(destination / f"{mode}.sqlite", args.out / "seed.sqlite",
                             work, info, source, arrays, args, mode)
                row.update(repeat=repeat + 1, order=order.index(mode) + 1)
                results.append(row)
            reference = destination / "v095.sqlite"
            completed = results[-len(modes):]
            baseline = next(row for row in completed if row["mode"] == "v095")
            for row in completed:
                row["comparison"] = compare(reference, destination / f"{row['mode']}.sqlite") if row["mode"] != "v095" else {}
                elapsed = row["wall_seconds"]
                comparable = not (baseline["resumed"] or row["resumed"])
                speedup = baseline["wall_seconds"] / elapsed if comparable else None
                row["speedup_vs_v095"] = speedup
                timing = f"{elapsed:.3f}s" if elapsed is not None else "untimed"
                ratio = f"; {speedup:.2f}x" if speedup is not None else ""
                print(f"{row['mode']}: {timing}{ratio}; {row['significant']:,} significant; "
                      f"{sum(row['comparison'].values()):,} discrepancies", flush=True)
            atomic_json(args.out / "summary.json", dict(selection=identity, runs=results))
        summary = dict(selection=identity, runs=results)
        failed = any(any(row["comparison"].values()) for row in results)
        if args.plot and not failed:
            plot(summary, args.out)
        return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
