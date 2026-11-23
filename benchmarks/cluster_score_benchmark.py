"""Replay sampled PC-screen candidates using saved KOVAR nulls."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sqlite3
import sys
import time
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import scipy
from numba import njit
from scipy.stats import chi2
from threadpoolctl import threadpool_limits

from ko_variation import coarse_screen, discovery_nulls
from ko_variation.discovery_store import atomic_json, digest, open_patterns
from ko_variation.glmm import KinshipEigensystem, score_predictor_block
from ko_variation.memory_budget import ByteLRU, MIB
from ko_variation.tree_null import TreeAlgebra


def sample_jobs(db, work, selected_pairs, responses, clusters, seed):
    roots, pairs, offsets = coarse_screen.open_candidates(work, selected_pairs)
    rng = np.random.default_rng(seed)
    nonempty = np.flatnonzero(offsets[1:] > offsets[:-1])
    sampled = sorted(rng.choice(nonempty, size=min(responses, len(nonempty)), replace=False))
    jobs = []
    for right in sampled:
        first, last = map(int, offsets[right:right + 2])
        positions = np.sort(rng.choice(last - first, min(clusters, last - first), replace=False)) + first
        response = representative(db, int(roots[right]))
        groups = []
        for left, observed_right in pairs[positions]:
            if observed_right != right or left >= right:
                raise ValueError("Saved candidate columns do not match the offsets")
            node = int(roots[left])
            anchor = representative(db, node)
            # KOVAR uses the higher pattern ID as response.
            if anchor >= response:
                raise ValueError("Saved representatives are out of response order")
            others = [pid for pid, in db.execute(
                "SELECT pid FROM members WHERE node=? AND pid<? AND pid!=? ORDER BY pid",
                (node, response, anchor))]
            groups.append(dict(node=node, members=[anchor, *others]))
        jobs.append(dict(response=response, groups=groups))
    return jobs


def representative(db, node):
    rows = db.execute("SELECT pid FROM reps WHERE node=?", (node,)).fetchall()
    if len(rows) != 1:
        raise ValueError("Expected one saved representative for every PC-screen root")
    return rows[0][0]


def read_selection(run, work, identity, args):
    fingerprints = dict(run_identity=digest(run / "run_identity.json"),
                        prepared=digest(work / "prepared.json"))
    if args.selection:
        frozen = json.loads(args.selection.read_text())
        if frozen["source"] != fingerprints:
            raise ValueError("Selection belongs to a different run or preparation")
        return frozen
    db = sqlite3.connect((work / "work.sqlite").as_uri() + "?mode=ro", uri=True)
    try:
        row = db.execute("SELECT value FROM meta WHERE key='coarse_screen_summary'").fetchone()
        if row is None:
            raise ValueError("Finish the source run's PC screen before sampling")
        screen = json.loads(row[0])
        jobs = sample_jobs(db, work, screen["selected_pairs"], args.responses,
                           args.clusters_per_response, args.seed)
        aliases = discovery_nulls.open_aliases(work)
        for job in jobs:
            response = job["response"]
            canonical = int(aliases[response]) if aliases is not None else response
            local = work / "null_models" / f"{canonical // 1000:06d}" / f"null_{canonical:09d}.json"
            row = db.execute("SELECT path FROM external_nulls WHERE pid=?", (canonical,)).fetchone()
            job["canonical"] = canonical
            job["null"] = str(local if local.is_file() or row is None else row[0])
    finally:
        db.close()
    if not jobs:
        raise ValueError("No retained PC-screen candidates available")
    return dict(source=fingerprints, settings=identity["settings"], screen=screen,
                seed=args.seed, jobs=jobs)


def decode(packed, ids, samples):
    return np.unpackbits(packed[ids], axis=1, count=samples, bitorder="little")


def load_null(job, packed, info, shared):
    path = Path(job["null"])
    if not path.is_file():
        raise ValueError(f"Missing saved null for response {job['response']}: {path}")
    record = json.loads(path.read_text())
    if record["status"] != "OK":
        return None, record["status"]

    def verify(product, expected):
        if digest(product) != expected:
            raise ValueError(f"Saved null checksum differs: {product}")

    K, eigen, tree = shared
    cache = discovery_nulls.load(record, path, eigen, verify, tree)
    canonical = job["canonical"]
    y = decode(packed, [canonical], info["n"])[0]
    if not np.array_equal(y, cache["y"]) or record["pattern"] != canonical:
        raise ValueError(f"Null response differs from saved pattern: {path}")
    if job["response"] != canonical:
        cache["y"] = decode(packed, [job["response"]], info["n"])[0].astype(float)
        if not np.array_equal(cache["y"], 1 - y):
            raise ValueError("Saved response alias is not a complement")
        cache["mu"] = 1 - cache["mu"]
    discovery_nulls.complete(cache, K, eigen)
    return cache, record["mode"]


@njit(cache=True)
def contrast_coefficients(left, right, tip, length, weights, tau):
    variance = np.empty(len(tip))
    left_fraction = np.zeros(len(tip))
    right_fraction = np.zeros(len(tip))
    scale = np.zeros(len(tip))
    for node in range(len(tip)):
        if tip[node] >= 0:
            variance[node] = 1.0 / weights[tip[node]] + tau * length[node]
        else:
            a, b = left[node], right[node]
            total = variance[a] + variance[b]
            left_fraction[node] = variance[b] / total
            right_fraction[node] = variance[a] / total
            variance[node] = variance[a] * left_fraction[node] + tau * length[node]
            scale[node] = math.sqrt(total)
    return left_fraction, right_fraction, scale, math.sqrt(variance[-1])


@njit(cache=True)
def contrast_norms(left, right, tip, fractions_a, fractions_b, scale, root_scale, data):
    width = data.shape[1]
    means = np.empty((len(tip), width))
    norms = np.zeros(width)
    for node in range(len(tip)):
        if tip[node] >= 0:
            means[node, :] = data[tip[node], :]
        else:
            a, b = left[node], right[node]
            for j in range(width):
                z = (means[a, j] - means[b, j]) / scale[node]
                norms[j] += z * z
                means[node, j] = fractions_a[node] * means[a, j] + fractions_b[node] * means[b, j]
    for j in range(width):
        z = means[-1, j] / root_scale
        norms[j] += z * z
    return norms


class TreeScore:
    def __init__(self, cache):
        tree = cache["decomp"].tree
        self.tree_indices = tree.left, tree.right, tree.tip
        self.coefficients = contrast_coefficients(*self.tree_indices, tree.length,
                                                   cache["weights"], cache["tau"])
        self.design, self.inverse_design, self.nuisance = (cache[k] for k in ("C", "Sinv_C", "M_inv"))
        self.residual = cache["y"] - cache["mu"]

    def __call__(self, X):
        coefficient = self.nuisance @ (self.inverse_design.T @ X)
        projected = np.ascontiguousarray(X - self.design @ coefficient)
        variance = contrast_norms(*self.tree_indices, *self.coefficients, projected)
        return X.T @ self.residual, variance


def reference(cache, X):
    results = score_predictor_block(X, cache, compute_spa_adjustment=False)
    return (np.array([r.score_u if r.status == "OK" else np.nan for r in results]),
            np.array([r.score_variance if r.status == "OK" else np.nan for r in results]))


def score_blocks(scorer, X, ids, width, u, v):
    for start in range(0, len(ids), width):
        batch = ids[start:start + width]
        u[batch], v[batch] = scorer(X[:, batch])


def summary_of_cluster(patterns):
    differences = patterns[1:] != patterns[0]
    return np.flatnonzero(differences.any(axis=0)), int(differences.sum(axis=1).max())


def top_sum(values, count):
    return np.partition(values, len(values) - count)[-count:].sum() if count else 0.


def cluster_upper(u, v, residual, weights, variable, radius):
    """Upper bound for U squared over V, with a margin for float64 arithmetic."""
    if not math.isfinite(u) or not math.isfinite(v) or v <= 1e-12:
        return math.inf
    numerator = abs(u) + top_sum(np.abs(residual[variable]), radius)
    difference_norm = math.sqrt(top_sum(weights[variable], radius))
    margin = 1e-9
    denominator = math.sqrt(v) - difference_norm - margin * max(1., math.sqrt(v), difference_norm)
    if denominator <= 0 or denominator * denominator <= 1e-12:
        return math.inf
    return ((numerator + margin * max(1., numerator)) / denominator) ** 2


def weighted_rejections(cache, X, largest_eigenvalue, cutoff):
    """Use the intercept-only weighted variance as a cheap lower bound on V."""
    weights = cache["weights"]
    moments = X.T @ np.column_stack((cache["y"] - cache["mu"], weights))
    u, present_weight = moments.T
    inflation = 1. + cache["tau"] * weights.max() * largest_eigenvalue
    lower = present_weight * (1. - present_weight / weights.sum()) / inflation
    lower -= 1e-9 * np.maximum(1., np.abs(lower))
    upper_u = np.abs(u) + 1e-9 * np.maximum(1., np.abs(u))
    return (lower > 1e-12) & (upper_u ** 2 < cutoff * lower)


def classify(u, v, eligible, gate, final, skipped):
    valid = np.isfinite(u) & np.isfinite(v) & (v > 1e-12)
    statistic = np.full(u.size, np.nan)
    statistic[valid] = (u[valid] * (1. / np.sqrt(v[valid]))) ** 2
    valid &= np.isfinite(statistic)
    route = np.where(skipped, -1, np.where(valid, np.where(statistic >= gate, 1, -1), 0))
    decided = np.where(skipped, -1, np.where(valid, np.where(statistic >= final, 1, -1), 0))
    return route, np.where(eligible, decided, -2), statistic


def compare_methods(cache, X, groups, eligible, args, gate, final, largest_eigenvalue, warm, use_bounds):
    methods = [("current_block", args.reference_block), ("wide_block", args.batch_size)]
    tree = hasattr(cache["decomp"], "score_variance")
    if tree:
        methods.extend((("fused_tree", args.batch_size), ("weighted_bound", args.batch_size)))
    if use_bounds:
        methods.append(("cluster_bound", args.batch_size))
    if args.scalar:
        methods.append(("single_pair", 1))
        if tree:
            methods.append(("fused_single_pair", 1))
    if warm:
        reference(cache, X[:, :1])
        if tree:
            TreeScore(cache)(X[:, :1])
    ids = np.arange(X.shape[1])
    scores = {}
    report = {}
    for method, width in methods:
        started = time.perf_counter()
        scorer = (TreeScore(cache) if tree and method in
                  ("fused_tree", "fused_single_pair", "weighted_bound", "cluster_bound")
                  else lambda x: reference(cache, x))
        u, v = np.full(X.shape[1], np.nan), np.full(X.shape[1], np.nan)
        skipped = np.zeros(X.shape[1], dtype=bool)
        scored = ids
        clusters_pruned = 0
        cutoff = min(gate, final)
        if method == "weighted_bound":
            skipped = weighted_rejections(cache, X, largest_eigenvalue, cutoff)
            scored = ids[~skipped]
        if method == "cluster_bound" and groups:
            anchors = np.array([group[0] for group in groups])
            score_blocks(scorer, X, anchors, width, u, v)
            residual = cache["y"] - cache["mu"]
            for first, last, variable, radius in groups:
                upper = cluster_upper(u[first], v[first], residual, cache["weights"], variable, radius)
                if upper < cutoff:
                    skipped[first + 1:last] = True
                    clusters_pruned += 1
            remaining = ~skipped
            remaining[anchors] = False
            scored = ids[remaining]
        score_blocks(scorer, X, scored, width, u, v)
        result = classify(u, v, eligible, gate, final, skipped)
        seconds = time.perf_counter() - started
        scores[method] = result
        report[method] = dict(seconds=seconds, pairs=X.shape[1], pruned_pairs=int(skipped.sum()),
                              pruned_clusters=clusters_pruned, significant=int((result[1] == 1).sum()))
    route, final_decisions, statistic = scores["current_block"]
    for method, (r, f, t) in scores.items():
        values = report[method]
        compared = np.isfinite(t)
        values.update(gate_mismatches=int((route != r).sum()), final_mismatches=int((final_decisions != f).sum()),
                      statistic_mismatches=int((~np.isclose(t[compared], statistic[compared],
                                                            rtol=1e-9, atol=1e-10)).sum()))
    return report


def workloads(job, packed, n, cluster_cache, timing):
    groups = [group["members"] for group in job["groups"]]
    yield "coarse_representatives", decode(packed, [ids[0] for ids in groups], n).T.astype(float), []
    predictors = decode(packed, [pid for ids in groups for pid in ids], n)
    envelopes, first = [], 0
    for ids in groups:
        last = first + len(ids)
        if len(ids) > 1:
            key = tuple(ids)
            if key in cluster_cache:
                envelope = cluster_cache.get(key)
            else:
                started = time.perf_counter()
                envelope = summary_of_cluster(predictors[first:last])
                cluster_cache.put(key, envelope, 128 + 32 * len(key) + envelope[0].nbytes)
                timing["cluster_summary_seconds"] += time.perf_counter() - started
            envelopes.append((first, last, *envelope))
        first = last
    yield "member_expansion", predictors.T.astype(float), envelopes


def benchmark(args):
    run = args.run.resolve()
    identity = json.loads((run / "run_identity.json").read_text())
    work = Path(identity.get("work_directory", run / ".kovar_work")).resolve()
    if args.out.resolve().is_relative_to(run) or args.out.resolve().is_relative_to(work):
        raise ValueError("Place --out outside the source results and work directories")
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Use an empty output directory; reuse a selection with --selection")
    info = json.loads((work / "prepared.json").read_text())
    frozen = read_selection(run, work, identity, args)
    args.out.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out / "selection.json", frozen)
    if args.prepare_only:
        return 0
    settings = identity["settings"]
    family = settings.get("bonferroni_tests") or math.comb(info["patterns"], 2)
    gate, final = (float(chi2.isf(p, 1)) for p in (settings["screen_p"], settings["alpha"] / family))
    tree = TreeAlgebra.load(work / "tree.npz") if (work / "tree.npz").is_file() else None
    shared = (np.load(work / "K.npy", mmap_mode="r"),
              KinshipEigensystem(np.load(work / "K_values.npy", mmap_mode="r"),
                                 np.load(work / "K_vectors.npy", mmap_mode="r"), info["shared_covariance"]),
              tree)
    largest_eigenvalue = max(0., float(shared[1].eigenvalues.max()))
    packed = open_patterns(work, info)
    rows, failed = [], []
    cluster_cache = ByteLRU(64 * MIB)
    timing = dict(null_load_seconds=0., panel_prepare_seconds=0., cluster_summary_seconds=0.)
    warmed = set()
    for job in frozen["jobs"]:
        started = time.perf_counter()
        cache, mode = load_null(job, packed, info, shared)
        timing["null_load_seconds"] += time.perf_counter() - started
        response = job["response"]
        if cache is None:
            failed.append(dict(response=response, status=mode))
            continue
        started = time.perf_counter()
        for label, X, groups in workloads(job, packed, info["n"], cluster_cache, timing):
            y = cache["y"]
            overlap, count = X.T @ y, X.sum(axis=0)
            eligible = np.minimum.reduce((overlap, count - overlap, y.sum() - overlap,
                                         len(y) - count - y.sum() + overlap)) >= settings["min_cell_count"]
            timing["panel_prepare_seconds"] += time.perf_counter() - started
            results = compare_methods(cache, X, groups, eligible, args, gate, final, largest_eigenvalue,
                                      mode not in warmed, label == "member_expansion")
            warmed.add(mode)
            rows.extend(dict(response=response, workload=label, null_mode=mode, method=method,
                             clusters=len(job["groups"]), non_singletons=len(groups), **values)
                        for method, values in results.items())
            started = time.perf_counter()
    if not rows:
        raise ValueError("No usable saved nulls in the selected slice")
    output = args.out / "response_results.tsv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    totals = defaultdict(lambda: defaultdict(int))
    for row in rows:
        key = row["workload"], row["null_mode"], row["method"]
        for field in ("seconds", "pairs", "clusters", "non_singletons", "pruned_pairs", "pruned_clusters",
                      "significant", "gate_mismatches", "final_mismatches", "statistic_mismatches"):
            totals[key][field] += row[field]
    summary = []
    for (workload, mode, method), values in sorted(totals.items()):
        baseline = totals[workload, mode, "current_block"]
        result = dict(workload=workload, null_mode=mode, method=method, **values,
                      pruned_fraction=values["pruned_pairs"] / values["pairs"],
                      speedup=baseline["seconds"] / values["seconds"])
        fused = totals.get((workload, mode, "fused_tree"))
        if fused and method in ("weighted_bound", "cluster_bound"):
            result["speedup_vs_fused"] = fused["seconds"] / values["seconds"]
        summary.append(result)
    atomic_json(args.out / "summary.json", dict(samples=info["n"], family=family, gate=gate, final=final,
                reference_block=args.reference_block, batch_size=args.batch_size, blas_threads=args.blas_threads,
                benchmark_sha256=digest(Path(__file__)),
                numpy=np.__version__, scipy=scipy.__version__, failed_responses=failed, **timing, methods=summary))
    for row in summary:
        print(f"{row['workload']} / {row['null_mode']} / {row['method']}: "
              f"{row['seconds']:.3f}s; {row['speedup']:.2f}x; {row['pruned_fraction']:.1%} pruned")
    return int(any(row[key] for row in summary for key in ("gate_mismatches", "final_mismatches", "statistic_mismatches")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Source discovery results directory")
    parser.add_argument("--out", type=Path, required=True, help="Independent empty benchmark directory")
    parser.add_argument("--selection", type=Path, help="Replay a saved selection.json")
    parser.add_argument("--prepare-only", action="store_true", help="Freeze the candidate selection and stop")
    parser.add_argument("--responses", type=int, default=64)
    parser.add_argument("--clusters-per-response", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference-block", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--scalar", action="store_true", help="Also time one predictor per call")
    args = parser.parse_args()
    if min(args.responses, args.clusters_per_response, args.reference_block, args.batch_size, args.blas_threads) < 1:
        parser.error("Counts and block sizes must be positive")
    with threadpool_limits(limits=args.blas_threads):
        return benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
