"""Four-cell benzylpenicillin MIC analysis of blaZ and first-neighbour unitigs."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from benchmarks import saureus_mic as mic


CELLS = ("00", "01", "10", "11")


def cell_counts(a, b, censor, groups):
    cells = 2 * a + b
    lineages = [set(groups[cells == index]) for index in range(4)]
    for index, cell in enumerate(CELLS):
        subset = cells == index
        exact = int(np.count_nonzero(subset & (censor == 0)))
        total = int(np.count_nonzero(subset))
        yield dict(cell=cell, n=total, n_exact=exact, n_censored=total - exact,
                   n_lineages=len(lineages[index]),
                   shared_lineages_within_blaZ=len(lineages[index] & lineages[index ^ 1]))


def support(cells, minimum):
    """An additive fit needs both A states and a within-A contrast in B."""
    counts = np.bincount(cells, minlength=4).reshape(2, 2)
    additive = np.all(counts.sum(axis=1) >= minimum) and np.any(
        np.all(counts >= minimum, axis=1)
    )
    return additive, bool(np.all(counts >= minimum))


def held_out_fit(columns, y, censor, train, test, alpha):
    raw_train = columns[train]
    mean = raw_train.mean(axis=0)
    sd = raw_train.std(axis=0)
    design_train = np.column_stack((np.ones(len(train)), (raw_train - mean) / sd))
    design_test = np.column_stack((np.ones(len(test)), (columns[test] - mean) / sd))
    beta, sigma = mic.fit_censored(design_train, y[train], censor[train], alpha, 1)
    predicted = design_test @ beta
    loss = mic.prediction_loss(y[test], censor[test], predicted, sigma)
    if not np.isfinite(loss).all():
        raise ValueError("Non-finite held-out likelihood")
    return loss, beta[1:] / sd


def evaluate_pair(a, b, y, censor, train, test, alpha, minimum, baseline_loss,
                  baseline_error):
    cells = 2 * a + b
    eligible = support(cells[train], minimum)
    rows = []
    for index, model in enumerate(("additive", "interaction")):
        row = dict(model=model, status="SPARSE_TRAIN", reason="")
        if not eligible[index]:
            row["reason"] = ("Both blaZ states and a within-state neighbour contrast required"
                             if index == 0 else "Too few observations in at least one training cell")
        elif baseline_error:
            row.update(status="FAILED", reason=f"blaZ baseline: {baseline_error}")
        else:
            columns = np.column_stack((a, b, a * b)) if index else np.column_stack((a, b))
            try:
                loss, beta = held_out_fit(columns, y, censor, train, test, alpha)
            except ValueError as error:
                row.update(status="FAILED", reason=str(error))
            else:
                effect = float(beta[2]) if index else 0.
                row.update(status="OK", neighbour_effect_A0=float(beta[1]),
                           neighbour_effect_A1=float(beta[1] + effect),
                           interaction_effect=effect)
                gain = baseline_loss - loss
                row["scores"] = []
                for cell in ("all", *CELLS):
                    mask = np.ones(len(test), dtype=bool) if cell == "all" else cells[test] == int(cell, 2)
                    if mask.any():
                        row["scores"].append(dict(
                            cell=cell, n=int(mask.sum()), delta_nll_sum=float(gain[mask].sum()),
                            nll_sum=float(loss[mask].sum()), baseline_nll_sum=float(baseline_loss[mask].sum())
                        ))
        rows.append(row)
    return rows


def match_controls(a, groups, matrix, targets, minimum, repeats, seed):
    """Rank by weighted differences of within-(blaZ, lineage) presence rates."""
    strata = list(zip(a, groups))
    levels = sorted(set(strata))
    masks = np.asarray([[s == group for s in strata] for group in levels], dtype=float)
    sizes = masks.sum(axis=1)
    arm = np.array([state for state, _ in levels])
    rates = (masks @ matrix) / sizes[:, None]
    target_rates = (masks @ targets) / sizes[:, None]
    eligible = [support(2 * a + b, minimum) for b in matrix.T]
    patterns = [mic.pattern_key(column) for column in matrix.T]
    excluded = {mic.pattern_key(a), *(mic.pattern_key(column) for column in targets.T)}
    rng = np.random.default_rng(seed)
    for index, target in enumerate(targets.T):
        add, interaction = support(2 * a + target, minimum)
        if not add or repeats == 0:
            yield []
            continue
        differences = abs(rates - target_rates[:, index, None])
        d0, d1 = (np.average(differences[arm == state], axis=0, weights=sizes[arm == state])
                  for state in (0, 1))
        distance = (d0 + d1) / 2
        order = np.lexsort((rng.random(len(distance)), distance))
        seen = excluded.copy()
        chosen = []
        for j in order:
            if patterns[j] in seen or eligible[j] != (add, interaction):
                continue
            seen.add(patterns[j])
            chosen.append((int(j), dict(match_distance=float(distance[j]),
                                        match_distance_A0=float(d0[j]), match_distance_A1=float(d1[j]))))
            if len(chosen) == repeats:
                break
        rng.shuffle(chosen)
        yield chosen


def analyze_fold(job):
    fold, train, test, endpoint = job
    args, info, matrix, _, names = mic._STATE
    indices = endpoint["indices"]
    y, censor, groups = (endpoint[k] for k in ("bound", "censor", "groups"))
    a = matrix[indices, names.index("blaZ_presence")].astype(int)
    first = sorted(set(info["neighbour_columns"]["penicillin"]))
    pool = sorted(set(info["random_pool"]) - set(first) - set(info["target_columns"]["penicillin"]))
    columns = matrix[np.ix_(indices, len(names) + np.arange(len(info["loci"])))].astype(int)
    matches = match_controls(a[train], groups[train], columns[np.ix_(train, pool)],
                             columns[np.ix_(train, first)], args.min_count,
                             args.random_sets, args.seed + fold)
    baseline_loss, baseline_error = None, ""
    if np.min(np.bincount(a[train], minlength=2)) < args.min_count:
        baseline_error = "Too few observations in a blaZ training state"
    else:
        try:
            baseline_loss, _ = held_out_fit(a[:, None], y, censor, train, test, args.pair_alpha)
        except ValueError as error:
            baseline_error = str(error)
    cells, results, controls = [], [], []
    cache = {}
    for target, selected in zip(first, matches):
        locus = info["loci"][target]
        candidates = [(target, 0, "neighbour", None)]
        candidates.extend((pool[j], repeat, "random", match)
                          for repeat, (j, match) in enumerate(selected, 1))
        for candidate, repeat, source, match in candidates:
            b = columns[:, candidate]
            key = dict(neighbour=locus, unitig=info["loci"][candidate], source=source, repeat=repeat, fold=fold)
            if source == "random":
                controls.append(dict(**key, **match))
            for split, rows in (("train", train), ("test", test)):
                cells.extend(dict(**key, split=split, **cell)
                             for cell in cell_counts(a[rows], b[rows], censor[rows], groups[rows]))
            if candidate not in cache:
                cache[candidate] = evaluate_pair(a, b, y, censor, train, test, args.pair_alpha,
                                                 args.min_count, baseline_loss, baseline_error)
            results.extend(dict(**key, **row) for row in cache[candidate])
    return dict(cells=cells, results=results, controls=controls)


def mean_scores(rows):
    n = sum(row["n"] for row in rows)
    return dict(n=n, **{metric: sum(row[f"{metric}_sum"] for row in rows) / n if n else None
                        for metric in ("nll", "baseline_nll", "delta_nll")})


def pair_reports(output, folds, total_folds, repeats):
    by_pair = defaultdict(list)
    fits, scores = [], []
    for fold in folds:
        for row in fold["results"]:
            key = {k: v for k, v in row.items() if k != "scores"}
            fits.append(key)
            by_pair[(row["neighbour"], row["source"], row["repeat"], row["model"])].append(row)
            scores.extend(dict(**key, cell=s["cell"], **mean_scores([s]))
                          for s in row.get("scores", []))
    pooled, complete = [], defaultdict(list)
    for (neighbour, source, repeat, model), rows in sorted(by_pair.items()):
        successful = [r for r in rows if r["status"] == "OK"]
        key = dict(neighbour=neighbour, source=source, repeat=repeat, model=model,
                   completed_folds=len(successful), total_folds=total_folds)
        if len(successful) != total_folds:
            pooled.append(dict(**key, cell="all", status="INCOMPLETE"))
            continue
        for cell in ("all", *CELLS):
            subset = [s for row in successful for s in row["scores"] if s["cell"] == cell]
            averages = mean_scores(subset)
            pooled.append(dict(**key, cell=cell, status="OK" if averages["n"] else "NO_CELL", **averages))
            if cell == "all":
                complete[(neighbour, model, source)].append(averages["delta_nll"])
    comparisons = []
    for neighbour in sorted({r["neighbour"] for r in fits if r["source"] == "neighbour"}):
        for model in ("additive", "interaction"):
            actual = complete[(neighbour, model, "neighbour")]
            random = complete[(neighbour, model, "random")]
            comparisons.append(dict(neighbour=neighbour, model=model, requested_random_sets=repeats,
                complete_random_sets=len(random), neighbour_delta_nll=actual[0] if actual else None,
                random_median_delta_nll=float(np.median(random)) if random else None,
                random_fraction_at_least_as_good=(sum(g >= actual[0] for g in random) / len(random)
                                                if actual and random and len(random) == repeats else None)))
        add = complete[(neighbour, "additive", "neighbour")]
        interaction = complete[(neighbour, "interaction", "neighbour")]
        if add and interaction:
            comparisons[-1]["delta_nll_vs_additive"] = interaction[0] - add[0]
    keys = ["neighbour", "unitig", "source", "repeat", "fold"]
    cell_fields = ["cell", "n", "n_exact", "n_censored", "n_lineages", "shared_lineages_within_blaZ"]
    score_fields = ["n", "baseline_nll", "nll", "delta_nll"]
    mic.write_tsv(output/"fold_cells.tsv", (r for f in folds for r in f["cells"]),
                  keys + ["split"] + cell_fields)
    mic.write_tsv(output/"random_matches.tsv", (r for f in folds for r in f["controls"]),
                  keys + ["match_distance", "match_distance_A0", "match_distance_A1"])
    mic.write_tsv(output/"pair_fit_status.tsv", fits,
                  keys + ["model", "status", "reason", "neighbour_effect_A0",
                          "neighbour_effect_A1", "interaction_effect"])
    mic.write_tsv(output/"fold_scores.tsv", scores, keys + ["model", "cell"] + score_fields)
    mic.write_tsv(output/"pair_scores.tsv", pooled,
                  ["neighbour", "source", "repeat", "model", "cell", "status",
                   "completed_folds", "total_folds"] + score_fields)
    mic.write_tsv(output/"random_comparison.tsv", comparisons,
                  ["neighbour", "model", "neighbour_delta_nll", "delta_nll_vs_additive",
                   "requested_random_sets", "complete_random_sets", "random_median_delta_nll",
                   "random_fraction_at_least_as_good"])


def run(args):
    root = args.out.resolve()
    audit = root/"audit_identity.json"
    if not audit.is_file() or json.loads(audit.read_text()).get("revision") != mic.AUDIT_REVISION:
        raise ValueError("Run the current allele audit before the pairs stage")
    settings = {name: getattr(args, name) for name in (
        "folds", "seed", "min_count", "pair_alpha", "random_sets",
        "penicillin_column", "penicillin_sign_column")}
    identity = dict(settings=settings, inputs={name: mic.sha256(root/name) for name in
        ("prepared.json", "prepared.npz", "genotypes.npz", "audit_identity.json")},
        code={Path(path).name: mic.sha256(path) for path in (__file__, mic.__file__)},
        dependencies={pkg: version(pkg) for pkg in ("numpy", "scipy", "scikit-learn")})
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    output = root/"pairs"/run_id
    output.mkdir(parents=True, exist_ok=True)
    mic.json_write(output/"pair_identity.json", identity)
    worker_settings = dict(settings, out=str(root), prediction_dir=str(output))
    mic.worker_init(worker_settings)
    endpoint = mic.endpoint_data("penicillin")
    _, info, matrix, _, names = mic._STATE
    first = sorted(set(info["neighbour_columns"]["penicillin"]))
    if not first:
        raise ValueError("The prepared map contains no first-neighbour coding unitigs for blaZ")
    indices = endpoint["indices"]
    a = matrix[indices, names.index("blaZ_presence")].astype(int)
    audit_rows = []
    for j in first:
        b = matrix[indices, len(names) + j].astype(int)
        for row in cell_counts(a, b, endpoint["censor"], endpoint["groups"]):
            locus = info["loci"][j]
            audit_rows.append(dict(neighbour=locus, gene=info["locus_gene"][str(locus)], **row))
    mic.write_tsv(output/"observed_cells.tsv", audit_rows,
                  ["neighbour", "gene", "cell", "n", "n_exact", "n_censored",
                   "n_lineages", "shared_lineages_within_blaZ"])
    jobs = [(i, train, test, {k: endpoint[k] for k in ("indices", "bound", "censor", "groups")})
            for i, (train, test) in enumerate(endpoint["folds"])]
    mic.log(f"Four-cell output: {output}; {len(first)} first-neighbour unitigs; "
            f"{len(jobs)} lineage folds; up to {args.random_sets} matched controls per fold")
    results = {}
    pending = []
    for job in jobs:
        path = output/f"fold{job[0]}.json"
        if path.is_file():
            results[job[0]] = json.loads(path.read_text())
        else:
            pending.append(job)
    if pending:
        with ProcessPoolExecutor(max_workers=min(args.threads, len(pending)),
                                 initializer=mic.worker_init, initargs=(worker_settings,)) as executor:
            for job, result in zip(pending, executor.map(analyze_fold, pending)):
                mic.json_write(output/f"fold{job[0]}.json", result)
                results[job[0]] = result
                mic.log(f"Four-cell fold {job[0]} saved")
    pair_reports(output, [results[job[0]] for job in jobs], len(jobs), args.random_sets)
    mic.json_write(root/"latest_pair_analysis.json", dict(path=str(output), run_id=run_id))
    mic.log(f"Four-cell analysis complete: {output}")
