"""Maximum-likelihood Gaussian phylogenetic models and identifiable pair contrasts.

Prediction uses conditional random-effect means for held-out tips. Interaction
testing compares ML fits, not penalized ridge coefficients or genomic KOVAR P-values.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
from scipy.stats import chi2, norm

from .amr_report import write_tsv


def covariance_cache(K, y):
    values, vectors = eigh(K, check_finite=False)
    if values.min() < -1e-7:
        raise ValueError("Phylogenetic covariance is not positive semidefinite")
    return dict(values=np.maximum(values, 0.), vectors=vectors, rotated_y=vectors.T @ y)


def design(X, candidates, min_count):
    """Keep baseline columns first; reject rare and linearly dependent columns.

    Standardization is fitted on training rows only. Products are made from raw
    binary states first; a product identical to a main effect is not an interaction.
    """
    kept, means, scales, columns, removed = [], [], [], [], []
    Q = np.ones((len(X), 1)) / np.sqrt(len(X))
    for name, values in candidates:
        values = np.asarray(values, dtype=float)
        count = int(values.sum())
        if min(count, len(values)-count) < min_count:
            removed.append(dict(feature=name, reason="rare_or_constant"))
            continue
        mean, scale = float(values.mean()), float(values.std())
        column = (values-mean)/scale
        residual = column-Q @ (Q.T @ column)
        # Reorthogonalize to avoid numeric drift when binary features are correlated.
        residual -= Q @ (Q.T @ residual)
        length = float(np.linalg.norm(residual))
        if length < 1e-8 * np.sqrt(len(X)):
            removed.append(dict(feature=name, reason="linear_dependency"))
            continue
        Q = np.column_stack([Q, residual/length])
        kept.append(name)
        means.append(mean)
        scales.append(scale)
        columns.append(column)
    return dict(F=np.column_stack([np.ones(len(X))] + columns), names=kept,
                means=np.array(means), scales=np.array(scales), removed=removed)


def candidate_values(X, columns, interactions=()):
    values = [(f"unitig_{c}", X[:, c]) for c in columns]
    values.extend((f"pair_{u}_{v}", X[:, u]*X[:, v]) for u, v in interactions)
    return values


def transformed(X, fitted):
    values = []
    for name in fitted["names"]:
        if name.startswith("unitig_"):
            values.append(X[:, int(name.split("_")[1])])
        else:
            _, u, v = name.split("_")
            values.append(X[:, int(u)]*X[:, int(v)])
    if not values:
        return np.ones((len(X), 1))
    return np.column_stack([np.ones(len(X)),
                            (np.column_stack(values)-fitted["means"])/fitted["scales"]])


def fit_lmm(X, y, columns, interactions, cache, min_count, phylogeny=True):
    """Profile ML over h in V=s²[(1-h)I+hK]; fixed effects are unpenalized GLS."""
    result = design(X, candidate_values(X, columns, interactions), min_count)
    F, Q = result["F"], cache["vectors"]
    if len(y) <= F.shape[1]+2:
        raise ValueError("Insufficient observations for the retained fixed-effect design")
    rotated = Q.T @ F
    ry = cache["rotated_y"]

    def evaluate(h):
        eigen = 1-h+h*cache["values"]
        weights = 1/eigen
        information = rotated.T @ (rotated*weights[:, None])
        beta = np.linalg.solve(information, rotated.T @ (ry*weights))
        residual = ry-rotated @ beta
        variance = max(float(np.dot(residual*weights, residual)/len(y)), 1e-12)
        nll = .5*(len(y)*(np.log(2*np.pi)+1+np.log(variance))+np.log(eigen).sum())
        return float(nll), beta, variance, information, residual*weights

    h = 0.
    if phylogeny:
        optimum = minimize_scalar(lambda value: evaluate(value)[0], bounds=(0., .999999),
                                  method="bounded", options={"xatol": 1e-6})
        if not optimum.success:
            raise ValueError("Phylogenetic variance optimization failed")
        h = min((0., float(optimum.x), .999999), key=lambda value: evaluate(value)[0])
    nll, beta, variance, information, rotated_residual = evaluate(h)
    result.update(beta=beta, variance=variance, h=h, nll=nll,
                  coefficient_covariance=variance*np.linalg.inv(information),
                  residual_weights=Q @ rotated_residual)
    return result


def predict(fitted, X, Kcross):
    return transformed(X, fitted) @ fitted["beta"] + fitted["h"] * Kcross @ fitted["residual_weights"]


def scores(y, prediction):
    error = y-prediction
    return dict(rmse=float(np.sqrt(np.mean(error**2))),
                within_one_dilution=float(np.mean(np.abs(error) <= 1)))


def matching_random(X, base, selected, pool, minimum, rng):
    """Approximate training minor-frequency match with equal independent main-effect count."""
    source = design(X, candidate_values(X, selected), minimum)
    baseline = design(X, candidate_values(X, base), minimum)
    wanted = [int(name.split("_")[1]) for name in source["names"] if name not in baseline["names"]]
    frequencies = np.minimum(X.mean(axis=0), 1-X.mean(axis=0))
    candidates = [int(c) for c in pool if min(int(X[:, c].sum()), len(X)-int(X[:, c].sum())) >= minimum]
    retained = [int(name.split("_")[1]) for name in baseline["names"]]
    extra, errors = [], []
    for column in rng.permutation(wanted):
        # Random tie-breaking is deterministic for this fold/replicate seed.
        rng.shuffle(candidates)
        ordered = sorted(candidates, key=lambda c: abs(frequencies[c]-frequencies[column]))
        for candidate in ordered:
            proposal = design(X, candidate_values(X, retained+extra+[candidate]), minimum)
            if len(proposal["names"]) == len(retained)+len(extra)+1:
                extra.append(candidate)
                errors.append(float(abs(frequencies[candidate]-frequencies[column])))
                candidates.remove(candidate)
                break
        else:
            raise ValueError("Random pool cannot match the independent main-effect count")
    return retained+extra, errors


def random_products(X, columns, base, needed, minimum, min_cell, rng):
    if needed <= 0:
        return []
    options = [(u, v) for u in base for v in columns if v not in base]
    rng.shuffle(options)
    accepted = []
    for u, v in options:
        count = np.bincount(X[:, u].astype(int)+2*X[:, v].astype(int), minlength=4)
        if count.min() < min_cell:
            continue
        proposal = design(X, candidate_values(X, columns, accepted+[(u, v)]), minimum)
        if f"pair_{u}_{v}" in proposal["names"]:
            accepted.append((u, v))
        if len(accepted) == needed:
            break
    return accepted


def fit_prediction(args, samples, X, loci, features, pool, pairs, phenotype, groups, K, out, identity):
    from sklearn.model_selection import GroupKFold
    from threadpoolctl import threadpool_limits
    from .phenotype import json_write, npz_write, log

    indices = np.array([i for i, p in enumerate(phenotype) if p["status"] == "exact"])
    if len(indices) < 20:
        raise ValueError("Fewer than 20 exact MIC measurements")
    y = np.array([phenotype[i]["log2_mic"] for i in indices])
    needed = sorted(set(pool.tolist()) | {f["fasta_column"] for f in features})
    remap = {column: i for i, column in enumerate(needed)}
    features = [dict(f, fasta_column=remap[f["fasta_column"]]) for f in features]
    pool = np.array([remap[c] for c in pool], dtype=int)
    loci = loci[needed]
    data, group = X[np.ix_(indices, needed)], groups[indices]
    K = K[np.ix_(indices, indices)]
    lookup = {int(locus): i for i, locus in enumerate(loci)}
    direct = [(lookup[p["u"]], lookup[p["v"]]) for p in pairs if p["direct_target"]]
    model_columns = {f"M{i}": [f["fasta_column"] for f in features if f["neighbourhood_order"] <= i]
                     for i in range(3)}
    nfolds = min(args.folds, len(set(group)))
    if nfolds < 3:
        raise ValueError("At least three phenotype-bearing groups required; do not change fastBAPS settings using MIC performance")
    splits = list(GroupKFold(n_splits=nfolds).split(data, y, group))
    labels = ["Mean_only", "Phylogeny_only", "M0", "M1", "M2", "M1_pair"] + [
        f"Random{size}_{r:03d}" for size in (1, 2) for r in range(1, args.random_sets+1)] + [
        f"RandomPair_{r:03d}" for r in range(1, args.random_sets+1)]
    predictions = np.full((len(labels), len(y)), np.nan)
    folds, training_rows, feature_rows, test_folds = [], [], [], np.zeros(len(y), dtype=int)
    checkpoint = out / "prediction_checkpoints"
    checkpoint.mkdir(exist_ok=True)
    with threadpool_limits(limits=1):
        for number, (train, test) in enumerate(splits, 1):
            test_folds[test] = number
            record, array = checkpoint / f"fold_{number}.json", checkpoint / f"fold_{number}.npz"
            if record.exists() and array.exists() and json.loads(record.read_text())["identity"] == identity:
                saved = json.loads(record.read_text())
                with np.load(array, allow_pickle=False) as loaded:
                    if not np.array_equal(loaded["test"], test):
                        raise ValueError("Checkpoint partition differs")
                    predictions[:, test] = loaded["predictions"]
                folds.extend(saved["folds"])
                training_rows.extend(saved["training"])
                feature_rows.extend(saved["features"])
                log(f"reuse completed mixed-model fold {number}/{nfolds}")
                continue
            log(f"mixed-model fold {number}/{nfolds}: train={len(train)}, test={len(test)}")
            x, target = data[train], y[train]
            ktrain, kcross = K[np.ix_(train, train)], K[np.ix_(test, train)]
            cache = covariance_cache(ktrain, target)
            usable_pairs = [(u, v) for u, v in direct if np.bincount(
                x[:, u].astype(int)+2*x[:, v].astype(int), minlength=4).min() >= args.interaction_min_cell]
            paired_design = design(x, candidate_values(x, model_columns["M1"], usable_pairs), args.min_count)
            n_interactions = sum(name.startswith("pair_") for name in paired_design["names"])
            saved_fold, saved_training, saved_features = [], [], []
            for m, label in enumerate(labels):
                differences, products, mismatch = [], [], False
                if label in {"Mean_only", "Phylogeny_only"}:
                    columns = []
                elif label.startswith("Random"):
                    family, replicate = label.split("_")
                    size = 2 if family == "Random2" else 1
                    rng = np.random.default_rng(np.random.SeedSequence([args.seed, number, size, int(replicate)]))
                    columns, differences = matching_random(x, model_columns["M0"], model_columns[f"M{size}"],
                                                           pool, args.min_count, rng)
                    if family == "RandomPair" and n_interactions:
                        base = [int(name.split("_")[1]) for name in design(
                            x, candidate_values(x, model_columns["M0"]), args.min_count)["names"]]
                        products = random_products(x, columns, base, n_interactions, args.min_count,
                                                   args.interaction_min_cell, rng)
                        mismatch = len(products) != n_interactions
                else:
                    columns = model_columns["M1" if label == "M1_pair" else label]
                    products = usable_pairs if label == "M1_pair" else []
                fitted = fit_lmm(x, target, columns, products, cache, args.min_count,
                                 phylogeny=label != "Mean_only")
                if label == "M1_pair":
                    for u, v in direct:
                        if (u, v) not in usable_pairs:
                            counts = np.bincount(x[:, u].astype(int)+2*x[:, v].astype(int), minlength=4)
                            fitted["removed"].append(dict(feature=f"pair_{u}_{v}", reason=
                                "missing_genotype_state" if not counts.all() else "sparse_genotype_states"))
                predictions[m, test] = predict(fitted, data[test], kcross)
                n_pair = sum(name.startswith("pair_") for name in fitted["names"])
                saved_fold.append(dict(fold=number, model=label, n_test=len(test), n_features=len(fitted["names"]),
                    n_interactions=n_pair, h=fitted["h"], variance=fitted["variance"],
                    max_frequency_mismatch=max(differences, default=0.),
                    control_count_match=not mismatch, **scores(y[test], predictions[m, test])))
                saved_training.append(dict(fold=number, model=label, n_train=len(train),
                    **scores(target, predict(fitted, x, ktrain))))

                def original(name):
                    parts = name.split("_")
                    return "unitig_"+str(int(loci[int(parts[1])])) if parts[0] == "unitig" else \
                           "pair_"+"_".join(str(int(loci[int(c)])) for c in parts[1:])

                saved_features.append(dict(fold=number, model=label, retained_features=";".join(
                    original(name) for name in fitted["names"]), removed_features=json.dumps([
                    dict(feature=original(row["feature"]), reason=row["reason"]) for row in fitted["removed"]])))
                if label.startswith("Random") and int(label.split("_")[1]) % 20 == 0:
                    log(f"fold {number}: {label} complete")
            npz_write(array, test=test, predictions=predictions[:, test])
            json_write(record, dict(identity=identity, folds=saved_fold, training=saved_training, features=saved_features))
            folds.extend(saved_fold)
            training_rows.extend(saved_training)
            feature_rows.extend(saved_features)
        if not np.isfinite(predictions).all():
            raise ValueError("Incomplete predictions")
        base = scores(y, predictions[labels.index("M0")])["rmse"]
        performance = []
        for label, prediction in zip(labels, predictions):
            matching = all(row["control_count_match"] for row in folds if row["model"] == label)
            result = scores(y, prediction)
            performance.append(dict(model=label, n_test=len(y), **result,
                                    rmse_reduction_vs_M0=base-result["rmse"], control_count_match=matching))
        for name, records in (("performance", performance), ("fold_performance", folds),
                              ("training_performance", training_rows), ("model_features", feature_rows)):
            write_tsv(out / f"{name}.tsv", records, list(records[0]))
        write_tsv(out / "predictions.tsv", (dict(sample=samples[indices[i]], fold=int(test_folds[i]),
                  model=label, observed_log2_mic=float(y[i]), predicted_log2_mic=float(predictions[m, i]))
                  for m, label in enumerate(labels) for i in range(len(y))),
                  ["sample", "fold", "model", "observed_log2_mic", "predicted_log2_mic"])
        interaction_report(args, data, y, K, loci, model_columns["M0"], pairs, lookup, out, identity)
    plots(args, y, predictions, labels, performance, folds, out)
    log(f"mixed-model prediction complete: {len(y)} exact MICs, {nfolds} grouped folds")


def interaction_report(args, X, y, K, loci, base, pairs, lookup, out, identity):
    from .phenotype import log, json_write
    direct = [p for p in pairs if p["direct_target"]]
    cache = covariance_cache(K, y)
    checkpoint = out / "interaction_checkpoints"
    checkpoint.mkdir(exist_ok=True)
    records = []
    for number, pair in enumerate(direct, 1):
        u, v = lookup[pair["u"]], lookup[pair["v"]]
        record = dict(pair, n_exact=len(y))
        states = X[:, u].astype(int)+2*X[:, v].astype(int)
        counts = np.bincount(states, minlength=4)
        for state, label in enumerate(("ab", "Ab", "aB", "AB")):
            record[f"n_{label}"] = int(counts[state])
            record[f"mean_log2_mic_{label}"] = float(y[states == state].mean()) if counts[state] else None
        record.update(gamma=None, gamma_se=None, gamma_ci_low=None, gamma_ci_high=None,
                      likelihood_ratio=None, p_lrt_asymptotic=None, p_bootstrap=None, q_value=None)
        stored = checkpoint / f"pair_{pair['u']}_{pair['v']}.json"
        if stored.exists() and json.loads(stored.read_text())["identity"] == identity:
            records.append(json.loads(stored.read_text())["record"])
            continue
        if not counts.all():
            record["status"] = "MISSING_GENOTYPE_STATE"
        elif counts.min() < args.interaction_min_cell:
            record["status"] = "SPARSE_GENOTYPE_STATES"
        else:
            columns = list(dict.fromkeys(base+[u, v]))
            null = fit_lmm(X, y, columns, [], cache, args.min_count)
            alternative = fit_lmm(X, y, columns, [(u, v)], cache, args.min_count)
            name = f"pair_{u}_{v}"
            if name not in alternative["names"]:
                record["status"] = "NON_IDENTIFIABLE_INTERACTION"
            else:
                index = alternative["names"].index(name)+1
                gamma = float(alternative["beta"][index]/alternative["scales"][index-1])
                se = float(np.sqrt(alternative["coefficient_covariance"][index, index]) /
                           alternative["scales"][index-1])
                statistic = max(0., 2*(null["nll"]-alternative["nll"]))
                record.update(status="TESTED", gamma=gamma, gamma_se=se,
                              gamma_ci_low=gamma-norm.ppf(.975)*se, gamma_ci_high=gamma+norm.ppf(.975)*se,
                              likelihood_ratio=statistic, p_lrt_asymptotic=float(chi2.sf(statistic, 1)))
                if args.interaction_bootstrap:
                    rng = np.random.default_rng(np.random.SeedSequence([args.seed, pair["u"], pair["v"]]))
                    eigen = 1-null["h"]+null["h"]*cache["values"]
                    mean = null["F"] @ null["beta"]
                    exceed = 0
                    for replicate in range(args.interaction_bootstrap):
                        simulated = mean + np.sqrt(null["variance"])*cache["vectors"] @ (
                            np.sqrt(eigen)*rng.standard_normal(len(y)))
                        simulated_cache = dict(cache, rotated_y=cache["vectors"].T @ simulated)
                        nfit = fit_lmm(X, simulated, columns, [], simulated_cache, args.min_count)
                        afit = fit_lmm(X, simulated, columns, [(u, v)], simulated_cache, args.min_count)
                        exceed += max(0., 2*(nfit["nll"]-afit["nll"])) >= statistic
                        if (replicate+1) % 50 == 0:
                            log(f"interaction {number}/{len(direct)}: bootstrap {replicate+1}/{args.interaction_bootstrap}")
                    record["p_bootstrap"] = (1+exceed)/(1+args.interaction_bootstrap)
        json_write(stored, dict(identity=identity, record=record))
        records.append(record)
        log(f"interaction {pair['u']} x {pair['v']}: {record['status']}")
    # Equivalent genotype products are not independent hypotheses. Correct over all
    # prespecified direct unitig pairs conservatively; untestable pairs contribute p=1.
    pvalues = [r["p_bootstrap"] if r["p_bootstrap"] is not None else
               r["p_lrt_asymptotic"] if r["p_lrt_asymptotic"] is not None else 1. for r in records]
    ranking = np.argsort(pvalues)
    adjusted = np.minimum.accumulate((np.array(pvalues)[ranking]*len(records)/np.arange(1, len(records)+1))[::-1])[::-1]
    for index, q in zip(ranking, adjusted):
        if records[index]["status"] == "TESTED":
            records[index]["q_value"] = float(min(1., q))
    if records:
        write_tsv(out / "interaction_report.tsv", records, list(records[0]))
        state_plots(args, X, y, direct, lookup, out)


def plots(args, y, predictions, labels, performance, folds, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .phenotype import save_figure
    plt.rcParams["svg.fonttype"] = "none"
    by_name = {row["model"]: row for row in performance}
    fixed = ["M0", "M1", "M2", "M1_pair"]
    fig, ax = plt.subplots(figsize=(9, 4))
    for x, name in enumerate(fixed):
        values = [row["rmse"] for row in folds if row["model"] == name]
        ax.scatter([x]*len(values), values, s=15, color="black", alpha=.3)
        ax.scatter(x, by_name[name]["rmse"], s=40, color="black", marker="D")
    for x, prefix in ((4, "Random1_"), (5, "Random2_"), (6, "RandomPair_")):
        values = [row["rmse"] for row in performance if row["model"].startswith(prefix) and row["control_count_match"]]
        if values:
            ax.boxplot(values, positions=[x], widths=.45, patch_artist=True,
                       boxprops=dict(facecolor="white", edgecolor="black"), medianprops=dict(color="black"),
                       flierprops=dict(markeredgecolor="black"))
        else:
            ax.text(x, .03, "No matched\ncontrol", transform=ax.get_xaxis_transform(), ha="center", fontsize=7)
    ax.axhline(by_name["Mean_only"]["rmse"], color="black", ls=":", label="Mean-only benchmark")
    ax.axhline(by_name["Phylogeny_only"]["rmse"], color="black", ls="--", label="Phylogeny-only benchmark")
    ax.set_xticks(range(7), [args.target, "+ direct", "+ extended", "+ direct\n+ interaction", "Random\ndirect", "Random\nextended", "Random\npair"])
    ax.set_ylabel("Held-out RMSE (log₂ MIC units)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, out / "mic_performance", args.dpi)
    plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.3), sharex=True, sharey=True)
    jitter = np.random.default_rng(args.seed).uniform(-.025, .025, len(y))
    selected = predictions[[labels.index(name) for name in fixed]]
    lower, upper = min(y.min(), selected.min())-.15, max(y.max(), selected.max())+.15
    for ax, name in zip(axes, fixed):
        ax.scatter(y+jitter, predictions[labels.index(name)], s=9, alpha=.3, color="black")
        ax.plot([lower, upper], [lower, upper], "k--", lw=.8)
        ax.set(xlim=(lower, upper), ylim=(lower, upper), xlabel="Observed log₂ MIC")
        ax.text(.04, .96, f"{name}; RMSE={by_name[name]['rmse']:.3f}", va="top", transform=ax.transAxes, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Predicted log₂ MIC")
    fig.tight_layout()
    save_figure(fig, out / "observed_predicted", args.dpi)
    plt.close(fig)


def state_plots(args, X, y, pairs, lookup, out):
    import matplotlib.pyplot as plt
    from .phenotype import save_figure
    fig, axes = plt.subplots(len(pairs), 1, figsize=(6, 2.8*len(pairs)), squeeze=False)
    rng = np.random.default_rng(args.seed)
    for ax, pair in zip(axes[:, 0], pairs):
        state = X[:, lookup[pair["u"]]].astype(int)+2*X[:, lookup[pair["v"]]].astype(int)
        counts = np.bincount(state, minlength=4)
        ax.scatter(state+rng.uniform(-.08, .08, len(y)), y, color="black", alpha=.25, s=9)
        for i in range(4):
            if counts[i]:
                ax.plot([i-.15, i+.15], [y[state == i].mean()]*2, color="black", lw=2)
        ax.set_xticks(range(4), [f"{label}\n(n={counts[i]})" for i, label in enumerate(("ab", "Ab", "aB", "AB"))])
        ax.set_ylabel("Observed log₂ MIC")
        ax.set_title(f"A: {pair['label_u']} ({pair['u']}); B: {pair['label_v']} ({pair['v']})", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, out / "interaction_states", args.dpi)
    plt.close(fig)
