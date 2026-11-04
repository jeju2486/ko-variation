"""Redraw screening-benchmark panels as PNG and editable SVG, without refitting."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def label(method):
    if method == "MI":
        return "MI"
    k = method.removeprefix('residual_k')
    return f"Tree residuals ({k} {'PC' if k == '1' else 'PCs'})"


def save(fig, prefix):
    import matplotlib.pyplot as plt
    fig.savefig(str(prefix)+".png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(str(prefix)+".svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot(directory, seed=42, max_points=30000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator

    directory = Path(directory)
    extension = (directory/"extension_source.json").exists()
    base = Path(json.loads((directory/"extension_source.json").read_text())["base_directory"]) if extension else directory
    figures = directory/"figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"svg.fonttype": "none", "font.family": "DejaVu Sans",
        "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelsize": 11, "legend.frameon": False})
    recovery = pd.read_csv(directory/"recovery.tsv", sep="\t")
    methods = recovery.method.drop_duplicates().tolist()
    styles = ["--", "-", ":", "-.", (0, (5, 1)), (0, (3, 1, 1, 1, 1, 1))]
    markers = ["o", "s", "^", "D", "v", "P"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True)
    for axis, metric, ylabel in zip(axes, ["discovery_recall", "priority_recall"],
            ["Significant KOVAR-pair recovery (%)", "Top-ranked KOVAR-pair recovery (%)"]):
        for i, method in enumerate(methods):
            subset = recovery.loc[recovery.method == method]
            aggregate = subset.groupby("budget_fraction")[metric].agg(["median", "min", "max"])
            # Use realized budgets after rounding to an integer number of pairs.
            x = subset.groupby("budget_fraction").actual_budget_fraction.median().to_numpy()*100
            y = aggregate["median"].to_numpy()*100
            if np.isfinite(y).any():
                axis.plot(x, y, color="black", linestyle=styles[i % len(styles)],
                          marker=markers[i % len(markers)], markersize=3, linewidth=1.4, label=label(method))
                # Computational panels are not biological replicates: show their
                # observed range, not a confidence interval.
                if subset.panel.nunique() > 1:
                    axis.fill_between(x, aggregate["min"].to_numpy()*100,
                        aggregate["max"].to_numpy()*100, color="black", alpha=.05)
        if not np.isfinite(recovery[metric]).any():
            axis.text(.5, .5, "No significant reference pairs\nRecovery is undefined",
                      ha="center", va="center", transform=axis.transAxes)
        axis.set(xscale="log", ylim=(-2, 102), xlabel="Candidate budget (% of eligible pairs)", ylabel=ylabel)
        axis.xaxis.set_major_locator(LogLocator(base=10))
        axis.grid(axis="y", color="0.9", linewidth=.5)
    handles, names = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, names, fontsize=8, loc="lower center", ncol=2 if len(methods) > 4 else 2)
    fig.tight_layout(rect=(0, .17 if len(methods) > 4 else .12, 1, 1))
    save(fig, figures/"A_candidate_recovery")

    # One subplot per screen, same eligible-and-successfully-tested population.
    # Plotting subsampling changes no ranks or recovery calculations.
    comparisons = []
    for path in sorted(directory.glob("panel_*/rank_comparison.tsv")):
        frame = pd.read_csv(path, sep="\t")
        frame = frame.loc[frame.eligible.eq(1) & frame.kovar_rank.notna()].copy()
        for method in methods:
            # Ranks were calculated over ALL eligible pairs, not this display subset.
            all_eligible = int(recovery.loc[recovery.panel == int(path.parent.name.split("_")[1]), "n_eligible_pairs"].iloc[0])
            frame[f"{method}_percentile"] = frame[f"{method}_rank"]/all_eligible*100
        frame["kovar_percentile"] = frame.kovar_rank/len(frame)*100
        frame["panel"] = path.parent.name
        comparisons.append(frame)
    combined = pd.concat(comparisons, ignore_index=True)
    if len(combined) > max_points:
        combined = combined.sample(max_points, random_state=seed)
    columns = min(3, len(methods))
    grid_rows = math.ceil(len(methods)/columns)
    fig, axes = plt.subplots(grid_rows, columns, figsize=(3.8*columns, 3.5*grid_rows), squeeze=False, sharey=True)
    for axis, method in zip(axes.ravel(), methods):
        subset = combined.loc[combined[f"{method}_percentile"].notna()]
        axis.scatter(subset[f"{method}_percentile"], subset.kovar_percentile,
                     s=6 if len(subset) < 5000 else 2, color="black",
                     alpha=.5 if len(subset) < 5000 else .12, linewidths=0, rasterized=False)
        axis.plot([0, 100], [0, 100], color="0.6", linestyle="--", linewidth=.8)
        axis.set(xlim=(-1, 101), ylim=(-1, 101), xlabel=label(method)+" rank percentile\n(lower = higher priority)")
    for row in range(grid_rows):
        axes[row, 0].set_ylabel("KOVAR P-value rank percentile\n(lower = higher priority)")
    for axis in axes.ravel()[len(methods):]:
        axis.set_visible(False)
    fig.tight_layout()
    save(fig, figures/"B_rank_comparison")

    # Actual measured stages; do not extrapolate panel timings to the full cohort.
    timing_frames = []
    prep = pd.read_csv(base/"preparation_timings.tsv", sep="\t")
    prep["panel"] = "shared"
    prep["timing_run"] = "original" if extension else "current"
    timing_frames.append(prep)
    for pattern in ["panel_*/screen_timings.tsv", "panel_*/reference_timings.tsv"]:
        for path in sorted(base.glob(pattern)):
            frame = pd.read_csv(path, sep="\t")
            frame["panel"] = path.parent.name
            frame["timing_run"] = "original" if extension else "current"
            timing_frames.append(frame)
    if extension:
        extra = pd.read_csv(directory/"projection_timings.tsv", sep="\t")
        extra["panel"], extra["timing_run"] = "shared", "extension"
        timing_frames.append(extra)
        for path in sorted(directory.glob("panel_*/screen_timings.tsv")):
            frame = pd.read_csv(path, sep="\t")
            frame["panel"], frame["timing_run"] = path.parent.name, "extension"
            timing_frames.append(frame)
    timings = pd.concat(timing_frames, ignore_index=True)
    timings.to_csv(directory/"runtime_comparison.tsv", sep="\t", index=False, na_rep="NA")
    def timing_axis(axis, values, names, xlabel):
        # Points have an unambiguous meaning on a log axis, unlike bars with
        # an invisible zero baseline. These remain measured operational times.
        axis.plot(values.to_numpy(), np.arange(len(values)), "ko", markersize=5)
        axis.set_yticks(np.arange(len(values)), labels=names)
        axis.set(xscale="log", xlabel=xlabel)
        axis.grid(axis="x", color="0.9", linewidth=.5)

    pretty = {"read_cohort_and_tree": "Cohort and tree", "exact_pattern_grouping": "Exact pattern grouping",
             "global_mi_weights": "Global MI weights", "tree_axes_and_all_pattern_projection": "Tree axes and projections"}
    if extension:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        new = timings.loc[timings.timing_run.eq("extension") & timings.stage.str.startswith("residual_")]
        aggregate = new.groupby("stage", sort=False).wall_seconds.median()
        timing_axis(axes[0], aggregate, [label(name) for name in aggregate.index], "Scoring wall time (s)\nSame allocation")
        extra = timings.loc[timings.timing_run.eq("extension") & timings.panel.eq("shared")]
        aggregate = extra.groupby("stage", sort=False).wall_seconds.sum()
        pretty["extend_tree_axes"] = "Extend tree axes"
        timing_axis(axes[1], aggregate, [pretty.get(name, name.replace("projection_", "PCs ").replace("_to_", " to "))
                                      for name in aggregate.index], "Additional preparation time (s)")
        historical = timings.loc[timings.timing_run.eq("original") & timings.stage.str.startswith(("MI_", "KOVAR_"))]
        aggregate = historical.groupby("stage", sort=False).wall_seconds.median()
        timing_axis(axes[2], aggregate, ["KOVAR reference" if name.startswith("KOVAR_") else "MI" for name in aggregate.index],
                    "Original-run wall time (s)")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        scoring = timings.loc[timings.stage.str.startswith(("MI_", "residual_", "KOVAR_"))]
        aggregate = scoring.groupby("stage", sort=False).wall_seconds.median()
        names = [label(name) if name.startswith("residual_") else
                 "KOVAR reference" if name.startswith("KOVAR_") else "MI" for name in aggregate.index]
        timing_axis(axes[0], aggregate, names, "Median wall time per panel (seconds)")
        shared = prep.groupby("stage", sort=False).wall_seconds.sum()
        timing_axis(axes[1], shared, [pretty.get(name, name) for name in shared.index], "Shared preparation wall time (seconds)")
    fig.tight_layout()
    save(fig, figures/"C_runtime")

    required_path = directory/"retention_summary.tsv"
    if required_path.exists():
        required = pd.read_csv(required_path, sep="\t")
        residual = required.loc[required.method.str.startswith("residual_k")].copy()
        residual["pcs"] = residual.method.str.removeprefix("residual_k").astype(int)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, sharex=True)
        for axis, reference_set, xlabel in zip(axes, ["significant", "priority"],
                ["PC count\nSignificant KOVAR reference pairs", "PC count\nTop-ranked KOVAR reference pairs"]):
            subset = residual.loc[residual.reference_set == reference_set]
            for i, target in enumerate(sorted(subset.target_recovery.unique())):
                group = subset.loc[subset.target_recovery == target].sort_values("pcs")
                y = group.median_candidate_fraction.to_numpy()*100
                if np.isfinite(y).any():
                    axis.plot(group.pcs, y, color="black", linestyle=styles[i % len(styles)],
                        marker=markers[i % len(markers)], markersize=4, label=f"{target*100:g}% recovery")
                    axis.fill_between(group.pcs, group.minimum_candidate_fraction.to_numpy()*100,
                        group.maximum_candidate_fraction.to_numpy()*100, color="black", alpha=.06)
            if not np.isfinite(subset.median_candidate_fraction).any():
                axis.text(.5, .5, "No significant reference pairs\nRequired budget is undefined", ha="center", va="center", transform=axis.transAxes)
            axis.set(xlabel=xlabel, yscale="log")
            axis.set_xticks(sorted(residual.pcs.unique()))
            axis.grid(axis="y", color="0.9", linewidth=.5)
        axes[0].set_ylabel("Minimum candidate budget (%)")
        handles, names = axes[1].get_legend_handles_labels()
        if handles:
            fig.legend(handles, names, loc="lower center", ncol=3, fontsize=9)
        fig.tight_layout(rect=(0, .1, 1, 1))
        save(fig, figures/"D_required_retention")

    # Exact figure semantics live beside the files rather than crowding panels.
    (figures/"figure_notes.txt").write_text(
        "A: Matched budgets over joint-cell-eligible pairs. Lines show medians across computational panels; "
        "shading, when present, is their observed range, not a confidence interval. Left: recovery of "
        "Bonferroni-significant KOVAR reference pairs. Right: recovery of the predefined top-ranked KOVAR fraction "
        "in benchmark_summary.json (default 1%); this is a ranking comparison, not evidence of significance. "
        "No significant reference pairs means discovery recall is NA.\n"
        "B: Midrank percentiles. Screen ranks use all joint-cell-eligible pairs; KOVAR ranks use finite tests. "
        "Displayed points are uniformly subsampled only for rendering, not analysis. Multiple computational "
        "panels are combined in this display. A diagonal indicates equal percentile ranks, not calibration.\n"
        "C: Measured median panel-stage wall times, excluding shared preparation shown separately. MI timing "
        "includes external execution/output parsing for the SpydrPick backend; residual-score timing reuses "
        "shared joint-state counts and precomputed projections. Shared count time and sampled sum-RSS memory "
        "are in runtime_comparison.tsv. KOVAR timing includes CLI input, fitting, scoring and output. These "
        "are operational timings, not pure kernel speed ratios. Original completed timing records are retained "
        "on cache reuse.\n"
        "For an extension, C separates re-timed residual scoring on the extension allocation, additional "
        "projection stages, and historical MI/KOVAR times on their original allocation. These panels must "
        "not be combined into a hardware-controlled speed ratio. The original global weighting and full "
        "preparation timings remain in runtime_comparison.tsv.\n"
        "D: Exact minimum candidate fractions needed for each target recovery, using deterministic rank "
        "positions. Lines are panel medians and shading is the observed panel range, not a confidence "
        "interval. The maximum required fraction in retention_summary.tsv is the worst observed defined "
        "panel. Neither is a guarantee for unseen or genome-wide candidates. NA budgets remain undefined.\n"
        "All figures: real-data KOVAR-reference benchmark, not epistasis validation; no genomic-distance or "
        "PAN-GWES graph-count filter. Unavailable residual scores and boundary ties are reported in recovery.tsv. "
        "SVG text remains editable.\n", encoding="utf-8")
    print(f"[screen benchmark] PNG and SVG figures: {figures}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Completed benchmark directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-points", type=int, default=30000)
    args = parser.parse_args(argv)
    if args.max_points < 1:
        parser.error("--max-points must be positive")
    plot(args.out, args.seed, args.max_points)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
