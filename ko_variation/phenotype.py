"""Binary FASTA export, tree-aligned heatmap and phylogenetic MIC interaction models.

This downstream helper reads an already selected network. It never reruns KOVAR
or changes its significance/LD selection. Gene names identify loci, not mechanisms.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re

import numpy as np

from . import __version__
from .amr_report import gene_key, label_tokens, mic_record, present, read_tsv, write_tsv

# Increment when numerical fitting/feature selection changes, to invalidate old folds.
ALGORITHM = "mic-phylogenetic-interaction-3"


def log(message):
    print(f"[phenotype] {message}", flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def npz_write(path, **arrays):
    temporary = Path(str(path) + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def binary_fasta(path):
    """Stream records; repeated contig headers are never accepted as sample IDs."""
    names, rows, name, chunks = [], [], None, []

    def finish():
        sequence = "".join(chunks).upper().encode("ascii")
        values = np.frombuffer(sequence, dtype=np.uint8)
        if not len(values) or np.any((values != ord("A")) & (values != ord("C"))):
            raise ValueError(f"{name}: expected a nonempty A/C binary sequence")
        if rows and len(values) != len(rows[0]):
            raise ValueError(f"{name}: unequal FASTA record length")
        names.append(name)
        rows.append((values == ord("C")).astype(np.uint8))
        if len(rows) % 50 == 0:
            log(f"read {len(rows)} FASTA records")

    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    finish()
                name, chunks = line[1:].split()[0], []
            else:
                if name is None:
                    raise ValueError("Sequence precedes first FASTA header")
                chunks.append(line)
    if name is not None:
        finish()
    if not names or len(names) != len(set(names)):
        raise ValueError("Use sample_ids.fasta with unique, nonempty isolate IDs")
    return names, np.stack(rows)


def export_matrix(args, out, sources):
    identity = {"fasta_sha256": sources["fasta"]["sha256"],
                "locus_map_sha256": sources.get("locus_map", {}).get("sha256"),
                "column_contract": "explicit_map" if args.locus_map else "zero_based_unitig_id"}
    cache_directory = args.matrix_cache or out
    cache_directory.mkdir(parents=True, exist_ok=True)
    cache, manifest = cache_directory / "matrix.npz", cache_directory / "matrix.json"
    if cache.exists() and manifest.exists() and json.loads(manifest.read_text()) == identity:
        log("reuse completed binary matrix")
        with np.load(cache, allow_pickle=False) as data:
            samples, X, loci = data["samples"].tolist(), data["X"], data["loci"]
    else:
        samples, X = binary_fasta(args.fasta)
        loci = np.arange(X.shape[1], dtype=np.int64)
        if args.locus_map:
            records = read_tsv(args.locus_map, ["fasta_column", "locus"])
            columns = {int(row["fasta_column"]): int(row["locus"]) for row in records}
            if len(columns) != len(records) or set(columns) != set(range(X.shape[1])):
                raise ValueError("Locus map must cover every zero-based FASTA column exactly once")
            loci = np.array([columns[i] for i in range(X.shape[1])], dtype=np.int64)
        if np.any(loci < 0) or len(set(loci.tolist())) != len(loci):
            raise ValueError("Original locus IDs must be unique and nonnegative")
        npz_write(cache, samples=np.asarray(samples), X=X, loci=loci)
        json_write(manifest, identity)
    samples, X, exclusions = filter_cohort(args, samples, X, out)
    export_identity = dict(identity, included_samples=samples, excluded_samples=sorted(exclusions))
    exported = out / "matrix_export.json"
    if not exported.exists() or json.loads(exported.read_text()) != export_identity:
        (out / "unitig_presence_absence.tsv.gz").unlink(missing_ok=True)
        npz_write(out / "analysis_matrix.npz", samples=np.asarray(samples), X=X, loci=loci)
        json_write(exported, export_identity)
    write_tsv(out / "column_mapping.tsv",
              ({"fasta_column": i, "locus": int(locus)} for i, locus in enumerate(loci)),
              ["fasta_column", "locus"])
    if not args.no_full_tsv and not (out / "unitig_presence_absence.tsv.gz").exists():
        target = out / "unitig_presence_absence.tsv.gz"
        temporary = Path(str(target) + ".partial")
        # Low compression avoids spending hours exporting a large text matrix.
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=1) as handle:
            handle.write("sample\t" + "\t".join(f"unitig_{i}" for i in loci) + "\n")
            for i, (sample, row) in enumerate(zip(samples, X), 1):
                handle.write(sample + "\t" + "\t".join(np.where(row, "1", "0")) + "\n")
                if i % 50 == 0 or i == len(samples):
                    log(f"full TSV export {i}/{len(samples)}")
        temporary.replace(target)
    log(f"matrix: {len(samples)} isolates x {X.shape[1]} unitigs")
    return samples, X, loci, exclusions


def filter_cohort(args, samples, X, out):
    """Exclude explicit non-target species consistently; never infer taxonomy from branch length."""
    rows = read_tsv(args.metadata, ["id", "isolate", "species"])
    if args.sample_map:
        mapping = read_tsv(args.sample_map, ["metadata_id", "sample"])
        lookup = {row["metadata_id"]: row["sample"] for row in mapping}
        if len(lookup) != len(mapping) or len(set(lookup.values())) != len(mapping):
            raise ValueError("Sample map must be one-to-one")
    else:
        lookup = {r["id"]: r["id"] + "_" + r["isolate"].replace(".", "_") for r in rows}
    requested = set(args.exclude_sample)
    if requested - set(samples):
        raise ValueError(f"Explicit exclusion absent from FASTA: {sorted(requested-set(samples))}")
    audit, excluded = [], set(requested)
    for row in rows:
        sample = lookup.get(row["id"])
        if sample in samples and row["species"].strip().casefold() != args.species.strip().casefold():
            excluded.add(sample)
        if sample in excluded:
            audit.append(dict(sample=sample, metadata_id=row["id"], species=row["species"],
                              reason="explicit_exclusion" if sample in requested else "non_target_species"))
    for sample in excluded - {r["sample"] for r in audit}:
        audit.append(dict(sample=sample, metadata_id="", species="", reason="explicit_exclusion"))
    write_tsv(out / "excluded_samples.tsv", audit, ["sample", "metadata_id", "species", "reason"])
    keep = [i for i, sample in enumerate(samples) if sample not in excluded]
    if not keep:
        raise ValueError("No isolates remain after species/exclusion filtering")
    selected = [samples[i] for i in keep]
    cleaned = [row for row in rows if lookup.get(row["id"]) in set(selected)]
    write_tsv(out / "metadata_filtered.tsv", cleaned, list(rows[0]))
    log(f"cohort: retained {len(selected)}; excluded {len(excluded)}: {sorted(excluded)}")
    return selected, X[keep], excluded


def neighbourhood(args, loci, out):
    """Join locus IDs to genes explicitly: sorted gene endpoints need not match u/v."""
    source = Path(args.network).read_text(encoding="utf-8")
    match = re.search(r'<script id="data" type="application/json">(.*?)</script>', source, re.S)
    network = json.loads(match[1] if match else source)
    nodes = {node["gene"]: node for node in network["nodes"]}
    target = [gene for gene, node in nodes.items()
              if gene == args.target or gene_key(args.target) in label_tokens(
                  node.get("display_label", node.get("label", gene)))]
    if len(target) != 1:
        raise ValueError(f"Target must resolve to one cluster; matches: {target}. Use its cluster ID if needed")
    target = target[0]
    adjacency = {gene: set() for gene in nodes}
    represented = set()
    for edge in network["edges"]:
        a, b = edge["gene_a"], edge["gene_b"]
        adjacency[a].add(b)
        adjacency[b].add(a)
        for pair in json.loads(edge["locus_pairs"]):
            represented.update(map(int, pair))
    first = adjacency[target] - {target}
    second = set().union(*(adjacency[g] for g in first)) - first - {target} if first else set()
    if not first or not second:
        raise ValueError("The four-family comparison requires both direct and second-order neighbours")
    order = {target: 0, **{g: 1 for g in first}, **{g: 2 for g in second}}
    available = {int(locus): column for column, locus in enumerate(loci)}
    annotation, features, pool, seen = read_tsv(args.annotation, ["locus", "gene", "annotation_class"]), [], [], set()
    for row in annotation:
        locus = int(row["locus"])
        if locus in seen:
            raise ValueError(f"Duplicate annotation for locus {locus}")
        seen.add(locus)
        if locus not in represented or locus not in available:
            continue
        if "coding" not in {c.strip() for c in row["annotation_class"].split(";")}:
            continue
        gene = row["gene"]
        if gene in order:
            features.append(dict(locus=locus, fasta_column=available[locus], gene=gene,
                                 label=nodes[gene].get("display_label", gene),
                                 neighbourhood_order=order[gene], annotation_class=row["annotation_class"]))
        elif gene in nodes:
            pool.append(available[locus])
    # Distinct IDs may later collapse to identical training-set genotype patterns.
    features.sort(key=lambda r: (r["neighbourhood_order"], r["label"], r["locus"]))
    for gene in order:
        if not any(row["gene"] == gene for row in features):
            raise ValueError(f"No coding-overlap, original-ID-matched unitigs for {gene}")
    write_tsv(out / "features.tsv", features, list(features[0]))
    if args.random_pool:
        requested = {int(row["locus"]) for row in read_tsv(args.random_pool, ["locus"])}
        if not requested.issubset(available):
            raise ValueError("Random pool includes locus IDs absent from the FASTA")
        excluded = {available[int(row["locus"])] for row in annotation if row["gene"] in order
                    and int(row["locus"]) in available}
        pool = sorted({available[locus] for locus in requested} - excluded)
    if not pool:
        raise ValueError("No random-control background unitigs remain")
    json_write(out / "neighbourhood.json", dict(target=target, first=sorted(first), second=sorted(second),
               unitig_definition="binary exact-sequence presence; not a named resistance mutation",
               pool_definition="supplied locus pool" if args.random_pool else
               "coding-overlap unitigs represented by other nodes in the selected distal map",
               distance_to_target="not certified by this helper; preserve upstream selection provenance"))
    # Gene endpoints are sorted, but u/v need not be: join through annotation IDs.
    by_locus = {row["locus"]: row for row in features}
    pairs, seen_pairs = [], set()
    for edge in network["edges"]:
        for u, v in json.loads(edge["locus_pairs"]):
            u, v = int(u), int(v)
            if u not in by_locus or v not in by_locus or tuple(sorted((u, v))) in seen_pairs:
                continue
            a, b = by_locus[u], by_locus[v]
            if {a["gene"], b["gene"]} != {edge["gene_a"], edge["gene_b"]}:
                raise ValueError(f"Edge/annotation conflict for loci {u}, {v}")
            seen_pairs.add(tuple(sorted((u, v))))
            pairs.append(dict(u=u, v=v, label_u=a["label"], label_v=b["label"],
                              direct_target=int(a["neighbourhood_order"] == 0 or b["neighbourhood_order"] == 0)))
    write_tsv(out / "selected_pairs.tsv", pairs, ["u", "v", "label_u", "label_v", "direct_target"])
    return features, np.asarray(pool, dtype=int), pairs


def phenotype_rows(args, samples, out):
    metadata = read_tsv(args.metadata, ["id", "isolate", args.mic_column, args.sign_column])
    if args.sample_map:
        mapping = read_tsv(args.sample_map, ["metadata_id", "sample"])
        lookup = {r["metadata_id"]: r["sample"] for r in mapping}
        if len(lookup) != len(mapping) or len(set(lookup.values())) != len(mapping):
            raise ValueError("Sample mapping must be one-to-one")
    else:
        # Matches the established ARC assembly basename; save the exact join below.
        lookup = {r["id"]: r["id"] + "_" + r["isolate"].replace(".", "_") for r in metadata}
    by_sample = {}
    for row in metadata:
        sample = lookup.get(row["id"])
        if sample is not None:
            if sample in by_sample:
                raise ValueError(f"Duplicate metadata mapping to {sample}")
            by_sample[sample] = row
    result = []
    for sample in samples:
        row = by_sample.get(sample)
        value = row.get(args.mic_column, "") if row else ""
        sign = row.get(args.sign_column, "") if row else ""
        number, status = mic_record(value, sign) if present(value) else (None, "missing_mic")
        if not row:
            status = "missing_metadata"
        result.append(dict(sample=sample, metadata_id=row["id"] if row else "", mic=value, sign=sign,
                           status=status, log2_mic=float(np.log2(number)) if status == "exact" else None))
    if not any(r["metadata_id"] for r in result):
        raise ValueError("No metadata/sample match; supply --sample-map")
    write_tsv(out / "phenotypes.tsv", result, list(result[0]))
    extra = [dict(sample=sample, metadata_id=row["id"], status="metadata_without_fasta")
             for sample, row in by_sample.items() if sample not in set(samples)]
    write_tsv(out / "unmatched_metadata.tsv", extra, ["sample", "metadata_id", "status"])
    return result


def tree_and_groups(args, samples, out, exclusions=()):
    from Bio import Phylo

    tree = Phylo.read(args.tree, "newick")
    tips = tree.get_terminals()
    names = [tip.name for tip in tips]
    if len(names) != len(set(names)) or any(name is None for name in names):
        raise ValueError("Tree tips must have unique isolate names")
    for name in sorted(set(exclusions) & set(names)):
        tree.prune(name)
    names = [tip.name for tip in tree.get_terminals()]
    if set(samples) != set(names):
        raise ValueError(f"Tree/FASTA mismatch: {len(set(samples)-set(names))} FASTA-only, "
                         f"{len(set(names)-set(samples))} tree-only. Supply a deliberately matched tree/FASTA.")
    for node in tree.find_clades():
        if node is not tree.root and (node.branch_length is None or
                not np.isfinite(node.branch_length) or node.branch_length < 0):
            raise ValueError("Midpoint rooting requires finite nonnegative branch lengths")
    tree.root_at_midpoint()
    tree.root.branch_length = 0.
    tree.ladderize()
    Phylo.write(tree, out / "midpoint.treefile", "newick", format_branch_length="%.12g")
    tips = tree.get_terminals()
    paths = {tip.name: [tree.root] + tree.get_path(tip) for tip in tips}
    depths = tree.depths()
    D = np.zeros((len(samples), len(samples)))
    K = np.zeros_like(D)
    for i, a in enumerate(samples):
        K[i, i] = depths[paths[a][-1]]
        for j in range(i):
            b = samples[j]
            common = tree.root
            for u, v in zip(paths[a], paths[b]):
                if u is not v:
                    break
                common = u
            D[i, j] = D[j, i] = max(0., depths[paths[a][-1]] + depths[paths[b][-1]] - 2 * depths[common])
            K[i, j] = K[j, i] = depths[common]
    scale = float(np.trace(K)/len(samples))
    if scale <= 0:
        raise ValueError("Tree has no positive shared-branch covariance scale")
    K /= scale
    npz_write(out / "phylogenetic_covariance.npz", samples=np.asarray(samples), K=K)
    groups = analysis_groups(args, samples, D, exclusions)
    write_tsv(out / "tree_order.tsv", ({"sample": tip.name, "tree_row": i,
              "phylogenetic_group": str(groups[samples.index(tip.name)]),
              "grouping_method": args.group_method} for i, tip in enumerate(tips)),
              ["sample", "tree_row", "phylogenetic_group", "grouping_method"])
    log(f"midpoint-rooted tree: {len(tips)} tips; {len(set(groups))} {args.group_method} groups")
    return tree, groups, K


def analysis_groups(args, samples, distances, exclusions=()):
    """Join external fastBAPS labels strictly by isolate ID, independently of MIC."""
    if args.group_method == "fastbaps":
        records = read_tsv(args.clusters, ["sample", "cluster"])
        lookup = {row["sample"]: row["cluster"].strip() for row in records}
        if len(lookup) != len(records) or any(not key or not value for key, value in lookup.items()):
            raise ValueError("Cluster table requires unique sample IDs and nonempty labels")
        missing = set(samples)-set(lookup)
        extra = set(lookup)-set(samples)-set(exclusions)
        if missing or extra:
            raise ValueError(f"Cluster/cohort mismatch: missing={sorted(missing)[:8]}, extra={sorted(extra)[:8]}")
        return np.asarray([lookup[sample] for sample in samples])
    # Explicit legacy option for reproducing the previous patristic partition.
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    return fcluster(linkage(squareform(distances), method="complete"),
                    t=min(args.groups, len(samples)), criterion="maxclust")


def save_figure(fig, prefix, dpi):
    fig.savefig(str(prefix) + ".png", dpi=dpi, facecolor="white", bbox_inches="tight")
    fig.savefig(str(prefix) + ".svg", facecolor="white", bbox_inches="tight")


def tree_plot(tree, samples, X, features, phenotypes, groups, out, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    plt.rcParams["svg.fonttype"] = "none"
    tips = tree.get_terminals()
    idx = {sample: i for i, sample in enumerate(samples)}
    rows = [idx[tip.name] for tip in tips]
    cols = [f["fasta_column"] for f in features]
    fig, axes = plt.subplots(1, 3, figsize=(11, max(5, len(tips) * .025)),
                             gridspec_kw={"width_ratios": [4, 5, .55], "wspace": .12})
    y = {tip: i for i, tip in enumerate(tips)}
    for node in tree.get_nonterminals(order="postorder"):
        y[node] = (y[node.clades[0]] + y[node.clades[-1]]) / 2
    depths = tree.depths()
    for node in tree.find_clades():
        for child in node.clades:
            axes[0].plot([depths[node], depths[child]], [y[child], y[child]], color="black", lw=.45)
        if node.clades:
            axes[0].plot([depths[node]] * 2, [y[node.clades[0]], y[node.clades[-1]]], color="black", lw=.45)
    axes[0].set_xlabel("Substitutions per site")
    axes[0].set_title("Midpoint-rooted core tree", fontsize=10)
    # Vector cells keep the heatmap itself editable in SVG, not only its labels.
    boundaries = np.arange(len(tips)+1) - .5
    centers, offset = [], 0.
    for i, feature in enumerate(features):
        if i and feature["neighbourhood_order"] != features[i-1]["neighbourhood_order"]:
            offset += .35
        center = i + offset
        centers.append(center)
        axes[1].pcolormesh([center-.5, center+.5], boundaries, X[rows, cols[i]][:, None],
                           cmap=ListedColormap(["white", "#222222"]), vmin=0, vmax=1,
                           shading="flat", edgecolors="none", rasterized=False)
    axes[1].set_xticks(centers,
                      [f"{f['label']} | {f['locus']}" for f in features], rotation=90, fontsize=7)
    axes[1].xaxis.tick_top()
    axes[1].set_xlim(-.5, centers[-1]+.5)
    axes[1].legend(handles=[Patch(facecolor="white", edgecolor="black", label="Absent"),
                            Patch(facecolor="#222222", label="Present")], loc="lower center",
                   bbox_to_anchor=(.5, -.065), ncol=2, frameon=False, fontsize=8)
    values = np.array([p["log2_mic"] if p["status"] == "exact" else np.nan for p in phenotypes])
    cmap = plt.get_cmap("Reds").copy()
    cmap.set_bad("white")
    image = axes[2].pcolormesh([-.5, .5], boundaries, np.ma.masked_invalid(values[rows, None]),
                              cmap=cmap, shading="flat", rasterized=False)
    axes[2].set_title("logâ‚‚ MIC", fontsize=8)
    # An ordinary colourbar shrinks just the MIC axes and breaks tip/row alignment.
    bar = axes[2].inset_axes([0, -.045, 1, .012])
    fig.colorbar(image, cax=bar, orientation="horizontal")
    for ax in axes:
        ax.set_ylim(len(tips)-.5, -.5)
        ax.set_yticks([])
        if ax is not axes[0]:
            ax.tick_params(bottom=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(.5, .005, "White MIC cells: no exact measurement. Rows follow tree tips. Neighbourhood blocks are separated by whitespace.",
             ha="center", fontsize=8)
    save_figure(fig, out / "tree_presence_absence", dpi)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fasta", "tree", "metadata", "network", "annotation", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    contract = parser.add_mutually_exclusive_group(required=True)
    contract.add_argument("--fasta-columns-are-locus-ids", action="store_true",
                          help="Declare unfiltered FASTA column 0 == original unitig 0")
    contract.add_argument("--locus-map", type=Path, help="TSV: zero-based fasta_column, original locus")
    parser.add_argument("--target", default="mprF")
    parser.add_argument("--mic-column", default="daptomycin_mic")
    parser.add_argument("--sign-column", default="daptomycin_mic_sign")
    parser.add_argument("--sample-map", type=Path, help="Override id_isolate basename join: metadata_id, sample")
    parser.add_argument("--species", default="Staphylococcus aureus", help="Retain this metadata species")
    parser.add_argument("--exclude-sample", action="append", default=[], help="Additional exact FASTA sample ID to exclude")
    parser.add_argument("--matrix-cache", type=Path, help="Directory containing a reusable raw matrix.npz and matrix.json")
    parser.add_argument("--random-pool", type=Path, help="Optional independently defined eligible locus pool")
    parser.add_argument("--group-method", choices=["fastbaps", "complete"], default="fastbaps")
    parser.add_argument("--clusters", type=Path, help="fastBAPS TSV with sample,cluster; required for default grouping")
    parser.add_argument("--groups", type=int, default=10, help="Maximum groups for explicit legacy --group-method complete only")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-sets", type=int, default=100)
    parser.add_argument("--min-count", type=int, default=5, help="Minimum present AND absent training isolates")
    parser.add_argument("--interaction-min-cell", type=int, default=5, help="Minimum observations in each of four genotype states")
    parser.add_argument("--interaction-bootstrap", type=int, default=200, help="Null parametric bootstrap draws per estimable pair; 0 uses asymptotic P")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-full-tsv", action="store_true", help="Cache full matrix; export only neighbourhood TSV")
    parser.add_argument("--stage", choices=["all", "tree"], default="all", help="tree skips MIC model fitting")
    args = parser.parse_args(argv)
    if args.folds < 3 or args.random_sets < 1 or args.min_count < 1 or args.dpi < 72:
        parser.error("folds >=3, random-sets/min-count >=1, dpi >=72 required")
    if args.group_method == "fastbaps" and args.clusters is None:
        parser.error("Grouping requires --clusters FILE with externally inferred sample,cluster labels")
    if args.group_method == "complete" and (args.groups < 3 or args.clusters is not None):
        parser.error("Legacy complete grouping requires groups >=3 and does not accept --clusters")
    if args.seed < 0 or args.interaction_bootstrap < 0 or args.interaction_min_cell < 1:
        parser.error("seed/bootstrap must be nonnegative; interaction-min-cell must be positive")
    sources = {}
    for name in ("fasta", "tree", "metadata", "network", "annotation", "locus_map", "sample_map", "random_pool", "clusters"):
        path = getattr(args, name)
        if path:
            if not path.is_file():
                parser.error(f"Missing {name}: {path}")
            sources[name] = dict(path=str(path.resolve()), sha256=sha256(path))
    args.out.mkdir(parents=True, exist_ok=True)
    versions = {p: importlib.metadata.version(p) for p in
                ("numpy", "scipy", "scikit-learn", "biopython", "matplotlib")}
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    # Package versions are part of restart identity, so a changed environment cannot reuse fits.
    analysis_settings = {k: v for k, v in settings.items()
                         if k not in {"out", "dpi", "stage", "no_full_tsv"}}
    model_code = [sha256(Path(__file__)), sha256(Path(__file__).with_name("phenotype_lmm.py"))]
    identity = hashlib.sha256(json.dumps(dict(algorithm=ALGORITHM, model_code=model_code, sources=sources,
                  settings=analysis_settings, versions=versions), sort_keys=True).encode()).hexdigest()
    summary = dict(status="started", algorithm=ALGORITHM, version=__version__, sources=sources,
                   settings=settings, packages=versions, identity=identity,
                   evaluation="fixed cohort-derived network; exact-MIC grouped held-out prediction",
                   network_discovery="not repeated inside folds; not independent end-to-end discovery validation",
                   gene_variation="exact unitig-sequence presence; named substitutions not inferred",
                   model="Gaussian phylogenetic mixed model fitted by maximum likelihood; selected direct pair products",
                   grouping=args.group_method,
                   covariance="shared root-to-MRCA branch length normalized by mean diagonal",
                   cohort="non-target metadata species excluded before midpoint rooting, grouping and fitting",
                   frozen_network="exclusion does not rescan or reselect the previously constructed network",
                   censoring="only explicit = measurements fitted; boundaries never treated as exact")
    json_write(args.out / "run.json", summary)
    samples, X, loci, exclusions = export_matrix(args, args.out, sources)
    features, pool, pairs = neighbourhood(args, loci, args.out)
    write_tsv(args.out / "neighbourhood_presence_absence.tsv",
              (dict(sample=sample, **{f"unitig_{f['locus']}": int(X[i, f['fasta_column']]) for f in features})
               for i, sample in enumerate(samples)), ["sample"] + [f"unitig_{f['locus']}" for f in features])
    phenotype = phenotype_rows(args, samples, args.out)
    tree, groups, K = tree_and_groups(args, samples, args.out, exclusions)
    tree_plot(tree, samples, X, features, phenotype, groups, args.out, args.dpi)
    if args.stage == "all":
        from .phenotype_lmm import fit_prediction
        fit_prediction(args, samples, X, loci, features, pool, pairs, phenotype, groups, K, args.out, identity)
    summary.update(status="complete", n_samples=len(samples), n_groups=len(set(groups)), excluded_samples=sorted(exclusions), n_unitigs=len(loci), n_features=len(features),
                   phenotype_counts={s: sum(p["status"] == s for p in phenotype)
                                     for s in sorted({p["status"] for p in phenotype})})
    json_write(args.out / "run.json", summary)
    (args.out / "figure_captions.txt").write_text(
        "Tree heatmap. A midpoint-rooted copy of the core-genome tree is aligned with binary "
        "unitig-sequence presence and exact log2 MIC. Coding-overlap "
        "unitigs are assigned through the locus annotation table, not the order of gene-edge endpoints. "
        "White MIC cells lack an exact measurement. Whitespace separates neighbourhood blocks. "
        "Midpoint rooting is a display convention, not evidence of ancestral states.\n\n"
        "MIC model comparison. Diamonds show pooled held-out RMSE; small dots show individual grouped-fold "
        "RMSE and are not independent replicates. White boxes summarize random-set RMSE, not confidence "
        "intervals. Dotted and dashed lines show mean-only and phylogeny-only benchmarks. Each random family matches "
        "the number of added retained predictors and uses nearest training minor-frequency matching; "
        "deviations are recorded in fold_performance.tsv. Default controls are other coding unitigs "
        "represented in the selected map; their physical distance to mprF is not independently verified. "
        "All models use identical held-out isolates and explicit exact MIC measurements only.\n\n"
        "Genotype states. Raw exact MIC values are shown for each direct target pair's four binary states. "
        "Interaction inference compares nested main-effect and product-term Gaussian phylogenetic mixed models "
        "by maximum likelihood. Missing, sparse and linearly dependent states are reported without forced interaction "
        "estimates. Reported coefficient intervals use nominal GLS standard errors; bootstrap P values simulate "
        "the fitted null model.\n\n"
        f"Observed versus predicted. Predictions are held out by {args.group_method} group. Horizontal jitter "
        "is display-only. The diagonal marks equality. MICs and predictions are expressed in log2 units. "
        "The existing network was discovered from the cohort and frozen before modelling; this is "
        "not an independent end-to-end evaluation of network discovery. Improved prediction does not "
        "establish mechanistic epistasis.\n", encoding="utf-8")
    log(f"outputs: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
