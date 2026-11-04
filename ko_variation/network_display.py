"""Compact effect-magnitude layout and shared static/D3 exports."""
from __future__ import annotations

import html
import json
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from .functional_annotation import annotate_nodes, FUNCTION_COLORS

COLORS = dict(positive="#C62828", negative="#2166AC", mixed="#7B3294",
              neutral="#AAB2BC", unknown="#AAB2BC")


def add_network_arguments(parser):
    parser.add_argument("--gene-catalog", help="Refresh annotation names/products by stable cluster ID")
    parser.add_argument("--eggnog", help="eggNOG .emapper.annotations with cluster IDs as query IDs; node fill uses broad COG categories")
    parser.add_argument("--node-radius", type=float, nargs=2, default=[3, 5], metavar=("MIN", "MAX"))
    parser.add_argument("--link-distances", type=float, nargs=3, default=[20, 35, 55], metavar=("STRONG", "MEDIUM", "WEAK"))
    parser.add_argument("--strength-cutoffs", type=float, nargs=2, metavar=("LOW", "HIGH"), help="Fixed |representative beta| boundaries; default within-map tertiles")
    parser.add_argument("--component-gap", type=float, default=16)
    parser.add_argument("--labels", choices=["none", "hubs", "all"], default="none", help="Static labels; interactive labels appear on hover/search")


def network_options(args):
    return dict(node_radius=args.node_radius, link_distances=args.link_distances,
                strength_cutoffs=args.strength_cutoffs, component_gap=args.component_gap, labels=args.labels,
                gene_catalog=args.gene_catalog, eggnog=args.eggnog)


def _records(frame):
    # Preserve tiny P-values and full float precision; only missing values become null.
    records = frame.to_dict("records")
    for row in records:
        for key, value in row.items():
            if isinstance(value, np.generic):
                value = value.item()
            row[key] = None if pd.isna(value) else value
    return records


def _display_edges(edges, cutoffs, distances):
    display = edges.copy()
    beta = pd.to_numeric(display.representative_adjusted_beta, errors="raise")
    magnitude = beta.abs().where(np.isfinite(beta) & display.n_adjusted_pairs.gt(0))
    values = magnitude.dropna().to_numpy()
    if cutoffs is None:
        boundaries = np.quantile(values, [1/3, 2/3]).tolist() if len(values) else None
    else:
        boundaries = list(map(float, cutoffs))
        if len(boundaries) != 2 or not np.isfinite(boundaries).all() or not 0 <= boundaries[0] <= boundaries[1]:
            raise ValueError("Strength cutoffs must be finite and 0 <= LOW <= HIGH")
    categories = []
    for value in magnitude:
        categories.append("unestimated" if pd.isna(value) else "weak" if value <= boundaries[0]
                          else "medium" if value <= boundaries[1] else "strong")
    target = dict(zip(("strong", "medium", "weak"), distances))
    target["unestimated"] = distances[1]
    display["representative_abs_beta"] = magnitude
    display["strength_category"] = categories
    display["target_link_distance"] = [target[category] for category in categories]
    # Continuous widths keep ties identical; clipping prevents thick dense webs.
    upper = float(np.quantile(values, 0.95)) if len(values) else 0
    display["display_width"] = [0.7 if pd.isna(value) else 0.7 + 1.5 * min(value / upper, 1)
                                if upper > 0 else 0.7 for value in magnitude]
    display["color"] = [COLORS[direction] for direction in display.adjusted_direction]
    return display, boundaries


def _positions(nodes, edges, seed, radius, gap):
    import networkx as nx
    from scipy.spatial import cKDTree
    graph = nx.Graph()
    graph.add_nodes_from(sorted(nodes.gene))
    for row in edges.itertuples():
        graph.add_edge(row.gene_a, row.gene_b, target=float(row.target_link_distance),
                       attraction=1 / float(row.target_link_distance)**2)
    blocks = []
    for component_id, genes in enumerate(sorted(nx.connected_components(graph), key=lambda group: (-len(group), sorted(group)))):
        genes = sorted(genes)
        subgraph = graph.subgraph(genes)
        if len(genes) == 1:
            xy = np.zeros((1, 2))
        elif len(genes) == 2:
            distance = subgraph[genes[0]][genes[1]]["target"]
            xy = np.array([[-distance/2, 0], [distance/2, 0]])
        else:
            if len(genes) <= 500:
                layout = nx.kamada_kawai_layout(subgraph, weight="target", pos=nx.circular_layout(subgraph))
            else:
                layout = nx.spring_layout(subgraph, seed=seed, weight="attraction", iterations=100)
            xy = np.array([layout[gene] for gene in genes])
            index = {gene: i for i, gene in enumerate(genes)}
            ratios = [attrs["target"] / max(np.linalg.norm(xy[index[a]] - xy[index[b]]), 1e-9)
                      for a, b, attrs in subgraph.edges(data=True) if a != b]
            xy *= float(np.median(ratios)) if ratios else 1
        # Separate overlapping circles while retaining the compact spring geometry.
        rng = np.random.default_rng(seed + component_id)
        for _ in range(40):
            pairs = sorted(cKDTree(xy).query_pairs(2 * max(radius.values(), default=5) + 2))
            moved = False
            for i, j in pairs:
                delta = xy[j] - xy[i]
                length = np.linalg.norm(delta)
                minimum = radius[genes[i]] + radius[genes[j]] + 2
                if length < minimum:
                    direction = delta / length if length > 1e-10 else rng.normal(size=2)
                    direction /= np.linalg.norm(direction)
                    shift = direction * (minimum - length + 0.05) / 2
                    xy[i] -= shift
                    xy[j] += shift
                    moved = True
            if not moved:
                break
        pad = max(radius.values(), default=5) + 4
        xy -= xy.min(axis=0)
        xy += pad
        extent = xy.max(axis=0) + pad
        blocks.append((genes, xy, extent, component_id))
    area = sum(float(extent[0]*extent[1]) for _, _, extent, _ in blocks)
    shelf_width = max([np.sqrt(max(area, 1)*1.5)] + [extent[0] for _, _, extent, _ in blocks])
    positions, centers, component_ids = {}, {}, {}
    x = y = row_height = 0.0
    used_width = 0.0
    for genes, xy, extent, component_id in blocks:
        if x and x + extent[0] > shelf_width:
            x = 0
            y += row_height + gap
            row_height = 0
        for gene, coordinate in zip(genes, xy):
            positions[gene] = (coordinate + [x+20, y+20]).tolist()
            centers[gene] = [x + extent[0]/2 + 20, y + extent[1]/2 + 20]
            component_ids[gene] = component_id
        used_width = max(used_width, x + extent[0])
        x += extent[0] + gap
        row_height = max(row_height, extent[1])
    return positions, centers, component_ids, max(used_width+40, 300), max(y+row_height+40, 160)


def export_gene_network(nodes, edges, prefix, *, seed=42, dpi=300, title="Distal gene covariation",
                        node_radius=(3, 5), link_distances=(20, 35, 55), strength_cutoffs=None,
                        component_gap=16, labels="none", gene_catalog=None, eggnog=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Arc
    if dpi < 72 or len(node_radius) != 2 or not np.isfinite(node_radius).all() or not 0 < node_radius[0] <= node_radius[1]:
        raise ValueError("DPI must be >=72 and node radii finite, positive and ordered")
    if len(link_distances) != 3 or not np.isfinite(link_distances).all() or not 0 < link_distances[0] <= link_distances[1] <= link_distances[2]:
        raise ValueError("Link distances must be finite and 0 < strong <= medium <= weak")
    if not np.isfinite(component_gap) or component_gap < 0 or labels not in {"none", "hubs", "all"}:
        raise ValueError("Invalid component gap or label mode")
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {key: Path(str(prefix) + suffix) for key, suffix in (
        ("png", ".png"), ("svg", ".svg"), ("html", ".html"), ("nodes", ".nodes.tsv"),
        ("edges", ".edges.tsv"), ("layout", ".layout.json"))}
    edge_display, boundaries = _display_edges(edges, strength_cutoffs, link_distances)
    degree_max = max(float(nodes.degree.max()), 1) if len(nodes) else 1
    radii = {row.gene: node_radius[0] + (node_radius[1]-node_radius[0]) * np.sqrt(max(row.degree-1, 0) / max(degree_max-1, 1))
             for row in nodes.itertuples()}
    positions, centers, components, width, height = _positions(nodes, edge_display, seed, radii, component_gap)
    node_display = annotate_nodes(nodes, gene_catalog, eggnog)
    for name, values in (
        ("x", [positions[gene][0] for gene in nodes.gene]), ("y", [positions[gene][1] for gene in nodes.gene]),
        ("radius", [radii[gene] for gene in nodes.gene]), ("component", [components[gene] for gene in nodes.gene]),
        ("component_x", [centers[gene][0] for gene in nodes.gene]), ("component_y", [centers[gene][1] for gene in nodes.gene]),
        ):
        node_display[name] = values
    function_legend = {category: color for category, color in FUNCTION_COLORS.items()
                       if eggnog and category in set(node_display.function_category)}
    fig_width = max(6, min(14, width/100))
    legend_columns = 2 if fig_width < 9 else 3
    legend_rows = int(np.ceil(len(function_legend) / legend_columns))
    # Reserve physical footer space so many functional categories cannot overlap
    # the graph, including on narrow maps with only a few connected components.
    footer_inches = .7 + (.22 * (legend_rows + 1) if function_legend else 0)
    fig_height = max(3.5, fig_width * height/width) + footer_inches
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.subplots_adjust(left=.02, right=.98, bottom=footer_inches/fig_height, top=.94 if title else .99)
    unit_points = fig_width * .96 * 72 / width
    for row in edge_display.itertuples():
        a, b = positions[row.gene_a], positions[row.gene_b]
        style = "--" if row.adjusted_direction in {"mixed", "unknown"} else "-"
        if row.gene_a == row.gene_b:
            ax.add_patch(Arc((a[0], a[1]-10), 20, 20, theta1=10, theta2=350, color=row.color,
                             lw=row.display_width, alpha=.7, linestyle=style))
        else:
            ax.plot([a[0], b[0]], [a[1], b[1]], color=row.color,
                    lw=row.display_width, alpha=.7, linestyle=style, zorder=1)
    if len(nodes):
        ax.scatter(node_display.x, node_display.y, s=[(2*r*unit_points)**2 for r in node_display.radius],
                   c=node_display.color.tolist(), edgecolors="white", linewidths=.35, zorder=2)
        hub_threshold = float(nodes.degree.quantile(.95))
        for row in node_display.itertuples():
            if (row.annotation_names or row.eggnog_preferred_name) and (labels == "all" or labels == "hubs" and row.degree >= max(3, hub_threshold)):
                ax.annotate(row.display_label, (row.x, row.y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    else:
        ax.text(width/2, height/2, "No gene pairs pass selection", ha="center")
    ax.set(xlim=(0, width), ylim=(height, 0), aspect="equal")
    ax.axis("off")
    if title:
        ax.set_title(title)
    handles = [Line2D([], [], color=color, lw=1.4, label=direction,
                      linestyle="--" if direction in {"mixed", "unknown"} else "-")
               for direction, color in COLORS.items() if direction in set(edge_display.adjusted_direction)]
    if handles:
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .32/fig_height), ncol=len(handles), frameon=False, fontsize=8)
    if function_legend:
        function_handles = [Line2D([], [], marker="o", color="none", markerfacecolor=color,
                                  markeredgecolor="white", label=category, markersize=6)
                            for category, color in function_legend.items()]
        fig.legend(handles=function_handles, loc="lower center", bbox_to_anchor=(.5, .65/fig_height),
                   ncol=legend_columns, frameon=False, fontsize=8, title="Node fill: broad COG function")
    threshold_text = "No finite representative coefficients" if boundaries is None else (
        f"|representative β|: weak ≤ {boundaries[0]:.3g}; medium > {boundaries[0]:.3g} to {boundaries[1]:.3g}; strong > {boundaries[1]:.3g}")
    fig.text(.5, .08/fig_height, threshold_text + "\nSpacing/width: representative |β|; node size: degree; positions are diagram coordinates.",
             ha="center", fontsize=7)
    with matplotlib.rc_context({"svg.fonttype": "none"}):
        fig.savefig(paths["png"], dpi=dpi, facecolor="white")
        fig.savefig(paths["svg"], facecolor="white")
    plt.close(fig)
    sources = {key: dict(path=str(Path(path).resolve()), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
               for key, path in (("gene_catalog", gene_catalog), ("eggnog", eggnog)) if path}
    payload = dict(nodes=_records(node_display), edges=_records(edge_display),
                   meta=dict(width=width, height=height, seed=seed, node_radius=list(node_radius),
                             link_distances=list(link_distances), strength_cutoffs=boundaries,
                             cutoff_mode="fixed" if strength_cutoffs is not None else "within_map_tertiles",
                             component_gap=component_gap, labels=labels, title=title, colors=COLORS,
                             function_colors=function_legend, annotation_sources=sources,
                             function_mapping="broad COG display bins; multiple bins retained; R/S or missing are unassigned",
                             representative_rule="smallest-P successful locus refit; ties u,v",
                             layout_method="packed connected components; beta-weighted shortest-path springs",
                             layout_distance_units="diagram units, not genomic bp"))
    serialized = json.dumps(payload, allow_nan=False)
    paths["layout"].write_text(serialized, encoding="utf-8")
    package = Path(__file__).parent
    document = (package / "templates/network_d3.html").read_text(encoding="utf-8")
    d3 = (package / "vendor/d3.v7.9.0.min.js").read_text(encoding="utf-8").replace("</script", "<\\/script")
    safe_payload = serialized.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    document = document.replace("__TITLE__", html.escape(title or "Gene covariation map"))
    document = document.replace("__DATA__", safe_payload).replace("__D3__", d3)
    paths["html"].write_text(document, encoding="utf-8")
    node_display.to_csv(paths["nodes"], sep="\t", index=False)
    edge_display.to_csv(paths["edges"], sep="\t", index=False)
    return paths
