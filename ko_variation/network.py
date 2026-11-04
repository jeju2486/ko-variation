"""Undirected gene covariation summaries and standalone visual exports."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from .postprocess import _require, annotate_pairs


EDGE_COLUMNS = ["gene_a", "gene_b", "n_locus_pairs", "min_significance", "significance_column", "representative_u",
                "representative_v", "representative_adjusted_odds_ratio", "adjusted_direction", "n_adjusted_pairs", "locus_pairs",
                "representative_adjusted_beta", "representative_adjusted_beta_se", "representative_significance",
                "representative_physical_distance", "representative_u_annotation_class",
                "representative_v_annotation_class", "representative_u_annotation_distance_max_bp",
                "representative_v_annotation_distance_max_bp", "n_nearby_locus_pairs"]
NODE_COLUMNS = ["gene", "label", "product", "group", "n_loci", "degree", "n_supporting_pairs",
                "annotation_classes", "n_coding_loci", "n_nearby_loci", "annotation_distance_max_bp"]


def discovery_network_input(frame: pd.DataFrame, summary: dict) -> pd.DataFrame:
    """Adapt certified v0.9 edges without inventing exact P-values or refits.

    The network significance column stays explicitly `p_upper`. Local and
    unknown-distance candidates remain in the source, outside the distal map.
    """
    _require(frame,["u","v","significance","p_lower","p_upper","distance_status","min_distance","beta","beta_status"])
    threshold=float(summary["raw_p_threshold"])
    alpha=float(summary["alpha"])
    family=int(summary["bonferroni_tests"])
    if family<1 or not np.isfinite(threshold) or threshold<=0 or not np.isclose(threshold,alpha/family,rtol=1e-12,atol=0):
        raise ValueError("Invalid discovery significance provenance")
    result=frame.loc[frame.significance.eq("SIGNIFICANT") & frame.distance_status.eq("DISTAL")].copy()
    upper=pd.to_numeric(result.p_upper,errors="raise")
    lower=pd.to_numeric(result.p_lower,errors="raise")
    if not np.isfinite(upper).all() or not np.isfinite(lower).all() or (lower<0).any() or (upper<lower).any() or (upper>threshold*(1+1e-12)).any():
        raise ValueError("Discovery edge is not certified at the recorded Bonferroni cutoff")
    result["adjusted_beta"]=pd.to_numeric(result.beta,errors="raise")
    result["effect_status"]=result.beta_status
    fitted=result.effect_status.eq("OK")
    if not np.isfinite(result.loc[fitted,"adjusted_beta"]).all():
        raise ValueError("Successful discovery beta is not finite")
    result["adjusted_odds_ratio"]=np.nan
    with np.errstate(over="ignore",under="ignore"):
        result.loc[fitted,"adjusted_odds_ratio"]=np.exp(result.loc[fitted,"adjusted_beta"])
    result["physical_distance"]=pd.to_numeric(result.min_distance,errors="raise")
    return result


def _annotation_classes(value):
    return set(str(value).split("; ")) if pd.notna(value) else set()


def build_gene_network(signals: pd.DataFrame, annotation: pd.DataFrame | None = None,
                       *, significance_column: str = "p_primary", include_self: bool = False,
                       missing_genes: str = "error") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate unique locus pairs per unordered gene pair, without pooling ORs.

    The representative is the smallest-significance successful refit (ties: u, v). Its OR
    remains a locus effect. min_significance is descriptive, not a gene-level test.
    """
    if missing_genes not in {"error", "drop"}:
        raise ValueError("missing_genes must be error or drop")
    data = annotate_pairs(signals, annotation) if annotation is not None else signals.copy()
    _require(data, ["u", "v", "u_gene", "v_gene", significance_column, "adjusted_odds_ratio", "adjusted_beta", "effect_status"])
    if data.duplicated(["u", "v"]).any():
        raise ValueError("Duplicate locus pairs cannot be counted as independent support")
    missing = data.u_gene.isna() | data.v_gene.isna() | data.u_gene.astype(str).str.strip().eq("") | data.v_gene.astype(str).str.strip().eq("")
    if missing.any() and missing_genes == "error":
        raise ValueError("Missing gene annotations; supply mappings or choose missing_genes='drop'")
    data = data.loc[~missing].copy()
    data["u_gene"] = data.u_gene.astype(str)
    data["v_gene"] = data.v_gene.astype(str)
    sig = pd.to_numeric(data[significance_column], errors="raise")
    if not np.isfinite(sig).all() or ((sig < 0) | (sig > 1)).any():
        raise ValueError("Network significance values must be finite and in [0, 1]")
    data[significance_column] = sig
    data["gene_a"] = [min(a, b) for a, b in zip(data.u_gene, data.v_gene)]
    data["gene_b"] = [max(a, b) for a, b in zip(data.u_gene, data.v_gene)]
    if not include_self:
        data = data.loc[data.gene_a.ne(data.gene_b)]
    edges = []
    for (a, b), block in data.groupby(["gene_a", "gene_b"], sort=True):
        block = block.sort_values([significance_column, "u", "v"], kind="stable")
        fitted = block.loc[block.effect_status.eq("OK")]
        representative = fitted.iloc[0] if len(fitted) else block.iloc[0]
        effects = pd.to_numeric(fitted.adjusted_beta, errors="raise").to_numpy(float)
        direction = ("mixed" if (effects > 0).any() and (effects < 0).any() else
                     "positive" if (effects > 0).any() else "negative" if (effects < 0).any() else
                     "neutral" if (effects == 0).any() else "unknown")
        nearby = np.zeros(len(block), dtype=bool)
        for side in ("u", "v"):
            column = f"{side}_annotation_class"
            if column in block:
                nearby |= block[column].map(lambda value: bool(_annotation_classes(value) & {"upstream", "downstream"})).to_numpy()
        edges.append([a, b, len(block), float(block.iloc[0][significance_column]), significance_column,
                      int(representative.u), int(representative.v), float(representative.adjusted_odds_ratio),
                      direction, len(fitted), json.dumps([[int(row.u), int(row.v)] for row in block.itertuples()]),
                      float(representative.adjusted_beta) if len(fitted) else np.nan,
                      float(representative.get("adjusted_beta_se", np.nan)) if len(fitted) else np.nan,
                      float(representative[significance_column]), float(representative.get("physical_distance", np.nan)),
                      representative.get("u_annotation_class", ""), representative.get("v_annotation_class", ""),
                      representative.get("u_annotation_distance_max_bp", np.nan),
                      representative.get("v_annotation_distance_max_bp", np.nan), int(nearby.sum())])
    edge_frame = pd.DataFrame(edges, columns=EDGE_COLUMNS)
    nodes = []
    for gene in sorted(set(edge_frame.gene_a) | set(edge_frame.gene_b)):
        mapped = []
        loci = set()
        for side in ("u", "v"):
            block = data.loc[data[f"{side}_gene"].eq(gene)]
            loci.update(block[side].tolist())
            mapped.append(block)
        attributes = {}
        for attribute in ("label", "product", "group"):
            values = set()
            for side, block in zip(("u", "v"), mapped):
                column = f"{side}_{attribute}"
                if column in block:
                    values.update(str(v) for v in block[column].dropna() if str(v).strip())
            attributes[attribute] = "; ".join(sorted(values)) or (gene if attribute == "label" else "")
        classes, coding_loci, nearby_loci, distances = set(), set(), set(), []
        for side, block in zip(("u", "v"), mapped):
            if f"{side}_annotation_class" in block:
                for locus, value in zip(block[side], block[f"{side}_annotation_class"]):
                    categories = _annotation_classes(value)
                    classes.update(categories)
                    if "coding" in categories:
                        coding_loci.add(locus)
                    if categories & {"upstream", "downstream"}:
                        nearby_loci.add(locus)
            if f"{side}_annotation_distance_max_bp" in block:
                distances.extend(pd.to_numeric(block[f"{side}_annotation_distance_max_bp"], errors="raise").dropna().tolist())
        incident = edge_frame.loc[edge_frame.gene_a.eq(gene) | edge_frame.gene_b.eq(gene)]
        neighbours = (set(incident.gene_a) | set(incident.gene_b)) - {gene}
        nodes.append([gene, attributes["label"], attributes["product"], attributes["group"],
                      len(loci), len(neighbours), int(incident.n_locus_pairs.sum()), "; ".join(sorted(classes)),
                      len(coding_loci), len(nearby_loci), max(distances) if distances else np.nan])
    return pd.DataFrame(nodes, columns=NODE_COLUMNS), edge_frame


from .network_display import export_gene_network
