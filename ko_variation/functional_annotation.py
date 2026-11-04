"""Prepare mapped representative proteins and read eggNOG functional annotations.

Cluster IDs join tables; annotation names never merge distinct clusters. This
module annotates an existing network and does not change its selected edges.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import pandas as pd

# Broad display bins combine documented COG letter categories. These are display
# summaries, not GO terms or evidence of a resistance mechanism.
FUNCTION_COLORS = {
    "Metabolism": "#009E73",
    "Transport / secretion": "#0072B2",
    "Cell envelope": "#E69F00",
    "Genetic information processing": "#CC79A7",
    "Regulation / signalling": "#56B4E9",
    "Cellular processes": "#6B6B3E",
    "Multiple categories": "#8172B2",
    "Unassigned": "#AAB2BC",
}
COG_BINS = {**dict.fromkeys("CGEFHIPQ", "Metabolism"),
            **dict.fromkeys("UW", "Transport / secretion"),
            "M": "Cell envelope",
            **dict.fromkeys("JALB", "Genetic information processing"),
            **dict.fromkeys("TK", "Regulation / signalling"),
            **dict.fromkeys("DNYZOV", "Cellular processes")}
# K is transcription; put it with regulation rather than general information flow.


def biological_names(label):
    """Keep every named alias, suppressing only placeholder cluster identifiers."""
    values = re.split(r";|~~~", str(label or ""))
    return "; ".join(dict.fromkeys(value.strip() for value in values
        if value.strip() and value.strip().lower() not in {"nan", "-"}
        and not re.fullmatch(r"group_\d+", value.strip(), re.I)))


def _clean(value):
    return "" if value is None or str(value).strip() in {"", "-", "nan"} else str(value).strip()


def read_eggnog(path):
    """Read v2/v3 annotations by column names, never by version-specific positions."""
    rows, columns = {}, None
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for line in handle:
            if line.startswith("##") or not line.strip():
                continue
            fields = line.rstrip("\r\n").split("\t")
            if fields[0].lstrip("#") == "query":
                columns = [fields[0].lstrip("#")] + fields[1:]
                if not {"query", "COG_category", "GOs"}.issubset(columns):
                    raise ValueError("eggNOG header must include query, COG_category and GOs")
                continue
            if line.startswith("#"):
                continue
            if columns is None or len(fields) != len(columns):
                raise ValueError("Missing eggNOG header or inconsistent annotation column count")
            row = dict(zip(columns, fields))
            query = row["query"]
            if query in rows:
                raise ValueError(f"Duplicate eggNOG query ID: {query}")
            rows[query] = row
    if columns is None:
        raise ValueError("No eggNOG query header found")
    return rows


def annotate_nodes(nodes, gene_catalog=None, eggnog=None):
    """Enrich display attributes; retain all node IDs and original annotations."""
    display = nodes.copy()
    catalogue = {}
    if gene_catalog:
        table = pd.read_csv(gene_catalog, sep="\t", dtype=str, keep_default_na=False)
        if not {"gene", "label"}.issubset(table.columns) or table.gene.duplicated().any():
            raise ValueError("Gene catalogue needs unique gene IDs and a label column")
        catalogue = table.set_index("gene").to_dict("index")
    functions = read_eggnog(eggnog) if eggnog else {}
    if functions and not (set(display.gene) & set(functions)):
        raise ValueError("No eggNOG query IDs match network cluster IDs; use prepared proteins")
    enriched = []
    for node in display.to_dict("records"):
        original = biological_names(node.get("label"))
        catalog = catalogue.get(node["gene"], {})
        names = biological_names("; ".join(filter(None, [original, catalog.get("label", "")])))
        product = _clean(catalog.get("product")) or _clean(node.get("product"))
        annotation = functions.get(node["gene"], {})
        preferred = biological_names(_clean(annotation.get("Preferred_name")))
        node["annotation_names"] = names
        node["label_source"] = "existing annotation" if names else "eggNOG predicted name" if preferred else "product description"
        node["display_label"] = names or (preferred + " (predicted)" if preferred else product or "Unannotated protein")
        if node.get("n_coding_loci") == 0 and node.get("n_nearby_loci", 0) > 0:
            node["display_label"] = "near " + node["display_label"]
        node["product"] = product
        node["eggnog_preferred_name"] = preferred
        node["cog_category"] = _clean(annotation.get("COG_category"))
        node["go_terms"] = _clean(annotation.get("GOs"))
        node["eggnog_description"] = _clean(annotation.get("Description"))
        node["eggnog_seed_ortholog"] = _clean(annotation.get("seed_ortholog"))
        categories = sorted({COG_BINS[c] for c in node["cog_category"] if c in COG_BINS})
        node["function_categories"] = "; ".join(categories)
        node["function_category"] = categories[0] if len(categories) == 1 else "Multiple categories" if categories else "Unassigned"
        node["color"] = FUNCTION_COLORS[node["function_category"]] if eggnog else "#3F4A5A"
        enriched.append(node)
    added = ["annotation_names", "label_source", "display_label", "eggnog_preferred_name",
             "cog_category", "go_terms", "eggnog_description", "eggnog_seed_ortholog",
             "function_categories", "function_category", "color"]
    return pd.DataFrame(enriched, columns=list(dict.fromkeys(list(nodes.columns) + added)))


def fasta_records(path):
    name, sequence = None, []
    with Path(path).open() as handle:
        for line in handle:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(sequence)
                name, sequence = line[1:].split()[0], []
            elif line.strip():
                if name is None:
                    raise ValueError(f"Sequence before FASTA header: {path}")
                sequence.append(line.strip())
    if name is not None:
        yield name, "".join(sequence)


def prepare_proteins(effects, panaroo, bakta, out):
    """Use the first available unambiguous Bakta protein per mapped cluster.

    Iterate Panaroo sample columns and FASTA order deterministically. A single
    representative describes a cluster's broad function, not every allele.
    Unmatched clusters remain in the network with an explicit missing status.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(effects, sep="\t", usecols=["u_gene", "v_gene"], dtype=str)
    wanted = set(table.u_gene.dropna()) | set(table.v_gene.dropna())
    wanted.discard("")
    memberships, available, labels = {}, set(), {}
    with Path(panaroo).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "Gene" not in (reader.fieldnames or []):
            raise ValueError("Panaroo table lacks Gene column")
        # Match the established Bakta SAMPLE/SAMPLE.faa layout, not metadata fields.
        samples = [sample for sample in reader.fieldnames
                   if (Path(bakta) / sample / f"{sample}.faa").is_file()]
        if not samples:
            raise ValueError("No Bakta SAMPLE/SAMPLE.faa files match Panaroo sample columns")
        for row in reader:
            cluster = row["Gene"]
            if cluster not in wanted:
                continue
            available.add(cluster)
            labels[cluster] = row.get("Non-unique Gene name", "")
            for sample in samples:
                for feature in re.split(r"[;\t]", row.get(sample, "") or ""):
                    if feature:
                        memberships.setdefault((sample, feature), set()).add(cluster)
    # Block a feature that also belongs to an unselected cluster. Checking this
    # second pass keeps memory bounded to selected memberships without silently
    # resolving conflicting Panaroo memberships in favour of network genes.
    with Path(panaroo).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["Gene"] in wanted:
                continue
            for sample in samples:
                for feature in re.split(r"[;\t]", row.get(sample, "") or ""):
                    if (sample, feature) in memberships:
                        memberships[sample, feature].add(row["Gene"])
    selected = {}
    for sample in samples:
        path = Path(bakta) / sample / f"{sample}.faa"
        for feature, sequence in fasta_records(path):
            clusters = set().union(*(memberships.get((sample, feature + suffix), set())
                for suffix in ("", "_len", "_pseudo", "_len_pseudo")))
            if len(clusters) != 1 or not sequence:
                continue
            cluster = next(iter(clusters))
            selected.setdefault(cluster, (sample, feature, sequence, path))
        if len(selected) == len(available):
            break
    with (out / "network_proteins.faa").open("w", encoding="utf-8") as fasta, \
            (out / "protein_manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["gene", "label", "status", "sample", "feature", "protein_sha256", "source_faa"])
        for cluster in sorted(wanted):
            if cluster in selected:
                sample, feature, sequence, path = selected[cluster]
                if any(c.isspace() for c in cluster):
                    raise ValueError(f"Cluster ID contains FASTA-unsafe whitespace: {cluster}")
                fasta.write(f">{cluster}\n{sequence}\n")
                writer.writerow([cluster, labels.get(cluster, ""), "REPRESENTATIVE", sample, feature,
                                 hashlib.sha256(sequence.encode()).hexdigest(), str(path.resolve())])
            else:
                writer.writerow([cluster, labels.get(cluster, ""), "NO_PROTEIN_MATCH" if cluster in available else "NOT_IN_PANAROO", "", "", "", ""])
    summary = dict(network_clusters=len(wanted), representative_proteins=len(selected),
        missing_proteins=len(wanted)-len(selected), effects=str(Path(effects).resolve()),
        panaroo=str(Path(panaroo).resolve()), bakta=str(Path(bakta).resolve()),
        representative_rule="first available unambiguous Bakta protein in Panaroo sample-column / FASTA order")
    (out / "protein_preparation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[function] prepared {len(selected)}/{len(wanted)} representative proteins; audit: {out / 'protein_manifest.tsv'}", flush=True)
    if not selected:
        raise ValueError("No proteins resolved; inspect Panaroo membership and Bakta protein IDs")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effects", required=True)
    parser.add_argument("--panaroo", required=True, help="gene_presence_absence[_roary].csv")
    parser.add_argument("--bakta", required=True, help="SAMPLE/SAMPLE.faa directory")
    parser.add_argument("--out", required=True, help="Separate functional annotation directory")
    args = parser.parse_args(argv)
    output = Path(args.out).resolve()
    if any(Path(path).resolve().is_relative_to(output) for path in (args.effects, args.panaroo)) or output.is_relative_to(Path(args.bakta).resolve()):
        parser.error("Output must be separate from biological inputs and Bakta directory")
    try:
        prepare_proteins(args.effects, args.panaroo, args.bakta, output)
    except (ValueError, OSError) as exc:
        sys.stderr.write(f"Functional annotation error: {exc}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
