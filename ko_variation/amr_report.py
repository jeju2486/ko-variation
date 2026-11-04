"""Read-only, species-specific AMR phenotype/determinant/network reporting.

Annotation labels identify biological candidates; cluster IDs only join tables.
This helper neither calls resistance genotypes nor changes network selection.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re

from . import __version__

SPECIES = {
    "saureus": "Staphylococcus aureus",
    "klebsiella": "Klebsiella pneumoniae",
    "neisseria": "Neisseria gonorrhoeae",
}
DEFAULT_CATALOGUE = Path(__file__).with_name("data") / "amr_determinants.tsv"
MISSING = {"", "na", "n/a", "nan", "none", "null", "unknown", "not available"}
SIGNS = {"=", "<", ">", "<=", ">="}


def read_tsv(path, required=()):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: required columns {', '.join(required)}")
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f"{path}: malformed TSV row")
    return rows


def write_tsv(path, rows, columns):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def present(value):
    return str(value).strip().lower() not in MISSING


def drug_key(value):
    return re.sub(r"[\s/\-]+", "_", value.strip().lower())


def gene_key(value):
    # Ignore nomenclature punctuation, retaining letters/numbers: tet(M) == tetM.
    # Exact tokens prevent matching tetR, tet38 or gyrA to unrelated names.
    return re.sub(r"[^a-z0-9]", "", value.lower())


def label_tokens(label):
    return {gene_key(token) for token in re.split(r";|~~~", str(label))
            if present(token) and not re.fullmatch(r"group_\d+", token.strip(), re.I)}


def annotation_label(record):
    """Do not present a fallback cluster ID as a biological annotation name."""
    tokens = [token.strip() for token in re.split(r";|~~~", str(record.get("label", "")))
              if present(token) and not re.fullmatch(r"group_\d+", token.strip(), re.I)]
    return "; ".join(tokens) or "unannotated"


def mic_record(value, qualifier):
    """Keep censoring distinct from exact measurements; never impute bounds."""
    value, qualifier = str(value).strip(), str(qualifier).strip()
    value = value.replace("≤", "<=").replace("≥", ">=")
    qualifier = qualifier.replace("≤", "<=").replace("≥", ">=")
    inline = re.match(r"^(<=|>=|<|>|=)\s*(.*)$", value)
    if inline:
        if qualifier and qualifier != inline[1]:
            return None, "conflicting_sign"
        qualifier, value = inline[1], inline[2]
    try:
        number = float(value)
    except ValueError:
        return None, "invalid_value"
    if not math.isfinite(number) or number <= 0:
        return None, "invalid_value"
    if qualifier not in SIGNS and qualifier:
        return number, "unknown_sign"
    return number, "exact" if qualifier == "=" else "censored" if qualifier else "missing_sign"


def phenotype_inventory(metadata, species, id_column="id", sample_map=None, sample_inclusion=None):
    """Count isolates once per drug, preferring dedicated fields over JSON comments.

    Species-mismatched records are counted in provenance and excluded. Optional
    cohort restriction requires an explicit metadata_id/sample map, not guessing.
    """
    rows = read_tsv(metadata, [id_column, "species"])
    ids = [row[id_column] for row in rows]
    if any(not present(item) for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("Metadata IDs must be nonempty and unique")
    n_source = len(rows)
    rows = [row for row in rows if row["species"].strip().casefold() == species.casefold()]
    n_species = len(rows)
    if not rows:
        raise ValueError("Metadata contains no records matching the requested species")
    if bool(sample_map) != bool(sample_inclusion):
        raise ValueError("Supply both --sample-map and --sample-inclusion to restrict the cohort")
    unmatched = 0
    if sample_map:
        mapping_rows = read_tsv(sample_map, ["metadata_id", "sample"])
        mapping = {row["metadata_id"]: row["sample"] for row in mapping_rows}
        if len(mapping) != len(mapping_rows) or len(set(mapping.values())) != len(mapping_rows):
            raise ValueError("Sample map must be one-to-one")
        if any(not present(k) or not present(v) for k, v in mapping.items()):
            raise ValueError("Sample map IDs must be nonempty")
        included = {row["sample"] for row in read_tsv(sample_inclusion, ["sample", "status"])
                    if row["status"] == "included"}
        unmatched = sum(row[id_column] not in mapping for row in rows)
        rows = [row for row in rows if mapping.get(row[id_column]) in included]
        if not rows:
            raise ValueError("No species-matched metadata records match included samples")
    totals, values, sources, category_columns = defaultdict(Counter), defaultdict(set), defaultdict(set), defaultdict(set)
    for row in rows:
        records = {}
        # SIR fields remain categorical; they never become quantitative MICs.
        for column, value in row.items():
            if column.endswith("_mic") and present(value):
                drug = drug_key(column[:-4])
                records[drug] = (value, row.get(column + "_sign", ""), "dedicated_field")
            if column.endswith("_SIR") and present(value):
                drug = drug_key(re.split(r"_CLSI|_EUCAST", column)[0].removesuffix("_SIR"))
                category_columns[drug].add(column)
        try:
            embedded = json.loads(row.get("comments", ""))
        except (ValueError, TypeError):
            embedded = {}
        categorical = {drug_key(re.split(r"_CLSI|_EUCAST", column)[0].removesuffix("_SIR"))
                       for column, value in row.items() if column.endswith("_SIR") and present(value)}
        if isinstance(embedded, dict):
            for raw_drug, item in embedded.items():
                if not isinstance(item, dict):
                    continue
                drug = drug_key(raw_drug)
                if present(item.get("SIR", "")):
                    categorical.add(drug)
                    category_columns[drug].add("comment_json:SIR")
                if present(item.get("mic", "")):
                    record = (str(item["mic"]), str(item.get("mic_sign", "")), "comment_json")
                    if drug in records:
                        if mic_record(*record[:2]) != mic_record(*records[drug][:2]):
                            totals[drug]["n_source_conflicts"] += 1
                    else:
                        records[drug] = record
        for drug in categorical:
            totals[drug]["n_categorical_records"] += 1
        for drug, (value, sign, source) in records.items():
            count = totals[drug]
            count["n_mic_recorded"] += 1
            number, status = mic_record(value, sign)
            sources[drug].add(source)
            count["n_" + status] += 1
            if number is not None and status != "conflicting_sign":
                count["n_positive_numeric_mic"] += 1
            if status == "exact":
                values[drug].add(number)
    result = []
    for drug, count in sorted(totals.items()):
        result.append(dict(species=species, antimicrobial=drug, n_metadata_records=len(rows),
            **{field: count[field] for field in PHENOTYPE_COLUMNS[3:13]},
            n_distinct_exact_mic=len(values[drug]), record_sources=";".join(sorted(sources[drug])),
            categorical_fields=";".join(sorted(category_columns[drug]))))
    return result, dict(n_source_records=n_source, n_species_records=n_species,
        n_excluded_other_species=n_source-n_species, n_report_records=len(rows),
        n_metadata_without_sample_mapping=unmatched,
        cohort="included_scanner_samples" if sample_map else "species_matched_metadata_only")


def load_network(path):
    """Read data from helper exports without executing the HTML or JavaScript."""
    if path is None:
        return {}, None, {}
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix.lower() == ".html":
        match = re.search(r'<script\b[^>]*\bid=[\"\']data[\"\'][^>]*>(.*?)</script>', text, re.S | re.I)
        if not match:
            raise ValueError("Network HTML has no embedded data script; use a helper layout JSON")
        text = match[1]
    payload = json.loads(text)
    nodes = {str(row["gene"]): row for row in payload["nodes"]}
    if len(nodes) != len(payload["nodes"]) or any(not present(gene) for gene in nodes):
        raise ValueError("Network node IDs must be unique and nonempty")
    adjacency = {gene: set() for gene in nodes}
    for edge in payload["edges"]:
        a, b = str(edge["gene_a"]), str(edge["gene_b"])
        if a not in nodes or b not in nodes:
            raise ValueError("Network edge refers to a missing node")
        if a != b:
            adjacency[a].add(b)
            adjacency[b].add(a)
    return nodes, adjacency, payload.get("meta", {})


def candidate_status(record, rule):
    """Match labels only, then check explicit product requirements/homonyms."""
    if not label_tokens(record.get("label", "")) & label_tokens(rule["aliases"]):
        return None
    product = record.get("product", "")
    if rule.get("product_exclude") and re.search(rule["product_exclude"], product, re.I):
        return "rejected_product"
    if rule.get("product_require") and not re.search(rule["product_require"], product, re.I):
        return "product_unverified"
    return "annotation_match"


def audit_matches(path, candidates):
    """Track coding, nearby and ambiguous evidence without promoting nearby hits."""
    evidence = defaultdict(Counter)
    if path:
        for row in read_tsv(path, ["locus", "status", "gene", "candidate_evidence"]):
            items = json.loads(row["candidate_evidence"] or "[]")
            seen = set()
            for item in items:
                gene = str(item.get("gene", ""))
                if gene in candidates:
                    category = "ambiguous" if row["status"] != "RESOLVED" else item.get("annotation_class", "unknown")
                    if (gene, category) not in seen:
                        evidence[gene][category] += 1
                        seen.add((gene, category))
    return evidence


PHENOTYPE_COLUMNS = ["species", "antimicrobial", "n_metadata_records", "n_mic_recorded",
    "n_positive_numeric_mic", "n_exact", "n_censored", "n_missing_sign", "n_unknown_sign",
    "n_invalid_value", "n_conflicting_sign", "n_source_conflicts", "n_categorical_records",
    "n_distinct_exact_mic", "record_sources", "categorical_fields"]
REPORT_COLUMNS = ["species", "antimicrobial", "determinant", "determinant_type",
    "n_positive_numeric_mic", "n_exact", "n_censored", "n_missing_sign", "n_distinct_exact_mic",
    "annotation_status", "map_status", "n_annotation_clusters", "n_map_clusters",
    "n_direct_neighbours", "n_second_order_neighbours", "screening_status", "next_action",
    "reference", "notes"]
MATCH_COLUMNS = ["species", "antimicrobial", "determinant", "annotation_label", "cluster_id",
    "product", "match_status", "in_map", "n_coding_loci", "n_nearby_loci", "n_ambiguous_loci"]
NEIGHBOUR_COLUMNS = ["species", "antimicrobial", "determinant", "order", "annotation_label", "cluster_id"]


def build_report(*, species, metadata, out, determinants=None, gene_catalog=None,
                 network=None, annotation_status=None, id_column="id", sample_map=None,
                 sample_inclusion=None):
    """Join availability and annotation evidence, retaining negative/deferred cases."""
    species = SPECIES.get(species, species)
    if species not in SPECIES.values():
        raise ValueError("Unsupported species")
    catalogue = Path(determinants or DEFAULT_CATALOGUE)
    rules = read_tsv(catalogue, ["species", "antimicrobial", "determinant", "aliases", "determinant_type", "reference", "notes"])
    keys = [(r["species"], r["antimicrobial"], r["determinant"]) for r in rules]
    if len(set(keys)) != len(keys) or any(r["determinant_type"] not in {"presence_absence", "mutation_deferred"} for r in rules):
        raise ValueError("Catalogue rows must be unique and use presence_absence or mutation_deferred")
    if any(not present(r[field]) for r in rules for field in ["species", "antimicrobial", "determinant", "aliases", "reference"]):
        raise ValueError("Catalogue identifiers, aliases and evidence references must be nonempty")
    if any(r["species"] not in SPECIES.values() for r in rules):
        raise ValueError("Catalogue species must use a supported full scientific name")
    for rule in rules:
        for field in ("product_require", "product_exclude"):
            if rule.get(field):
                re.compile(rule[field], re.I)
    phenotypes, cohort = phenotype_inventory(metadata, species, id_column, sample_map, sample_inclusion)
    nodes, adjacency, network_meta = load_network(network)
    annotations = read_tsv(gene_catalog, ["gene", "label", "product"]) if gene_catalog else []
    records = {r["gene"]: r for r in annotations}
    if len(records) != len(annotations):
        raise ValueError("Gene catalogue cluster IDs must be unique")
    # Network labels can establish representation even without a full catalogue.
    # A missing node cannot establish annotation absence in that situation.
    if gene_catalog and nodes.keys() - records.keys():
        raise ValueError("Network cluster IDs missing from the supplied complete gene catalogue; verify input provenance")
    records.update({gene: row for gene, row in nodes.items() if gene not in records})
    evidence = audit_matches(annotation_status, records)
    reports, matches, neighbours, todos = [], [], [], []
    for phenotype in phenotypes:
        drug = phenotype["antimicrobial"]
        selected_rules = [r for r in rules if r["species"] == species and drug_key(r["antimicrobial"]) == drug]
        if not selected_rules:
            selected_rules = [dict(determinant="", determinant_type="not_curated", aliases="", reference="",
                                   notes="No mapping in the supplied starter/custom catalogue; no biological absence inferred.")]
        for rule in selected_rules:
            accepted, uncertain = set(), set()
            for gene, record in sorted(records.items()):
                status = candidate_status(record, rule)
                if status is None:
                    continue
                if status == "annotation_match":
                    accepted.add(gene)
                elif status == "product_unverified":
                    uncertain.add(gene)
                counts = evidence[gene]
                matches.append(dict(species=species, antimicrobial=drug, determinant=rule["determinant"],
                    annotation_label=annotation_label(record), cluster_id=gene, product=record.get("product", ""),
                    match_status=status, in_map=gene in nodes if network else "not_checked",
                    n_coding_loci=counts["coding"] if annotation_status else "not_checked",
                    n_nearby_loci=sum(counts[k] for k in ("upstream", "downstream", "unoriented_nearby", "nearby")) if annotation_status else "not_checked",
                    n_ambiguous_loci=counts["ambiguous"] if annotation_status else "not_checked"))
            mapped = accepted & nodes.keys()
            n1 = set().union(*(adjacency[gene] for gene in mapped)) - accepted if mapped else set()
            n2 = set().union(*(adjacency[gene] for gene in n1)) - accepted - n1 if n1 else set()
            annotation = "annotation_match" if accepted else "product_unverified" if uncertain else "not_in_catalogue" if gene_catalog else "not_checked"
            map_status = "present" if mapped else "product_unverified" if uncertain & nodes.keys() else "absent" if network else "not_checked"
            kind = rule["determinant_type"]
            if kind == "not_curated":
                annotation, map_status = "not_curated", "not_curated"
                stage, action = "CATALOGUE_NOT_COVERED", "Curate species-specific determinants with evidence references."
            elif kind == "mutation_deferred":
                stage, action = "MUTATION_ANALYSIS_DEFERRED", "Call relevant alleles/mutations; gene presence is not the resistance state."
            elif not phenotype["n_positive_numeric_mic"]:
                stage, action = "NO_QUANTITATIVE_MIC", "Obtain quantitative MIC; categorical susceptibility is insufficient."
            elif mapped and n1:
                stage = "NETWORK_CONTEXT_AVAILABLE"
                action = "Verify determinant genotype variation in MIC-matched isolates, units/censoring, full baseline and sample size."
                if annotation_status and not any(evidence[g]["coding"] for g in mapped):
                    action = "Verify coding overlap: map evidence is nearby or unverified. " + action
            else:
                stage = "FOLLOW_UP_REQUIRED"
                action = "Inspect annotation/selection coverage; absence from this selected map is not absence of association."
                if mapped:
                    action = "Mapped determinant has no additional direct neighbours; inspect the supplied network."
                if not network or not gene_catalog:
                    action = "Supply the species gene catalogue and selected distal network. " + action
            reports.append(dict(species=species, antimicrobial=drug, determinant=rule["determinant"], determinant_type=kind,
                **{key: phenotype[key] for key in REPORT_COLUMNS[4:9]}, annotation_status=annotation,
                map_status=map_status, n_annotation_clusters=len(accepted) if gene_catalog else "not_checked",
                n_map_clusters=len(mapped) if network else "not_checked",
                n_direct_neighbours=len(n1) if network else "not_checked",
                n_second_order_neighbours=len(n2) if network else "not_checked",
                screening_status=stage, next_action=action, reference=rule["reference"], notes=rule["notes"]))
            todos.append(dict(species=species, antimicrobial=drug, determinant=rule["determinant"], status=stage, action=action))
            for order, genes in ((1, n1), (2, n2)):
                for gene in sorted(genes):
                    neighbours.append(dict(species=species, antimicrobial=drug, determinant=rule["determinant"], order=order,
                        annotation_label=annotation_label(records[gene]), cluster_id=gene))
    prefix = Path(out)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix, rows, columns in [("phenotypes", phenotypes, PHENOTYPE_COLUMNS), ("determinants", reports, REPORT_COLUMNS),
        ("matches", matches, MATCH_COLUMNS), ("neighbours", neighbours, NEIGHBOUR_COLUMNS),
        ("todo", todos, ["species", "antimicrobial", "determinant", "status", "action"])]:
        write_tsv(str(prefix) + f".{suffix}.tsv", rows, columns)
    inputs = dict(metadata=metadata, determinants=catalogue, gene_catalog=gene_catalog, network=network,
                  annotation_status=annotation_status, sample_map=sample_map, sample_inclusion=sample_inclusion)
    fingerprints = {key: dict(path=str(Path(path).resolve()), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
                    for key, path in inputs.items() if path}
    summary = dict(schema_version=1, helper_version=__version__, species=species, cohort=cohort,
        inputs=fingerprints, n_available_phenotypes=len(phenotypes), n_network_nodes=len(nodes) if network else None,
        screening_counts=dict(Counter(row["screening_status"] for row in reports)), network_metadata=network_meta,
        interpretation="Annotation-label screen; no sequence validation, genotype calling, phenotype modelling or network reselection.",
        network_selection="Uses supplied selected map as-is; significance/distance provenance must be verified upstream.",
        neighbourhoods="Cluster unions across all accepted labels; N2 excludes N1/targets; two graph steps do not imply target-to-N2 physical distance.",
        catalogue_scope="Curated starter mappings are incomplete; no drug-class extrapolation or prediction-readiness claim.",
        mic_policy="Positive numeric limits remain censored; missing qualifiers remain unknown; no log transform or imputation.")
    Path(str(prefix) + ".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", required=True, choices=list(SPECIES) + list(SPECIES.values()))
    parser.add_argument("--metadata", required=True, help="Collection TSV with species, id and *_mic / *_mic_sign fields")
    parser.add_argument("--out", required=True, help="Output prefix; five TSVs and provenance JSON")
    parser.add_argument("--determinants", help="Custom curated TSV; replaces the bundled starter catalogue")
    parser.add_argument("--gene-catalog", help="Complete annotation gene/label/product TSV")
    parser.add_argument("--network", help="Selected distal helper network .html or .layout.json")
    parser.add_argument("--annotation-status", help="Optional locus annotation audit for coding/nearby/ambiguous evidence")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--sample-map", help="Optional one-to-one metadata_id/sample TSV; requires --sample-inclusion")
    parser.add_argument("--sample-inclusion", help="Optional scanner sample/status TSV; requires --sample-map")
    args = parser.parse_args(argv)
    try:
        summary = build_report(**vars(args))
    except (ValueError, KeyError, OSError, re.error) as error:
        parser.exit(2, f"AMR report: {error}\n")
    print(f"AMR report: {summary['species']}; {summary['n_available_phenotypes']} phenotypes; "
          f"{summary['screening_counts']}; output {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
