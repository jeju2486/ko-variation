"""Bounded, strand-aware annotations of exact pyseer draft-reference hits.

This helper uses BWA's one-based inclusive hit coordinates and GFF CDS geometry,
not pyseer's unbounded nearest-gene names. It never changes pair selection.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import sqlite3


def create_geometry(path):
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE contigs (contig TEXT PRIMARY KEY, reference TEXT, original_contig TEXT, length INTEGER)")
    db.execute("CREATE TABLE cds (contig TEXT, start INTEGER, end INTEGER, strand TEXT, gene TEXT, feature_id TEXT)")
    db.execute("CREATE INDEX cds_location ON cds (contig, start)")
    return db


def qualify_reference(raw_fasta, raw_gff, output_dir, reference, index, members, metadata, db):
    """Copy sequence unchanged; qualify contig IDs to distinguish draft assemblies."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fasta, gff = output_dir / "assembly.fna", output_dir / "genes.gff"
    contigs = {}
    lengths = Counter()
    current = None
    with Path(raw_fasta).open() as source, fasta.open("w") as target:
        for line in source:
            if line.startswith(">"):
                original = line[1:].split()[0]
                if original in contigs:
                    raise ValueError(f"Duplicate FASTA contig: {reference}/{original}")
                # No original-ID delimiters leak into pyseer's comma/semicolon format.
                current = f"kvref{index}_ctg{len(contigs)}"
                contigs[original] = current
                target.write(f">{current}\n")
            else:
                if current is None and line.strip():
                    raise ValueError("Sequence before first FASTA header")
                lengths[current] += len(line.strip())
                target.write(line)
    for original, qualified in contigs.items():
        db.execute("INSERT INTO contigs VALUES (?, ?, ?, ?)",
                   (qualified, reference, original, lengths[qualified]))
    from urllib.parse import unquote
    mapped = total = 0
    with Path(raw_gff).open() as source, gff.open("w") as target:
        target.write("##gff-version 3\n")
        for line in source:
            if line.startswith("##FASTA"):
                break
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "CDS":
                continue
            if fields[0] not in contigs:
                raise ValueError(f"GFF contig missing from FASTA: {reference}/{fields[0]}")
            attributes = dict(item.split("=", 1) for item in fields[8].split(";") if "=" in item)
            feature = unquote(attributes.get("ID", ""))
            clusters = members.get(feature, set())
            if not clusters:
                clusters = set().union(*(members.get(feature + suffix, set())
                                         for suffix in ("_len", "_pseudo", "_len_pseudo")))
            if len(clusters) == 1:
                cluster = next(iter(clusters))
                mapped += 1
                # Retain named Bakta genes even when Panaroo's display-name field is empty.
                alias = unquote(attributes.get("gene", ""))
                if alias:
                    values = metadata[cluster]
                    names = {x.strip() for x in values.get("label", "").split(";") if x.strip() and x != cluster}
                    names.add(alias)
                    values["label"] = "; ".join(sorted(names))
            else:
                # Unresolved CDS must block false overlap/nearby assignments.
                cluster = f"__UNRESOLVED_CDS_{index}_{total}"
            if any(char in cluster for char in ";|,\t\n\r"):
                raise ValueError(f"Unsupported delimiter in cluster ID: {cluster}")
            contig = contigs[fields[0]]
            start, end = int(fields[3]), int(fields[4])
            if not 1 <= start <= end <= lengths[contig]:
                raise ValueError(f"Invalid CDS coordinates: {reference}/{feature}")
            db.execute("INSERT INTO cds VALUES (?, ?, ?, ?, ?, ?)",
                       (contig, start, end, fields[6], cluster, feature))
            fields[0] = contig
            attributes["gene"] = cluster
            fields[8] = ";".join(f"{key}={value}" for key, value in attributes.items())
            target.write("\t".join(fields) + "\n")
            total += 1
    return fasta, gff, mapped


def hit_evidence(hit, db, nearby_bp):
    fields = hit.split(";")
    if len(fields) != 4:
        raise ValueError(f"Unexpected pyseer hit: {hit}")
    match = re.fullmatch(r"(.+):(\d+)-(\d+)", fields[0])
    if not match:
        raise ValueError(f"Expected contig:start-end: {fields[0]}")
    contig, start, end = match[1], int(match[2]), int(match[3])
    reference = db.execute("SELECT reference, original_contig, length FROM contigs WHERE contig=?", (contig,)).fetchone()
    if reference is None or not 1 <= start <= end <= reference[2]:
        raise ValueError(f"Hit does not match reference geometry: {fields[0]}")
    rows = db.execute("SELECT start, end, strand, gene, feature_id FROM cds WHERE contig=? AND start<=? AND end>=?",
                      (contig, end + nearby_bp, start - nearby_bp)).fetchall()
    overlaps = [row for row in rows if row[0] <= end and row[1] >= start]
    candidates = []
    for gene_start, gene_end, strand, gene, feature_id in overlaps or rows:
        if overlaps:
            category, distance = "coding", 0
        elif gene_start > end:
            category = "upstream" if strand == "+" else "downstream" if strand == "-" else "unoriented_nearby"
            distance = gene_start - end
        else:
            category = "downstream" if strand == "+" else "upstream" if strand == "-" else "unoriented_nearby"
            distance = start - gene_end
        candidates.append(dict(gene=gene, annotation_class=category, distance_bp=distance,
                               reference=reference[0], annotation_contig=reference[1],
                               hit_start=start, hit_end=end, gene_start=gene_start,
                               gene_end=gene_end, strand=strand, feature_id=feature_id))
    return candidates


def resolve_hits(raw, db, metadata, nearby_bp):
    if not raw:
        return "UNMAPPED", [], ""
    hits = [hit_evidence(hit, db, nearby_bp) for hit in raw.split(",")]
    evidence = [row for rows in hits for row in rows]
    for row in evidence:
        row["label"] = metadata.get(row["gene"], {}).get("label", row["gene"])
    genes = {row["gene"] for row in evidence}
    # Every reported hit must agree; no nearest-gene tie breaking.
    if not evidence:
        return "INTERGENIC_OUTSIDE_WINDOW", [], ""
    if any(not rows for rows in hits) or len(genes) != 1 or not genes.issubset(metadata):
        return "AMBIGUOUS", evidence, ""
    if any(row["annotation_class"] == "unoriented_nearby" for row in evidence):
        return "AMBIGUOUS", evidence, ""
    return "RESOLVED", evidence, next(iter(genes))


def annotate_work(work, output, nearby_bp=500):
    if nearby_bp < 0:
        raise ValueError("nearby_bp must be nonnegative")
    work, output = Path(work), Path(output)
    loci = json.loads((work / "selected_loci.json").read_text())
    if len(set(loci)) != len(loci):
        raise ValueError("Duplicate selected locus IDs")
    wanted = set(loci)
    metadata_file = work / "gene_metadata.json"
    metadata = json.loads(metadata_file.read_text()) if metadata_file.exists() else {}
    raw_hits = {}
    with (work / "pyseer_hits.tsv").open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 3:
                raise ValueError("Expected pyseer sequence, locus, annotations")
            locus = int(fields[1])
            if locus in raw_hits or locus not in wanted:
                raise ValueError(f"Duplicate or unexpected pyseer locus: {locus}")
            # Empty positions (e.g. fastmap repeat-limit hits) remain unmapped.
            raw_hits[locus] = fields[2]
    counts, classes = Counter(), Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + ".partial")
    db = sqlite3.connect(f"{(work / 'gene_geometry.sqlite').resolve().as_uri()}?mode=ro", uri=True)
    columns = ["locus", "gene", "label", "product", "annotation_class", "annotation_distance_min_bp",
               "annotation_distance_max_bp", "annotation_hit_count", "annotation_evidence"]
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle, (work / "annotation_status.tsv").open("w", newline="", encoding="utf-8") as audit:
            writer, audit_writer = csv.writer(handle, delimiter="\t"), csv.writer(audit, delimiter="\t")
            writer.writerow(columns)
            audit_writer.writerow(["locus", "status", "gene", "annotation_class", "candidate_evidence", "pyseer_hits"])
            for locus in loci:
                raw = raw_hits.get(locus, "")
                status, evidence, gene = resolve_hits(raw, db, metadata, nearby_bp)
                category = "; ".join(sorted({row["annotation_class"] for row in evidence}))
                encoded = json.dumps(evidence, separators=(",", ":"))
                counts[status] += 1
                audit_writer.writerow([locus, status, gene, category, encoded, raw])
                if status == "RESOLVED":
                    distances = [row["distance_bp"] for row in evidence]
                    values = metadata[gene]
                    classes[category] += 1
                    writer.writerow([locus, gene, values.get("label", gene), values.get("product", ""),
                                     category, min(distances), max(distances), len(raw.split(",")), encoded])
        temporary.replace(output)
    finally:
        db.close()
        temporary.unlink(missing_ok=True)
    with (work / "gene_catalog.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["gene", "label", "product"])
        for gene, values in sorted(metadata.items()):
            writer.writerow([gene, values.get("label", gene), values.get("product", "")])
    summary = json.loads((work / "annotation_preparation.json").read_text())
    summary.update(annotation_status_counts=dict(counts), annotation_class_counts=dict(classes),
                   annotation_nearby_bp=nearby_bp, output_annotation=str(output),
                   gene_assignment="one known cluster across every reported hit; CDS overlap takes priority; ambiguous loci excluded",
                   coordinates="one-based inclusive, reference-specific; not used for distal-pair selection",
                   distance_definition="nearest hit/CDS endpoint separation in bp; coding overlap=0; adjacent bases=1")
    (work / "annotation_summary.json").write_text(json.dumps(summary, indent=2))
    print("Annotation status counts:", dict(counts), flush=True)
    print("Resolved annotation classes:", dict(classes), flush=True)
    print("Generated locus-to-gene mapping:", output, flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, help="Prepared pyseer annotation directory")
    parser.add_argument("--out", required=True, help="Locus-to-gene TSV")
    parser.add_argument("--nearby-bp", type=int, default=500, help="Maximum same-contig distance; 0 retains coding overlaps only [500]")
    args = parser.parse_args(argv)
    if args.nearby_bp < 0:
        parser.error("--nearby-bp must be nonnegative")
    annotate_work(args.work, args.out, args.nearby_bp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
