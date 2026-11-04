"""Portable preparation and annotation of selected DNA unitigs with Pyseer/Bakta."""
from __future__ import annotations

import argparse
from contextlib import closing
import csv
import json
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile

import pandas as pd

from .locus_annotation import annotate_work, create_geometry, qualify_reference
from .workflow_cache import cache_matches, save_cache, stage_identity


def prepare_inputs(selected, fasta, unitigs, panaroo, bakta, work):
    """Prepare exact-match queries and uniquely qualified reference coordinates.

    Selected pairs retain zero-based scanner IDs. Annotation cannot change their
    significance or physical-distance rule. CDS memberships stay on disk to avoid
    loading the full pangenome membership table into memory.
    """
    selected, fasta, unitigs, panaroo, bakta, work = map(Path,
        (selected, fasta, unitigs, panaroo, bakta, work))
    pairs = pd.read_csv(selected, sep="\t", usecols=["u", "v"])
    values = pairs[["u", "v"]].apply(pd.to_numeric, errors="raise")
    if values.isna().any().any() or (values < 0).any().any() or (values % 1 != 0).any().any():
        raise ValueError("Selected locus IDs must be nonnegative integers")
    loci = sorted(set(values.u.astype(int)) | set(values.v.astype(int)))
    wanted, found = set(loci), {}
    with unitigs.open() as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 2 or fields[0].startswith("#"):
                continue
            locus = int(fields[0])
            if locus in wanted:
                if locus in found:
                    raise ValueError(f"Duplicate unitig ID: {locus}")
                sequence = fields[1].upper()
                if set(sequence) - set("ACGT"):
                    raise ValueError(f"Invalid DNA sequence for unitig {locus}")
                found[locus] = sequence
    if missing := wanted - set(found):
        raise ValueError(f"Unitig sequences missing for loci: {sorted(missing)[:10]}")
    work.mkdir(parents=True, exist_ok=True)
    with (work / "queries.tsv").open("w") as handle:
        handle.write("sequence\tlocus\n")
        for locus in loci:
            handle.write(f"{found[locus]}\t{locus}\n")
    (work / "selected_loci.json").write_text(json.dumps(loci))
    with fasta.open() as handle:
        samples = [line[1:].split()[0] for line in handle if line.startswith(">")]
    if not samples or len(samples) != len(set(samples)):
        raise ValueError("Scanner FASTA must have unique sample names")
    metadata, reference_count = {}, 0
    with closing(sqlite3.connect(work / "panaroo_membership.sqlite")) as membership, \
            closing(create_geometry(work / "gene_geometry.sqlite")) as geometry, \
            (work / "references.txt").open("w") as references:
        membership.execute("CREATE TABLE membership (sample TEXT, feature TEXT, cluster TEXT)")
        pending = []
        with panaroo.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if "Gene" not in (reader.fieldnames or []):
                raise ValueError("Panaroo table lacks the Gene column")
            sample_set = set(samples)
            sample_columns = [name for name in reader.fieldnames if name in sample_set]
            for row in reader:
                cluster = row["Gene"]
                metadata[cluster] = dict(label=row.get("Non-unique Gene name", "") or cluster,
                                         product=row.get("Annotation", ""))
                for sample in sample_columns:
                    for feature in re.split(r"[;\t]", row.get(sample, "") or ""):
                        if feature:
                            pending.append((sample, feature, cluster))
                            if len(pending) >= 50000:
                                membership.executemany("INSERT INTO membership VALUES (?, ?, ?)", pending)
                                pending.clear()
        if pending:
            membership.executemany("INSERT INTO membership VALUES (?, ?, ?)", pending)
        membership.execute("CREATE INDEX membership_sample ON membership (sample)")
        membership.commit()
        # Match the original workflow: scan FASTA order, first matching draft assembly.
        if loci:
            for index, sample in enumerate(samples):
                raw_gff, raw_fasta = bakta / sample / f"{sample}.gff3", bakta / sample / f"{sample}.fna"
                if not raw_gff.is_file() or not raw_fasta.is_file():
                    continue
                members = {}
                for feature, cluster in membership.execute(
                        "SELECT feature, cluster FROM membership WHERE sample = ?", (sample,)):
                    members.setdefault(feature, set()).add(cluster)
                linked_fasta, mapped_gff, count = qualify_reference(raw_fasta, raw_gff,
                    work / "references" / str(index), sample, index, members, metadata, geometry)
                if not count:
                    continue
                if any(c.isspace() for c in str(linked_fasta) + str(mapped_gff)):
                    raise ValueError("Pyseer reference paths must not contain whitespace")
                references.write(f"{linked_fasta.resolve()}\t{mapped_gff.resolve()}\tdraft\n")
                reference_count += 1
            if reference_count == 0:
                raise ValueError("No Bakta CDS IDs matched Panaroo clusters for scanner samples")
        geometry.commit()
    (work / "gene_metadata.json").write_text(json.dumps(metadata))
    details = dict(selected_input=str(selected.resolve()), selected_pairs=len(pairs),
        selected_loci=len(loci), reference_count=reference_count,
        unitigs=str(unitigs.resolve()), panaroo_clusters=str(panaroo.resolve()),
        mapping="pyseer draft/full-length exact match; first matching assembly",
        gene_assignment="CDS overlap or bounded strand-aware nearby region; all reported hits must resolve to one cluster")
    (work / "annotation_preparation.json").write_text(json.dumps(details, indent=2))
    print(f"[locus annotation] prepared {len(loci):,} loci and {reference_count:,} references", flush=True)
    return details


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", required=True, type=Path, help="Selected-pair TSV from ko-variation-select")
    parser.add_argument("--fasta", required=True, type=Path, help="Original scanner binary FASTA; supplies sample order")
    parser.add_argument("--unitigs", required=True, type=Path, help="Original DNA table: zero-based locus ID, sequence")
    parser.add_argument("--panaroo", required=True, type=Path, help="gene_presence_absence.csv")
    parser.add_argument("--bakta", required=True, type=Path, help="Directory with SAMPLE/SAMPLE.gff3 and SAMPLE.fna")
    parser.add_argument("--out", required=True, type=Path, help="Separate annotation output directory")
    parser.add_argument("--nearby-bp", type=int, default=500)
    parser.add_argument("--pyseer-python", default=sys.executable, help="Python environment with Pyseer installed")
    parser.add_argument("--resume", action="store_true", help="Reuse an identical completed annotation stage")
    args = parser.parse_args(argv)
    if args.nearby_bp < 0:
        parser.error("nearby-bp must be nonnegative")
    args.out = args.out.resolve()
    inputs = [args.selected, args.fasta, args.unitigs, args.panaroo]
    if any(path.resolve().is_relative_to(args.out) for path in inputs) or args.out.is_relative_to(args.bakta.resolve()):
        parser.error("Annotation output must be separate from source inputs and Bakta directory")
    try:
        args.out.mkdir(parents=True, exist_ok=True)
        outputs = [args.out / name for name in ("locus_to_gene.tsv", "annotation_status.tsv",
                    "gene_catalog.tsv", "annotation_summary.json")]
        identity = stage_identity("annotation", inputs + [Path(__file__),
            Path(__file__).with_name("locus_annotation.py")],
            dict(nearby_bp=args.nearby_bp, pyseer_python=args.pyseer_python), [args.bakta])
        manifest = args.out / "annotation.cache.json"
        if args.resume and cache_matches(manifest, identity, outputs):
            print(f"[locus annotation] reusing completed annotation: {outputs[0]}", flush=True)
            return 0
        # Separate scratch cwd avoids Pyseer's fixed temporary-file collisions.
        with tempfile.TemporaryDirectory(prefix="work-", dir=args.out) as folder:
            work = Path(folder)
            details = prepare_inputs(args.selected, args.fasta, args.unitigs, args.panaroo, args.bakta, work)
            if details["selected_loci"]:
                for executable in ("bwa", "bedtools", "gff2bed"):
                    if shutil.which(executable) is None:
                        raise ValueError(f"Missing annotation dependency on PATH: {executable}")
                print("[locus annotation] mapping exact unitig sequences with Pyseer", flush=True)
                with (args.out / "pyseer.log").open("w") as log:
                    subprocess.run([args.pyseer_python, "-m", "pyseer.kmer_mapping.annotate_hits",
                        str(work / "queries.tsv"), str(work / "references.txt"),
                        str(work / "pyseer_hits.tsv"), "--tmp-prefix", str(work)],
                        cwd=work, stdout=log, stderr=subprocess.STDOUT, check=True)
            else:
                (work / "pyseer_hits.tsv").touch()
            annotate_work(work, outputs[0], args.nearby_bp)
            for name in ("annotation_status.tsv", "gene_catalog.tsv", "annotation_summary.json"):
                shutil.copyfile(work / name, args.out / name)
        save_cache(manifest, identity, outputs)
    except (ValueError, OSError, sqlite3.Error, subprocess.CalledProcessError) as exc:
        sys.stderr.write(f"Locus annotation error: {exc}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
