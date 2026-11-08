"""Portable preparation and annotation of selected DNA unitigs with Pyseer/Bakta."""
from __future__ import annotations

import argparse
from contextlib import closing
import csv
import json
import os
from pathlib import Path
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile

import pandas as pd

from .locus_annotation import annotate_work, create_geometry, qualify_reference
from .workflow_cache import cache_matches, save_cache, stage_identity


def annotation_environment(python, smoke_test=False):
    """Use one environment for Pyseer and its native mapping executables.

    Probe all dependencies before preparing hundreds of reference assemblies.
    The optional tiny mapping exercises actual BWA/BEDOPS/BEDTools and Pyseer;
    it does not use the biological data or alter any cached annotation stage.
    """
    executable = shutil.which(str(python))
    if not executable:
        raise ValueError(f'Annotation Python not found: {python}')
    executable = str(Path(executable).absolute())
    environment = dict(os.environ)
    environment['PATH'] = str(Path(executable).parent)+os.pathsep+environment.get('PATH','')
    environment.pop('PYTHONPATH',None)
    tools = {name:shutil.which(name,path=environment['PATH']) for name in ('bwa','bedtools','gff2bed')}
    errors = [f'Missing executable: {name}' for name,path in tools.items() if path is None]
    code = ('import json,sys; import pyseer.kmer_mapping.annotate_hits; import pyseer,pybedtools; '
            'print(json.dumps(dict(python=sys.executable,pyseer=getattr(pyseer,"__version__","unknown"),'
            'pybedtools=getattr(pybedtools,"__version__","unknown"))))')
    probe = subprocess.run([executable,'-c',code],env=environment,capture_output=True,text=True,timeout=60)
    packages = {}
    if probe.returncode:
        errors.append('Pyseer annotation import failed: '+probe.stderr.strip()[-3000:])
    else:
        packages = json.loads(probe.stdout.strip().splitlines()[-1])
    if errors:
        raise ValueError('Annotation environment incomplete ('+executable+'):\n  '+
                         '\n  '.join(errors)+'\nInstall pyseer, bwa, bedtools and bedops in the annotation environment.')
    report = dict(python=executable,packages=packages,executables=tools,smoke_test='NOT_RUN')
    if smoke_test:
        with tempfile.TemporaryDirectory(prefix='kovar-mapping-check-') as temporary:
            work = Path(temporary).resolve()
            if any(c.isspace() for c in str(work)):
                raise ValueError('Mapping smoke test needs a TMPDIR without whitespace (Pyseer reference-file constraint)')
            rng = random.Random(73)
            dna = ''.join(rng.choice('ACGT') for _ in range(500))
            query = dna[100:161]
            fasta,gff,queries,references,hits = [work/name for name in
                ('reference.fna','reference.gff','queries.tsv','references.tsv','hits.tsv')]
            fasta.write_text('>smoke\n'+dna+'\n')
            gff.write_text('##gff-version 3\nsmoke\tcheck\tCDS\t1\t500\t.\t+\t0\tID=smoke_cds;gene=smoke_gene\n')
            queries.write_text('sequence\tlocus\n'+query+'\t0\n')
            references.write_text(f'{fasta}\t{gff}\tdraft\n')
            run = subprocess.run([executable,'-m','pyseer.kmer_mapping.annotate_hits',
                str(queries),str(references),str(hits),'--tmp-prefix',str(work)],
                cwd=work,env=environment,capture_output=True,text=True,timeout=120)
            if run.returncode or not hits.is_file():
                raise ValueError('Real-tool mapping smoke test failed:\n'+run.stderr[-6000:])
            rows = [line.split('\t') for line in hits.read_text().splitlines()]
            if (len(rows)!=1 or len(rows[0])!=3 or rows[0][:2]!=[query,'0'] or
                    'smoke:101-161;' not in rows[0][2] or 'smoke_gene' not in rows[0][2]):
                raise ValueError('Mapping smoke test produced unexpected coordinates/annotation: '+hits.read_text()[:2000])
        report['smoke_test'] = 'PASSED'
    return environment,report


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
        # Native executables must come from the selected Pyseer environment,
        # and missing dependencies must fail before reference preparation.
        annotation_env, environment_report = None, dict(mapping_required=False)
        if len(pd.read_csv(args.selected,sep='\t',usecols=['u','v'],nrows=1)):
            annotation_env, environment_report = annotation_environment(args.pyseer_python)
            print('[locus annotation] environment:',json.dumps(environment_report),flush=True)
        # Separate scratch cwd avoids Pyseer's fixed temporary-file collisions.
        with tempfile.TemporaryDirectory(prefix="work-", dir=args.out) as folder:
            work = Path(folder)
            details = prepare_inputs(args.selected, args.fasta, args.unitigs, args.panaroo, args.bakta, work)
            if details["selected_loci"]:
                print("[locus annotation] mapping exact unitig sequences with Pyseer", flush=True)
                print('[locus annotation] Pyseer log:',args.out/'pyseer.log',flush=True)
                with (args.out / "pyseer.log").open("w") as log:
                    subprocess.run([environment_report['python'], "-m", "pyseer.kmer_mapping.annotate_hits",
                        str(work / "queries.tsv"), str(work / "references.txt"),
                        str(work / "pyseer_hits.tsv"), "--tmp-prefix", str(work)],
                        cwd=work, env=annotation_env, stdout=log, stderr=subprocess.STDOUT, check=True)
            else:
                (work / "pyseer_hits.tsv").touch()
            annotate_work(work, outputs[0], args.nearby_bp)
            summary = json.loads((work/'annotation_summary.json').read_text())
            summary['annotation_environment'] = environment_report
            (work/'annotation_summary.json').write_text(json.dumps(summary,indent=2))
            for name in ("annotation_status.tsv", "gene_catalog.tsv", "annotation_summary.json"):
                shutil.copyfile(work / name, args.out / name)
        save_cache(manifest, identity, outputs)
    except (ValueError, OSError, sqlite3.Error, subprocess.SubprocessError) as exc:
        sys.stderr.write(f"Locus annotation error: {exc}\n")
        log = args.out/'pyseer.log'
        if isinstance(exc,subprocess.CalledProcessError) and log.is_file():
            with log.open('rb') as handle:
                handle.seek(0,2)
                handle.seek(max(0,handle.tell()-65536))
                tail = handle.read().decode('utf-8',errors='replace').splitlines()[-40:]
            sys.stderr.write(f'Last Pyseer log lines ({log}):\n'+'\n'.join(tail)+'\n')
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
