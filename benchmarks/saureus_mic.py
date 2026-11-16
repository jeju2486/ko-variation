"""Frozen-map S. aureus allele audit and lineage-held-out MIC prediction.

Research helper, not part of discovery. AMRFinderPlus defines published marker
states; an exact unitig match plus reference-protein alignment establishes site
overlap. Prediction uses censored Gaussian ridge with training-derived tree PCs,
not the binary-response KOVAR test. No bound is imputed as an exact MIC.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr

from ko_variation.amr_report import gene_key, label_tokens, mic_record, read_tsv, write_tsv
from ko_variation.phenotype import json_write, npz_write, sha256

REVISION = "saureus-mic-1"
# The initial 2eb840d helper hashed the entire script for matrix preparation.
# Its preparation algorithm is unchanged; accept its LF/CRLF identities only
# when every input hash, cohort setting and preparation revision still match.
LEGACY_PREPARATION_CODES = {
    "252043a3a00e3f0556c0c36446f0f9e8817c2bd554e613ec925c8c048d051fd3",
    "9c0e1ad44fd5d7720597f418cac5ee6a682702067bfcb8cbf3263931fe5059ea",
}
ALIASES = {"blaZ": {"blaz", "blapc1"}, "gyrA": {"gyra"}, "parC": {"parc", "grla"}}
TARGETS = {"penicillin": ["blaZ"], "fluoroquinolone": ["gyrA", "parC"]}


def log(message):
    print(f"[S. aureus MIC] {message}", flush=True)


def canonical(symbol):
    key = gene_key(symbol.split("_")[0])
    if key in {"meca","mecc"}: return {"meca":"mecA","mecc":"mecC"}[key]
    return next((g for g, aliases in ALIASES.items() if key in aliases), None)


def records(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream, delimiter="\t")


def load_map(path):
    text = Path(path).read_text(encoding="utf-8")
    match = re.search(r'<script id="data" type="application/json">(.*?)</script>', text, re.S)
    data = json.loads(match[1] if match else text)
    nodes = {n["gene"]: n for n in data["nodes"]}
    groups = {g: set() for g in ALIASES}
    adjacency = defaultdict(set)
    for name, node in nodes.items():
        tokens = label_tokens(name) | label_tokens(node.get("label", ""))
        for gene, aliases in ALIASES.items():
            if tokens & aliases:
                groups[gene].add(name)
    for edge in data["edges"]:
        a, b = edge["gene_a"], edge["gene_b"]
        adjacency[a].add(b)
        adjacency[b].add(a)
    missing = [g for g, names in groups.items() if not names]
    if missing:
        raise ValueError(f"No named map nodes for {missing}; annotation/allele calls cannot be inferred")
    rows = []
    for gene, names in groups.items():
        for edge in data["edges"]:
            for target, other in [(edge["gene_a"], edge["gene_b"]),
                                  (edge["gene_b"], edge["gene_a"])]:
                if target in names:
                    rows.append(dict(target=gene, target_cluster=target, neighbour_cluster=other,
                        neighbour_label=nodes[other].get("label", other),
                        supporting_pairs=edge.get("n_locus_pairs", ""),
                        representative_p=edge.get("representative_significance", ""),
                        representative_beta=edge.get("representative_adjusted_beta", ""),
                        representative_distance_bp=edge.get("representative_physical_distance", ""),
                        representative_u=edge.get("representative_u", ""),
                        representative_v=edge.get("representative_v", "")))
    return data, nodes, groups, adjacency, rows


def map_report(args):
    data, nodes, groups, adjacency, rows = load_map(args.network)
    args.out.mkdir(parents=True, exist_ok=True)
    write_tsv(args.out / "neighbours.tsv", rows, list(rows[0]))
    json_write(args.out / "map_summary.json", dict(nodes=len(nodes), edges=len(data["edges"]),
        targets={g: dict(clusters=sorted(names), direct_neighbours=sorted(set().union(
            *(adjacency[n] for n in names))-names)) for g, names in groups.items()},
        limitation="Gene-edge association and beta are not MIC effects; site identity needs DNA evidence."))
    focal_plots(data, nodes, groups, adjacency, args.out)
    return data, nodes, groups, adjacency


def focal_plots(data, nodes, groups, adjacency, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import textwrap
    plt.rcParams["svg.fonttype"] = "none"
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, genes in zip(axes, TARGETS.values()):
        anchors = set().union(*(groups[g] for g in genes))
        others = sorted(set().union(*(adjacency[g] for g in anchors))-anchors)
        pos = {g: (0., (i-(len(anchors)-1)/2)*.6) for i, g in enumerate(sorted(anchors))}
        for angle, g in zip(np.linspace(0, 2*np.pi, len(others), endpoint=False), others):
            pos[g] = (2*np.cos(angle), 2*np.sin(angle))
        for edge in data["edges"]:
            a, b = edge["gene_a"], edge["gene_b"]
            if a in pos and b in pos and (a in anchors or b in anchors):
                x, y = zip(pos[a], pos[b])
                ax.plot(x, y, color="black" if a in anchors and b in anchors else ".7",
                        linewidth=2. if a in anchors and b in anchors else .8,
                        linestyle="--" if edge.get("adjusted_direction") == "negative" else "-", zorder=1)
        for gene, (x, y) in pos.items():
            label = nodes[gene].get("label", gene)
            unknown = bool(re.fullmatch(r"group_\d+", label))
            ax.scatter([x], [y], s=140 if gene in anchors else 45 if not unknown else 15,
                       color="black" if gene in anchors else ".45" if not unknown else ".75", zorder=2)
            # Radial labels and a fixed ring make the small focal panel legible.
            offset = 0 if gene in anchors else .18
            tx, ty = x*(1+offset), y*(1+offset)
            if gene in anchors: ty += .18
            if gene in anchors and len(anchors)>1: tx,ty=x-.2,y
            if unknown: continue
            ax.text(tx, ty, textwrap.fill(label, 20), fontsize=10 if gene in anchors else 8,
                    ha="right" if gene in anchors and len(anchors)>1 else "center" if gene in anchors else "left" if x>.1 else "right" if x<-.1 else "center",
                    va="center", bbox=dict(facecolor="white", edgecolor="none", alpha=.8, pad=1))
        ax.set(xlim=(-3.5, 3.5), ylim=(-3.1, 3.1), aspect="equal", title=" + ".join(genes))
        ax.axis("off")
    fig.text(.5, .025, "Direct neighbours only; small pale nodes = unannotated. Dashed = negative association. Positions are diagrammatic.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    for suffix in ["png", "svg"]: fig.savefig(out/f"focal_neighbours.{suffix}", dpi=300)
    plt.close(fig)


def fasta_records(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        name, chunks = None, []
        for line in stream:
            if line.startswith(">"):
                if name is not None: yield name, "".join(chunks).upper()
                name, chunks = line[1:].split()[0], []
            elif line.strip(): chunks.append(line.strip())
        if name is not None: yield name, "".join(chunks).upper()


def metadata_samples(path, fasta_names, mapping=None):
    rows = read_tsv(path, ["id", "isolate", "species"])
    explicit = {r["metadata_id"]: r["sample"] for r in read_tsv(mapping)} if mapping else None
    result = {}
    for row in rows:
        if row["species"].strip().casefold() != "staphylococcus aureus": continue
        candidates = [explicit.get(row["id"])] if explicit is not None else [
            row["id"]+"_"+row["isolate"].replace(".", "_"), row["id"], row["isolate"]]
        matches = set(candidates) & fasta_names
        if len(matches)>1: raise ValueError(f"Ambiguous metadata join for {row['id']}; use --sample-map")
        if matches:
            sample = matches.pop()
            if sample in result: raise ValueError(f"Duplicate metadata sample: {sample}")
            result[sample] = row
    if not result: raise ValueError("No S. aureus metadata/sample overlap; use --sample-map")
    return result


def preparation_code():
    """Database discovery and plotting edits must not invalidate the input matrix."""
    source="\n".join(inspect.getsource(fn).replace("\r\n","\n") for fn in
                     [load_map,fasta_records,metadata_samples,prepare,gene_key,label_tokens,read_tsv])
    source+=json.dumps(dict(aliases={g:sorted(a) for g,a in ALIASES.items()},targets=TARGETS),sort_keys=True)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def compatible_preparation(saved, current):
    if saved==current: return True
    return (saved.get("code") in LEGACY_PREPARATION_CODES and
            {k:v for k,v in saved.items() if k!="code"}==
            {k:v for k,v in current.items() if k!="code"})


def prepare(args, mapped):
    from Bio import Phylo
    data, nodes, groups, adjacency = mapped
    sources = {key: dict(path=str(getattr(args, key).resolve()), sha256=sha256(getattr(args, key)))
               for key in ["network", "annotation", "fasta", "unitigs", "tree", "metadata"]}
    for key in ["sample_map", "clusters", "locus_map"]:
        if getattr(args, key): sources[key] = dict(path=str(getattr(args, key).resolve()), sha256=sha256(getattr(args, key)))
    identity = dict(revision=REVISION, code=preparation_code(), sources=sources, groups=args.groups)
    state = args.out / "prepared.json"
    if state.exists():
        info = json.loads(state.read_text())
        if not compatible_preparation(info["identity"],identity): raise ValueError("Prepared inputs/settings changed; choose a new --out")
        log("reuse prepared selected-column matrix")
        return info
    annotation = {}
    for r in records(args.annotation):
        if "coding" in {x.strip() for x in r.get("annotation_class", "").split(";")} and r["gene"] in nodes:
            locus = int(r["locus"])
            if locus in annotation: raise ValueError(f"Duplicate coding annotation: {locus}")
            annotation[locus] = r["gene"]
    loci = sorted(annotation)
    if not loci: raise ValueError("No coding map loci in --annotation")
    index = {u: j for j, u in enumerate(loci)}
    target_columns, neighbour_columns, excluded = {}, {}, set()
    for name, genes in TARGETS.items():
        anchors = set().union(*(groups[g] for g in genes))
        first = set().union(*(adjacency[g] for g in anchors))-anchors
        second = set().union(*(adjacency[g] for g in first))-first-anchors
        target_columns[name] = [index[u] for u in loci if annotation[u] in anchors]
        neighbour_columns[name] = [index[u] for u in loci if annotation[u] in first]
        excluded.update(anchors | first | second)
        if not target_columns[name] or not neighbour_columns[name]:
            raise ValueError(f"No coding target/neighbour features for {name}")
    pool = [index[u] for u in loci if annotation[u] not in excluded]
    if not pool: raise ValueError("No coding random-control pool outside both two-hop neighbourhoods")
    columns = np.asarray(loci)
    if args.locus_map:
        mappings=read_tsv(args.locus_map, ["locus", "fasta_column"])
        lm = {int(r["locus"]): int(r["fasta_column"]) for r in mappings}
        if len(lm)!=len(mappings) or len(set(lm.values()))!=len(lm): raise ValueError("Locus/FASTA columns must be one-to-one")
        columns = np.array([lm[u] for u in loci])
    elif not args.fasta_columns_are_locus_ids:
        raise ValueError("Confirm --fasta-columns-are-locus-ids or provide --locus-map")
    samples, matrix, length = [], [], None
    for sample, sequence in fasta_records(args.fasta):
        if not sequence or set(sequence)-set("AC"): raise ValueError(f"Not A/C pseudo-FASTA: {sample}")
        if length is None: length = len(sequence)
        if len(sequence)!=length or max(columns)>=length or min(columns)<0:
            raise ValueError(f"FASTA length/locus-column mismatch: {sample}")
        samples.append(sample)
        matrix.append((np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)[columns]==ord("C")).astype(np.uint8))
    if not samples or len(samples)!=len(set(samples)): raise ValueError("FASTA needs unique sample IDs, not repeated contig headers")
    meta = metadata_samples(args.metadata, set(samples), args.sample_map)
    tree = Phylo.read(args.tree, "newick")
    tips = [t.name for t in tree.get_terminals()]
    if len(tips)!=len(set(tips)): raise ValueError("Duplicate tree tips")
    retained = [s for s in samples if s in meta and s in set(tips)]
    omitted = [dict(sample=s, reason="missing_tree" if s not in tips else "not_species_matched_metadata")
               for s in samples if s not in retained]
    if len(retained)<20: raise ValueError("Fewer than 20 species/metadata/tree-matched samples")
    X = np.stack(matrix)[[samples.index(s) for s in retained]]
    for tip in list(tree.get_terminals()):
        if tip.name not in set(retained): tree.prune(tip)
    for clade in tree.find_clades():
        if clade is not tree.root and (clade.branch_length is None or not np.isfinite(clade.branch_length) or clade.branch_length<0):
            raise ValueError("Tree needs finite nonnegative branch lengths")
    tree.root_at_midpoint()
    tree.root.branch_length = 0.
    paths = {t.name: [tree.root]+tree.get_path(t) for t in tree.get_terminals()}
    depths = tree.depths()
    K = np.zeros((len(retained), len(retained)))
    for i, a in enumerate(retained):
        for j, b in enumerate(retained[:i+1]):
            common = tree.root
            for u, v in zip(paths[a], paths[b]):
                if u is not v: break
                common = u
            K[i,j] = K[j,i] = depths[common]
    scale = np.trace(K)/len(K)
    if not np.isfinite(scale) or scale<=0: raise ValueError("No positive tree covariance scale")
    K /= scale
    if args.clusters:
        cluster_rows=read_tsv(args.clusters, ["sample", "cluster"])
        lookup = {r["sample"]: r["cluster"] for r in cluster_rows}
        if len(lookup)!=len(cluster_rows) or any(s not in lookup or not lookup[s] for s in retained):
            raise ValueError("Cluster labels must uniquely cover every retained sample")
        labels = [lookup[s] for s in retained]
        group_method = "supplied labels (e.g. fastBAPS)"
    else:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
        distance = np.maximum(0, K.diagonal()[:,None]+K.diagonal()[None,:]-2*K)
        labels = fcluster(linkage(squareform(distance, checks=False), method="complete"),
                          min(args.groups, len(retained)), criterion="maxclust").astype(str).tolist()
        group_method = "explicit complete-linkage patristic partition; not fastBAPS"
    target_loci = {loci[j] for cols in target_columns.values() for j in cols}
    dna = {}
    with args.unitigs.open() as stream:
        for line in stream:
            fields = line.split()
            if len(fields)>=2 and fields[0].isdigit() and int(fields[0]) in target_loci:
                if int(fields[0]) in dna: raise ValueError("Duplicate target DNA unitig ID")
                dna[int(fields[0])] = fields[1].upper()
    if set(dna)!=target_loci: raise ValueError("Some target unitig IDs have no DNA in --unitigs")
    info = dict(identity=identity, samples=retained, groups=labels, group_method=group_method,
        loci=loci, locus_gene={str(u): annotation[u] for u in loci}, target_columns=target_columns,
        neighbour_columns=neighbour_columns, random_pool=pool, target_dna={str(k):v for k,v in dna.items()},
        target_gene={str(u): next(g for g in groups if annotation[u] in groups[g]) for u in target_loci},
        metadata=[meta[s] for s in retained])
    npz_write(args.out/"prepared.npz", X=X, K=K)
    write_tsv(args.out/"features.tsv", [dict(locus=u,gene=annotation[u],
        label=nodes[annotation[u]].get("label",annotation[u]),
        role=";".join(name+":"+("target" if index[u] in target_columns[name] else "neighbour")
                      for name in TARGETS if index[u] in target_columns[name]+neighbour_columns[name]) or "random_pool")
        for u in loci], ["locus","gene","label","role"])
    write_tsv(args.out/"excluded_samples.tsv", omitted, ["sample", "reason"])
    write_tsv(args.out/"sample_groups.tsv", [dict(sample=s, cluster=g) for s,g in zip(retained, labels)], ["sample", "cluster"])
    Phylo.write(tree, args.out/"analysis.treefile", "newick")
    json_write(state, info)
    log(f"prepared {len(retained)} isolates; {X.shape[1]} selected coding columns; {len(set(labels))} groups")
    return info


def source_index(root, suffixes, wanted):
    found = {}
    if root is None: return found
    for path in root.rglob("*"):
        for suffix in suffixes:
            if path.name.endswith(suffix) and path.is_file():
                name = path.name[:-len(suffix)]
                if name not in wanted: break
                if name in found: raise ValueError(f"Multiple files for sample {name} under {root}")
                found[name] = path
                break
    return found


def genome_dna(path):
    if re.search(r"\.gff3?(\.gz)?$", str(path)):
        opener = gzip.open if str(path).endswith(".gz") else open
        from Bio import SeqIO
        with opener(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                if line.strip()=="##FASTA":
                    return {r.id: str(r.seq).upper() for r in SeqIO.parse(stream, "fasta")}
        raise ValueError(f"GFF has no embedded ##FASTA: {path}; supply --assemblies")
    return dict(fasta_records(path))


def sample_directory(root, sample):
    # Keep normal isolate names readable; unsafe names receive an unambiguous suffix.
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", sample)
    if safe != sample: safe += "_"+hashlib.sha256(sample.encode()).hexdigest()[:10]
    return root/safe


def call_amrfinder(job):
    sample, source, root, executable, database, version = job
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    out = sample_directory(root, sample)
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(sample=sample, source_sha256=sha256(source), amrfinder=version)
    done = out/"done.json"
    report, mutations = out/"amr.tsv", out/"mutations.tsv"
    if done.exists():
        if json.loads(done.read_text())!=identity: raise ValueError(f"AMRFinder inputs/database changed: {sample}; choose a new --out")
        if report.exists() and mutations.exists(): return sample
    dna = genome_dna(source)
    if not dna: raise ValueError(f"No assembly DNA for {sample}")
    temporary = out/"input.fna"
    with temporary.open("w") as stream:
        for name, sequence in dna.items(): stream.write(f">{name}\n{sequence}\n")
    command = [executable, "-n", str(temporary), "-O", "Staphylococcus_aureus", "--threads", "1",
               "--mutation_all", str(out/"mutations.partial.tsv"), "-o", str(out/"amr.partial.tsv")]
    if database: command += ["--database", str(database)]
    try:
        with (out/"amrfinder.log").open("w") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        for filename in ["amr", "mutations"]:
            partial = out/f"{filename}.partial.tsv"
            if not partial.exists(): raise ValueError(f"AMRFinder did not create {partial}")
            partial.replace(out/f"{filename}.tsv")
        json_write(done, identity)
    finally:
        temporary.unlink(missing_ok=True)
    return sample


def point_symbol(symbol):
    match = re.fullmatch(r"(.+)_([A-Z*])(\d+)([A-Z*])", symbol)
    if match and canonical(match[1]) in {"gyrA", "parC"}:
        return canonical(match[1]), match[2], int(match[3]), match[4]
    return None


def known_matrix(samples, main, calls, min_coverage):
    symbols = sorted({r["Gene symbol"] for rs in main.values() for r in rs
                      if r.get("Element subtype")=="POINT" and point_symbol(r["Gene symbol"])})
    carriage = ["blaZ","mecA","mecC"]
    names = [g+"_presence" for g in carriage]+symbols
    G = np.full((len(samples),len(names)), np.nan)
    for i,sample in enumerate(samples):
        for j,gene in enumerate(carriage):
            hits = [r for r in main[sample] if canonical(r.get("Gene symbol", ""))==gene]
            intact = [r for r in hits if float(r.get("% Coverage of reference sequence",0))>=min_coverage
                      and not re.search("PARTIAL|INTERNAL_STOP|FRAME", r.get("Method", ""))]
            G[i,j] = 1. if intact else np.nan if hits else 0.
        by_site = defaultdict(list)
        for r in calls[sample]:
            parsed = point_symbol(r.get("Gene symbol", ""))
            if not parsed: continue
            gene,wild,position,allele = parsed
            by_site[(gene,wild,position)].append(allele)
        for j,name in enumerate(symbols,len(carriage)):
            gene,wild,position,allele = point_symbol(name)
            observed = by_site.get((gene,wild,position), [])
            if len(observed)==1: G[i,j] = float(observed[0]==allele)
    return G,names


def reference_mapping(reference, query):
    """Map reference AA offsets to query offsets; ambiguous mappings stay missing."""
    from Bio.Align import PairwiseAligner
    from itertools import islice
    aligner = PairwiseAligner(mode="global", match_score=2., mismatch_score=-2.,
                              open_gap_score=-6., extend_gap_score=-.5)
    maps = []
    for alignment in islice(aligner.align(reference, query), 2):
        mapped = {}
        for (a,b), (c,d) in zip(*alignment.aligned):
            mapped.update(zip(range(a,b), range(c,d)))
        maps.append(mapped)
    if not maps: return {}
    return {a:b for a,b in maps[0].items() if all(m.get(a)==b for m in maps[1:])}


def audit_sample(job):
    sample, source, all_rows, target_dna, target_gene, reference = job
    from Bio.Seq import Seq
    dna = genome_dna(source)
    alignments, matches, rows = {}, {}, []
    for call in all_rows:
        symbol = point_symbol(call.get("Gene symbol", ""))
        if not symbol: continue
        gene, wild, position, observed = symbol
        contig, start, stop, strand = (call["Contig id"], int(call["Start"])-1,
                                      int(call["Stop"]), call["Strand"])
        accession = call["Accession of closest sequence"]
        key = contig, start, stop, strand, accession
        if key not in alignments:
            seq = dna.get(contig, "")[start:stop]
            seq = str(Seq(seq).reverse_complement()) if strand=="-" else seq
            protein = str(Seq(seq[:len(seq)//3*3]).translate(table=11))
            ref = reference.get(accession)
            alignments[key] = (protein, reference_mapping(ref, protein) if ref else {}, ref)
        protein, mapping, ref = alignments[key]
        offset = mapping.get(position-1)
        resolved = (ref is not None and offset is not None and position<=len(ref)
                    and ref[position-1]==wild and protein[offset]==observed)
        codon_start = (start+3*offset if strand=="+" else stop-3*(offset+1)) if resolved else None
        for locus, sequence in target_dna.items():
            if target_gene[locus]!=gene: continue
            local_key = locus, contig, start, stop
            if local_key not in matches:
                left, right = max(0,start-len(sequence)), min(len(dna.get(contig,"")),stop+len(sequence))
                window = dna.get(contig, "")[left:right]
                hits = set()
                for query in {sequence, str(Seq(sequence).reverse_complement())}:
                    offset_q = window.find(query)
                    while offset_q>=0:
                        hits.add((left+offset_q, left+offset_q+len(query)))
                        offset_q = window.find(query, offset_q+1)
                matches[local_key] = hits
            hits = matches[local_key]
            rows.append(dict(sample=sample, locus=int(locus), gene=gene,
                marker=call["Gene symbol"], site=f"{gene}_{wild}{position}",
                contig=contig, reference_accession=accession,
                local_exact_matches=len(hits), coordinate_resolved=int(resolved),
                covers_known_codon=int(resolved and any(a<=codon_start and b>=codon_start+3 for a,b in hits)),
                codon_start_1based=codon_start+1 if resolved else "",
                observed_amino_acid=observed))
    return rows


def database_from_version(text):
    """AMRFinder -V prints a shell-quoted path; normal runs may colour it."""
    text=re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]","",text)
    match=re.search(r"^\s*Database directory:\s*(.+?)\s*$",text,re.I|re.M)
    if not match: return None
    value=match[1].strip()
    if value.startswith(("'",'"')):
        try:
            parts=shlex.split(value)
        except ValueError:
            return None
        if len(parts)!=1: return None
        value=parts[0]
    return Path(value).expanduser() if value else None


def allele_audit(args, info):
    assemblies = source_index(args.assemblies, [".fasta.gz", ".fna.gz", ".fa.gz", ".fasta", ".fna", ".fa"], set(info["samples"]))
    gffs = source_index(args.gff_dir, [".gff3.gz", ".gff.gz", ".gff3", ".gff"], set(info["samples"]))
    sources = {s: assemblies.get(s, gffs.get(s)) for s in info["samples"]}
    missing = [s for s,p in sources.items() if p is None]
    if missing: raise ValueError(f"No exact-name assembly/embedded-FASTA GFF for {len(missing)} samples, e.g. {missing[:5]}")
    executable = shutil.which(args.amrfinder)
    if not executable: raise ValueError("AMRFinderPlus not found; set --amrfinder or install ncbi-amrfinderplus")
    command = [executable, "-V"]
    if args.amrfinder_db: command += ["--database", str(args.amrfinder_db)]
    version = subprocess.run(command, capture_output=True, text=True, check=True)
    version_text = version.stdout+version.stderr
    (args.out/"amrfinder_version.txt").write_text(version_text)
    database = args.amrfinder_db or database_from_version(version_text)
    reference_file = args.reference_proteins or (database/"AMRProt.fa" if database else None)
    if reference_file is None or not reference_file.is_file():
        raise ValueError(f"AMRProt.fa not found at {reference_file or 'an undetected database directory'}. "
                         f"Database output saved to {args.out/'amrfinder_version.txt'}. "
                         "Set AMRFINDER_DB=/actual/database/directory or --reference-proteins.")
    audit_identity = dict(prepared=info["identity"], amrfinder=version_text,
                          reference_proteins_sha256=sha256(reference_file), min_gene_coverage=args.min_gene_coverage)
    state = args.out/"audit_identity.json"
    if state.exists():
        if json.loads(state.read_text())!=audit_identity: raise ValueError("Allele audit identity changed; choose a new --out")
        if (args.out/"genotypes.npz").exists():
            log("reuse completed allele audit")
            return
    root = args.out/"amrfinder"
    jobs = [(s, p, root, executable, database, version_text) for s,p in sources.items()]
    with ProcessPoolExecutor(max_workers=args.threads) as pool:
        for i,sample in enumerate(pool.map(call_amrfinder, jobs, chunksize=1), 1):
            if i%10==0 or i==len(jobs): log(f"AMRFinder reports {i}/{len(jobs)} saved; per-sample restart enabled")
    main, calls = {}, {}
    for sample in info["samples"]:
        directory = sample_directory(root, sample)
        main[sample] = list(records(directory/"amr.tsv"))
        calls[sample] = [r for r in records(directory/"mutations.tsv") if point_symbol(r.get("Gene symbol", ""))]
    accessions = {r["Accession of closest sequence"] for rs in calls.values() for r in rs}
    reference = {}
    for accession, protein in fasta_records(reference_file):
        for key in set(accession.split("|")) & accessions: reference[key] = protein
    absent = accessions-set(reference)
    if absent: raise ValueError(f"AMRProt.fa does not match AMRFinder reports; missing reference proteins {sorted(absent)}")
    G,names = known_matrix(info["samples"], main, calls, args.min_gene_coverage)
    symbols = [name for name in names if point_symbol(name)]
    genotypes = [dict(sample=s, **{name: "" if not np.isfinite(G[i,j]) else int(G[i,j])
                                  for j,name in enumerate(names)}) for i,s in enumerate(info["samples"])]
    write_tsv(args.out/"known_genotypes.tsv", genotypes, ["sample"]+names)
    with np.load(args.out/"prepared.npz") as f: X = f["X"]
    sample_index = {s:i for i,s in enumerate(info["samples"])}
    locus_index = {u:j for j,u in enumerate(info["loci"])}
    detail, aggregate = [], defaultdict(lambda: defaultdict(int))
    jobs = [(s,sources[s],calls[s],info["target_dna"],info["target_gene"],reference) for s in info["samples"]]
    with ProcessPoolExecutor(max_workers=args.threads) as pool:
        for i,result in enumerate(pool.map(audit_sample, jobs, chunksize=1),1):
            for row in result:
                present = int(X[sample_index[row["sample"]],locus_index[row["locus"]]])
                row["pseudo_fasta_present"] = present
                key = row["locus"], row["site"]
                value = aggregate[key]
                value["callable_rows"] += 1
                value["unitig_present_rows"] += present
                value["present_and_local_exact_match"] += present*int(row["local_exact_matches"]>0)
                value["present_and_covers_known_codon"] += present*row["covers_known_codon"]
                value["coordinate_unresolved_rows"] += int(not row["coordinate_resolved"])
                parsed = point_symbol(row["marker"])
                resistant = row["marker"] in symbols
                value[("present" if present else "absent")+"_"+("known_mutant" if resistant else "wildtype" if parsed[1]==parsed[3] else "other_allele")] += 1
                detail.append(row)
            if i%50==0 or i==len(jobs): log(f"target unitig/site audit {i}/{len(jobs)}")
    fields = ["sample", "locus", "gene", "marker", "site", "contig", "reference_accession", "local_exact_matches",
              "coordinate_resolved", "covers_known_codon", "codon_start_1based", "observed_amino_acid", "pseudo_fasta_present"]
    with gzip.open(args.out/"unitig_site_evidence.tsv.gz", "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fields, delimiter="\t"); writer.writeheader(); writer.writerows(detail)
    summaries = []
    for (locus,site), value in sorted(aggregate.items()):
        summaries.append(dict(locus=locus, gene=info["target_gene"][str(locus)], site=site,
            status="DIRECT_KNOWN_SITE_OVERLAP" if value["present_and_covers_known_codon"] else "NO_CONFIRMED_SITE_OVERLAP",
            **value))
    for locus in map(int,info["target_dna"]):
        if not any(r["locus"]==locus for r in summaries):
            summaries.append(dict(locus=locus,gene=info["target_gene"][str(locus)],site="",
                status="GENE_CARRIAGE_MARKER" if info["target_gene"][str(locus)]=="blaZ" else "NO_CALLABLE_KNOWN_SITE"))
    columns = ["locus","gene","site","status"]+sorted({k for r in summaries for k in r}-{"locus","gene","site","status"})
    write_tsv(args.out/"unitig_allele_audit.tsv", summaries, columns)
    npz_write(args.out/"genotypes.npz", G=G, names=np.asarray(names))
    json_write(state, audit_identity)
    log(f"allele audit saved: {len(symbols)} observed curated mutation features; blaZ gene detection separate")


def censored_loss(theta, design, bound, censor, penalty):
    """Gaussian exact densities or one-sided probabilities, with analytic gradient."""
    beta, logs = theta[:-1], theta[-1]
    sigma = np.exp(logs)
    z = (bound-design@beta)/sigma
    exact = censor==0
    loss = np.empty(len(bound)); dmean = np.empty(len(bound)); dscale = np.empty(len(bound))
    loss[exact] = .5*z[exact]**2+logs+.5*np.log(2*np.pi)
    dmean[exact] = -z[exact]/sigma
    dscale[exact] = 1-z[exact]**2
    t = -censor[~exact]*z[~exact]
    logcdf = log_ndtr(t)
    mills = np.exp(-.5*t*t-.5*np.log(2*np.pi)-logcdf)
    loss[~exact] = -logcdf
    dmean[~exact] = -censor[~exact]*mills/sigma
    dscale[~exact] = mills*t
    value = loss.sum()+.5*np.dot(penalty,beta*beta)
    gradient = np.r_[design.T@dmean+penalty*beta, dscale.sum()]
    return value, gradient


def fit_censored(design, bound, censor, alpha, nuisance):
    if np.count_nonzero(censor==0)<3: raise ValueError("Fewer than 3 exact MICs in a training partition")
    init = np.zeros(design.shape[1]+1)
    init[0] = np.median(bound[censor==0])
    init[-1] = np.log(max(.5,np.std(bound[censor==0])))
    penalty = np.r_[np.zeros(nuisance), np.full(design.shape[1]-nuisance,alpha)]
    fitted = minimize(censored_loss, init, args=(design,bound,censor,penalty), jac=True,
        method="L-BFGS-B", bounds=[(None,None)]*design.shape[1]+[(np.log(.05),np.log(32.))],
        options=dict(maxiter=500,ftol=1e-10,gtol=1e-6))
    if not fitted.success or not np.isfinite(fitted.fun): raise ValueError("Censored fit failed: "+fitted.message)
    if abs(fitted.x[-1]-np.log(.05))<1e-5 or abs(fitted.x[-1]-np.log(32.))<1e-5:
        raise ValueError("Residual scale reached a numerical search bound")
    return fitted.x[:-1], float(np.exp(fitted.x[-1]))


def prediction_loss(bound, censor, mean, sigma):
    z = (bound-mean)/sigma
    return np.where(censor==0, .5*z*z+np.log(sigma)+.5*np.log(2*np.pi),
                    -log_ndtr(-censor*z))


def tree_axes(K, train, test, count):
    """Training-only kernel PCA, with out-of-sample projection of held-out isolates."""
    block = K[np.ix_(train,train)]
    means, grand = block.mean(axis=0), block.mean()
    centered = block-means[None,:]-means[:,None]+grand
    eigenvalues, vectors = np.linalg.eigh(centered)
    selected = np.flatnonzero(eigenvalues>max(1e-10,float(eigenvalues.max())*1e-8))[-count:][::-1] if count else []
    if not len(selected): return np.empty((len(train),0)), np.empty((len(test),0))
    values, vectors = eigenvalues[selected], vectors[:,selected]
    a = vectors*np.sqrt(values)
    cross = K[np.ix_(test,train)]
    cross = cross-cross.mean(axis=1)[:,None]-means[None,:]+grand
    b = cross@vectors/np.sqrt(values)
    std = a.std(axis=0)
    return a/std, b/std


def pattern_key(values):
    packed = np.packbits(values.astype(np.uint8)).tobytes()
    inverse = np.packbits(1-values.astype(np.uint8)).tobytes()
    return min(packed,inverse)


def independent_columns(F, train, candidates, minimum):
    selected, seen = [], set()
    for j in candidates:
        values = F[train,j]
        if min(values.sum(),len(values)-values.sum())<minimum: continue
        key = pattern_key(values)
        if key not in seen: selected.append(j); seen.add(key)
    return selected


def model_design(F, train, test, candidates, axes, minimum):
    selected = independent_columns(F,train,candidates,minimum)
    mean = F[train][:,selected].mean(axis=0)
    std = F[train][:,selected].std(axis=0)
    a = np.column_stack([np.ones(len(train)),axes[0],(F[train][:,selected]-mean)/std])
    b = np.column_stack([np.ones(len(test)),axes[1],(F[test][:,selected]-mean)/std])
    return a,b,1+axes[0].shape[1],selected


def matched_partners(F, train, base, neighbours, pool, minimum, seed):
    rng = np.random.default_rng(seed)
    baseline = independent_columns(F,train,base,minimum)
    selected = independent_columns(F,train,base+neighbours,minimum)[len(baseline):]
    wanted = np.minimum(F[train][:,selected].mean(axis=0),1-F[train][:,selected].mean(axis=0))
    eligible = independent_columns(F,train,baseline+list(pool),minimum)[len(baseline):]
    rng.shuffle(eligible)
    maf = {j:min(F[train,j].mean(),1-F[train,j].mean()) for j in eligible}
    chosen, differences = [], []
    for value in wanted:
        if not eligible: raise ValueError("Not enough independent MAF-matched random partners")
        j = min(eligible,key=lambda k:abs(maf[k]-value))
        differences.append(abs(maf[j]-value)); chosen.append(j); eligible.remove(j)
    return chosen, float(np.mean(differences)) if differences else 0.


_STATE = None
_ENDPOINTS = {}


def worker_init(settings):
    global _STATE, _ENDPOINTS
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    args = argparse.Namespace(**settings)
    args.out = Path(args.out)
    info = json.loads((args.out/"prepared.json").read_text())
    with np.load(args.out/"prepared.npz") as f: X,K = f["X"].astype(float),f["K"]
    with np.load(args.out/"genotypes.npz") as f: G,names = f["G"],f["names"].tolist()
    _STATE = args,info,np.column_stack([G,X]),K,names
    _ENDPOINTS = {}


def endpoint_data(endpoint):
    if endpoint in _ENDPOINTS: return _ENDPOINTS[endpoint]
    from sklearn.model_selection import GroupKFold
    args,info,F,K,names = _STATE
    baseline = [j for j,name in enumerate(names) if name in {"blaZ_presence","mecA_presence","mecC_presence"}] if endpoint=="penicillin" else [j for j,name in enumerate(names) if point_symbol(name)]
    if not baseline: raise ValueError("No observed curated gyrA/parC mutation features; inspect the allele audit first")
    drug = "benzylpenicillin" if endpoint=="penicillin" else args.quinolone
    column = args.penicillin_column if endpoint=="penicillin" else drug+"_mic"
    sign_column = args.penicillin_sign_column if endpoint=="penicillin" else drug+"_mic_sign"
    if column not in info["metadata"][0] or sign_column not in info["metadata"][0]:
        raise ValueError(f"Metadata lacks {column} / {sign_column}")
    indices, bounds, censor, audit = [], [], [], []
    for i,row in enumerate(info["metadata"]):
        value,sign = str(row.get(column,"")),str(row.get(sign_column,""))
        number,status = mic_record(value,sign)
        sign = sign.strip().replace("≤","<=").replace("≥",">=")
        inline = re.match(r"^(<=|>=|<|>|=)",value.strip().replace("≤","<=").replace("≥",">="))
        if inline: sign = inline[1]
        if not np.isfinite(F[i,baseline]).all(): status="uncallable_baseline_genotype"
        if status in {"exact","censored"}:
            indices.append(i); bounds.append(np.log2(number)); censor.append(0 if status=="exact" else -1 if sign.startswith("<") else 1)
        audit.append(dict(sample=info["samples"][i],mic=value,sign=sign,status=status))
    if len(indices)<30 or sum(c==0 for c in censor)<10:
        raise ValueError(f"{drug}: fewer than 30 usable MICs or 10 exact MICs; do not substitute limits as exact values")
    indices = np.array(indices)
    groups = np.asarray(info["groups"])[indices]
    if len(set(groups))<3: raise ValueError("Fewer than three represented phylogenetic groups")
    folds = list(GroupKFold(n_splits=min(args.folds,len(set(groups)))).split(indices,groups=groups))
    state = dict(indices=indices,bound=np.asarray(bounds),censor=np.asarray(censor),groups=groups,
                 baseline=baseline,folds=folds,audit=audit,drug=drug,axes={})
    _ENDPOINTS[endpoint] = state
    return state


def prediction_job(job):
    from sklearn.model_selection import GroupKFold
    endpoint,fold,model,repeat = job
    args,info,F,K,names = _STATE
    directory = args.out/"prediction_checkpoints"/endpoint
    directory.mkdir(parents=True,exist_ok=True)
    path = directory/f"fold{fold}_{model}_{repeat}.json"
    if path.exists(): return json.loads(path.read_text())
    result = dict(endpoint=endpoint,fold=fold,model=model,repeat=repeat,status="FAILED")
    try:
        state = endpoint_data(endpoint)
        indices,y,censor,groups = (state[k] for k in ["indices","bound","censor","groups"])
        train,test = state["folds"][fold]
        real_train,real_test = indices[train],indices[test]
        base = state["baseline"]
        target = [len(names)+j for j in info["target_columns"][endpoint]]
        neighbours = [len(names)+j for j in info["neighbour_columns"][endpoint]]
        if model in {"Mean","Tree"}: candidates=[]
        elif model=="Background": candidates=[j for j in base if names[j]!="blaZ_presence"]
        elif model=="Known_gyrA": candidates=[j for j in base if canonical(names[j])=="gyrA"]
        elif model=="Known_parC": candidates=[j for j in base if canonical(names[j])=="parC"]
        elif model=="Known": candidates=base
        elif model=="Target_unitigs": candidates=base+target
        elif model=="Neighbours": candidates=base+target+neighbours
        elif model=="Random_partners":
            random,mafdiff = matched_partners(F,real_train,base+target,neighbours,
                [len(names)+j for j in info["random_pool"]],args.min_count,args.seed+100003*fold+repeat)
            candidates=base+target+random
            result.update(random_loci=[info["loci"][j-len(names)] for j in random],
                          mean_maf_difference=mafdiff,random_partner_count=len(random))
        else: raise ValueError(model)
        inner_groups = groups[train]
        if len(set(inner_groups))<2: raise ValueError("No grouped inner validation partition")
        inner = list(GroupKFold(n_splits=min(3,len(set(inner_groups)))).split(train,groups=inner_groups))
        blocks = []
        for t,v in inner:
            a,b = indices[train[t]],indices[train[v]]
            key=(fold,tuple(t),tuple(v),model=="Mean")
            if key not in state["axes"]: state["axes"][key]=tree_axes(K,a,b,0 if model=="Mean" else args.pcs)
            design,held,nuisance,_ = model_design(F,a,b,candidates,state["axes"][key],args.min_count)
            blocks.append((design,held,nuisance,train[t],train[v]))
        losses = []
        for alpha in args.alphas:
            scored=[]
            try:
                for design,held,nuisance,t,v in blocks:
                    beta,sigma = fit_censored(design,y[t],censor[t],alpha,nuisance)
                    scored.extend(prediction_loss(y[v],censor[v],held@beta,sigma))
                losses.append((float(np.mean(scored)),alpha))
            except ValueError: continue
        if not losses: raise ValueError("No ridge penalty completed all grouped inner folds")
        alpha = min(losses)[1]
        key=(fold,"outer",model=="Mean")
        if key not in state["axes"]: state["axes"][key]=tree_axes(K,real_train,real_test,0 if model=="Mean" else args.pcs)
        design,held,nuisance,selected = model_design(F,real_train,real_test,candidates,state["axes"][key],args.min_count)
        beta,sigma = fit_censored(design,y[train],censor[train],alpha,nuisance)
        mean = held@beta
        result.update(status="OK",alpha=alpha,sigma=sigma,n_train=len(train),n_test=len(test),
            n_genomic_features=len(selected),
            selected_features=[names[j] if j<len(names) else "unitig:"+str(info["loci"][j-len(names)]) for j in selected],
            predictions=[dict(sample=info["samples"][i],group=str(groups[j]),bound=float(y[j]),
                censor=int(censor[j]),prediction=float(p),sigma=sigma,nll=float(loss))
                for i,j,p,loss in zip(real_test,test,mean,prediction_loss(y[test],censor[test],mean,sigma))])
    except ValueError as error:
        result["reason"] = str(error)
    json_write(path,result)
    return result


def metrics(predictions):
    exact = [r for r in predictions if r["censor"]==0]
    error = np.array([r["prediction"]-r["bound"] for r in exact])
    observed = np.array([r["bound"] for r in exact])
    return dict(n=len(predictions),n_exact=len(exact),nll=float(np.mean([r["nll"] for r in predictions])),
        rmse_exact=float(np.sqrt(np.mean(error**2))) if len(error) else None,
        within_one_dilution_exact=float(np.mean(abs(error)<=1)) if len(error) else None,
        r2_exact=float(1-np.sum(error**2)/np.sum((observed-observed.mean())**2)) if len(error) and np.var(observed)>0 else None)


def prediction_report(args, results):
    summaries, predictions, comparisons, features = [], [], [], []
    grouped = defaultdict(list)
    for r in results:
        grouped[(r["endpoint"],r["model"],r["repeat"])].append(r)
        for row in r.get("predictions",[]): predictions.append(dict(endpoint=r["endpoint"],model=r["model"],repeat=r["repeat"],fold=r["fold"],**row))
        for name in r.get("selected_features",[]):
            features.append(dict(endpoint=r["endpoint"],model=r["model"],repeat=r["repeat"],fold=r["fold"],feature=name))
    for (endpoint,model,repeat), folds in sorted(grouped.items()):
        success = all(f["status"]=="OK" for f in folds)
        row=dict(endpoint=endpoint,model=model,repeat=repeat,status="OK" if success else "INCOMPLETE",
                 failed_folds=";".join(str(f["fold"]) for f in folds if f["status"]!="OK"))
        if success: row.update(metrics([p for f in folds for p in f["predictions"]]))
        summaries.append(row)
    for endpoint in TARGETS:
        selected = [r for r in summaries if r["endpoint"]==endpoint and r["status"]=="OK"]
        baseline = next((r for r in selected if r["model"]=="Target_unitigs"),None)
        network = next((r for r in selected if r["model"]=="Neighbours"),None)
        controls = [r for r in selected if r["model"]=="Random_partners"]
        if baseline and network:
            comparisons.append(dict(endpoint=endpoint,delta_nll=baseline["nll"]-network["nll"],
                delta_rmse_exact=baseline["rmse_exact"]-network["rmse_exact"],valid_random_sets=len(controls),
                random_fraction_at_least_as_good=(1+sum(r["nll"]<=network["nll"] for r in controls))/(1+len(controls)) if controls else "",
                interpretation="positive delta favours neighbours; random fraction is a descriptive benchmark, not an epistasis P-value"))
    for name,rows,first in [("performance",summaries,["endpoint","model","repeat","status"]),
                           ("predictions",predictions,["endpoint","model","repeat","fold","sample"]),
                           ("neighbour_improvement",comparisons,["endpoint"])]:
        fields=first+sorted({k for r in rows for k in r}-set(first))
        write_tsv(args.out/f"{name}.tsv",rows,fields)
    write_tsv(args.out/"fit_status.tsv",[{k:v for k,v in r.items() if k not in {"predictions","selected_features","random_loci"}} for r in results],
              ["endpoint","fold","model","repeat","status","reason","alpha","n_train","n_test","n_genomic_features","random_partner_count","mean_maf_difference"])
    write_tsv(args.out/"model_features.tsv",features,["endpoint","model","repeat","fold","feature"])
    performance_plot(args.out,summaries)


def performance_plot(out, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["svg.fonttype"]="none"
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for panel,endpoint in enumerate(TARGETS):
        successful=[r for r in rows if r["endpoint"]==endpoint and r["status"]=="OK"]
        fixed=[r for r in successful if r["model"]!="Random_partners"]
        controls=[r for r in successful if r["model"]=="Random_partners"]
        for col,measure in enumerate(["nll","rmse_exact"]):
            ax=axes[panel,col]
            for j,r in enumerate(fixed): ax.plot(j,r[measure],"ko")
            labels=[r["model"].replace("_","\n") for r in fixed]
            if controls:
                ax.boxplot([[r[measure] for r in controls]],positions=[len(fixed)],widths=.5,
                    boxprops=dict(color="black"),medianprops=dict(color="black"),whiskerprops=dict(color="black"),capprops=dict(color="black"))
                labels.append("Random\npartners")
            ax.set_xticks(range(len(labels)),labels,rotation=35,ha="right",fontsize=8)
            ax.set_title(endpoint)
            ax.set_ylabel("Held-out censored Gaussian NLL" if measure=="nll" else "Exact-MIC RMSE (log2 units)")
            ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    for suffix in ["png","svg"]: fig.savefig(out/f"mic_performance.{suffix}",dpi=300)
    plt.close(fig)


def predict(args, info):
    if not (args.out/"audit_identity.json").exists(): raise ValueError("Run the allele audit first")
    settings = dict(out=str(args.out),quinolone=args.quinolone,penicillin_column=args.penicillin_column,
        penicillin_sign_column=args.penicillin_sign_column,folds=args.folds,pcs=args.pcs,
        min_count=args.min_count,seed=args.seed,alphas=args.alphas)
    import importlib.metadata
    identity=dict(settings=settings,revision=REVISION,code=sha256(__file__),
        audit=json.loads((args.out/"audit_identity.json").read_text()),
        dependencies={name:importlib.metadata.version(name) for name in ["numpy","scipy","scikit-learn"]})
    state=args.out/"prediction_identity.json"
    if state.exists() and json.loads(state.read_text())!=identity: raise ValueError("Prediction settings changed; use a new --out")
    json_write(state,identity)
    worker_init(settings)
    jobs=[]
    for endpoint in TARGETS:
        try:
            endpoint_state=endpoint_data(endpoint)
            write_tsv(args.out/f"{endpoint}_phenotypes.tsv",endpoint_state["audit"],["sample","mic","sign","status"])
            models=["Mean","Tree","Known","Target_unitigs","Neighbours"]
            if endpoint=="penicillin": models.insert(2,"Background")
            if endpoint=="fluoroquinolone": models[2:2]=["Known_gyrA","Known_parC"]
            for fold in range(len(endpoint_state["folds"])):
                jobs.extend((endpoint,fold,model,0) for model in models)
                jobs.extend((endpoint,fold,"Random_partners",repeat) for repeat in range(args.random_sets))
            log(f"{endpoint}: {len(endpoint_state['indices'])} callable MICs; {sum(endpoint_state['censor']==0)} exact")
        except ValueError as error:
            log(f"{endpoint}: NOT RUN: {error}")
            json_write(args.out/f"{endpoint}_not_run.json",dict(reason=str(error)))
    if not jobs: raise ValueError("Neither endpoint is eligible; inspect *_not_run.json")
    results=[]
    with ProcessPoolExecutor(max_workers=args.threads,initializer=worker_init,initargs=(settings,)) as pool:
        for i,result in enumerate(pool.map(prediction_job,jobs,chunksize=1),1):
            results.append(result)
            if i%10==0 or i==len(jobs): log(f"prediction partitions saved {i}/{len(jobs)}; failed={sum(r['status']!='OK' for r in results)}")
    prediction_report(args,results)
    log("prediction finished; inspect fit_status.tsv before interpreting improvement")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["map","prepare","audit","predict","all"])
    parser.add_argument("--network",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    for name in ["annotation","fasta","unitigs","tree","metadata","sample-map","clusters","locus-map",
                 "assemblies","gff-dir","amrfinder-db","reference-proteins"]:
        parser.add_argument("--"+name,type=Path)
    parser.add_argument("--fasta-columns-are-locus-ids",action="store_true")
    parser.add_argument("--amrfinder",default="amrfinder")
    parser.add_argument("--min-gene-coverage",type=float,default=90.)
    parser.add_argument("--quinolone",default="levofloxacin",choices=["levofloxacin","ciprofloxacin","moxifloxacin"])
    parser.add_argument("--penicillin-column",default="benzylpenicillin_mic")
    parser.add_argument("--penicillin-sign-column",default="benzylpenicillin_mic_sign")
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--groups",type=int,default=10,help="Tree partition only when --clusters is absent; not fastBAPS")
    parser.add_argument("--folds",type=int,default=5)
    parser.add_argument("--pcs",type=int,default=10)
    parser.add_argument("--random-sets",type=int,default=100)
    parser.add_argument("--min-count",type=int,default=5)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--alphas",type=float,nargs="+",default=[.1,1.,10.,100.])
    args=parser.parse_args(argv)
    if min(args.threads,args.folds,args.groups,args.min_count)<1 or args.folds<3 or args.pcs<0 or args.random_sets<0 or min(args.alphas)<=0:
        parser.error("Need positive CPU/group/count/penalty settings, >=3 folds, nonnegative PCs/random sets")
    if args.stage in {"map","prepare","all"}:
        required=["network"] if args.stage=="map" else ["network","annotation","fasta","unitigs","tree","metadata"]
        for name in required:
            path=getattr(args,name)
            if path is None or not path.is_file(): parser.error(f"--{name} must name an existing input")
        mapped=map_report(args)
        if args.stage=="map": return
        info=prepare(args,mapped)
    else:
        info=json.loads((args.out/"prepared.json").read_text())
    if args.stage in {"audit","all"}: allele_audit(args,info)
    if args.stage in {"predict","all"}: predict(args,info)


if __name__=="__main__":
    main()
