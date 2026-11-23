"""Disk-backed preparation and atomic restart records for unitig discovery.

FASTA is read one isolate at a time. Only a bounded locus block is transposed;
exact patterns and the original unitig mapping are indexed in SQLite.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import shutil
import tempfile

import numpy as np

from .kinship import build_tree_covariance
from .glmm import prepare_kinship, prepare_kinship_eigensystem


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def connect(path, memory_mb=32):
    db = sqlite3.connect(path, timeout=120)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("PRAGMA temp_store=FILE")
    db.execute(f"PRAGMA cache_size=-{max(1024, int(memory_mb * 1024))}")
    return db


def schema(db):
    db.executescript("""
    CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS nodes(id INTEGER PRIMARY KEY,parent INTEGER,level INTEGER,
      anchor INTEGER,size INTEGER DEFAULT 0,ready INTEGER DEFAULT 0);
    CREATE INDEX IF NOT EXISTS node_parent ON nodes(parent,id);
    CREATE TABLE IF NOT EXISTS members(node INTEGER,pid INTEGER,PRIMARY KEY(node,pid)) WITHOUT ROWID;
    CREATE TABLE IF NOT EXISTS assigned(parent INTEGER,pid INTEGER,child INTEGER,PRIMARY KEY(parent,pid)) WITHOUT ROWID;
    CREATE TABLE IF NOT EXISTS reps(node INTEGER,pid INTEGER,PRIMARY KEY(node,pid)) WITHOUT ROWID;
    CREATE TABLE IF NOT EXISTS tasks(a INTEGER,b INTEGER,done INTEGER DEFAULT 0,PRIMARY KEY(a,b)) WITHOUT ROWID;
    CREATE INDEX IF NOT EXISTS pending_tasks ON tasks(done,a,b);
    CREATE TABLE IF NOT EXISTS block_eligibility(node INTEGER,min_cell INTEGER,center INTEGER,
      radius INTEGER,eligible_members INTEGER,PRIMARY KEY(node,min_cell)) WITHOUT ROWID;
    CREATE TABLE IF NOT EXISTS decisions(u INTEGER,v INTEGER,gate INTEGER,final INTEGER,
      t_low REAL,t_high REAL,p REAL,status TEXT,PRIMARY KEY(u,v)) WITHOUT ROWID;
    CREATE TABLE IF NOT EXISTS compact_rejections(u INTEGER,v INTEGER,final INTEGER,
      PRIMARY KEY(u,v)) WITHOUT ROWID;
    CREATE INDEX IF NOT EXISTS discovery_hits ON decisions(final,u,v);
    CREATE TABLE IF NOT EXISTS external_nulls(pid INTEGER PRIMARY KEY,path TEXT);
    CREATE TABLE IF NOT EXISTS null_aliases(pid INTEGER PRIMARY KEY,canonical INTEGER);
    CREATE TABLE IF NOT EXISTS null_inventory(pid INTEGER PRIMARY KEY,status TEXT,mode TEXT,
      fit_seconds REAL,source TEXT,local INTEGER,signature TEXT);
    CREATE TABLE IF NOT EXISTS beta(u INTEGER,v INTEGER,value REAL,status TEXT,PRIMARY KEY(u,v)) WITHOUT ROWID;
    CREATE TABLE IF NOT EXISTS distance(u INTEGER,v INTEGER,pu INTEGER,pv INTEGER,
      distance REAL,connected INTEGER,status TEXT DEFAULT 'PENDING',PRIMARY KEY(u,v)) WITHOUT ROWID;
    CREATE INDEX IF NOT EXISTS distance_pending ON distance(status,u,v);
    CREATE INDEX IF NOT EXISTS distance_pattern ON distance(pu,pv,distance);
    """)


def get_meta(db, key, default=None):
    row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else default


def set_meta(db, key, value):
    db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, json.dumps(value, allow_nan=False)))


def records(path):
    """Yield unique-ID pseudo-FASTA records, validating A/C strictly."""
    name, chunks = None, []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for raw in stream:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, b"".join(chunks)
                fields = line[1:].split()
                if not fields:
                    raise ValueError("Empty FASTA sample ID")
                name, chunks = fields[0], []
            else:
                if name is None:
                    raise ValueError("Sequence before FASTA header")
                sequence = line.upper().encode("ascii")
                if set(sequence) - {65, 67}:
                    raise ValueError(f"Unexpected characters for {name}; expected A/C")
                chunks.append(sequence)
        if name is not None:
            yield name, b"".join(chunks)


def prepare_pangwes(raw_fasta, raw_tree, assembly_list, gfa, metadata, fasta,
                    tree_file, exclude, missing_policy, work):
    """Validate Cuttlefish row order and publish a sample-labelled FASTA and tree."""
    if fasta.parent != tree_file.parent or fasta == tree_file:
        raise ValueError("Prepared FASTA and tree need distinct paths in the same directory")
    prep = fasta.parent
    manifest = prep / "preparation.json"
    mapping = prep / "sample_mapping.tsv"
    outputs = [fasta, tree_file, mapping]
    sources = [raw_fasta, assembly_list, gfa, metadata, raw_tree] + ([exclude] if exclude else [])
    if len(set(outputs + [manifest])) != 4 or set(outputs + [manifest]).intersection(sources):
        raise ValueError("Prepared inputs must have their own paths")

    def stamp(path):
        stat = path.stat()
        return [str(path.resolve()), stat.st_size, stat.st_mtime_ns]

    for source in sources:
        if not source.is_file() or not source.stat().st_size:
            raise ValueError(f"Missing or empty upstream input: {source}")
    identity = dict(sources=[stamp(p) for p in sources], missing_policy=missing_policy,
                    preparation="cuttlefish-ids-matched-midpoint-v1")
    prep.mkdir(parents=True, exist_ok=True)
    lock = sqlite3.connect(prep / ".preparation.lock.sqlite", timeout=0)
    try:
        lock.execute("BEGIN EXCLUSIVE")
        if manifest.exists():
            old = json.loads(manifest.read_text())
            if (old["identity"] == identity and all(p.is_file() for p in outputs)
                    and old["outputs"] == [stamp(p) for p in outputs]):
                print(f"[prepare] Reusing prepared inputs: {prep}", flush=True)
                return
        if (work / "run_identity.json").exists():
            raise ValueError("Upstream inputs or generated files changed; choose a new output")

        assemblies = [line.strip() for line in assembly_list.read_text().splitlines() if line.strip()]
        recorded = json.loads(metadata.read_text())["parameters info"]["input"]
        if isinstance(recorded, str):
            recorded = [name.strip() for name in recorded.split(",")]
        if recorded != assemblies:
            raise ValueError("Cuttlefish JSON input order differs from the assembly list")
        names = [Path(name).stem for name in assemblies]
        if not names or len(set(names)) != len(names) or any(re.search(r"\s", name) for name in names):
            raise ValueError("Assembly filename stems must be unique sample IDs without whitespace")

        references = {}
        pattern = re.compile(r"Reference:(\d+)_Sequence:")
        print("[prepare] Reading original GFA sample/reference order", flush=True)
        with gfa.open() as stream:
            for line in stream:
                if line.startswith("P\t"):
                    match = pattern.search(line.split("\t", 2)[1])
                    if match is None:
                        raise ValueError("Unexpected GFA path name; cannot infer sample IDs")
                    references[int(match[1])] = None
        if list(references) not in (list(range(len(names))), list(range(1, len(names) + 1))):
            raise ValueError("GFA references must follow contiguous zero-based or one-based input order")

        try:
            from Bio import Phylo
        except ImportError as error:
            raise ValueError("PAN-GWES input preparation requires Biopython") from error
        excluded = {line.strip() for line in exclude.read_text().splitlines() if line.strip()} if exclude else set()
        if excluded - set(names):
            raise ValueError(f"Excluded sample IDs absent from FASTA: {sorted(excluded - set(names))}")
        tree = Phylo.read(raw_tree, "newick")
        tips = [tip.name for tip in tree.get_terminals()]
        if None in tips or len(set(tips)) != len(tips):
            raise ValueError("Tree tips must have unique sample IDs")
        desired = set(names) - excluded
        missing = sorted(desired - set(tips))
        if missing and missing_policy == "error":
            raise ValueError(f"{len(missing)} sample IDs absent from tree, e.g. {missing[:5]}")
        keep = desired & set(tips)
        if len(keep) < 4:
            raise ValueError("At least four retained isolates are required")
        for tip in list(tree.get_terminals()):
            if tip.name not in keep:
                tree.prune(tip)
        tree.root_at_midpoint()

        with tempfile.TemporaryDirectory(prefix=".prepare-", dir=prep) as temporary:
            temp_fasta, temp_tree, temp_map = [Path(temporary) / path.name for path in outputs]
            Phylo.write(tree, temp_tree, "newick", format_branch_length="%0.15g")
            index = 0
            print(f"[prepare] Streaming {len(names):,} isolates to sample-ID FASTA", flush=True)
            with raw_fasta.open() as source, temp_fasta.open("w") as target:
                for line in source:
                    if line.startswith(">"):
                        if index >= len(names):
                            raise ValueError("More pseudo-FASTA records than GFA sample references")
                        target.write(f">{names[index]}\n")
                        index += 1
                    else:
                        target.write(line)
            if index != len(names):
                raise ValueError(f"FASTA/GFA sample count mismatch: {index} != {len(names)}")
            with temp_map.open("w", newline="") as stream:
                writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
                writer.writerow(["fasta_row_0based", "gfa_reference", "sample_id", "assembly", "retained"])
                for i, (ref, name, assembly) in enumerate(zip(references, names, assemblies)):
                    writer.writerow([i, ref, name, assembly, int(name in keep)])
            for temporary_file, target in zip((temp_fasta, temp_tree, temp_map), outputs):
                os.replace(temporary_file, target)
        atomic_json(manifest, dict(identity=identity, outputs=[stamp(p) for p in outputs],
                    samples=len(names), retained=len(keep), excluded=sorted(excluded),
                    missing_in_tree=missing, tree_root="midpoint after matching/exclusions",
                    unitig_columns="unchanged"))
        print(f"[prepare] Complete: {len(names):,} FASTA rows; {len(keep):,} retained tree tips", flush=True)
    finally:
        lock.close()


def localize_graph_paths(source, target, rebase):
    """Resolve copied graph paths with explicit project-prefix mappings."""
    if source == target:
        raise ValueError("Local graph path list must differ from the source file")
    resolved = []
    for raw in source.read_text().splitlines():
        if not (raw := raw.strip()):
            continue
        path = Path(raw).expanduser()
        candidates = [Path(root) / raw.rsplit(marker, 1)[1] for marker, root in rebase
                      if marker and marker in raw]
        candidates.append(path if path.is_absolute() else source.parent / path)
        match = next((p.resolve() for p in candidates if p.is_file()), None)
        if match is None:
            raise ValueError(f"Copied graph path not found locally: {raw}")
        resolved.append(str(match))
    if not resolved:
        raise ValueError("The graph path list is empty")
    content = "\n".join(resolved) + "\n"
    if not target.exists() or target.read_text() != content:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".paths-", dir=target.parent) as temporary:
            pending = Path(temporary) / target.name
            pending.write_text(content)
            os.replace(pending, target)
    print(f"[prepare] Local graph-path list: {target}; {len(resolved):,} graph files", flush=True)


def reuse_prepared(args, db, work, expected):
    """Copy immutable preparation/hierarchy; never copy old search decisions."""
    if (work/'prepared.json').exists(): return
    source=args.reuse_prepared_from.resolve()
    prior=json.loads((source/'run_identity.json').read_text())
    source_work=Path(prior.get('work_directory',source/'.kovar_work'))
    if source_work.resolve()==work.resolve():
        raise ValueError('--reuse-prepared-from needs a separate new output directory')
    if prior.get('inputs')!=expected['inputs']:
        raise ValueError('Reuse source FASTA/tree/exclusions differ from this run')
    for key in ('numpy','scipy','null_model_source_sha256'):
        if prior.get(key)!=expected[key]: raise ValueError(f'Reuse source {key} differs')
    for key in ('min_maf','tree_missing_samples'):
        if prior['settings'].get(key)!=expected['settings'].get(key):
            raise ValueError(f'Reuse source preparation setting differs: {key}')
    info=json.loads((source_work/'prepared.json').read_text())
    for name in ('patterns.bin','K.npy','K_values.npy','K_vectors.npy'):
        if digest(source_work/name)!=info['products'][name]:
            raise ValueError(f'Reuse source prepared product changed: {name}')
        shutil.copyfile(source_work/name,work/name)
    reader=sqlite3.connect((source_work/'work.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    try:
        reader.execute('BEGIN')
        row=reader.execute("SELECT value FROM meta WHERE key='hierarchy_complete'").fetchone()
        hierarchy=bool(row and json.loads(row[0]))
        if hierarchy:
            keys=('coarse_r2','refine_r2','leaf_size','rep_policy','max_reps')
            hierarchy=all(prior['settings'].get(k)==expected['settings'].get(k) for k in keys)
        tables=['patterns','unitigs']+(['nodes','members','reps','block_eligibility'] if hierarchy else [])
        available={r[0] for r in reader.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'null_aliases' in available: tables.append('null_aliases')
        # One transaction avoids partially imported tables. Immutable arrays are
        # small relative to the null cache and can be recopied after interruption.
        with db:
            for table in tables:
                definition=reader.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?",(table,)).fetchone()[0]
                exists=db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",(table,)).fetchone()
                if not exists: db.execute(definition)
                db.execute(f'DELETE FROM {table}')
                cursor=reader.execute(f'SELECT * FROM {table}')
                while True:
                    rows=cursor.fetchmany(4096)
                    if not rows: break
                    marks=','.join('?' for _ in rows[0])
                    db.executemany(f'INSERT INTO {table} VALUES ({marks})',rows)
            for sql, in reader.execute("SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name IN ('patterns','unitigs') AND sql IS NOT NULL"):
                db.execute(sql.replace('CREATE INDEX ','CREATE INDEX IF NOT EXISTS ',1))
            db.execute("INSERT OR REPLACE INTO meta VALUES ('hierarchy_complete',?)",(json.dumps(hierarchy),))
            db.execute("INSERT OR REPLACE INTO meta VALUES ('reused_prepared_from',?)",(json.dumps(str(source)),))
    finally:
        reader.close()
    atomic_json(work/'prepared.json',info)
    print(f'[KOVAR] reused prepared patterns/tree; hierarchy reused={hierarchy}; source={source}',flush=True)


def prepare(args, db, work):
    if args.memory_gb is None:
        from .memory_budget import detect_budget
        args.memory_gb=detect_budget(None,args.threads)['working_gb']
    ready = work / "prepared.json"
    if ready.exists():
        info = json.loads(ready.read_text())
        # Large arrays stay read-only and are paged by the operating system.
        for name, expected in info["products"].items():
            if digest(work / name) != expected:
                raise ValueError(f"Prepared product changed: {name}")
        return info
    names, width = [], None
    for name, sequence in records(args.fasta):
        if width is None:
            width = len(sequence)
        if len(sequence) != width:
            raise ValueError("Unequal FASTA sequence lengths")
        names.append(name)
    if not names or not width or len(names) != len(set(names)):
        raise ValueError("Use nonempty sample_ids.fasta with unique sample IDs")
    excluded = set()
    if args.exclude_samples:
        excluded = {line.strip() for line in args.exclude_samples.read_text().splitlines() if line.strip()}
        unknown = excluded - set(names)
        if unknown:
            raise ValueError(f"Excluded samples absent from FASTA: {sorted(unknown)}")
    candidates = [name for name in names if name not in excluded]
    estimate = (160*len(candidates)**2+48*(1<<20))*args.threads+160*(1<<20)
    if estimate > args.memory_gb*(1<<30):
        raise ValueError(f"Tree/worker memory estimate is {estimate/(1<<30):.2f} GiB; increase --memory-gb or reduce --threads")
    tree = build_tree_covariance(args.tree, candidates, missing_samples=args.tree_missing_samples)
    retained = [candidates[int(i)] for i in tree.sample_indices]
    if len(retained) < 4:
        raise ValueError("At least four matched isolates are required")
    n = len(retained)
    required = n*width+width*((n+7)//8)+32*n*n+(64<<20)
    if shutil.disk_usage(work).free < required:
        raise ValueError(f"Insufficient scratch space for preparation: need approximately {required/(1<<30):.2f} GiB")
    # Guard against silent renumbering of graph unitigs: the pseudo-alignment
    # column is the zero-based graph unitig ID, including filtered columns.
    if args.unitigs:
        with args.unitigs.open() as stream:
            count = 0
            for line in stream:
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 2 or int(fields[0]) != count:
                    raise ValueError("--unitigs must have consecutive zero-based IDs")
                count += 1
        if count != width:
            raise ValueError("Unitig sequence count differs from FASTA columns")
    if n * width > args.memory_gb * (1 << 30) * 50:
        print(f"[KOVAR] large scratch matrix: {n*width/(1<<30):.2f} GiB", flush=True)
    raw_path=work/"samples.partial.bin"
    raw_state=work/"samples_progress.json"
    raw_cursor=json.loads(raw_state.read_text())["rows"] if raw_state.exists() and raw_path.exists() else 0
    raw = np.memmap(raw_path, mode="r+" if raw_cursor else "w+", dtype="u1", shape=(n, width))
    lookup = {name: i for i, name in enumerate(retained)}
    for name, sequence in records(args.fasta):
        if name in lookup and lookup[name]>=raw_cursor:
            raw[lookup[name]] = np.frombuffer(sequence, dtype="u1") == 67
            raw.flush()
            atomic_json(raw_state,dict(rows=lookup[name]+1))
    raw.flush()
    row=db.execute("SELECT value FROM meta WHERE key='preparation_cursor'").fetchone()
    cursor=json.loads(row[0]) if row else None
    if cursor is None:
        db.executescript("""
      DROP TABLE IF EXISTS unitigs; DROP TABLE IF EXISTS patterns;
      CREATE TABLE patterns(id INTEGER PRIMARY KEY, bits BLOB UNIQUE NOT NULL,
        representative INTEGER NOT NULL, present INTEGER NOT NULL);
      CREATE TABLE unitigs(id INTEGER PRIMARY KEY, pattern INTEGER, present INTEGER NOT NULL);
      CREATE INDEX unitig_pattern ON unitigs(pattern,id);
      CREATE INDEX pattern_present ON patterns(present,id);
        """)
    packed_width = (n + 7) // 8
    patterns = cursor["patterns"] if cursor else 0
    first_locus=cursor["locus"] if cursor else 0
    # Limit this transpose independently of the total number of loci.
    block_size = min(4096, max(1, int(args.memory_gb*(1<<30)*.05/max(n, 1))))
    with (work / "patterns.bin").open("r+b" if cursor else "wb") as out:
        # An interrupted transaction may have appended uncommitted bytes.
        # Truncation restores the last atomic database/file boundary.
        if cursor and out.seek(0,2)<patterns*packed_width:
            raise ValueError("Committed pattern bytes are missing; checkpoint is corrupted")
        out.truncate(patterns*packed_width)
        out.seek(patterns*packed_width)
        for first in range(first_locus, width, block_size):
            block = np.array(raw[:, first:first+block_size].T, copy=True)
            packed = np.packbits(block, axis=1, bitorder="little")
            counts = block.sum(axis=1, dtype=np.int64)
            with db:
                for offset, (bits, present) in enumerate(zip(packed, counts)):
                    locus, present = first+offset, int(present)
                    maf = min(present, n-present)/n
                    pattern = None
                    if 0 < present < n and maf >= args.min_maf:
                        key = bits.tobytes()
                        row = db.execute("SELECT id FROM patterns WHERE bits=?", (key,)).fetchone()
                        if row is None:
                            pattern = patterns
                            db.execute("INSERT INTO patterns VALUES (?,?,?,?)", (pattern, key, locus, present))
                            out.write(key)
                            patterns += 1
                        else:
                            pattern = row[0]
                    db.execute("INSERT INTO unitigs VALUES (?,?,?)", (locus, pattern, present))
                out.flush()
                os.fsync(out.fileno())
                db.execute("INSERT OR REPLACE INTO meta VALUES ('preparation_cursor',?)",
                           (json.dumps(dict(locus=first+len(block),patterns=patterns)),))
            if first % (block_size*25) == 0:
                print(f"[KOVAR] preparation: {min(width,first+len(block)):,}/{width:,} loci; {patterns:,} patterns", flush=True)
        out.flush()
        os.fsync(out.fileno())
    del raw
    if patterns < 2:
        raise ValueError("Fewer than two variable patterns pass filtering")
    K, _ = prepare_kinship(tree.K)
    np.save(work / "K.npy", K)
    eigen = prepare_kinship_eigensystem(K)
    np.save(work / "K_values.npy", eigen.eigenvalues)
    np.save(work / "K_vectors.npy", eigen.eigenvectors)
    info = dict(samples=retained, excluded=[name for name in names if name not in lookup],
                n=n, loci=width, patterns=patterns, packed_width=packed_width,
                shared_covariance=bool(eigen.has_shared_covariance))
    info["products"] = {name: digest(work/name) for name in
                        ("patterns.bin", "K.npy", "K_values.npy", "K_vectors.npy")}
    atomic_json(ready, info)
    raw_path.unlink()
    raw_state.unlink()
    return info


def open_patterns(work, info):
    return np.memmap(Path(work)/"patterns.bin", mode="r", dtype="u1",
                     shape=(info["patterns"], info["packed_width"]))


def write_gzip(path, header, rows):
    temporary = Path(str(path)+".partial")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow("" if value is None else value for value in row)
    os.replace(temporary, path)


def import_benchmark_nulls(db, work, info, root, prior):
    """Import historical benchmark nulls only when explicitly requested."""
    import zipfile
    import scipy
    old_preparation = Path(prior["settings"]["benchmark_dir"])/"preparation.npz"
    if not old_preparation.exists():
        raise ValueError(f"Previous benchmark preparation is required for safe reuse: {old_preparation}")
    from .null_identity import null_source_hash, LEGACY_GLMM_HASHES, LEGACY_NULL_SOURCE
    current = null_source_hash()
    source = prior.get("null_model_source_sha256")
    if source != current:
        hashes = [r.get("sha256") for r in prior.get("inputs",[]) if r.get("name")=="glmm.py"]
        # Some legacy identities store the scorer hash separately.
        hashes += [prior.get("glmm_source_sha256"), prior.get("glmm_sha256"),prior.get("code",{}).get("glmm.py")]
        if not (current == LEGACY_NULL_SOURCE and any(h in LEGACY_GLMM_HASHES for h in hashes)):
            raise ValueError("Prior benchmark null-fitting code is not verified compatible")
    versions = prior.get("versions",prior)
    if versions.get("numpy") != np.__version__ or versions.get("scipy") != scipy.__version__:
        raise ValueError("Prior benchmark NumPy/SciPy versions differ")
    if prior["settings"].get("null_max_iter") != 100 or prior["settings"].get("null_tolerance") != 1e-7:
        raise ValueError("Prior null fitting settings differ")
    expected = [r["sha256"] for r in prior["inputs"] if r["name"] == "preparation.npz"]
    if expected != [digest(old_preparation)]:
        raise ValueError("Previous benchmark preparation changed")
    extracted = work/"import_preparation"
    extracted.mkdir(exist_ok=True)
    with zipfile.ZipFile(old_preparation) as archive:
        for name in ("samples.npy", "K.npy", "packed.npy"):
            with archive.open(name) as source_file, (extracted/name).open("wb") as target:
                shutil.copyfileobj(source_file,target,1<<20)
    if (list(np.load(extracted/"samples.npy", mmap_mode="r")) != info["samples"] or
            not np.array_equal(np.load(extracted/"K.npy", mmap_mode="r"), np.load(work/"K.npy", mmap_mode="r"))):
        raise ValueError("Previous sample order/covariance differs; reuse rejected")
    packed = np.load(extracted/"packed.npy", mmap_mode="r")
    for meta in (root/"null_models").glob("null_*.json"):
        old_pattern = json.loads(meta.read_text())["pattern"]
        row = db.execute("SELECT id FROM patterns WHERE bits=?", (packed[old_pattern].tobytes(),)).fetchone()
        if row:
            with db:
                db.execute("INSERT OR IGNORE INTO external_nulls VALUES (?,?)", (row[0],str(meta)))
    del packed
    for path in extracted.iterdir():
        path.unlink()
    extracted.rmdir()
