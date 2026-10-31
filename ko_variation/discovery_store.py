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
import sqlite3
import shutil

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
