"""Shared completed-stage cache for portable downstream helpers.

Source identities use path, size and mtime to avoid hashing huge scan inputs.
Completed outputs are hashed; failed stages never receive a cache manifest.
"""
from __future__ import annotations

import json
from pathlib import Path

from .checkpoint import fingerprint_file, _atomic_write_bytes


def file_signature(path):
    source = Path(path).resolve()
    stat = source.stat()
    return dict(path=str(source), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def stage_identity(stage, inputs, settings, directories=()):
    # Avoid rereading huge original pair/genotype files just to reuse a stage.
    # File path, byte size and nanosecond mtime identify unchanged source files.
    inventory = []
    for directory in directories:
        root = Path(directory).resolve()
        for pattern in ("*.gff3", "*.fna"):
            inventory.extend(file_signature(path) for path in sorted(root.rglob(pattern)))
    return dict(stage=stage, cache_schema=1, inputs=[file_signature(path) for path in inputs],
                settings=settings, directory_files=inventory)


def cache_matches(manifest, identity, outputs):
    try:
        saved = json.loads(Path(manifest).read_text(encoding="utf-8"))
        return (saved.get("cache_identity") == identity and
                saved.get("cache_outputs") == [fingerprint_file(path) for path in outputs])
    except (OSError, ValueError):
        return False


def cache_fields(identity, outputs):
    return dict(cache_identity=identity, cache_outputs=[fingerprint_file(path) for path in outputs])


def save_cache(manifest, identity, outputs, **details):
    """Publish provenance only after every completed output can be fingerprinted."""
    payload = dict(details, **cache_fields(identity, outputs))
    _atomic_write_bytes(Path(manifest), json.dumps(payload, indent=2).encode("utf-8"))
