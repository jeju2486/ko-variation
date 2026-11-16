from __future__ import annotations

"""Shared content fingerprints and atomic writes for downstream checkpoints."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class CheckpointError(ValueError):
    """Raised when a checkpoint is missing, corrupt, or incompatible."""


def _normalise_json(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _normalise_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise_json(item) for item in value]
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _normalise_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=True,
    ).encode("utf-8")


def fingerprint_file(path: str | Path, block_size: int = 8 * 1024 * 1024) -> dict[str, Any]:
    """Return a content fingerprint suitable for strict resume validation."""

    source = Path(path)
    if not source.is_file():
        raise CheckpointError(f"Cannot fingerprint missing input file: {source}")
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
            size += len(block)
    return {
        "name": source.name,
        "size_bytes": size,
        "sha256": digest.hexdigest(),
    }


def run_fingerprint(identity: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(identity)).hexdigest()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


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
