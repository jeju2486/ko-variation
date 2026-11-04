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
