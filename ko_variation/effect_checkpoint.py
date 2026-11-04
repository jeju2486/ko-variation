"""Durable per-pair downstream checkpoints and periodic progress messages."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time

import numpy as np

from .checkpoint import run_fingerprint

EFFECT_COLUMNS = (
    "adjusted_beta", "adjusted_beta_se", "adjusted_odds_ratio",
    "adjusted_or_ci_low", "adjusted_or_ci_high", "alternative_tau",
    "effect_status", "effect_iterations", "effect_message",
)


def array_fingerprint(array: np.ndarray) -> dict:
    data = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    if data.size:
        view = memoryview(data).cast("B")
        for start in range(0, len(view), 8 * 1024 * 1024):
            digest.update(view[start:start + 8 * 1024 * 1024])
    return dict(shape=list(data.shape), dtype=str(data.dtype), sha256=digest.hexdigest())


class EffectCheckpoint:
    """Commit every completed pair; retain failures as completed attempts.

    DELETE journaling avoids the shared-memory files required by SQLite WAL.
    An OS lock excludes concurrent writers and releases automatically on exit.
    """

    def __init__(self, path: str | Path, identity: dict, *, resume: bool = False):
        self.path = Path(path)
        self.identity = identity
        self.resume = resume
        self.connection = None
        self.lock_handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_handle = Path(str(self.path) + ".lock").open("a+b")
        try:
            import os
            if os.name == "nt":
                import msvcrt
                self.lock_handle.seek(0, 2)
                if self.lock_handle.tell() == 0:
                    self.lock_handle.write(b"0")
                    self.lock_handle.flush()
                self.lock_handle.seek(0)
                msvcrt.locking(self.lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.lock_handle.close()
            raise ValueError(f"Another refit process holds checkpoint {self.path}") from exc
        try:
            exists = self.path.exists()
            if exists and not self.resume:
                raise ValueError(f"Checkpoint already exists: {self.path}; use --resume or a new output prefix")
            if not exists and self.resume:
                raise ValueError(f"Resume checkpoint does not exist: {self.path}")
            self.connection = sqlite3.connect(self.path, timeout=30)
            self.connection.execute("PRAGMA journal_mode=DELETE")
            self.connection.execute("PRAGMA synchronous=FULL")
            fingerprint = run_fingerprint(self.identity)
            if exists:
                row = self.connection.execute("SELECT schema_version, fingerprint FROM metadata").fetchone()
                if row is None or row[0] != 1 or row[1] != fingerprint:
                    raise ValueError("Refit checkpoint inputs, fitting code or numerical settings do not match")
            else:
                with self.connection:
                    self.connection.execute("CREATE TABLE metadata (schema_version INTEGER, fingerprint TEXT, identity TEXT)")
                    self.connection.execute("CREATE TABLE effects (u INTEGER, v INTEGER, payload TEXT, PRIMARY KEY (u, v))")
                    self.connection.execute("INSERT INTO metadata VALUES (?, ?, ?)",
                                            (1, fingerprint, json.dumps(self.identity, sort_keys=True)))
        except Exception as exc:
            self.__exit__(None, None, None)
            if isinstance(exc, sqlite3.Error):
                raise ValueError(f"Invalid refit checkpoint: {exc}") from exc
            raise
        return self

    def load(self) -> dict:
        recovered = {}
        try:
            for u, v, text in self.connection.execute("SELECT u, v, payload FROM effects"):
                payload = json.loads(text)
                if not isinstance(payload, dict) or set(payload) != set(EFFECT_COLUMNS) or payload["effect_status"] == "NOT_FITTED":
                    raise ValueError(f"Invalid saved effect for pair {u}, {v}")
                recovered[(int(u), int(v))] = {key: np.nan if value is None else value for key, value in payload.items()}
        except (sqlite3.Error, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read refit checkpoint: {exc}") from exc
        return recovered

    def record(self, u: int, v: int, payload: dict) -> None:
        cleaned = {}
        for key in EFFECT_COLUMNS:
            value = payload[key]
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float) and not np.isfinite(value):
                value = None
            cleaned[key] = value
        with self.connection:
            self.connection.execute("INSERT INTO effects VALUES (?, ?, ?)",
                                    (int(u), int(v), json.dumps(cleaned, allow_nan=False)))

    def __exit__(self, *_):
        if self.connection is not None:
            self.connection.close()
        if self.lock_handle is not None:
            self.lock_handle.close()


class RefitProgress:
    def __init__(self, total: int, recovered: dict, *, enabled=True, every=10, seconds=60):
        if every < 1 or not np.isfinite(seconds) or seconds <= 0:
            raise ValueError("Progress intervals must be positive")
        self.total = total
        self.recovered = len(recovered)
        self.done = self.recovered
        self.statuses = Counter(payload["effect_status"] for payload in recovered.values())
        self.enabled, self.every, self.seconds = enabled, every, seconds
        self.started = time.monotonic()
        self.current = "preparing"
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = None

    def _print(self):
        if not self.enabled:
            return
        with self.lock:
            elapsed = time.monotonic() - self.started
            new = self.done - self.recovered
            remaining = elapsed / new * (self.total - self.done) if new else None
            eta = f"{remaining / 3600:.2f}h" if remaining is not None else "pending"
            percent = 100 * self.done / self.total if self.total else 100
            print(f"[refit] {self.done}/{self.total} ({percent:.1f}%) recovered={self.recovered} "
                  f"elapsed={elapsed / 3600:.2f}h remaining~{eta} current={self.current} "
                  f"statuses={dict(self.statuses)}", flush=True)

    def _heartbeat(self):
        while not self.stop.wait(self.seconds):
            self._print()

    def __enter__(self):
        self._print()
        if self.enabled:
            self.thread = threading.Thread(target=self._heartbeat, daemon=True)
            self.thread.start()
        return self

    def start_pair(self, u, v):
        with self.lock:
            self.current = f"pair({u},{v})"

    def completed(self, status):
        with self.lock:
            self.done += 1
            self.statuses[status] += 1
            self.current = "between pairs"
            report = (self.done - self.recovered) % self.every == 0
        if report:
            self._print()

    def __exit__(self, exc_type, *_):
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
        with self.lock:
            self.current = "complete" if exc_type is None else "interrupted"
        self._print()
