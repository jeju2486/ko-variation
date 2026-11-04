"""Allocation-aware memory planning without optional system dependencies.

The budget is a working target, not an OS-enforced limit. Numerical temporaries
and a reserve are deducted before assigning per-worker byte-limited caches.
"""
from __future__ import annotations

from collections import OrderedDict
import os
from pathlib import Path
import re
import sys

BLAS_ENVIRONMENT_VARIABLES = (
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def guard_blas_environment(default_threads: int = 1) -> dict[str, str]:
    """Set conservative native-thread defaults before NumPy/SciPy import.

    Explicit user settings are preserved. The returned values are suitable for
    later reporting, but the loaded-library inspection in
    :func:`collect_parallel_runtime` is the authoritative runtime view.
    """

    if default_threads < 1:
        raise ValueError("default_threads must be positive")
    default = str(default_threads)
    configured: dict[str, str] = {}
    for variable in BLAS_ENVIRONMENT_VARIABLES:
        if not os.environ.get(variable, "").strip():
            os.environ[variable] = default
        configured[variable] = os.environ[variable]
    return configured


GIB = 1 << 30
MIB = 1 << 20


def _read(path):
    try:
        return Path(path).read_text().strip()
    except (OSError, ValueError):
        return ""


def _positive(text):
    try:
        value = int(text)
        return value if 0 < value < 1 << 60 else None
    except (TypeError, ValueError):
        return None


def slurm_bytes(value):
    """Slurm memory environment values are MiB unless a suffix is present."""
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGT]?)B?\s*", str(value), re.I)
    if not match:
        return None
    number, suffix = match.groups()
    result = int(float(number) * {"": MIB, "K": 1 << 10, "M": MIB, "G": GIB, "T": 1 << 40}[suffix.upper()])
    return result or None


def cgroup_limits(root=Path("/sys/fs/cgroup"), membership=Path("/proc/self/cgroup")):
    """Read v2/v1 limits at the process path and its ancestors, including root.

    Checking both the membership path and mount root also covers common cgroup
    namespaces whose process-relative path differs from the mounted root.
    """
    locations = [(root, "memory.max", "memory.current", "memory.stat")]
    for line in _read(membership).splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        _, controllers, relative = fields
        if controllers == "":
            base, limit, usage = root, "memory.max", "memory.current"
        elif "memory" in controllers.split(","):
            base, limit, usage = root/"memory", "memory.limit_in_bytes", "memory.usage_in_bytes"
        else:
            continue
        candidate = base/relative.lstrip("/")
        while candidate == base or base in candidate.parents:
            locations.append((candidate, limit, usage, "memory.stat"))
            if candidate == base:
                break
            candidate = candidate.parent
    limits, seen = [], set()
    for folder, name, usage, stat in locations:
        path = folder/name
        if path in seen:
            continue
        seen.add(path)
        value = _positive(_read(path))
        if value:
            limits.append(dict(bytes=value, source=str(path), usage=str(folder/usage), stat=str(folder/stat)))
    return limits


def physical_bytes():
    if sys.platform == "win32":
        import ctypes
        class Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in
                ("total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "extended")]
        status = Status()
        status.length = ctypes.sizeof(status)
        return int(status.total) if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)) else None
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def detect_budget(requested_gb, threads, *, environ=None, groups=None, physical=None):
    env = os.environ if environ is None else environ
    limits = list(cgroup_limits() if groups is None else groups)
    node = slurm_bytes(env.get("SLURM_MEM_PER_NODE", ""))
    if not node:
        per_cpu = slurm_bytes(env.get("SLURM_MEM_PER_CPU", ""))
        cpu_text = env.get("SLURM_CPUS_ON_NODE", env.get("SLURM_JOB_CPUS_PER_NODE", str(threads)))
        match = re.match(r"\d+", cpu_text)
        node = per_cpu * int(match.group()) if per_cpu and match else None
    if node:
        limits.append(dict(bytes=node, source="Slurm allocation"))
    host = physical_bytes() if physical is None else physical
    candidates = limits + ([dict(bytes=host, source="host physical RAM")] if host else [])
    detected = min(candidates, key=lambda item: item["bytes"]) if candidates else None
    # Host RAM alone is not an allocation on a shared machine. Without an
    # explicit budget or job limit, stay conservative instead of using it all.
    ceiling = int(.8*detected["bytes"]) if detected else None
    requested = int(requested_gb*GIB) if requested_gb is not None else None
    working = requested if requested is not None else ceiling if limits and ceiling else 4*GIB
    if ceiling:
        working = min(working, ceiling)
    if working < 256*MIB:
        raise ValueError("Detected working-memory budget is below 0.25 GiB")
    monitors = [item for item in limits if item.get("usage")]
    monitor = min(monitors, key=lambda item: item["bytes"]) if monitors else None
    return dict(working_bytes=working, working_gb=working/GIB,
                requested_gb=requested_gb, limiting_source=detected["source"] if detected else "conservative fallback",
                detected_limit_bytes=detected["bytes"] if detected else None,
                headroom_fraction=.2, monitor=monitor)


def plan_workers(budget, samples, threads, batch_size):
    working = budget["working_bytes"]
    coordinator = min(160*MIB, int(.15*working))
    active = 160*samples*samples + 80*samples*batch_size + 48*MIB
    reserve = max(32*MIB, int(.1*working))
    available = working-coordinator-threads*active-reserve
    if available < 0:
        required = coordinator+threads*active+reserve
        raise ValueError(f"Working budget {working/GIB:.2f} GiB cannot fit {threads} workers; "
                         f"estimated minimum {required/GIB:.2f} GiB. Reduce --threads or --batch-size.")
    cache = min(int(.6*working), available)
    return dict(cache_bytes_per_worker=cache//threads, total_cache_bytes=cache,
                active_bytes_per_worker=active, worker_target_bytes=(working-coordinator)//threads,
                total_working_bytes=working, monitor=budget["monitor"],
                query_buffer_pairs=max(1024,min(65536,int(.01*working)//800)),
                frontier_tasks=max(128,min(4096,threads*64)))


def working_usage(monitor):
    if not monitor:
        return None
    usage = _positive(_read(monitor["usage"]))
    if usage is None:
        return None
    # Reclaimable inactive file cache should not force eviction of useful model
    # arrays just because memory-mapped/checkpoint files were read recently.
    stats = dict(line.split() for line in _read(monitor["stat"]).splitlines() if len(line.split()) == 2)
    # v1 usage includes descendants: use its matching hierarchical counter.
    # v2 exposes the hierarchical value without the total_ prefix.
    inactive = int(stats.get("total_inactive_file", stats.get("inactive_file", "0")))
    return max(0, usage-inactive)


def process_rss():
    if sys.platform.startswith("linux"):
        fields = _read("/proc/self/statm").split()
        if len(fields) > 1:
            return int(fields[1])*os.sysconf("SC_PAGE_SIZE")
    return None


class ByteLRU:
    """An eviction cache whose charge is supplied in bytes, not entry counts."""
    def __init__(self, limit):
        self.limit = max(0, int(limit))
        self.bytes = self.peak_bytes = self.evictions = self.refused = 0
        self.items = OrderedDict()

    def get(self, key):
        value, charge = self.items.pop(key)
        self.items[key] = value, charge
        return value

    def __contains__(self, key):
        return key in self.items

    def trim(self, limit=None):
        if limit is not None:
            self.limit = max(0, int(limit))
        while self.bytes > self.limit and self.items:
            _, (_, charge) = self.items.popitem(last=False)
            self.bytes -= charge
            self.evictions += 1

    def put(self, key, value, charge):
        if key in self.items:
            _, old = self.items.pop(key)
            self.bytes -= old
        if charge <= self.limit:
            self.items[key] = value, charge
            self.bytes += charge
        else:
            self.refused += 1
        self.peak_bytes = max(self.peak_bytes, self.bytes)
        self.trim()
