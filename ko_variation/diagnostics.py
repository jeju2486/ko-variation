from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
import os
import sys
from typing import Any, Iterable, Mapping


from .memory_budget import BLAS_ENVIRONMENT_VARIABLES, guard_blas_environment


@dataclass(frozen=True)
class NativeLibrary:
    user_api: str
    internal_api: str
    prefix: str
    version: str
    num_threads: int

    @property
    def label(self) -> str:
        implementation = self.internal_api or self.prefix or "unknown"
        version = self.version or "unknown"
        return f"{self.user_api}:{implementation}:{version}:{self.num_threads}"


@dataclass(frozen=True)
class ParallelRuntime:
    worker_processes: int
    logical_cpus: int
    effective_blas_threads: int
    estimated_native_threads: int
    oversubscribed: bool
    libraries: tuple[NativeLibrary, ...]
    environment: tuple[tuple[str, str], ...]
    warnings: tuple[str, ...]

    def summary_fields(self) -> dict[str, Any]:
        return {
            "runtime_cpu_count": self.logical_cpus,
            "runtime_worker_processes": self.worker_processes,
            "runtime_blas_threads": self.effective_blas_threads,
            "runtime_estimated_native_threads": self.estimated_native_threads,
            "runtime_blas_oversubscribed": int(self.oversubscribed),
            "runtime_blas_libraries": ",".join(library.label for library in self.libraries) or "unresolved",
            "runtime_blas_environment": ",".join(f"{key}={value}" for key, value in self.environment),
            "runtime_warnings": ",".join(self.warnings) or "none",
        }


@dataclass(frozen=True)
class ScanMemoryEstimate:
    available_bytes: int
    shared_parent_bytes: int
    private_bytes_per_worker: int
    requested_private_worker_bytes: int
    recommended_max_workers_by_memory: int
    start_method: str
    warnings: tuple[str, ...]

    def summary_fields(self) -> dict[str, Any]:
        return {
            "available_bytes": self.available_bytes,
            "shared_parent_bytes": self.shared_parent_bytes,
            "private_bytes_per_worker": self.private_bytes_per_worker,
            "requested_private_worker_bytes": self.requested_private_worker_bytes,
            "recommended_max_workers_by_memory": self.recommended_max_workers_by_memory,
            "multiprocessing_start_method": self.start_method,
            "warnings": ",".join(self.warnings) or "none",
        }


def available_memory_bytes() -> int:
    """Best-effort available physical memory using only the standard library."""

    if sys.platform == "win32":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.available_physical)
        except (AttributeError, OSError, ValueError):
            pass
    try:
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        if pages > 0 and page_size > 0:
            return pages * page_size
    except (AttributeError, OSError, ValueError):
        pass
    return 0


def estimate_scan_memory(
    *,
    n_samples: int,
    n_loci: int,
    worker_processes: int,
    predictor_batch_size: int,
    logical_cpus: int | None = None,
    available_bytes: int | None = None,
    start_method: str | None = None,
) -> ScanMemoryEstimate:
    """Conservatively estimate dense-solver worker memory.

    This is a safety bound rather than a performance model. It intentionally
    leaves 40% of currently available RAM for pandas tables, input parsing,
    operating-system cache, and numerical-library workspace not represented by
    the simple dense-array count.
    """

    if n_samples < 2 or n_loci < 1 or worker_processes < 1:
        raise ValueError("Sample, locus and worker counts must be positive")
    if predictor_batch_size < 1:
        raise ValueError("predictor_batch_size must be positive")
    available = int(available_memory_bytes() if available_bytes is None else available_bytes)
    method = start_method
    if method is None:
        method = "fork" if "fork" in mp.get_all_start_methods() else mp.get_start_method()

    float_bytes = 8
    dense_matrix_bytes = n_samples * n_samples * float_bytes
    batch_workspace_bytes = n_samples * predictor_batch_size * float_bytes * 3
    # Weighted K, eigenvectors, solver copy and LAPACK work arrays. Actual use
    # is backend-dependent, so this remains deliberately conservative.
    private_per_worker = 6 * dense_matrix_bytes + batch_workspace_bytes
    shared_parent = n_samples * n_loci + 2 * dense_matrix_bytes
    if method != "fork":
        # Spawn workers receive independent copies of X, K and the reusable
        # eigensystem through the pool initializer.
        private_per_worker += shared_parent

    requested = private_per_worker * worker_processes
    cpus = max(1, int(logical_cpus or os.cpu_count() or 1))
    if available > 0:
        memory_budget = int(available * 0.60)
        recommended = max(1, min(cpus, memory_budget // max(1, private_per_worker)))
    else:
        recommended = cpus

    warnings: list[str] = []
    if available > 0 and requested > int(available * 0.60):
        warnings.append("dense_worker_memory_pressure")
    if method != "fork" and worker_processes > 1:
        warnings.append("spawn_duplicates_input_arrays")
    if worker_processes > recommended:
        warnings.append("workers_exceed_memory_recommendation")

    return ScanMemoryEstimate(
        available_bytes=available,
        shared_parent_bytes=shared_parent,
        private_bytes_per_worker=private_per_worker,
        requested_private_worker_bytes=requested,
        recommended_max_workers_by_memory=recommended,
        start_method=str(method),
        warnings=tuple(warnings),
    )


def _positive_environment_threads(environment: Mapping[str, str]) -> list[int]:
    values: list[int] = []
    for variable in BLAS_ENVIRONMENT_VARIABLES:
        raw = environment.get(variable, "").strip()
        try:
            value = int(raw)
        except ValueError:
            continue
        if value > 0:
            values.append(value)
    return values


def assess_parallel_runtime(
    worker_processes: int,
    *,
    logical_cpus: int | None,
    libraries: Iterable[NativeLibrary],
    environment: Mapping[str, str],
) -> ParallelRuntime:
    """Combine process and native-thread settings into actionable diagnostics."""

    if worker_processes < 1:
        raise ValueError("worker_processes must be positive")

    library_tuple = tuple(libraries)
    blas_threads = [
        library.num_threads
        for library in library_tuple
        if library.user_api == "blas" and library.num_threads > 0
    ]
    thread_source = blas_threads or _positive_environment_threads(environment)
    effective_blas_threads = max(thread_source, default=1)
    estimated_native_threads = worker_processes * effective_blas_threads
    cpus = max(1, int(logical_cpus or 1))
    oversubscribed = estimated_native_threads > cpus

    warnings: list[str] = []
    if worker_processes > 1 and effective_blas_threads > 1:
        warnings.append("nested_blas_parallelism")
    if oversubscribed:
        warnings.append("native_thread_oversubscription")

    return ParallelRuntime(
        worker_processes=worker_processes,
        logical_cpus=cpus,
        effective_blas_threads=effective_blas_threads,
        estimated_native_threads=estimated_native_threads,
        oversubscribed=oversubscribed,
        libraries=library_tuple,
        environment=tuple((variable, environment.get(variable, "unset")) for variable in BLAS_ENVIRONMENT_VARIABLES),
        warnings=tuple(warnings),
    )


def collect_parallel_runtime(worker_processes: int) -> ParallelRuntime:
    """Inspect loaded BLAS/OpenMP libraries and the process-pool configuration."""

    loaded: list[NativeLibrary] = []
    try:
        from threadpoolctl import threadpool_info

        for information in threadpool_info():
            num_threads = information.get("num_threads")
            if not isinstance(num_threads, int):
                continue
            loaded.append(
                NativeLibrary(
                    user_api=str(information.get("user_api") or "unknown"),
                    internal_api=str(information.get("internal_api") or "unknown"),
                    prefix=str(information.get("prefix") or "unknown"),
                    version=str(information.get("version") or "unknown"),
                    num_threads=num_threads,
                )
            )
    except (ImportError, RuntimeError):
        # Environment reporting remains useful if library introspection is not
        # available in an unusual installation.
        pass

    return assess_parallel_runtime(
        worker_processes,
        logical_cpus=os.cpu_count(),
        libraries=loaded,
        environment=os.environ,
    )
