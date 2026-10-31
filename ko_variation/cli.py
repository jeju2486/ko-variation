from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

# Set conservative defaults before importing NumPy/SciPy so response-level
# worker processes do not silently multiply a second native thread pool. User
# settings remain authoritative and are inspected and reported at runtime.
from .memory_budget import guard_blas_environment

guard_blas_environment(default_threads=1)

from . import __version__


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "KOVAR: unordered pangenome covariation scanning with a "
            "phylogeny-adjusted logistic mixed-model score test. "
            "For direct unitig discovery without --pairs: ko-variation discover --help."
        )
    )
    parser.add_argument("--fasta", required=True, help="Binary fake FASTA with A=0 and C=1.")
    parser.add_argument("--pairs", required=True, help="Zero-based candidate-locus pairs (u and v columns).")
    parser.add_argument("--tree", required=True, help="Rooted Newick tree with branch lengths.")
    parser.add_argument("--out", required=True, help="Output directory.")

    input_group = parser.add_argument_group("input and filtering")
    input_group.add_argument(
        "--tree-missing-samples", choices=["error", "drop"], default="error",
        help="Missing FASTA isolates in tree: error [default] or explicitly drop them.",
    )
    input_group.add_argument(
        "--min-maf",
        type=float,
        default=0.05,
        help="Minimum minor-state frequency for predictor and response [0.05].",
    )
    input_group.add_argument(
        "--min-cell-count",
        type=int,
        default=1,
        help="Minimum count in each 2x2 table cell [1].",
    )

    model_group = parser.add_argument_group("logistic mixed model")
    model_group.add_argument(
        "--spa-mode",
        choices=["off", "auto", "always"],
        default="auto",
        help="Saddlepoint calibration policy [auto]. Not a SAIGE reproduction.",
    )
    model_group.add_argument("--threads", type=int, default=1, help="Response-wise worker processes [1].")

    output_group = parser.add_argument_group("execution and output")
    output_group.add_argument("--no-progress", action="store_true", help="Suppress progress messages.")
    output_group.add_argument("--overwrite", action="store_true", help="Write into a non-empty output directory.")
    output_group.add_argument(
        "--checkpoint-dir",
        help="Checkpoint directory [OUT/.kovar_checkpoint].",
    )
    output_group.add_argument(
        "--resume",
        action="store_true",
        help="Resume an exactly matching checkpoint.",
    )
    output_group.add_argument("--version", action="version", version=f"KO-Variation {__version__}")
    args = parser.parse_args(argv)

    if not 0.0 <= args.min_maf <= 0.5:
        parser.error("--min-maf must be between 0 and 0.5")
    if args.min_cell_count < 0:
        parser.error("--min-cell-count must be non-negative")
    if args.threads < 1:
        parser.error("--threads must be positive")
    # Low-level legacy candidate-scan tuning remains internal.
    args.presence_char = "C"
    args.absence_char = "A"
    args.tree_missing_length = "error"
    args.null_max_iter = 100
    args.null_tolerance = 1e-7
    args.worker_chunk_size = 1
    args.predictor_batch_size = 256
    args.checkpoint_every = 100
    return args


def _step(started: float, label: str, message: str = "") -> None:
    suffix = f" {message}" if message else ""
    sys.stderr.write(f"[KOVAR] step={label} elapsed={time.time() - started:.1f}s{suffix}\n")
    sys.stderr.flush()


def _atomic_dataframe_write(frame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, sep="\t", index=False, na_rep="NA")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_text_write(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _checkpoint_identity(
    args,
    *,
    input_fingerprints: dict[str, object],
    n_samples: int,
    n_loci: int,
    n_pairs: int,
    blas_libraries: str,
) -> dict[str, object]:
    """Scientific inputs/settings that must match before work can be reused."""
    import numpy as np
    import pandas as pd
    import scipy

    return {
        "tool": "KO-Variation",
        "version": __version__,
        "inputs": input_fingerprints,
        "dimensions": {
            "n_samples": n_samples,
            "n_loci": n_loci,
            "n_pairs": n_pairs,
        },
        "numerical_environment": {
            "python": ".".join(str(value) for value in sys.version_info[:3]),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "blas_libraries": blas_libraries,
        },
        "settings": {
            "binary_encoding": "A=0,C=1",
            "tree_missing_length": args.tree_missing_length,
            **({"tree_missing_samples": args.tree_missing_samples}
               if args.tree_missing_samples != "error" else {}),
            "min_maf": args.min_maf,
            "min_cell_count": args.min_cell_count,
            "graph_count_fraction": 0.05,
            "spa_mode": args.spa_mode,
            "null_max_iter": args.null_max_iter,
            "null_tolerance": args.null_tolerance,
            "predictor_batch_size": args.predictor_batch_size,
        },
    }


def _metadata_rows(
    category: str,
    fields: dict[str, object],
    *,
    unit: str = "",
) -> list[dict[str, object]]:
    return [
        {"category": category, "metric": key, "value": value, "unit": unit}
        for key, value in sorted(fields.items())
    ]


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "discover":
        from .discovery import main as discover
        return discover(arguments[1:])
    started = time.time()
    args = parse_args(arguments)
    # Discovery does not use the legacy pair-table scanner/checkpoint stack.
    from .diagnostics import collect_parallel_runtime, estimate_scan_memory
    import numpy as np
    import pandas as pd
    from .checkpoint import CheckpointError, CheckpointStore, fingerprint_inputs
    from .glmm import prepare_kinship
    from .io_utils import read_fake_fasta, read_pairs, validate_pairs
    from .kinship import build_tree_covariance
    from .scan import ScanConfig, ScanMetrics, scan_pairs_glmm
    runtime = collect_parallel_runtime(args.threads)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.overwrite and not args.resume:
        raise SystemExit(
            f"Output directory is not empty: {out}. Use --overwrite for a new run "
            "or --resume for an exactly matching checkpoint."
        )
    out.mkdir(parents=True, exist_ok=True)
    progress = not args.no_progress

    _step(
        started,
        "runtime",
        f"workers={runtime.worker_processes} blas_threads={runtime.effective_blas_threads} "
        f"estimated_native_threads={runtime.estimated_native_threads} "
        f"logical_cpus={runtime.logical_cpus}",
    )
    for warning in runtime.warnings:
        sys.stderr.write(
            f"[KOVAR] warning={warning} workers={runtime.worker_processes} "
            f"blas_threads={runtime.effective_blas_threads}; "
            "prefer one BLAS thread per response worker unless benchmarked otherwise\n"
        )
    sys.stderr.flush()

    stage_seconds: dict[str, float] = {}

    stage_started = time.monotonic()
    _step(started, "read_fasta")
    fasta = read_fake_fasta(
        args.fasta,
        presence_char=args.presence_char,
        absence_char=args.absence_char,
    )
    X = fasta.X
    n_samples, n_loci = X.shape
    stage_seconds["read_fasta"] = time.monotonic() - stage_started
    _step(started, "read_fasta", f"samples={n_samples} loci={n_loci}")

    stage_started = time.monotonic()
    _step(started, "build_covariance", "source=tree")
    kinship = build_tree_covariance(
        args.tree,
        fasta.sample_names,
        dtype="float64",
        missing_length=args.tree_missing_length,
        missing_samples=args.tree_missing_samples,
    )
    input_sample_names = fasta.sample_names
    if kinship.excluded_samples:
        X = X[kinship.sample_indices, :]
        fasta.X = X
        fasta.sample_names = [input_sample_names[i] for i in kinship.sample_indices]
    n_samples = X.shape[0]
    K, kdiag = prepare_kinship(kinship.K)
    del kinship.K
    stage_seconds["build_covariance"] = time.monotonic() - stage_started
    _step(
        started,
        "build_covariance",
        f"source={kinship.source} rank={kdiag.rank}/{n_samples} "
        f"eig_min={kdiag.eigen_min:.3g} eig_max={kdiag.eigen_max:.3g}",
    )

    stage_started = time.monotonic()
    _step(started, "read_pairs")
    pairs = validate_pairs(
        read_pairs(args.pairs),
        n_loci=n_loci,
        drop_invalid=False,
    )
    if pairs.empty:
        raise SystemExit("No candidate pairs remain after input validation")
    stage_seconds["read_pairs"] = time.monotonic() - stage_started
    _step(started, "read_pairs", f"pairs={len(pairs)}")

    memory_estimate = estimate_scan_memory(
        n_samples=n_samples,
        n_loci=n_loci,
        worker_processes=args.threads,
        predictor_batch_size=args.predictor_batch_size,
        logical_cpus=runtime.logical_cpus,
    )
    _step(
        started,
        "memory",
        f"available_gib={memory_estimate.available_bytes / 2**30:.2f} "
        f"private_worker_mib={memory_estimate.private_bytes_per_worker / 2**20:.1f} "
        f"requested_private_gib={memory_estimate.requested_private_worker_bytes / 2**30:.2f} "
        f"recommended_max_workers={memory_estimate.recommended_max_workers_by_memory} "
        f"start_method={memory_estimate.start_method}",
    )
    for warning in memory_estimate.warnings:
        sys.stderr.write(
            f"[KOVAR] warning={warning} requested_workers={args.threads} "
            f"memory_recommended_max={memory_estimate.recommended_max_workers_by_memory}\n"
        )
    sys.stderr.flush()

    stage_started = time.monotonic()
    _step(started, "checkpoint_fingerprint")
    try:
        input_fingerprints = fingerprint_inputs([
            ("fasta", args.fasta), ("pairs", args.pairs), ("tree", args.tree)
        ])
        identity = _checkpoint_identity(
            args,
            input_fingerprints=input_fingerprints,
            n_samples=n_samples,
            n_loci=n_loci,
            n_pairs=len(pairs),
            blas_libraries=str(runtime.summary_fields().get("runtime_blas_libraries", "unresolved")),
        )
        checkpoint_path = Path(args.checkpoint_dir) if args.checkpoint_dir else out / ".kovar_checkpoint"
        resolved_checkpoint = checkpoint_path.resolve()
        resolved_out = out.resolve()
        if resolved_checkpoint == resolved_out or resolved_out.is_relative_to(resolved_checkpoint):
            raise CheckpointError("Checkpoint directory must not be the output directory or one of its parents")
        checkpoint = CheckpointStore(
            checkpoint_path, identity, tool_version=__version__,
            every=args.checkpoint_every, resume=args.resume,
        )
    except CheckpointError as exc:
        raise SystemExit(f"Checkpoint error: {exc}") from exc
    stage_seconds["checkpoint_fingerprint"] = time.monotonic() - stage_started
    _step(started, "checkpoint_fingerprint", f"resume={int(args.resume)} directory={checkpoint.root}")

    config = ScanConfig(
        min_maf=args.min_maf,
        min_cell_count=args.min_cell_count,
        graph_count_fraction=0.05,
        spa_mode=args.spa_mode,
        threads=args.threads,
        worker_chunk_size=args.worker_chunk_size,
        predictor_batch_size=args.predictor_batch_size,
        null_max_iter=args.null_max_iter,
        null_tolerance=args.null_tolerance,
        progress=progress,
    )
    _step(
        started,
        "pair_scan",
        f"unordered=1 min_maf={args.min_maf:g} graph_count_fraction=0.05 "
        f"min_cell_count={args.min_cell_count} spa={args.spa_mode}",
    )
    scan_metrics = ScanMetrics()
    stage_started = time.monotonic()
    try:
        results, response_models = scan_pairs_glmm(
            pairs,
            X,
            K,
            config,
            checkpoint=checkpoint,
            metrics=scan_metrics,
        )
    except CheckpointError as exc:
        raise SystemExit(f"Checkpoint error: {exc}") from exc
    stage_seconds["pair_scan"] = time.monotonic() - stage_started

    cache_diagnostics: dict[str, object] = {
        "response_pattern_cache": "exact_identical_complement",
        "n_response_loci": len(response_models),
    }
    required_cache_columns = {
        "response_pattern_id",
        "response_pattern_flipped",
        "response_pattern_size",
        "null_fit_reused",
    }
    if required_cache_columns.issubset(response_models.columns):
        reused = response_models["null_fit_reused"].fillna(False).astype(bool)
        flipped = response_models["response_pattern_flipped"].fillna(False).astype(bool)
        cache_diagnostics.update(
            {
                "n_unique_response_patterns": int(response_models["response_pattern_id"].nunique()),
                "n_null_glmm_fits": int((~reused).sum()),
                "n_null_glmm_reused": int(reused.sum()),
                "n_complement_response_loci": int(flipped.sum()),
                "max_response_pattern_size": (
                    int(response_models["response_pattern_size"].max()) if len(response_models) else 0
                ),
                "response_pattern_reuse_fraction": float(reused.mean()) if len(reused) else 0.0,
            }
        )
        _step(
            started,
            "pair_scan",
            f"duration={stage_seconds['pair_scan']:.1f}s "
            f"responses={len(response_models)} "
            f"unique_patterns={cache_diagnostics['n_unique_response_patterns']} "
            f"null_fits_reused={cache_diagnostics['n_null_glmm_reused']}",
        )
    else:
        cache_diagnostics["response_pattern_cache"] = "unavailable"
        _step(started, "pair_scan", f"duration={stage_seconds['pair_scan']:.1f}s")

    stage_started = time.monotonic()
    _step(started, "write_results")
    result_path = out / "ko_variation.tsv"
    model_path = out / "response_models.tsv"
    summary_path = out / "run_summary.txt"
    metadata_path = out / "execution_metadata.tsv"
    _atomic_dataframe_write(results, result_path)
    _atomic_dataframe_write(response_models, model_path)
    retained_samples = set(fasta.sample_names)
    _atomic_dataframe_write(pd.DataFrame({
        "sample": input_sample_names,
        "status": ["included" if name in retained_samples else "excluded_missing_tree"
                   for name in input_sample_names],
    }), out / "sample_inclusion.tsv")
    stage_seconds["write_primary_outputs"] = time.monotonic() - stage_started
    tested = int(np.isfinite(results["p_primary"].to_numpy(dtype=np.float64)).sum())

    summary_lines = [
        f"version\t{__version__}",
        "tool\tKO-Variation",
        "acronym\tKOVAR",
        "release_status\texperimental",
        "model\tunordered_logistic_mixed_model_pql_score",
        "interpretation\tphylogeny_adjusted_covariation_significance_not_effect_size",
        f"n_samples\t{n_samples}",
        f"n_input_samples\t{len(input_sample_names)}",
        f"n_excluded_samples\t{len(kinship.excluded_samples)}",
        f"tree_missing_samples\t{args.tree_missing_samples}",
        "sample_inclusion\tsample_inclusion.tsv",
        f"n_loci\t{n_loci}",
        f"n_input_pairs\t{len(pairs)}",
        f"n_pair_rows\t{len(results)}",
        f"n_tests\t{tested}",
        "unordered_pairs\t1",
        "graph_count_fraction\t0.05",
        f"min_maf\t{args.min_maf}",
        f"min_cell_count\t{args.min_cell_count}",
        f"spa_mode\t{args.spa_mode}",
        f"kinship_source\t{kinship.source}",
        f"tree\t{args.tree}",
        f"threads\t{args.threads}",
        "solver_backend\tdense_weighted_eigen_pql",
        "tau_profile_backend\texact_spectral_coordinates",
        "checkpoint_enabled\t1",
        f"checkpoint_resumed\t{int(checkpoint.statistics.resumed)}",
        "checkpoint_retained\t0",
        f"execution_metadata\t{metadata_path.name}",
    ]
    _atomic_text_write(summary_path, "\n".join(summary_lines) + "\n")

    metadata_rows: list[dict[str, object]] = []
    metadata_rows.extend(_metadata_rows("configuration", {
        "tree_missing_samples": args.tree_missing_samples,
        "unordered_pairs": 1,
        "graph_count_fraction": 0.05,
        "min_maf": args.min_maf,
        "min_cell_count": args.min_cell_count,
        "spa_mode": args.spa_mode,
        "null_max_iter": args.null_max_iter,
        "null_tolerance": args.null_tolerance,
        "threads": args.threads,
        "worker_chunk_size": args.worker_chunk_size,
        "predictor_batch_size": args.predictor_batch_size,
        "checkpoint_every": args.checkpoint_every,
    }))
    metadata_rows.extend(_metadata_rows("runtime", runtime.summary_fields()))
    metadata_rows.extend(_metadata_rows("memory", memory_estimate.summary_fields()))
    metadata_rows.extend(_metadata_rows("response_cache", cache_diagnostics))
    metadata_rows.extend(_metadata_rows("stage_timing", stage_seconds, unit="seconds"))
    metadata_rows.extend(
        _metadata_rows("scan_timing", scan_metrics.seconds, unit="seconds")
    )
    metadata_rows.extend(_metadata_rows("scan_counts", scan_metrics.counts, unit="count"))
    metadata_rows.extend(_metadata_rows("kinship", {
        "rank": kdiag.rank,
        "eigen_min": kdiag.eigen_min,
        "eigen_max": kdiag.eigen_max,
        "roundoff_correction": kdiag.roundoff_correction,
        "mean_diag_before_norm": kinship.mean_diag_before_norm,
        "loci_used": kinship.n_loci_used,
        **{f"detail_{key}": value for key, value in kinship.details.items()},
    }))
    metadata_rows.extend(_metadata_rows("checkpoint", checkpoint.metadata_fields()))
    _atomic_dataframe_write(
        pd.DataFrame(metadata_rows),
        metadata_path,
    )

    sys.stderr.write("[KOVAR] result_status_counts\n")
    sys.stderr.write(results["status"].fillna("NA").value_counts().to_string() + "\n")
    checkpoint_finalized = checkpoint.complete(cleanup=True)
    if not checkpoint_finalized:
        sys.stderr.write(
            f"[KOVAR] warning=checkpoint_cleanup_failed directory={checkpoint.root}; "
            "final outputs are complete and the checkpoint is marked complete\n"
        )
        sys.stderr.flush()
    _step(
        started,
        "complete",
        f"results={result_path} response_models={model_path} metadata={metadata_path}",
    )


if __name__ == "__main__":
    main()
