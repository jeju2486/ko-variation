"""Benchmark MI and tree-PC residual screening on truth-blind pattern panels.

Every unordered pair in each sampled panel reaches the unchanged KOVAR CLI.
No original PAN-GWES shortlist, annotation, phenotype or KOVAR p-value is used
to choose the panel. This is an all-pair computational benchmark, not a distal
network: new pairs have no verified genomic distances or PAN-GWES graph counts.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd
import scipy
from scipy import linalg
from scipy.special import xlogy
from threadpoolctl import threadpool_limits

from ko_variation import __version__
from ko_variation.checkpoint import fingerprint_file
from ko_variation.glmm import prepare_kinship
from ko_variation.kinship import build_tree_covariance
from ko_variation.phenotype import binary_fasta, filter_cohort, json_write
from ko_variation.amr_report import read_tsv
from ko_variation.workflow_cache import stage_identity, cache_matches, save_cache

ALGORITHM = "tree-pc-screen-benchmark-2"
POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1)


def log(message):
    print(f"[screen benchmark] {message}", flush=True)


def table(path, frame):
    """Publish complete tables only; never reuse a half-written result."""
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    frame.to_csv(temporary, sep="\t", index=False, na_rep="NA")
    temporary.replace(path)


def npz(path, **arrays):
    temporary = Path(str(path) + ".partial")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def versions():
    return dict(python=platform.python_version(), numpy=np.__version__,
                scipy=scipy.__version__, pandas=pd.__version__, kovar=__version__)


def code_identity():
    # Shared numerical code lives in ko_variation; benchmark code is optional.
    import ko_variation
    core = Path(ko_variation.__file__).parent
    files = [Path(__file__), Path(__file__).with_name("screen_benchmark_extend.py")]
    files += [core/name for name in ("glmm.py", "kinship.py", "io_utils.py", "phenotype.py")]
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


@contextmanager
def measure(directory, stage, records):
    """Wall time plus sampled sum of RSS for this process and its children.

    Sum-RSS can double-count shared pages; it is not unique physical memory.
    psutil is optional. Missing measurements stay NA, never zero.
    """
    stop, peak = threading.Event(), [None]
    try:
        import psutil
        process = psutil.Process()

        def sample():
            while not stop.is_set():
                total = 0
                for child in [process] + process.children(recursive=True):
                    try:
                        total += child.memory_info().rss
                    except psutil.Error:
                        pass
                peak[0] = max(peak[0] or 0, total)
                stop.wait(0.25)
        worker = threading.Thread(target=sample, daemon=True)
        worker.start()
    except ImportError:
        worker = None
    started = time.perf_counter()
    try:
        yield
    finally:
        stop.set()
        if worker:
            worker.join()
        records.append(dict(stage=stage, wall_seconds=time.perf_counter()-started,
                            sampled_sum_rss_bytes=peak[0], source="measured_run"))
        table(Path(directory)/"timings.tsv", pd.DataFrame(records))


def input_files(args):
    return [Path(value) for value in [args.fasta, args.tree, args.metadata,
            args.sample_map, args.sample_inclusion, args.locus_map, args.sample_weights] if value]


def prepare(args):
    out = args.out
    manifest = out/"preparation.json"
    settings = {key: getattr(args, key) for key in ["patterns", "replicates", "seed", "pcs",
                "min_maf", "mi_weighting", "weight_distance", "species", "exclude_sample",
                "tree_missing_samples", "rooting", "fasta_columns_are_locus_ids"]}
    settings.update(algorithm=ALGORITHM, versions=versions(), code=code_identity())
    identity = stage_identity("screen_benchmark_prepare", input_files(args), settings)
    products = [out/name for name in ["preparation.npz", "pattern_mapping.tsv.gz",
                "patterns.tsv", "sample_inclusion.tsv", "analysis.treefile", "sample_weights.tsv",
                "spydrpick_weights.txt", "tree_axes.tsv", "preparation_timings.tsv"]]
    if cache_matches(manifest, identity, products):
        log("reuse cohort, unique-pattern mapping, global weights, tree axes and sampled panels")
        return json.loads(manifest.read_text(encoding="utf-8"))
    if manifest.exists():
        raise ValueError("Preparation inputs/settings changed. Use a different --out directory; old results are preserved.")
    records = []
    with measure(out, "read_cohort_and_tree", records):
        from Bio import Phylo
        samples, X = binary_fasta(args.fasta)
        original = list(samples)
        excluded = set(args.exclude_sample)
        if excluded-set(samples):
            raise ValueError(f"Explicit exclusions absent from FASTA: {sorted(excluded-set(samples))}")
        if args.metadata:
            metadata = read_tsv(args.metadata, ["id", "isolate", "species"])
            if args.sample_map:
                mapping = read_tsv(args.sample_map, ["metadata_id", "sample"])
                known = {row["sample"] for row in mapping}
            else:
                known = {row["id"]+"_"+row["isolate"].replace(".", "_") for row in metadata}
            if unknown := set(samples)-known-excluded:
                raise ValueError(f"Metadata does not identify {len(unknown)} FASTA samples: {sorted(unknown)[:5]}; supply --sample-map")
            samples, X, excluded = filter_cohort(args, samples, X, out)
        else:
            keep = [i for i, sample in enumerate(samples) if sample not in excluded]
            samples, X = [samples[i] for i in keep], X[keep]
        if args.sample_inclusion:
            inclusion = read_tsv(args.sample_inclusion, ["sample", "status"])
            if len({row["sample"] for row in inclusion}) != len(inclusion):
                raise ValueError("Duplicate sample in --sample-inclusion")
            status = {row["sample"]: row["status"] for row in inclusion}
            if set(samples)-set(status):
                raise ValueError("--sample-inclusion must describe every remaining FASTA sample")
            keep = [i for i, sample in enumerate(samples) if status[sample] == "included"]
            samples, X = [samples[i] for i in keep], X[keep]
        tree = Phylo.read(args.tree, "newick")
        tips = [tip.name for tip in tree.get_terminals()]
        if None in tips or len(tips) != len(set(tips)):
            raise ValueError("Tree tip IDs must be nonempty and unique")
        missing = set(samples)-set(tips)
        if missing and args.tree_missing_samples == "error":
            raise ValueError(f"Tree misses {len(missing)} samples: {sorted(missing)[:5]}")
        if missing:
            keep = [i for i, sample in enumerate(samples) if sample not in missing]
            samples, X = [samples[i] for i in keep], X[keep]
        if len(samples) < max(args.pcs)+3:
            raise ValueError("Not enough retained isolates for the requested number of tree PCs")
        for node in tree.find_clades():
            if node is not tree.root and (node.branch_length is None or
                    not np.isfinite(node.branch_length) or node.branch_length < 0):
                raise ValueError("Tree needs finite, nonnegative branch lengths")
        for name in sorted(set(tips)-set(samples)):
            tree.prune(name)
        if args.rooting == "midpoint":
            tree.root_at_midpoint()
        tree.root.branch_length = 0.0
        Phylo.write(tree, out/"analysis.treefile", "newick", format_branch_length="%.17g")
        table(out/"sample_inclusion.tsv", pd.DataFrame([
            dict(sample=name, status="included" if name in set(samples) else "excluded",
                 reason="" if name in set(samples) else "metadata_explicit_cohort_or_tree_policy")
            for name in original]))
        loci = np.arange(X.shape[1], dtype=np.int64)
        if args.locus_map:
            mapping = read_tsv(args.locus_map, ["fasta_column", "locus"])
            by_column = {int(row["fasta_column"]): int(row["locus"]) for row in mapping}
            if len(by_column) != len(mapping) or set(by_column) != set(range(X.shape[1])):
                raise ValueError("Locus map must cover every zero-based FASTA column exactly once")
            loci = np.array([by_column[i] for i in range(X.shape[1])], dtype=np.int64)
        if len(np.unique(loci)) != len(loci) or np.any(loci < 0):
            raise ValueError("Original locus IDs must be unique and nonnegative")
        n = len(samples)
        present = X.sum(axis=0, dtype=np.int64)
        maf = np.minimum(present, n-present)/n
        eligible = np.flatnonzero((maf+1e-12 >= args.min_maf) & (present > 0) & (present < n))
        if not len(eligible):
            raise ValueError("No variable loci meet the frequency threshold")
        log(f"cohort: {n} isolates; {len(eligible):,}/{len(loci):,} columns pass MAF")
    with measure(out, "exact_pattern_grouping", records):
        # Exact states only. Complements remain distinct. Every original unitig
        # retains a mapping; a pattern is not a gene or a genomic coordinate.
        packed = np.packbits(X[:, eligible], axis=0, bitorder="little").T.copy()
        unique, first, inverse, multiplicity = np.unique(
            packed, axis=0, return_index=True, return_inverse=True, return_counts=True)
        representatives = eligible[first]
        U = len(first)
        if args.patterns > U:
            raise ValueError(f"Requested {args.patterns} unique patterns, but only {U} pass filtering")
        with gzip.open(out/"pattern_mapping.tsv.gz", "wt", encoding="utf-8", compresslevel=1) as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["fasta_column", "locus", "pattern_id", "representative_locus"])
            for offset, column in enumerate(eligible):
                pattern = inverse[offset]
                writer.writerow([column, loci[column], pattern, loci[representatives[pattern]]])
        stats_frame = pd.DataFrame(dict(pattern_id=np.arange(U),
            representative_locus=loci[representatives], fasta_column=representatives,
            n_unitigs=multiplicity, n_present=present[representatives], maf=maf[representatives]))
        table(out/"patterns.tsv", stats_frame)
        log(f"exact patterns: {U:,}; sampling {args.patterns:,} per panel without looking at MI/KOVAR")
    with measure(out, "global_mi_weights", records):
        if args.sample_weights:
            supplied = read_tsv(args.sample_weights, ["sample", "weight"])
            weights_by_sample = {row["sample"]: float(row["weight"]) for row in supplied}
            if len(weights_by_sample) != len(supplied) or set(samples) != set(weights_by_sample):
                raise ValueError("Supplied sample weights must cover retained samples exactly once")
            weights = np.array([weights_by_sample[sample] for sample in samples])
            weighting = "supplied_sample_weights"
        elif args.mi_weighting == "sequence":
            # Use every frequency-eligible original column, BEFORE deduplication
            # or panel selection. Unitig multiplicity therefore remains intact.
            row_bits = np.packbits(X[:, eligible], axis=1, bitorder="little")
            neighbours = np.ones(n, dtype=np.int64)
            for i in range(n-1):
                differences = POPCOUNT[np.bitwise_xor(row_bits[i+1:], row_bits[i])].sum(axis=1)
                close = differences/len(eligible) < args.weight_distance
                neighbours[i] += close.sum()
                neighbours[i+1:] += close
                if (i+1) % 25 == 0 or i == n-2:
                    log(f"global sequence weights: {i+1}/{n-1} sample comparisons")
            weights = 1.0/neighbours
            weighting = "global_hamming_sequence_reweighting"
        else:
            weights = np.ones(n)
            weighting = "unweighted"
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("Sample weights must be finite and positive")
        table(out/"sample_weights.tsv", pd.DataFrame(dict(sample=samples, weight=weights)))
        np.savetxt(out/"spydrpick_weights.txt", weights, fmt="%.17g")
    with measure(out, "tree_axes_and_all_pattern_projection", records):
        K, _ = prepare_kinship(build_tree_covariance(out/"analysis.treefile", samples).K)
        # Center covariance so the intercept is not selected again as a PC.
        means = K.mean(axis=0)
        centered = K-means[:, None]-means[None, :]+K.mean()
        eigenvalues, eigenvectors = linalg.eigh(centered, check_finite=False)
        positive = np.flatnonzero(eigenvalues > max(1e-12, eigenvalues[-1]*1e-10))[::-1]
        if len(positive) < max(args.pcs):
            raise ValueError(f"Centered tree covariance has only {len(positive)} positive axes; reduce --pcs")
        Q, _ = linalg.qr(np.column_stack([np.ones(n)/np.sqrt(n),
                 eigenvectors[:, positive[:max(args.pcs)]]]), mode="economic")
        table(out/"tree_axes.tsv", pd.DataFrame(dict(axis=np.arange(1, len(positive)+1),
            eigenvalue=eigenvalues[positive], variance_fraction=eigenvalues[positive]/eigenvalues[positive].sum())))
        projections = np.empty((U, Q.shape[1]))
        for start in range(0, U, 8192):
            stop = min(U, start+8192)
            bits = np.unpackbits(unique[start:stop], axis=1, bitorder="little")[:, :n]
            projections[start:stop] = bits.astype(np.float64) @ Q
            if start % 65536 == 0 or stop == U:
                log(f"project unique patterns once: {stop:,}/{U:,}")
        panels = []
        for replicate in range(args.replicates):
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, replicate]))
            selected = rng.choice(U, args.patterns, replace=False)
            # Preserve original-locus ordering for KOVAR's canonical direction.
            selected = selected[np.argsort(loci[representatives[selected]], kind="stable")]
            panels.append(selected)
        npz(out/"preparation.npz", samples=np.array(samples), K=K, Q=Q,
            packed=unique, projections=projections, representatives=loci[representatives],
            present=present[representatives], maf=maf[representatives],
            panels=np.array(panels), weights=weights)
    shutil.copyfile(out/"timings.tsv", out/"preparation_timings.tsv")
    save_cache(manifest, identity, products, algorithm=ALGORITHM, versions=versions(),
        n_samples=n, n_original_loci=len(loci), n_maf_eligible_loci=len(eligible),
        n_unique_patterns=U, mi_weighting=weighting, pattern_selection="uniform_unique_patterns_without_replacement",
        replicate_seeds=[dict(seed=args.seed, panel=i+1) for i in range(args.replicates)],
        covariance="same tree MRCA covariance as KOVAR; centered only for screening eigenvectors",
        scope="all unordered pairs in sampled panels; distances and graph-count filters unavailable",
        source_identity="source path/size/mtime; completed cache products are SHA256 checked")
    return json.loads(manifest.read_text(encoding="utf-8"))


def binary_mi(X, u, v, weights, pseudocount=0.5):
    """Explicit 2x2 reweighted MI; not claimed bit-for-bit SpydrPick output."""
    weighted = X.T @ weights
    n11 = np.empty(len(u))
    for start in range(0, len(u), 32768):
        stop = min(len(u), start+32768)
        n11[start:stop] = np.sum(X[:, u[start:stop]]*X[:, v[start:stop]]*weights[:, None], axis=0)
    cells = np.column_stack([weights.sum()-weighted[u]-weighted[v]+n11,
                            weighted[v]-n11, weighted[u]-n11, n11])+pseudocount
    cells /= weights.sum()+4*pseudocount
    cube = cells.reshape(-1, 2, 2)
    expected = cube.sum(axis=2)[:, :, None]*cube.sum(axis=1)[:, None, :]
    return np.sum(xlogy(cube, cube/expected), axis=(1, 2))


def residual_scores(n11, u, v, present, projection, k, tolerance=1e-10):
    """Exact linear partial correlation via binary counts minus PC projections."""
    z = projection[:, :k+1]
    norms = present.astype(float)-np.sum(z*z, axis=1)
    floor = tolerance*np.maximum(present, 1)
    valid_loci = norms > floor
    scores = np.full(len(u), np.nan)
    for start in range(0, len(u), 65536):
        stop = min(len(u), start+65536)
        a, b = u[start:stop], v[start:stop]
        valid = valid_loci[a] & valid_loci[b]
        cross = n11[start:stop]-np.einsum("ij,ij->i", z[a], z[b])
        values = np.full(stop-start, np.nan)
        values[valid] = cross[valid]/np.sqrt(norms[a[valid]]*norms[b[valid]])
        scores[start:stop] = np.clip(values, -1, 1)
    return scores, norms, valid_loci


def run_command(command, directory, log_name, *, blas_threads=1, reference=False):
    env = dict(os.environ, OMP_NUM_THREADS=str(blas_threads), OPENBLAS_NUM_THREADS=str(blas_threads),
               MKL_NUM_THREADS=str(blas_threads), NUMEXPR_NUM_THREADS=str(blas_threads))
    if reference:
        env.pop("PYTHONPATH", None)
    log(f"start {log_name}; progress/log: {directory/log_name}")
    with (directory/log_name).open("a", encoding="utf-8") as handle:
        handle.write("\nCOMMAND "+json.dumps([str(part) for part in command])+"\n")
        handle.flush()
        process = subprocess.Popen([str(part) for part in command], cwd=directory, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        try:
            for line in process.stdout:
                handle.write(line)
                handle.flush()
                print(line, end="", flush=True)
            returncode = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
        finally:
            process.stdout.close()
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)


def spydrpick_mi(args, directory, u, v, m):
    work = directory/"spydrpick"
    work.mkdir(exist_ok=True)
    shutil.copyfile(directory/"panel.fasta", work/"panel.fasta")
    # Retry only our private incomplete SpydrPick output, never original scans.
    for path in work.glob("panel*spydrpick_couplings*"):
        path.unlink()
    command = [args.spydrpick, "--alignmentfile", "panel.fasta", "--no-filter-alignment",
               "--mi-threshold", "0", "--no-aracne", "--output-indexing-base", "0",
               "--linear-genome", "--threads", args.threads, "--verbose",
               "--sample-weights", args.out/"spydrpick_weights.txt"]
    run_command(command, work, "spydrpick.log")
    files = list(work.glob("*.spydrpick_couplings.0-based.*edges"))
    if len(files) != 1:
        raise ValueError(f"Expected one zero-based SpydrPick edge file in {work}, found {len(files)}")
    values = np.full(len(u), np.nan)
    observed = 0
    with files[0].open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            a, b, score = int(fields[0]), int(fields[1]), float(fields[4])
            a, b = min(a, b), max(a, b)
            if not 0 <= a < b < m or not np.isfinite(score) or score < 0:
                raise ValueError("Invalid SpydrPick pair/index/MI")
            index = a*(2*m-a-1)//2+b-a-1
            if np.isfinite(values[index]):
                raise ValueError("Duplicate pair in SpydrPick output")
            values[index] = score
            observed += 1
    # Its strict threshold<MI skips exact-zero MI; keep these pairs in the
    # benchmark universe at MI=0 rather than silently removing them.
    missing = np.isnan(values)
    values[missing] = 0.
    json_write(work/"mi_inventory.json", dict(reported_pairs=observed,
        threshold_zero_unreported_pairs_restored=int(missing.sum()), eligible_panel_pairs=len(u),
        interpretation="unreported at explicit threshold 0 restored as MI 0; no pair removed"))
    return values


def panel(args, preparation, replicate, do_reference):
    directory = args.out/f"panel_{replicate+1:03d}"
    directory.mkdir(exist_ok=True)
    selected = preparation["panels"][replicate]
    m, n = len(selected), len(preparation["samples"])
    X = np.unpackbits(preparation["packed"][selected], axis=1, bitorder="little")[:, :n].T.copy()
    present = preparation["present"][selected]
    projection = preparation["projections"][selected]
    original_loci = preparation["representatives"][selected]
    u, v = np.triu_indices(m, 1)
    # Use float64 here: uint8 matrix products overflow above 255 observations.
    operation = time.perf_counter()
    cross = X.astype(np.float64).T @ X.astype(np.float64)
    n11 = np.rint(cross[u, v]).astype(np.int64)
    counts_seconds = time.perf_counter()-operation
    min_cell = np.minimum.reduce([n11, present[u]-n11, present[v]-n11,
                                 n-present[u]-present[v]+n11])
    score_path = directory/"screen_scores.tsv"
    identity = stage_identity("panel_screens", [args.out/"preparation.npz"], dict(
        algorithm=ALGORITHM, code=code_identity(), panel=replicate+1, pcs=args.pcs,
        min_cell_count=args.min_cell_count, mi_backend=args.mi_backend, spydrpick=args.spydrpick,
        threads=args.threads, pseudocount=args.mi_pseudocount))
    manifest = directory/"screens.json"
    products = [score_path, directory/"locus_diagnostics.tsv", directory/"screen_timings.tsv",
                directory/"panel.fasta", directory/"panel_loci.tsv"]
    if not cache_matches(manifest, identity, products):
        if manifest.exists():
            raise ValueError(f"Screen configuration changed in {directory}; use a new --out")
        with (directory/"panel.fasta").open("w", encoding="utf-8") as handle:
            for name, states in zip(preparation["samples"], X):
                handle.write(f">{name}\n"+"".join(np.where(states, "C", "A"))+"\n")
        table(directory/"panel_loci.tsv", pd.DataFrame(dict(panel_locus=np.arange(m),
            pattern_id=selected, original_locus=original_loci, n_present=present,
            maf=preparation["maf"][selected])))
        frame = pd.DataFrame(dict(u=u, v=v, original_u=original_loci[u], original_v=original_loci[v],
            min_pair_maf=np.minimum(preparation["maf"][selected[u]], preparation["maf"][selected[v]]),
            min_cell=min_cell, eligible=(min_cell >= args.min_cell_count).astype(int)))
        records = [dict(stage="shared_joint_state_counts", wall_seconds=counts_seconds,
                        sampled_sum_rss_bytes=None, source="measured_run")]
        with measure(directory, "MI_"+args.mi_backend, records):
            frame["MI"] = (spydrpick_mi(args, directory, u, v, m) if args.mi_backend == "spydrpick"
                else binary_mi(X, u, v, preparation["weights"], args.mi_pseudocount))
        diagnostics = pd.DataFrame(dict(panel_locus=np.arange(m), original_locus=original_loci))
        for k in args.pcs:
            with measure(directory, f"residual_k{k}", records):
                signed, norms, valid = residual_scores(n11, u, v, present, projection, k)
                frame[f"residual_k{k}"] = np.abs(signed)
                frame[f"signed_residual_k{k}"] = signed
                diagnostics[f"residual_ss_k{k}"] = norms
                diagnostics[f"scorable_k{k}"] = valid.astype(int)
        table(score_path, frame)
        table(directory/"locus_diagnostics.tsv", diagnostics)
        shutil.copyfile(directory/"timings.tsv", directory/"screen_timings.tsv")
        save_cache(manifest, identity, products)
    else:
        log(f"panel {replicate+1}: reuse completed MI and residual scores")
    if do_reference:
        reference = directory/"reference"
        reference.mkdir(exist_ok=True)
        reference_manifest = directory/"reference.json"
        reference_products = [reference/"ko_variation.tsv", reference/"response_models.tsv",
                              reference/"execution_metadata.tsv", directory/"reference_timings.tsv"]
        ref_identity = stage_identity("all_pair_kovar_reference",
            [directory/"panel.fasta", args.out/"analysis.treefile"], dict(
                code=code_identity(), versions=versions(), min_maf=args.min_maf,
                min_cell_count=args.min_cell_count, spa_mode=args.spa_mode, reference_python=str(args.reference_python) if args.reference_python else None,
                threads=args.threads, original_locus_order=original_loci.tolist()))
        if not cache_matches(reference_manifest, ref_identity, reference_products):
            if reference_manifest.exists():
                raise ValueError("Reference settings changed; use a new --out directory")
            pairs_path = directory/"reference_pairs.tsv"
            table(pairs_path, pd.DataFrame(dict(u=u, v=v)))
            if not args.reference_python:
                raise ValueError("Fresh historical references require --reference-python pointing to an independent v0.8.3 installation; completed references can be extended or plotted directly.")
            command = [args.reference_python, "-m", "ko_variation.cli", "--fasta", directory/"panel.fasta",
                "--pairs", pairs_path, "--tree", args.out/"analysis.treefile", "--out", reference,
                "--threads", args.threads, "--min-maf", args.min_maf,
                "--min-cell-count", args.min_cell_count, "--spa-mode", args.spa_mode]
            if (reference/".kovar_checkpoint"/"manifest.json").exists():
                command.append("--resume")
            else:
                # Dedicated benchmark directory only; originals are untouched.
                command.append("--overwrite")
            records = []
            with measure(directory, "KOVAR_all_pairs_CLI", records):
                run_command(command, directory, "reference.log", reference=True)
            shutil.copyfile(directory/"timings.tsv", directory/"reference_timings.tsv")
            save_cache(reference_manifest, ref_identity, reference_products)
        else:
            log(f"panel {replicate+1}: reuse completed exhaustive KOVAR reference")
    log(f"panel {replicate+1}: {m:,} patterns; {len(u):,} pairs; {np.count_nonzero(min_cell >= args.min_cell_count):,} pass joint-cell filter")


def evaluate(args, base_out=None):
    """Keep the reference family fixed, including for legacy PC extensions."""
    base_out = args.out if base_out is None else Path(base_out)
    rows, summaries, panel_settings, retention, variance = [], [], [], [], []
    for replicate in range(args.replicates):
        directory = args.out/f"panel_{replicate+1:03d}"
        reference_directory = base_out/directory.name
        scores = pd.read_csv(directory/"screen_scores.tsv", sep="\t")
        reference = pd.read_csv(reference_directory/"reference"/"ko_variation.tsv", sep="\t")
        if reference.duplicated(["u", "v"]).any() or len(reference) != len(scores):
            raise ValueError("Reference must cover every panel pair exactly once, including untested pairs")
        frame = scores.merge(reference[["u", "v", "p_primary", "status"]],
                             on=["u", "v"], how="left", validate="one_to_one")
        if frame["status"].isna().any():
            raise ValueError("Reference and screen pair universes differ")
        frame["p_primary"] = pd.to_numeric(frame.p_primary, errors="coerce")
        finite = np.isfinite(frame.p_primary) & frame.p_primary.between(0, 1)
        eligible = frame.eligible.eq(1)
        tested = finite & eligible
        n_eligible, n_tested = int(eligible.sum()), int(tested.sum())
        if not n_eligible or not n_tested:
            raise ValueError(f"Panel {replicate+1} has no eligible/successfully tested pairs; inspect reference.log and min-cell-count")
        denominator = args.bonferroni_tests or n_tested
        if denominator < n_tested:
            raise ValueError("--bonferroni-tests cannot be smaller than the number of finite reference tests")
        discovery = tested & (frame.p_primary <= args.alpha/denominator)
        reference_order = frame.loc[tested].sort_values(["p_primary", "u", "v"], kind="stable").index
        top_count = max(1, math.ceil(n_tested*args.reference_top_fraction))
        priorities = set(reference_order[:top_count])
        priority_boundary_p = float(frame.loc[reference_order[top_count-1], "p_primary"])
        priority_boundary_ties = int((tested & frame.p_primary.eq(priority_boundary_p)).sum())
        # Midranks for visual comparisons; deterministic locus order breaks ties
        # only for fixed-budget selection, with boundary tie size reported.
        frame["kovar_rank"] = frame.p_primary.where(tested).rank(method="average")
        methods = ["MI"]+[f"residual_k{k}" for k in args.pcs]
        for method in methods:
            frame[f"{method}_rank"] = frame[method].where(eligible).rank(ascending=False, method="average")
            ordered = frame.loc[eligible].sort_values([method, "u", "v"], ascending=[False, True, True],
                                                       kind="stable", na_position="last").index
            positions = pd.Series(np.arange(1, n_eligible+1), index=ordered)
            # Calculate exact minimum budgets from rank positions, rather than
            # interpolating a sparse budget grid. This is in-panel agreement,
            # not a threshold certified for unseen or genome-wide candidates.
            for reference_set, indices in [("significant", frame.index[discovery]),
                                           ("priority", reference_order[:top_count])]:
                target_positions = positions.loc[indices].to_numpy(dtype=int)
                for target in args.recovery_targets:
                    required = math.ceil(target*len(indices))
                    if required:
                        budget = int(np.partition(target_positions, required-1)[required-1])
                        boundary = frame.loc[ordered[budget-1], method]
                        ties = int(frame.loc[eligible, method].eq(boundary).sum()) if pd.notna(boundary) else int(frame.loc[eligible, method].isna().sum())
                        attained = float(np.count_nonzero(target_positions <= budget)/len(indices))
                    else:
                        budget, boundary, ties, attained = None, None, None, None
                    retention.append(dict(panel=replicate+1, method=method, reference_set=reference_set,
                        target_recovery=target, n_reference_pairs=len(indices),
                        n_required_to_recover=required, min_candidates=budget,
                        candidate_fraction=budget/n_eligible if budget else np.nan,
                        attained_recovery=attained, score_threshold=boundary, boundary_tie_size=ties,
                        n_unscorable_reference_pairs=int(frame.loc[indices, method].isna().sum()),
                        n_eligible_pairs=n_eligible))
            for fraction in args.budgets:
                budget = min(n_eligible, max(1, math.ceil(n_eligible*fraction)))
                selected = set(ordered[:budget])
                recovered = int(discovery.loc[list(selected)].sum())
                finite_selected = int(tested.loc[list(selected)].sum())
                boundary = frame.loc[ordered[budget-1], method]
                tie_size = int((frame.loc[eligible, method] == boundary).sum()) if pd.notna(boundary) else int(frame.loc[eligible, method].isna().sum())
                rows.append(dict(panel=replicate+1, method=method, budget_fraction=fraction,
                    actual_budget_fraction=budget/n_eligible,
                    n_budget_pairs=budget, n_eligible_pairs=n_eligible, n_reference_tested=n_tested,
                    n_reference_significant=int(discovery.sum()), n_significant_recovered=recovered,
                    discovery_recall=recovered/int(discovery.sum()) if discovery.any() else np.nan,
                    discovery_precision=recovered/finite_selected if finite_selected else np.nan,
                    n_reference_priority=top_count, n_priority_recovered=len(selected & priorities),
                    priority_recall=len(selected & priorities)/top_count,
                    n_selected_unscorable=int(frame.loc[list(selected), method].isna().sum()),
                    boundary_tie_size=tie_size, bonferroni_tests=denominator,
                    reference_priority_boundary_ties=priority_boundary_ties,
                    alpha=args.alpha, reference_top_fraction=args.reference_top_fraction))
                # Report sensitivity by the rarer feature's MAF, rather than
                # hiding losses concentrated in low-frequency patterns.
                for low, high in [(0, .05), (.05, .1), (.1, .2), (.2, .5)]:
                    in_bin = tested & (frame.min_pair_maf >= low) & (
                        (frame.min_pair_maf < high) if high < .5 else (frame.min_pair_maf <= high))
                    targets = discovery & in_bin
                    target_indices = set(frame.index[targets])
                    summaries.append(dict(panel=replicate+1, method=method, budget_fraction=fraction,
                        min_maf=low, max_maf=high, n_reference_significant=len(target_indices),
                        n_recovered=len(selected & target_indices),
                        discovery_recall=len(selected & target_indices)/len(target_indices) if target_indices else np.nan))
        table(directory/"rank_comparison.tsv", frame)
        table(directory/"reference_status_counts.tsv", reference.status.value_counts().rename_axis("status").reset_index(name="n_pairs"))
        diagnostics = pd.read_csv(directory/"locus_diagnostics.tsv", sep="\t")
        for k in args.pcs:
            norms = diagnostics[f"residual_ss_k{k}"].to_numpy()
            scorable = diagnostics[f"scorable_k{k}"].eq(1)
            variance.append(dict(panel=replicate+1, pcs=k, n_loci=len(diagnostics),
                n_near_zero=int((~scorable).sum()), fraction_near_zero=float((~scorable).mean()),
                minimum_residual_ss=float(norms.min()), median_residual_ss=float(np.median(norms))))
        panel_settings.append(dict(panel=replicate+1,
            screen=json.loads((directory/"screens.json").read_text())["cache_identity"]["settings"],
            reference=json.loads((reference_directory/"reference.json").read_text())["cache_identity"]["settings"]))
        log(f"panel {replicate+1}: {n_tested:,} finite KOVAR tests; {int(discovery.sum()):,} Bonferroni discoveries; reference-top set {top_count:,}")
    table(args.out/"recovery.tsv", pd.DataFrame(rows))
    table(args.out/"recovery_by_maf.tsv", pd.DataFrame(summaries))
    required = pd.DataFrame(retention)
    table(args.out/"required_retention.tsv", required)
    aggregate = required.groupby(["method", "reference_set", "target_recovery"], sort=False).agg(
        n_panels=("panel", "count"), n_defined_panels=("candidate_fraction", "count"),
        median_candidate_fraction=("candidate_fraction", "median"),
        minimum_candidate_fraction=("candidate_fraction", "min"),
        maximum_candidate_fraction=("candidate_fraction", "max"),
        smallest_reference_set=("n_reference_pairs", "min"), largest_reference_set=("n_reference_pairs", "max")).reset_index()
    table(args.out/"retention_summary.tsv", aggregate)
    table(args.out/"residual_variance_diagnostics.tsv", pd.DataFrame(variance))
    config = json.loads((base_out/"preparation.json").read_text(encoding="utf-8"))
    json_write(args.out/"benchmark_summary.json", dict(algorithm=ALGORITHM, versions=versions(),
        n_samples=config["n_samples"], n_unique_patterns=config["n_unique_patterns"],
        settings=vars_for_json(args), preparation_settings=config["cache_identity"]["settings"],
        panel_settings=panel_settings,
        base_directory=str(base_out),
        interpretation=["Observed KOVAR-reference recovery, not biological precision or epistasis validation",
            "Independent panels are computational resamples, not biological replicates",
            "No original MI shortlist was used to select patterns or filter pair enumeration",
            "All pair budgets use joint-cell-eligible pairs, including failed reference fits",
            "No physical-distance or PAN-GWES graph-count exclusion; do not call these distal discoveries",
            "Priority recall uses top-ranked KOVAR pairs and does not establish significance",
            "Zero-discovery panels retain NA discovery recall; no fabricated zero recall",
            "Required retention uses deterministic in-panel rank positions, not an unseen-data guarantee",
            "Maximum required fraction summarizes the worst observed defined panel, not all possible cohorts",
            "Ties broken by original-locus order; unavailable residual scores placed last and counted",
            "Sum-RSS memory sampling may double-count shared pages; missing psutil means unmeasured memory"]))


def vars_for_json(args):
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=["all", "prepare", "screen", "reference", "evaluate", "extend"], default="all")
    parser.add_argument("--add-pcs", nargs="+", type=int, default=[80, 120], help="PC settings to add with --stage extend")
    parser.add_argument("--retime-pcs", nargs="+", type=int, default=[40], help="Existing settings to re-time in the extension allocation")
    parser.add_argument("--extension-out", type=Path, help="Separate extension output [OUT/extended_pcs]")
    parser.add_argument("--recovery-targets", nargs="+", type=float, default=[.95, .99, 1.])
    parser.add_argument("--fasta", type=Path)
    parser.add_argument("--tree", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--species", default="Staphylococcus aureus")
    parser.add_argument("--sample-map", type=Path)
    parser.add_argument("--sample-inclusion", type=Path)
    parser.add_argument("--exclude-sample", action="append", default=[])
    parser.add_argument("--locus-map", type=Path)
    parser.add_argument("--fasta-columns-are-locus-ids", action="store_true")
    parser.add_argument("--tree-missing-samples", choices=["error", "drop"], default="error")
    parser.add_argument("--rooting", choices=["midpoint", "as-is"], default="midpoint")
    parser.add_argument("--patterns", type=int, default=1000)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pcs", nargs="+", type=int, default=[10, 20, 40, 80, 120])
    parser.add_argument("--min-maf", type=float, default=.05)
    parser.add_argument("--min-cell-count", type=int, default=20,
                        help="20 matches the supplied S. aureus runner; 0 allows all joint-state counts")
    parser.add_argument("--spa-mode", choices=["off", "auto", "always"], default="off")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--reference-python", type=Path, help="Independent installed v0.8.3 Python for fresh historical reference fits")
    parser.add_argument("--mi-backend", choices=["spydrpick", "binary"], default="spydrpick")
    parser.add_argument("--spydrpick", default="SpydrPick")
    parser.add_argument("--mi-weighting", choices=["sequence", "none"], default="sequence")
    parser.add_argument("--sample-weights", type=Path, help="TSV sample/weight; overrides MI-weighting")
    parser.add_argument("--weight-distance", type=float, default=.1)
    parser.add_argument("--mi-pseudocount", type=float, default=.5, help="Explicit binary MI backend only")
    parser.add_argument("--budgets", nargs="+", type=float, default=[.0001, .0005, .001, .005, .01, .05, .1, .2, .5, 1.])
    parser.add_argument("--reference-top-fraction", type=float, default=.01)
    parser.add_argument("--alpha", type=float, default=.05)
    parser.add_argument("--bonferroni-tests", type=int,
                        help="Optional larger discovery-family denominator; default finite tests in each panel")
    args = parser.parse_args(argv)
    args.out = args.out.resolve()
    for key in ["fasta", "tree", "metadata", "sample_map", "sample_inclusion", "locus_map", "sample_weights", "extension_out"]:
        if getattr(args, key):
            setattr(args, key, getattr(args, key).resolve())
    args.pcs = sorted(set(args.pcs))
    args.add_pcs, args.retime_pcs = sorted(set(args.add_pcs)), sorted(set(args.retime_pcs))
    args.recovery_targets = sorted(set(args.recovery_targets))
    args.budgets = sorted(set(args.budgets))
    if args.patterns < 3 or args.patterns > 3000:
        parser.error("--patterns must be 3..3000 (use independent panels rather than an accidental enormous reference scan)")
    if args.replicates < 1 or args.threads < 1 or min(args.pcs) < 0 or args.seed < 0:
        parser.error("Replicates/threads must be positive; PCs and seed nonnegative")
    if min(args.add_pcs+args.retime_pcs) < 0 or any(not 0 < x <= 1 for x in args.recovery_targets):
        parser.error("Extension PCs must be nonnegative; recovery targets must be in (0,1]")
    if not 0 <= args.min_maf < .5 or args.min_cell_count < 0:
        parser.error("Invalid frequency or joint-cell threshold")
    if not 0 < args.weight_distance <= 1 or args.mi_pseudocount < 0:
        parser.error("Invalid MI weighting distance/pseudocount")
    if not 0 < args.alpha < 1 or not 0 < args.reference_top_fraction <= 1 or any(not 0 < x <= 1 for x in args.budgets):
        parser.error("Alpha must be (0,1); budgets and reference-top fraction must be (0,1]")
    if args.bonferroni_tests is not None and args.bonferroni_tests < 1:
        parser.error("--bonferroni-tests must be positive")
    if args.stage in {"all", "prepare"} and (not args.fasta or not args.tree or not (args.locus_map or args.fasta_columns_are_locus_ids)):
        parser.error("Preparation needs --fasta, --tree and explicit --fasta-columns-are-locus-ids or --locus-map")
    if args.sample_map and not args.metadata:
        parser.error("--sample-map requires --metadata")
    # Resolve relative executable paths before entering a private panel folder.
    if args.mi_backend == "spydrpick" and (executable := shutil.which(args.spydrpick)):
        args.spydrpick = str(Path(executable).resolve())
    return args


def main(argv=None):
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    # One BLAS pool for screens; reference subprocess uses response workers
    # with BLAS=1. Original pipeline scripts and outputs are never modified.
    with threadpool_limits(limits=args.threads):
        if args.stage == "extend":
            from .screen_benchmark_extend import extend
            return extend(args)
        if args.stage in {"all", "prepare"}:
            preparation = prepare(args)
        else:
            manifest = args.out/"preparation.json"
            if not manifest.is_file():
                raise ValueError("Run --stage prepare first")
            preparation = json.loads(manifest.read_text(encoding="utf-8"))
            saved = preparation["cache_identity"]["settings"]
            # Restart stages inherit the scientific preparation configuration.
            for key in ["pcs", "patterns", "replicates", "seed", "min_maf"]:
                setattr(args, key, saved[key])
            if preparation["cache_outputs"][0] != fingerprint_file(args.out/"preparation.npz"):
                raise ValueError("Prepared data checksum changed")
        if args.stage in {"all", "screen", "reference"}:
            with np.load(args.out/"preparation.npz", allow_pickle=False) as data:
                for replicate in range(args.replicates):
                    panel(args, data, replicate, args.stage in {"all", "reference"})
        if args.stage in {"all", "evaluate"}:
            evaluate(args)
            from .plot_screen_benchmark import plot
            plot(args.out, seed=args.seed)
    log(f"finished stage={args.stage}: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
