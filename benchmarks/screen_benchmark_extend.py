"""Extend a frozen completed benchmark without invoking SpydrPick or KOVAR.

The original cache is read-only. Additional projections, scores, timings and
combined evaluations live in a separate extension directory. Old reference
P-values and eligibility remain the computational reference, even if a newer
KOVAR version is installed. Legacy v1 benchmark manifests are supported.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg

from ko_variation.checkpoint import fingerprint_file
from ko_variation.phenotype import json_write
from ko_variation.workflow_cache import stage_identity, cache_matches, save_cache
from .screen_benchmark import (
    code_identity, evaluate, log, measure, npz, residual_scores, table, versions,
)


def checked_product(manifest, path):
    """Validate a required legacy output using its recorded content hash."""
    payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    recorded = [item for item in payload.get("cache_outputs", []) if item["name"] == path.name]
    if len(recorded) != 1 or recorded[0] != fingerprint_file(path):
        raise ValueError(f"Missing/changed cached product: {path}; original outputs are preserved")
    return payload


def grow_basis(K, original_Q, maximum):
    """Append tree axes while preserving the original adjustment subspace.

    This also handles tied eigenvalues: keep the old basis exactly rather than
    allowing a new eigensolver run to rotate the old cutoff within a tied space.
    Twice-orthogonalized eigenvectors are considered in descending eigenvalue.
    """
    n = K.shape[0]
    if maximum > n-3:
        raise ValueError(f"{maximum} PCs need at least {maximum+3} isolates; cache has {n}")
    if not np.allclose(original_Q.T @ original_Q, np.eye(original_Q.shape[1]), atol=1e-8):
        raise ValueError("Cached PC/intercept basis is not orthonormal")
    basis = [original_Q[:, i].copy() for i in range(original_Q.shape[1])]
    if len(basis) >= maximum+1:
        return original_Q.copy()
    means = K.mean(axis=0)
    centered = K-means[:, None]-means[None, :]+K.mean()
    values, vectors = linalg.eigh(centered, check_finite=False)
    positive = np.flatnonzero(values > max(1e-12, values[-1]*1e-10))[::-1]
    if len(positive) < maximum:
        raise ValueError(f"Only {len(positive)} positive tree axes are available; requested {maximum}")
    for index in positive:
        candidate = vectors[:, index].copy()
        Q = np.column_stack(basis)
        for _ in range(2):
            candidate -= Q @ (Q.T @ candidate)
        norm = np.linalg.norm(candidate)
        if norm > 1e-8:
            basis.append(candidate/norm)
        if len(basis) >= maximum+1:
            break
    if len(basis) < maximum+1:
        raise ValueError("Could not extend the cached tree basis to the requested rank")
    return np.column_stack(basis)


def extend(args):
    base = args.out.resolve()
    destination = (args.extension_out or base/"extended_pcs").resolve()
    if destination == base:
        raise ValueError("Extension output must differ from the read-only original benchmark directory")
    if (destination/"preparation.json").exists():
        raise ValueError("Extension output contains a primary benchmark; choose another --extension-out")
    preparation = checked_product(base/"preparation.json", base/"preparation.npz")
    if not (base/"benchmark_summary.json").is_file():
        raise ValueError("Run evaluation of the original completed reference before extending it")
    original_summary = json.loads((base/"benchmark_summary.json").read_text(encoding="utf-8"))
    settings = preparation["cache_identity"]["settings"]
    for name in ["patterns", "replicates", "seed", "min_maf"]:
        setattr(args, name, settings[name])
    # Freeze discovery definition; caller defaults must never replace the
    # original alpha, family denominator or ranked-reference fraction.
    for name in ["alpha", "bonferroni_tests", "reference_top_fraction"]:
        setattr(args, name, original_summary["settings"].get(name))
    old_pcs = list(settings["pcs"])
    combined_pcs = sorted(set(old_pcs) | set(args.add_pcs))
    retime = sorted(set(args.retime_pcs) & set(old_pcs))
    work_pcs = sorted(set(args.add_pcs) | set(retime))
    if not work_pcs:
        raise ValueError("No additional or retimed PC settings requested")
    destination.mkdir(parents=True, exist_ok=True)
    base_products = [base/"preparation.json", base/"preparation.npz", base/"benchmark_summary.json"]
    for replicate in range(args.replicates):
        folder = base/f"panel_{replicate+1:03d}"
        for manifest, names in [("screens.json", ["screen_scores.tsv", "locus_diagnostics.tsv", "panel_loci.tsv", "screen_timings.tsv"]),
                                ("reference.json", ["reference/ko_variation.tsv", "reference_timings.tsv"])]:
            for name in names:
                path = folder/name
                checked_product(folder/manifest, path)
                base_products.append(path)
            base_products.append(folder/manifest)
    frozen_screen = json.loads((base/"panel_001"/"screens.json").read_text())["cache_identity"]["settings"]
    frozen_reference = json.loads((base/"panel_001"/"reference.json").read_text())["cache_identity"]["settings"]
    for name in ["min_cell_count", "spa_mode"]:
        setattr(args, name, frozen_reference[name])
    for name in ["mi_backend", "spydrpick"]:
        setattr(args, name, frozen_screen[name])
    args.mi_weighting = settings["mi_weighting"]
    # Identity is content-based for legacy products, allowing a cache copied
    # from another machine to be extended without its original source paths.
    identity = stage_identity("frozen_benchmark_pc_extension", [], dict(
        algorithm="tree-pc-extension-1", code=code_identity(), versions=versions(),
        source_products=[fingerprint_file(path) for path in base_products],
        original_directory=str(base), add_pcs=args.add_pcs, retime_pcs=retime,
        combined_pcs=combined_pcs, threads=args.threads))
    if (destination/"extension_source.json").exists():
        previous = json.loads((destination/"extension_source.json").read_text())
        if previous["identity"] != identity:
            raise ValueError("Extension source/settings changed; choose a different --extension-out")
    else:
        json_write(destination/"extension_source.json", dict(base_directory=str(base), identity=identity))
    log(f"extend frozen reference: {base}; additional PCs={args.add_pcs}; re-time={retime}")
    projection_products = [destination/name for name in ["projections.npy", "axes.npz", "projection_timings.tsv", "global_residual_diagnostics.tsv"]]
    projection_manifest = destination/"projections.json"
    with np.load(base/"preparation.npz", allow_pickle=False) as data:
        n, U = len(data["samples"]), len(data["present"])
        if max(combined_pcs) > n-3:
            raise ValueError(f"Cache has {n} samples; at most {n-3} PCs are allowed")
        if not cache_matches(projection_manifest, identity, projection_products):
            records = []
            with measure(destination, "extend_tree_axes", records):
                old_Q = data["Q"]
                Q = grow_basis(data["K"], old_Q, max(combined_pcs))
                npz(destination/"axes.npz", Q=Q, samples=data["samples"])
            bits = data["packed"]
            old_projection = data["projections"]
            temporary = destination/"projections.partial.npy"
            projection = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float64,
                                                   shape=(U, Q.shape[1]))
            projection[:, :old_projection.shape[1]] = old_projection
            previous = old_Q.shape[1]-1
            for maximum in sorted(set(k for k in combined_pcs if k > previous)):
                with measure(destination, f"projection_{previous}_to_{maximum}", records):
                    for start in range(0, U, 8192):
                        stop = min(U, start+8192)
                        X = np.unpackbits(bits[start:stop], axis=1, bitorder="little")[:, :n]
                        projection[start:stop, previous+1:maximum+1] = X.astype(float) @ Q[:, previous+1:maximum+1]
                        if start % 65536 == 0 or stop == U:
                            log(f"projection {previous}->{maximum}: {stop:,}/{U:,} patterns")
                    projection.flush()
                previous = maximum
            present = data["present"]
            diagnostics = []
            baseline = present*(1-present/n)
            for k in combined_pcs:
                norms = present-np.sum(projection[:, :k+1]**2, axis=1)
                valid = norms > 1e-10*np.maximum(present, 1)
                fractions = norms/np.maximum(baseline, 1e-30)
                diagnostics.append(dict(pcs=k, n_patterns=U, n_near_zero=int((~valid).sum()),
                    fraction_near_zero=float((~valid).mean()), median_variance_fraction=float(np.median(fractions)),
                    minimum_variance_fraction=float(fractions.min())))
            table(destination/"global_residual_diagnostics.tsv", pd.DataFrame(diagnostics))
            projection.flush()
            del projection
            temporary.replace(destination/"projections.npy")
            table(destination/"projection_timings.tsv", pd.DataFrame(records))
            save_cache(projection_manifest, identity, projection_products)
        else:
            log("reuse completed additional projections")
        projection = np.load(destination/"projections.npy", mmap_mode="r")
        for replicate in range(args.replicates):
            original = base/f"panel_{replicate+1:03d}"
            folder = destination/original.name
            folder.mkdir(exist_ok=True)
            products = [folder/name for name in ["screen_scores.tsv", "locus_diagnostics.tsv", "screen_timings.tsv"]]
            panel_identity = dict(identity, panel=replicate+1)
            if cache_matches(folder/"screens.json", panel_identity, products):
                log(f"panel {replicate+1}: reuse completed extension scores/timings")
                continue
            frame = pd.read_csv(original/"screen_scores.tsv", sep="\t")
            # Reuse exact joint counts from the old exhaustive reference. There
            # is no binary pair recount, external executable or new GLMM fit.
            reference = pd.read_csv(original/"reference"/"ko_variation.tsv", sep="\t", usecols=["u", "v", "n11"])
            frame = frame.merge(reference, on=["u", "v"], how="left", validate="one_to_one")
            if len(reference) != len(frame) or frame.n11.isna().any():
                raise ValueError("Frozen screen/reference pair universes differ")
            selected = data["panels"][replicate]
            loci = data["representatives"][selected]
            u, v = frame.u.to_numpy(dtype=int), frame.v.to_numpy(dtype=int)
            if not (np.array_equal(frame.original_u, loci[u]) and np.array_equal(frame.original_v, loci[v])):
                raise ValueError("Frozen original-locus mapping differs from cached pattern panel")
            z, present = projection[selected], data["present"][selected]
            diagnostics = pd.read_csv(original/"locus_diagnostics.tsv", sep="\t")
            if not np.array_equal(diagnostics.original_locus, loci):
                raise ValueError("Locus diagnostics do not follow the frozen panel order")
            records = []
            for k in work_pcs:
                with measure(folder, f"residual_k{k}", records):
                    signed, norms, valid = residual_scores(frame.n11.to_numpy(), u, v, present, z, k)
                name = f"residual_k{k}"
                if name in frame and not np.allclose(frame[name], np.abs(signed), atol=1e-8, rtol=1e-7, equal_nan=True):
                    raise ValueError(f"Retimed {k}-PC scores changed beyond tolerance; old outputs preserved")
                frame[name], frame[f"signed_residual_k{k}"] = np.abs(signed), signed
                diagnostics[f"residual_ss_k{k}"], diagnostics[f"scorable_k{k}"] = norms, valid.astype(int)
            table(folder/"screen_scores.tsv", frame.drop(columns="n11"))
            table(folder/"locus_diagnostics.tsv", diagnostics)
            table(folder/"screen_timings.tsv", pd.DataFrame(records))
            save_cache(folder/"screens.json", panel_identity, products)
            log(f"panel {replicate+1}: scored {work_pcs}; reused every MI score and reference test")
        del projection
    args.out, args.pcs = destination, combined_pcs
    args.base_benchmark = str(base)
    evaluate(args, base_out=base)
    from .plot_screen_benchmark import plot
    plot(destination, seed=args.seed)
    log(f"extension complete; original benchmark unchanged; results: {destination}")
    return 0
