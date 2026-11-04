"""Stream significant distal pairs from a full scan without loading it into RAM."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from .postprocess import SelectionConfig, select_distal_signals
from .workflow_cache import cache_matches, save_cache, stage_identity


def select_file(source, output, config, *, chunk_rows=50000, pangwes_distances=False,
                resume=False):
    """Count original tests, then select in chunks using that same denominator.

    This is a memory-bounded adapter to the existing selection rule, not a new
    association test. PAN-GWES -1 distances are missing only when explicitly enabled.
    """
    source, output = Path(source), Path(output)
    manifest = Path(str(output) + ".selection.json")
    if source.resolve() in {output.resolve(), manifest.resolve()}:
        raise ValueError("Selection outputs must not overwrite the original scan")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    identity = stage_identity("selection", [source, Path(__file__),
        Path(__file__).with_name("postprocess.py")],
        dict(config=asdict(config), pangwes_distances=pangwes_distances))
    if resume and cache_matches(manifest, identity, [output]):
        print(f"[selection] reusing completed selection: {output}", flush=True)
        return json.loads(manifest.read_text(encoding="utf-8"))
    header = pd.read_csv(source, sep="\t", nrows=0).columns
    required = {"u", "v", "status", "p_primary", "n11", "n10", "n01", "n00", config.distance_column}
    if missing := required - set(header):
        raise ValueError(f"Missing scan columns: {sorted(missing)}")
    test_columns = ["p_primary"] + (["n_tests"] if "n_tests" in header else [])
    input_rows = nonmissing_p = 0
    declared_tests = None
    print("[selection] counting original tests", flush=True)
    with pd.read_csv(source, sep="\t", usecols=test_columns, chunksize=chunk_rows) as chunks:
        for chunk in chunks:
            p = pd.to_numeric(chunk.p_primary, errors="raise")
            if not np.isfinite(p.dropna()).all() or ((p.dropna() < 0) | (p.dropna() > 1)).any():
                raise ValueError("Significance values must lie in [0, 1] or be missing")
            input_rows += len(chunk)
            nonmissing_p += int(p.notna().sum())
            if "n_tests" in chunk and len(chunk):
                tests = pd.to_numeric(chunk.n_tests, errors="raise")
                if not np.isfinite(tests).all() or (tests < 0).any() or (tests != np.floor(tests)).any() or tests.nunique() != 1:
                    raise ValueError("n_tests must consistently describe the original full scan")
                value = int(tests.iloc[0])
                if declared_tests is not None and declared_tests != value:
                    raise ValueError("Inconsistent n_tests across scan chunks")
                declared_tests = value
    n_tests = declared_tests if declared_tests is not None else nonmissing_p
    if n_tests < nonmissing_p:
        raise ValueError("n_tests is smaller than the original nonmissing P-value count")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + ".partial")
    selected_rows = undefined_rows = scanned_rows = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle, \
                pd.read_csv(source, sep="\t", chunksize=chunk_rows) as chunks:
            first = True
            for index, chunk in enumerate(chunks, 1):
                if pangwes_distances:
                    distance = pd.to_numeric(chunk[config.distance_column], errors="raise")
                    undefined = distance.eq(-1)
                    undefined_rows += int(undefined.sum())
                    # Retain the original sentinel for annotation/provenance audits.
                    chunk[config.distance_column + "_pangwes_raw"] = chunk[config.distance_column]
                    chunk[config.distance_column] = distance.mask(undefined)
                chunk["n_tests"] = n_tests
                selected = select_distal_signals(chunk, config)
                selected.to_csv(handle, sep="\t", index=False, header=first)
                first = False
                selected_rows += len(selected)
                scanned_rows += len(chunk)
                if index == 1 or index % 20 == 0:
                    print(f"[selection] {scanned_rows:,}/{input_rows:,} rows; retained {selected_rows:,}", flush=True)
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    details = dict(original_results=str(source.resolve()), selected_input=str(output.resolve()),
        input_rows=input_rows, n_tests=n_tests, nonmissing_primary_p=nonmissing_p,
        selected_rows=selected_rows, undefined_distance_rows=undefined_rows,
        config=asdict(config), pangwes_distances=pangwes_distances)
    save_cache(manifest, identity, [output], **details)
    return details


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True, help="Selected-pair TSV; includes original n_tests")
    parser.add_argument("--significance-threshold", type=float, default=0.05)
    parser.add_argument("--ld-distance", type=float, default=0)
    parser.add_argument("--distance-column", default="distance")
    parser.add_argument("--cross-contig", choices=["exclude", "distal"], default="exclude")
    parser.add_argument("--pangwes-distances", action="store_true", help="Treat -1 as missing; preserve original values")
    parser.add_argument("--chunk-rows", type=int, default=50000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        details = select_file(args.results, args.out, SelectionConfig(
            args.significance_threshold, args.distance_column, args.ld_distance, args.cross_contig),
            chunk_rows=args.chunk_rows, pangwes_distances=args.pangwes_distances, resume=args.resume)
    except (ValueError, OSError) as exc:
        sys.stderr.write(f"Selection error: {exc}\n")
        return 2
    print(f"[selection] selected {details['selected_rows']:,} of {details['input_rows']:,} rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
