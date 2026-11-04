"""Separate optional helpers; the scanner CLI is unchanged."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import sqlite3
import pandas as pd
from .postprocess import SelectionConfig, select_distal_signals, fit_selected_effects
from .network import build_gene_network, export_gene_network
from .network_display import add_network_arguments, network_options
from . import __version__
from .io_utils import read_fake_fasta
from .kinship import build_tree_covariance
from .glmm import prepare_kinship
from .workflow_cache import cache_matches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Select Bonferroni-significant distal pairs, refit adjusted effects, and optionally export gene networks")
    parser.add_argument("--results", required=True, help="Original ko_variation.tsv")
    parser.add_argument("--out", required=True, help="Output prefix; must differ from input")
    parser.add_argument("--fasta", required=True, help="Original scanner binary FASTA")
    parser.add_argument("--tree", required=True, help="Original rooted tree with branch lengths")
    parser.add_argument("--tree-missing-samples", choices=["error", "drop"], default="error",
                        help="Use the same missing-isolate policy as the scanner [error]")
    parser.add_argument("--annotation", help="TSV: locus, gene; optional contig, position, label, product, group")
    parser.add_argument("--significance-threshold", type=float, default=0.05, help="Bonferroni family-wise alpha; cutoff is alpha / original n_tests (default: 0.05)")
    parser.add_argument("--ld-distance", type=float, default=0, help="Exclude distances <= cutoff (default: 0); physical-distance proxy for linkage")
    parser.add_argument("--distance-column", default="distance")
    parser.add_argument("--cross-contig", choices=["exclude", "distal"], default="exclude")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--network", action="store_true", help="Export PNG, offline HTML, and node/edge TSVs")
    parser.add_argument("--missing-genes", choices=["error", "drop"], default="error")
    parser.add_argument("--include-self", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--title", default="Distal gene covariation")
    add_network_arguments(parser)
    parser.add_argument("--checkpoint-file", help="Per-pair refit checkpoint [OUT.refit.sqlite]; retained after completion")
    parser.add_argument("--resume", action="store_true", help="Reuse completed refits from an exactly matching checkpoint")
    parser.add_argument("--no-checkpoint", action="store_true", help="Disable refit checkpoints")
    parser.add_argument("--no-progress", action="store_true", help="Suppress stage and refit progress messages")
    parser.add_argument("--progress-every", type=int, default=10, help="Report after this many newly completed pairs [10]")
    parser.add_argument("--progress-seconds", type=float, default=60, help="Heartbeat interval, including during a slow fit [60]")
    args = parser.parse_args(argv)
    if args.resume and args.no_checkpoint:
        parser.error("--resume cannot be combined with --no-checkpoint")
    import math
    if args.progress_every < 1 or not math.isfinite(args.progress_seconds) or args.progress_seconds <= 0:
        parser.error("Progress intervals must be positive")
    config = SelectionConfig(**{key: getattr(args, key) for key in asdict(SelectionConfig())})
    prefix = Path(args.out)
    target = Path(str(prefix) + ".distal.tsv")
    destinations = [target, Path(str(prefix) + ".selection.json")]
    if args.network:
        destinations += [Path(str(prefix) + suffix) for suffix in (".png", ".svg", ".html", ".nodes.tsv", ".edges.tsv", ".layout.json")]
    inputs = [Path(args.results), Path(args.fasta), Path(args.tree)] + ([Path(args.annotation)] if args.annotation else [])
    checkpoint_file = None if args.no_checkpoint else Path(args.checkpoint_file or str(prefix) + ".refit.sqlite")
    if checkpoint_file is not None:
        destinations.append(checkpoint_file)
    if any(output.resolve() == source.resolve() for output in destinations for source in inputs):
        parser.error("Output paths must not overwrite input files")
    try:
        def announce(message):
            if not args.no_progress:
                print(f"[annotation] {message}", flush=True)
        announce("reading selected-pair input and annotation")
        results = pd.read_csv(args.results, sep="\t")
        # A preselected TSV retains n_tests; carry the full-scan provenance too.
        screening = None
        selection_manifest = Path(str(args.results) + ".selection.json")
        if selection_manifest.is_file():
            saved = json.loads(selection_manifest.read_text(encoding="utf-8"))
            if "cache_identity" in saved and "original_results" in saved:
                if not cache_matches(selection_manifest, saved["cache_identity"], [args.results]):
                    raise ValueError("Selected TSV does not match its selection manifest")
                screening = saved
        annotation = pd.read_csv(args.annotation, sep="\t", dtype={"gene": str, "contig": str, "label": str, "product": str, "group": str}) if args.annotation else None
        signals = select_distal_signals(results, config, annotation)
        announce(f"selected {len(signals)} pairs; reading scanner genotypes")
        fasta = read_fake_fasta(args.fasta)
        sample_provenance = dict(policy=args.tree_missing_samples,
                                 n_input_samples=len(fasta.sample_names),
                                 n_samples=len(fasta.sample_names), excluded_samples=[],
                                 matching_performed=False)
        # Empty selections require no tree covariance or alternative fits.
        if signals.empty:
            import numpy as np
            K = np.zeros((0, 0))
        else:
            announce(f"Refitting {len(signals)} selected pairs; preparing phylogenetic covariance")
            kinship = build_tree_covariance(args.tree, fasta.sample_names,
                                           missing_samples=args.tree_missing_samples)
            if kinship.excluded_samples:
                fasta.X = fasta.X[kinship.sample_indices, :]
                fasta.sample_names = [fasta.sample_names[i] for i in kinship.sample_indices]
            sample_provenance.update(n_samples=len(fasta.sample_names),
                                     excluded_samples=kinship.excluded_samples,
                                     matching_performed=True)
            K, _ = prepare_kinship(kinship.K)
        signals = fit_selected_effects(signals, fasta.X, K, confidence=args.confidence,
                                       checkpoint_file=checkpoint_file, resume=args.resume,
                                       progress=not args.no_progress, progress_every=args.progress_every,
                                       progress_seconds=args.progress_seconds,
                                       checkpoint_identity=dict(selection=asdict(config),
                                           **({"tree_missing_samples": args.tree_missing_samples}
                                              if args.tree_missing_samples != "error" else {})))
        checkpoint_details = signals.attrs.get("effect_checkpoint", {})
        if args.network:
            announce("aggregating gene edges and exporting figures")
            nodes, edges = build_gene_network(signals, significance_column="p_primary",
                                             include_self=args.include_self, missing_genes=args.missing_genes)
            export_gene_network(nodes, edges, prefix, seed=args.seed, dpi=args.dpi, title=args.title, **network_options(args))
        prefix.parent.mkdir(parents=True, exist_ok=True)
        signals.to_csv(target, sep="\t", index=False)
        provenance = dict(version=__version__, config=asdict(config),
                          input_rows=screening["input_rows"] if screening else len(results),
                          selected_input_rows=len(results), screening=screening, selected_rows=len(signals),
                          inputs=[str(source.resolve()) for source in inputs], confidence=args.confidence,
                          effect_method="alternative_logistic_mixed_pql", effect_status_counts=signals.effect_status.value_counts().to_dict(),
                          effect_checkpoint=checkpoint_details,
                          samples=sample_provenance,
                          network=dict(enabled=args.network, seed=args.seed, dpi=args.dpi, include_self=args.include_self,
                                       missing_genes=args.missing_genes, node_count=len(nodes) if args.network else 0,
                                       edge_count=len(edges) if args.network else 0, display=network_options(args)))
        Path(str(prefix) + ".selection.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    except (ValueError, OSError, ImportError, sqlite3.Error) as exc:
        sys.stderr.write(f"Annotation error: {exc}\n")
        return 2
    print(f"Selected {len(signals)} of {len(results)} pairs: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
