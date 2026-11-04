"""Redraw an existing selected-pair effect table without annotation or refitting."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import pandas as pd
from .network import build_gene_network, export_gene_network, discovery_network_input
from .network_display import add_network_arguments, network_options


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effects", required=True, help="Native OUT.distal.tsv or v0.9 associations.tsv.gz")
    parser.add_argument("--annotation", help="Locus-to-gene TSV for v0.9 unitig associations")
    parser.add_argument("--out", required=True)
    parser.add_argument("--missing-genes", choices=["drop", "error"], default="drop")
    parser.add_argument("--include-self", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--title", default="")
    add_network_arguments(parser)
    args = parser.parse_args(argv)
    outputs = [Path(args.out + suffix) for suffix in (".png", ".svg", ".html", ".nodes.tsv", ".edges.tsv", ".layout.json", ".network.json")]
    sources = [Path(path).resolve() for path in (args.effects, args.annotation,args.gene_catalog, args.eggnog) if path]
    if any(path.resolve() in sources for path in outputs):
        parser.error("Network outputs must not overwrite the effect table or annotation inputs")
    try:
        print(f"[network] reading saved effects: {args.effects}; no refitting", flush=True)
        signals = pd.read_csv(args.effects, sep="\t", dtype={"u_gene": str, "v_gene": str})
        significance_column="p_primary"
        if "significance" in signals and "p_upper" in signals:
            summary_path=Path(args.effects).parent/"run_summary.json"
            signals=discovery_network_input(signals,json.loads(summary_path.read_text()))
            significance_column="p_upper"
        annotation=pd.read_csv(args.annotation,sep="\t",dtype={"gene":str}) if args.annotation else None
        nodes, edges = build_gene_network(signals,annotation,significance_column=significance_column,
                                         missing_genes=args.missing_genes, include_self=args.include_self)
        paths = export_gene_network(nodes, edges, args.out, seed=args.seed, dpi=args.dpi, title=args.title, **network_options(args))
        metadata = dict(source_effects=str(Path(args.effects).resolve()), selected_pairs=len(signals),
                        nodes=len(nodes), edges=len(edges), refitted=False,significance_column=significance_column, display=network_options(args),
                        outputs={key: str(path) for key, path in paths.items()})
        Path(args.out + ".network.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"[network] exported {len(nodes)} genes and {len(edges)} edges: {args.out}", flush=True)
    except (ValueError, OSError, ImportError) as exc:
        sys.stderr.write(f"Network error: {exc}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
