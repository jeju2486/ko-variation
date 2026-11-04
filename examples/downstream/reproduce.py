"""Create synthetic scanner output, then run both downstream helpers.

Run from the repository root: python examples/downstream/reproduce.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from ko_variation.annotation_cli import main


def reproduce():
    root = Path(__file__).resolve().parent / "output"
    root.mkdir(parents=True, exist_ok=True)
    # Each quartet includes all four cells; synthetic columns repeat patterns.
    matrix = np.array([[1, 1]] * 30 + [[1, 0]] * 10 + [[0, 1]] * 10 + [[0, 0]] * 30, dtype=np.uint8)
    matrix = np.column_stack([matrix, matrix[:, 0], 1 - matrix[:, 1]])
    pairs = pd.DataFrame(dict(u=[0, 0, 2], v=[1, 3, 3], distance=[20000, 50000, 30000]))
    # Explicit synthetic selected evidence exercises the downstream workflow.
    # These invented demonstration values are not a statistical benchmark.
    results = pairs.assign(p_primary=1e-12, status="OK")
    results.to_csv(root / "ko_variation.tsv", sep="\t", index=False)
    (root / "binary.fa").write_text("".join(f">s{i}\n{''.join('C' if v else 'A' for v in row)}\n" for i, row in enumerate(matrix)), encoding="utf-8")
    (root / "tree.nwk").write_text("(" + ",".join(f"s{i}:1" for i in range(len(matrix))) + ");\n", encoding="utf-8")
    pd.DataFrame(dict(locus=[0, 1, 2, 3], gene=["gene_A", "gene_B", "gene_A", "gene_C"],
                      label=["Gene A", "Gene B", "Gene A", "Gene C"], group=["Group 1", "Group 2", "Group 1", "Group 2"],
                      product=["Synthetic annotation"] * 4)).to_csv(root / "annotation.tsv", sep="\t", index=False)
    return main(["--results", str(root / "ko_variation.tsv"), "--annotation", str(root / "annotation.tsv"),
                 "--fasta", str(root / "binary.fa"), "--tree", str(root / "tree.nwk"),
                 "--out", str(root / "network"), "--network"])


if __name__ == "__main__":
    raise SystemExit(reproduce())
