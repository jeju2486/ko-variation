from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class KinshipResult:
    """A normalized sample covariance and provenance for its construction."""

    K: np.ndarray
    n_loci_used: int = 0
    n_loci_filtered: int = 0
    mean_diag_before_norm: float = float("nan")
    source: str = "unknown"
    details: dict[str, str | int | float] = field(default_factory=dict)
    sample_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    excluded_samples: list[str] = field(default_factory=list)


@dataclass
class _Node:
    name: Optional[str] = None
    length: Optional[float] = None
    children: list["_Node"] = field(default_factory=list)
    parent: Optional["_Node"] = None
    depth: float = 0.0
    tip_index: Optional[int] = None
    desc_tips: Optional[np.ndarray] = None

    @property
    def is_leaf(self) -> bool:
        return not self.children


class _NewickParser:
    """Small standalone Newick parser for branch-length covariance trees."""

    def __init__(self, text: str):
        self.text = text.strip()
        self.i = 0
        self.n = len(self.text)

    def parse(self) -> _Node:
        self._skip_ws()
        root = self._parse_subtree()
        self._skip_ws()
        if self.i < self.n and self.text[self.i] == ";":
            self.i += 1
        self._skip_ws()
        if self.i != self.n:
            raise ValueError(
                f"Unexpected trailing Newick text near position {self.i}: "
                f"{self.text[self.i:self.i + 40]!r}"
            )
        root.parent = None
        return root

    def _skip_ws(self) -> None:
        while self.i < self.n and self.text[self.i].isspace():
            self.i += 1

    def _parse_subtree(self) -> _Node:
        self._skip_ws()
        if self.i >= self.n:
            raise ValueError("Unexpected end of Newick string")
        if self.text[self.i] != "(":
            return _Node(
                name=self._parse_label_required(),
                length=self._parse_length_optional(),
            )

        self.i += 1
        children: list[_Node] = []
        while True:
            children.append(self._parse_subtree())
            self._skip_ws()
            if self.i >= self.n:
                raise ValueError("Unclosed internal node in Newick string")
            if self.text[self.i] == ",":
                self.i += 1
                continue
            if self.text[self.i] == ")":
                self.i += 1
                break
            raise ValueError(
                f"Expected ',' or ')' at position {self.i}, "
                f"found {self.text[self.i]!r}"
            )
        node = _Node(
            name=self._parse_label_optional(),
            length=self._parse_length_optional(),
            children=children,
        )
        for child in children:
            child.parent = node
        return node

    def _parse_label_required(self) -> str:
        label = self._parse_label_optional()
        if not label:
            raise ValueError(f"Expected leaf label at position {self.i}")
        return label

    def _parse_label_optional(self) -> Optional[str]:
        self._skip_ws()
        if self.i >= self.n or self.text[self.i] in ":,();":
            return None
        if self.text[self.i] in "'\"":
            quote = self.text[self.i]
            self.i += 1
            out: list[str] = []
            while self.i < self.n:
                ch = self.text[self.i]
                self.i += 1
                if ch == quote:
                    if self.i < self.n and self.text[self.i] == quote:
                        out.append(quote)
                        self.i += 1
                        continue
                    return "".join(out)
                out.append(ch)
            raise ValueError("Unclosed quoted label in Newick string")
        start = self.i
        while self.i < self.n and self.text[self.i] not in ":,();":
            self.i += 1
        return self.text[start:self.i].strip()

    def _parse_length_optional(self) -> Optional[float]:
        self._skip_ws()
        if self.i >= self.n or self.text[self.i] != ":":
            return None
        self.i += 1
        self._skip_ws()
        start = self.i
        while self.i < self.n and self.text[self.i] not in ",();":
            self.i += 1
        raw = self.text[start:self.i].strip()
        if not raw:
            raise ValueError("Empty branch length in Newick string")
        try:
            length = float(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid branch length {raw!r}") from exc
        if not np.isfinite(length) or length < 0:
            raise ValueError(f"Branch lengths must be finite and nonnegative: {raw!r}")
        return length


def _walk_preorder(node: _Node):
    yield node
    for child in node.children:
        yield from _walk_preorder(child)


def _walk_postorder(node: _Node):
    for child in node.children:
        yield from _walk_postorder(child)
    yield node


def _assign_depths(root: _Node, missing_length: str) -> None:
    if missing_length not in {"error", "one", "zero"}:
        raise ValueError("missing_length must be one of: error, one, zero")

    def edge_length(node: _Node) -> float:
        if node.parent is None:
            return 0.0
        if node.length is None:
            if missing_length == "error":
                raise ValueError(
                    f"Missing branch length for node {node.name or '<internal>'!r}. "
                    "KOVAR requires branch lengths unless an explicit fallback is chosen."
                )
            return 1.0 if missing_length == "one" else 0.0
        return float(node.length)

    root.depth = 0.0
    stack = [root]
    while stack:
        node = stack.pop()
        for child in node.children:
            child.depth = node.depth + edge_length(child)
            stack.append(child)


def build_tree_covariance(
    tree_path: str | Path,
    sample_names: list[str],
    dtype: str = "float64",
    missing_length: str = "error",
    missing_samples: str = "error",
) -> KinshipResult:
    """Build root-to-MRCA covariance in FASTA sample order.

    This follows the tree-similarity construction used by pyseer's
    ``phylogeny_distance.py --lmm``: an off-diagonal element is the root-to-MRCA
    branch length, and a diagonal element is the root-to-tip length. The matrix
    is normalized to mean diagonal one before GLMM fitting.
    With missing_samples="drop", sample_indices selects the retained FASTA rows
    in their original order. Callers must apply it to genotypes before fitting.
    """
    if missing_samples not in {"error", "drop"}:
        raise ValueError("missing_samples must be one of: error, drop")
    if len(set(sample_names)) != len(sample_names):
        raise ValueError("Duplicate sample names found in FASTA")
    path = Path(tree_path)
    root = _NewickParser(path.read_text(encoding="utf-8").strip()).parse()
    _assign_depths(root, missing_length)
    tips = [node for node in _walk_preorder(root) if node.is_leaf]
    if not tips:
        raise ValueError(f"No tips found in tree: {path}")
    tip_by_name: dict[str, _Node] = {}
    for tip in tips:
        if not tip.name:
            raise ValueError("Tree contains an unnamed tip")
        if tip.name in tip_by_name:
            raise ValueError(f"Duplicate tip label in tree: {tip.name}")
        tip_by_name[tip.name] = tip

    sample_set = set(sample_names)
    missing = [sample for sample in sample_names if sample not in tip_by_name]
    if missing and missing_samples == "error":
        raise ValueError(
            f"Tree is missing {len(missing)} FASTA samples; examples: {missing[:5]}. "
            "Supply a tree covering all samples, or explicitly use "
            "--tree-missing-samples drop to analyze only matched isolates."
        )
    sample_indices = np.array(
        [i for i, sample in enumerate(sample_names) if sample in tip_by_name],
        dtype=np.int64,
    )
    n_input_samples = len(sample_names)
    if missing_samples == "drop" and len(sample_indices) < 2:
        raise ValueError("Fewer than two FASTA samples remain after matching tree tips")
    sample_names = [sample_names[i] for i in sample_indices]
    if missing:
        sys.stderr.write(
            f"[KOVAR] warning: excluding {len(missing)} FASTA samples absent from tree; "
            f"retained={len(sample_names)}/{n_input_samples}; examples: {missing[:5]}\n"
        )
    extra = [name for name in tip_by_name if name not in sample_set]
    if extra:
        sys.stderr.write(
            f"[KOVAR] warning: ignoring {len(extra)} tree tips absent from FASTA; "
            f"examples: {extra[:5]}\n"
        )

    for index, sample in enumerate(sample_names):
        tip_by_name[sample].tip_index = index
    for node in _walk_postorder(root):
        if node.is_leaf:
            node.desc_tips = (
                np.empty(0, dtype=np.int64)
                if node.tip_index is None
                else np.array([node.tip_index], dtype=np.int64)
            )
        else:
            arrays = [
                child.desc_tips
                for child in node.children
                if child.desc_tips is not None and child.desc_tips.size
            ]
            node.desc_tips = (
                np.concatenate(arrays) if arrays else np.empty(0, dtype=np.int64)
            )

    n = len(sample_names)
    K = np.zeros((n, n), dtype=np.float64)
    for sample in sample_names:
        tip = tip_by_name[sample]
        assert tip.tip_index is not None
        K[tip.tip_index, tip.tip_index] = tip.depth
    for node in _walk_preorder(root):
        if node.is_leaf:
            continue
        groups = [
            child.desc_tips
            for child in node.children
            if child.desc_tips is not None and child.desc_tips.size
        ]
        for i, left in enumerate(groups):
            for right in groups[i + 1:]:
                K[np.ix_(left, right)] = node.depth
                K[np.ix_(right, left)] = node.depth

    K = 0.5 * (K + K.T)
    mean_diag = float(np.mean(np.diag(K)))
    if not np.isfinite(mean_diag) or mean_diag <= 0:
        raise ValueError("Invalid tree covariance scale; check rooting and branch lengths")
    K /= mean_diag
    out_dtype = np.float64 if dtype == "float64" else np.float32
    return KinshipResult(
        K=K.astype(out_dtype, copy=False),
        mean_diag_before_norm=mean_diag,
        source="tree_mrca_covariance",
        details={"tree": str(path), "n_tree_tips": len(tips), "n_samples": n,
                 "n_input_samples": n_input_samples, "n_excluded_samples": len(missing),
                 "tree_missing_samples": missing_samples},
        sample_indices=sample_indices,
        excluded_samples=missing,
    )
