from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from ko_variation.glmm import prepare_kinship
from ko_variation.kinship import build_tree_covariance


class KinshipTests(unittest.TestCase):
    def test_intersection_preserves_fasta_order_and_covariance_scale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tree = Path(tmp) / "tree.nwk"
            tree.write_text("((a:1,b:3):2,extra:7,c:4);", encoding="utf-8")
            names = ["missing_first", "c", "b", "missing_last", "a"]
            with self.assertRaisesRegex(ValueError, "missing 2 FASTA samples"):
                build_tree_covariance(tree, names)
            actual = build_tree_covariance(tree, names, missing_samples="drop")
            expected = build_tree_covariance(tree, ["c", "b", "a"])
        np.testing.assert_array_equal(actual.sample_indices, [1, 2, 4])
        np.testing.assert_allclose(actual.K, expected.K)
        self.assertEqual(actual.excluded_samples, ["missing_first", "missing_last"])
        self.assertEqual(actual.details["n_input_samples"], 5)
        self.assertEqual(actual.mean_diag_before_norm, 4.0)

    def test_intersection_rejects_insufficient_overlap_and_duplicate_tips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tree = Path(tmp) / "tree.nwk"
            tree.write_text("(a:1,b:1);", encoding="utf-8")
            for names in (["missing"], ["a", "missing"]):
                with self.assertRaisesRegex(ValueError, "Fewer than two"):
                    build_tree_covariance(tree, names, missing_samples="drop")
            tree.write_text("(a:1,a:1,b:1);", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate tip label"):
                build_tree_covariance(tree, ["a", "b"], missing_samples="drop")

    def test_tree_covariance_matches_root_to_mrca_oracle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tree = Path(tmp) / "tree.nwk"
            tree.write_text("((s1:1,s2:1):1,(s3:1,s4:1):1);\n", encoding="utf-8")
            result = build_tree_covariance(
                tree,
                sample_names=["s1", "s2", "s3", "s4"],
            )

        expected = np.array(
            [
                [1.0, 0.5, 0.0, 0.0],
                [0.5, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.5],
                [0.0, 0.0, 0.5, 1.0],
            ]
        )
        np.testing.assert_allclose(result.K, expected, rtol=0, atol=1e-12)
        self.assertAlmostEqual(result.mean_diag_before_norm, 2.0)
        self.assertEqual(result.source, "tree_mrca_covariance")

    def test_prepare_kinship_accepts_singular_psd_without_scientific_shrinkage(self) -> None:
        raw = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
        prepared, diagnostics = prepare_kinship(raw)

        np.testing.assert_allclose(prepared, raw, rtol=0, atol=1e-12)
        self.assertEqual(diagnostics.rank, 1)
        self.assertEqual(diagnostics.roundoff_correction, 0.0)

    def test_prepare_kinship_accepts_explicit_no_relatedness_kernel(self) -> None:
        raw = np.zeros((4, 4), dtype=np.float64)
        prepared, diagnostics = prepare_kinship(raw)

        np.testing.assert_array_equal(prepared, raw)
        self.assertEqual(diagnostics.rank, 0)
        self.assertEqual(diagnostics.mean_diagonal, 0.0)

    def test_prepare_kinship_rejects_materially_indefinite_matrix(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive semidefinite"):
            prepare_kinship(np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float64))

if __name__ == "__main__":
    unittest.main()
