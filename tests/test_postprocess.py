"""Small downstream smoke checks; model numerics have existing core tests."""
import unittest
import numpy as np
import pandas as pd
from ko_variation.postprocess import SelectionConfig, select_distal_signals, fit_selected_effects
from ko_variation.network import build_gene_network


class DownstreamSmokeTests(unittest.TestCase):
    def test_default_and_configurable_thresholds(self):
        frame = pd.DataFrame(dict(u=[0, 0, 1], v=[1, 2, 2], status=["OK"]*3,
                                  p_primary=[0.005, 0.006, 0.0001], n_tests=[10]*3,
                                  distance=[1, 100, 0], n11=[30]*3, n10=[10]*3,
                                  n01=[10]*3, n00=[30]*3))
        selected = select_distal_signals(frame)
        self.assertEqual(selected[["u", "v"]].values.tolist(), [[0, 1]])
        self.assertAlmostEqual(selected.selection_threshold.iloc[0], 0.005)
        self.assertFalse(any(c.startswith("raw_") for c in selected))
        selected = select_distal_signals(frame, SelectionConfig(significance_threshold=0.1, ld_distance=10))
        self.assertEqual(selected[["u", "v"]].values.tolist(), [[0, 2]])

    def test_refit_and_failed_input(self):
        X = np.array([[1, 1]]*30 + [[1, 0]]*10 + [[0, 1]]*10 + [[0, 0]]*30, dtype=np.uint8)
        signals = pd.DataFrame(dict(u=[0], v=[1], n11=[30], n10=[10], n01=[10], n00=[30],
                                    p_primary=[0.001], u_gene=["A"], v_gene=["B"]))
        effects = fit_selected_effects(signals, X, np.zeros((80, 80)))
        self.assertEqual(effects.effect_status.iloc[0], "OK")
        self.assertAlmostEqual(effects.adjusted_odds_ratio.iloc[0], 9, places=5)
        self.assertAlmostEqual(effects.adjusted_beta_se.iloc[0], np.sqrt(2/30 + 2/10), places=5)
        nodes, edges = build_gene_network(effects)
        self.assertEqual(edges.n_adjusted_pairs.iloc[0], 1)
        self.assertEqual(edges.adjusted_direction.iloc[0], "positive")
        self.assertEqual(len(nodes), 2)
        wrong = X.copy()
        wrong[0, 0] = 0
        with self.assertRaisesRegex(ValueError, "joint counts differ"):
            fit_selected_effects(signals, wrong, np.zeros((80, 80)))


if __name__ == "__main__":
    unittest.main()
