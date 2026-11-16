"""Focused checks of exception selection and non-invasive iteration tracing."""
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from benchmarks import investigate_tree_null as investigation


class InvestigationTests(unittest.TestCase):
    def test_selection_keeps_status_and_other_accuracy_failures(self):
        b = investigation.b
        limits = dict(probability_tolerance=1e-7, intercept_tolerance=1e-6,
            objective_tolerance=1e-7, tau_tolerance=1e-5, variance_tolerance=1e-6)
        payloads = []
        for index, probability, intercept, status in (
                (0, 2e-7, 0., 'OK'), (1, 1e-8, 2e-6, 'OK'),
                (2, 0., 0., 'PQL_NOT_CONVERGED')):
            comparison = dict(index=index, status_match=status == 'OK', jointly_successful=status == 'OK',
                max_probability_difference=probability, intercept_abs_difference=intercept,
                objective_relative_difference=0., tau_relative_difference=0., max_variance_relative_difference=0.)
            payloads.append(dict(comparison=comparison, rows=[dict(method='dense', status='OK'),
                dict(method='tree', status=status)]))
        self.assertEqual(investigation.targets(payloads * 3, limits, 5e-7), [1, 2])

    def test_trace_matches_fitter_and_patches_are_restored(self):
        b = investigation.b
        with tempfile.TemporaryDirectory() as temporary:
            treefile = Path(temporary) / 'tree.nwk'
            treefile.write_text('((a:1,b:1):1,(c:1,d:1):1);')
            names = ['a', 'b', 'c', 'd']
            tree = b.TreeAlgebra.from_newick(treefile, names, 'python')
            K = b.kinship.build_tree_covariance(treefile, names).K
            y = np.array([0., 0., 1., 1.])
            b.STATE.update(tree=tree, K=K, eigen=b.glmm.prepare_kinship_eigensystem(K))
            originals = (b.glmm._weighted_eigen, b.glmm._profile_tau, b.optimize.minimize_scalar)
            reference, _ = b.glmm.fit_null_glmm(0, y, K, kinship_eigensystem=b.STATE['eigen'])
            for variant in investigation.VARIANTS:
                fit, _, row, trace = investigation.traced_fit(0, y, variant)
                self.assertEqual(fit.iterations, len(trace))
                self.assertAlmostEqual(fit.max_eta_change, trace[-1]['max_eta_change'], places=12)
                if fit.status == 'OK':
                    self.assertGreaterEqual(trace[-1]['consecutive_passes'], 2)
                self.assertEqual(originals, (b.glmm._weighted_eigen, b.glmm._profile_tau, b.optimize.minimize_scalar))
                if not variant[1]:
                    np.testing.assert_array_equal(fit.fitted_probability, reference.fitted_probability)
                    self.assertEqual(fit.tau, reference.tau)
                self.assertTrue(math.isfinite(row['seconds']))


if __name__ == '__main__':
    unittest.main()
