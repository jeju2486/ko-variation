"""Focused actual-pyseer comparison, resume and frozen-vector checks."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from benchmarks import diagnose_amr_pyseer as script


class PyseerDiagnosisTests(unittest.TestCase):
    def fixture(self, root):
        work, diag = root/'work', root/'diagnosis'
        work.mkdir(); diag.mkdir()
        rng = np.random.default_rng(31)
        matrix = rng.binomial(1, .4, size=(4, 80)).astype('u1')
        matrix[0] = np.where(rng.random(80) < .8, matrix[2], matrix[0])
        np.packbits(matrix, axis=1, bitorder='little').tofile(work/'patterns.bin')
        K = .2*np.eye(80) + .8*(np.arange(80)[:, None]//20 == np.arange(80)[None, :]//20)
        np.save(work/'K.npy', K)
        info = dict(n=80, samples=[f's{i}' for i in range(80)], patterns=4, packed_width=10,
                    products={k: script.digest(work/k) for k in ('patterns.bin', 'K.npy')})
        state = dict(prepared=info, source_identity=dict(settings={'alpha': .05}, work_directory=str(work)))
        script.atomic_json(diag/'diagnosis_identity.json', state)
        rows = []
        for u, v in [(0, 2), (1, 2), (0, 3)]:
            x, y = matrix[u].astype(int), matrix[v].astype(int)
            n11 = int(x @ y)
            rows.append(dict(target_pairs='mecA--graR', predictor_pattern=u, response_pattern=v,
                             n00=80-int(x.sum())-int(y.sum())+n11, n10=int(x.sum())-n11,
                             n01=int(y.sum())-n11, n11=n11, bonferroni_tests=6, p_cutoff=.05/6,
                             eligible_original=0, eligible_new=1, p_exact=.7, score_status='OK', beta=9.9))
        script.table(diag/'pair_comparison.tsv', rows, list(rows[0]))
        return work, diag, matrix

    def test_actual_lmm_matches_pyseer_api_and_resumes_without_fitting(self):
        import pandas as pd
        from pyseer.lmm import initialise_lmm, fit_lmm_block
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            work, diag, matrix = self.fixture(root)
            out = root/'comparison'
            args = ['--diagnosis', str(diag), '--work', str(work), '--out', str(out), '--threads', '2']
            original_hashes = {p: script.digest(p) for d in (work, diag) for p in d.iterdir()}
            with patch.object(Path, 'is_relative_to', side_effect=AttributeError('Python 3.8'), create=True):
                script.main(args)
            rows = script.read_table(out/'pyseer_comparison.tsv')
            self.assertEqual(len(rows), 3)
            phenotype = pd.Series(matrix[2].astype(float), index=[f's{i}' for i in range(80)])
            _, model, h2 = initialise_lmm(phenotype, pd.DataFrame(), str(out/'covariance.tsv'))
            reference = fit_lmm_block(model, h2, matrix[[0, 1]].T.astype(float))
            self.assertAlmostEqual(float(rows[0]['lmm_p']), reference['p_values'][0], places=12)
            self.assertAlmostEqual(float(rows[1]['lmm_p']), reference['p_values'][1], places=12)
            self.assertEqual(rows[0]['beta'], '9.9')  # Saved KOVAR coefficient stays intact.
            self.assertTrue(all(r['lmm_status'] == 'OK' for r in rows))
            checkpoint = script.digest(out/'pyseer_comparison.tsv')
            with patch.object(script, 'fit_response', side_effect=AssertionError('Unexpected refit')):
                script.main(args)
            self.assertEqual(script.digest(out/'pyseer_comparison.tsv'), checkpoint)
            self.assertEqual(json.loads((out/'summary.json').read_text())['resumed_response_models'], 2)
            self.assertTrue(all(script.digest(p) == sha for p, sha in original_hashes.items()))

    def test_mismatched_vectors_are_rejected_before_fitting(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            work, diag, _ = self.fixture(root)
            rows = script.read_table(diag/'pair_comparison.tsv')
            rows[0]['n11'] = int(rows[0]['n11']) + 1
            script.table(diag/'pair_comparison.tsv', rows, list(rows[0]))
            with self.assertRaisesRegex(ValueError, 'do not reproduce the saved cells'):
                script.main(['--diagnosis', str(diag), '--work', str(work), '--out', str(root/'output')])


if __name__ == '__main__':
    unittest.main()
