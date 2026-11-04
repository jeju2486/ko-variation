"""Focused interruption/resume regressions; no large genomic run required."""
import io
from itertools import product
import json
from pathlib import Path
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np
import pandas as pd

from ko_variation import postprocess
from ko_variation.annotation_cli import main as annotation_main
from ko_variation.workflow_cache import stage_identity, cache_fields, cache_matches


class DownstreamResumeTests(unittest.TestCase):
    def setUp(self):
        self.X = np.repeat(np.array(list(product([0, 1], repeat=3)), dtype=np.uint8), 10, axis=0)
        self.K = np.zeros((80, 80))
        self.signals = pd.DataFrame(dict(u=[0, 0, 1], v=[1, 2, 2], n11=[20]*3,
                                         n10=[20]*3, n01=[20]*3, n00=[20]*3))

    def test_interrupted_run_matches_uninterrupted_and_skips_saved_pair(self):
        original = postprocess._fit_pair_effect
        baseline = postprocess.fit_selected_effects(self.signals, self.X, self.K)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'fits.sqlite'
            calls = []
            def interrupt(row, *args):
                calls.append((row.u, row.v))
                if len(calls) == 2:
                    raise KeyboardInterrupt()
                return original(row, *args)
            with patch.object(postprocess, '_fit_pair_effect', side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    postprocess.fit_selected_effects(self.signals, self.X, self.K, checkpoint_file=checkpoint)
            with patch.object(postprocess, '_fit_pair_effect', wraps=original) as fitter:
                resumed = postprocess.fit_selected_effects(self.signals, self.X, self.K,
                                                          checkpoint_file=checkpoint, resume=True)
            self.assertEqual(fitter.call_count, 2)
            self.assertEqual(resumed.attrs['effect_checkpoint']['recovered_pairs'], 1)
            pd.testing.assert_frame_equal(resumed, baseline)
            with patch.object(postprocess, '_fit_pair_effect', side_effect=AssertionError('must not refit')):
                completed = postprocess.fit_selected_effects(self.signals, self.X, self.K,
                                                            checkpoint_file=checkpoint, resume=True)
            pd.testing.assert_frame_equal(completed, baseline)
            with self.assertRaisesRegex(ValueError, 'do not match'):
                postprocess.fit_selected_effects(self.signals, self.X, self.K, confidence=0.9,
                                                checkpoint_file=checkpoint, resume=True)

    def test_failed_attempt_is_saved_and_progress_reports_during_slow_fit(self):
        original = postprocess._fit_pair_effect
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'fits.sqlite'
            def failed(row, *args):
                payload = original(row, *args)
                payload['effect_status'] = 'PQL_NOT_CONVERGED'
                payload['adjusted_beta'] = np.nan
                time.sleep(0.04)
                return payload
            stream = io.StringIO()
            with patch.object(postprocess, '_fit_pair_effect', side_effect=failed), redirect_stdout(stream):
                result = postprocess.fit_selected_effects(self.signals.iloc[:1], self.X, self.K,
                                                         checkpoint_file=checkpoint, progress=True,
                                                         progress_every=1, progress_seconds=0.01)
            self.assertIn('current=pair(0,1)', stream.getvalue())
            self.assertIn('1/1 (100.0%)', stream.getvalue())
            with patch.object(postprocess, '_fit_pair_effect', side_effect=AssertionError('must not retry')):
                resumed = postprocess.fit_selected_effects(self.signals.iloc[:1], self.X, self.K,
                                                          checkpoint_file=checkpoint, resume=True)
            pd.testing.assert_frame_equal(resumed, result)

    def test_empty_selection_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'empty.sqlite'
            first = postprocess.fit_selected_effects(self.signals.iloc[:0], self.X, np.zeros((0, 0)),
                                                    checkpoint_file=checkpoint)
            second = postprocess.fit_selected_effects(self.signals.iloc[:0], self.X, np.zeros((0, 0)),
                                                     checkpoint_file=checkpoint, resume=True)
            pd.testing.assert_frame_equal(first, second)

    def test_stage_cache_rejects_modified_input_or_output(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output, manifest = [Path(folder) / name for name in ('source', 'output', 'cache.json')]
            source.write_text('input')
            output.write_text('result')
            identity = stage_identity('test', [source], dict(cutoff=10000))
            manifest.write_text(json.dumps(cache_fields(identity, [output])))
            self.assertTrue(cache_matches(manifest, identity, [output]))
            output.write_text('edited')
            self.assertFalse(cache_matches(manifest, identity, [output]))
            output.write_text('result')
            source.write_text('new input')
            self.assertFalse(cache_matches(manifest, stage_identity('test', [source], dict(cutoff=10000)), [output]))

    def test_cli_checkpoint_resume_and_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            X = np.repeat(np.array(list(product([0, 1], repeat=2)), dtype=np.uint8), 4, axis=0)
            fasta = folder / 'binary.fa'
            fasta.write_text(''.join(f'>s{i}\n' + ''.join('C' if state else 'A' for state in row) + '\n'
                                     for i, row in enumerate(X)))
            tree = folder / 'tree.nwk'
            tree.write_text('(' + ','.join(f's{i}:1' for i in range(len(X))) + ');')
            score = folder / 'score.tsv'
            pd.DataFrame(dict(u=[0], v=[1], n11=[4], n10=[4], n01=[4], n00=[4],
                              status=['OK'], p_primary=[1e-8], distance=[20000], n_tests=[100])).to_csv(score, sep='\t', index=False)
            prefix = folder / 'output'
            args = ['--results', str(score), '--fasta', str(fasta), '--tree', str(tree),
                    '--out', str(prefix), '--ld-distance', '10000', '--no-progress']
            with redirect_stdout(io.StringIO()):
                self.assertEqual(annotation_main(args), 0)
                with patch.object(postprocess, '_fit_pair_effect', side_effect=AssertionError('must reuse saved fit')):
                    self.assertEqual(annotation_main(args + ['--resume']), 0)
            provenance = json.loads(Path(str(prefix) + '.selection.json').read_text())
            self.assertEqual(provenance['effect_checkpoint']['recovered_pairs'], 1)
            self.assertEqual(provenance['selected_rows'], 1)

    def test_cli_uses_scanner_sample_intersection_for_refits(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            X = np.repeat(np.array(list(product([0, 1], repeat=2)), dtype=np.uint8), 4, axis=0)
            fasta, tree, score = [folder / name for name in ('binary.fa', 'tree.nwk', 'score.tsv')]
            fasta.write_text('>missing\nCC\n' + ''.join(
                f'>s{i}\n' + ''.join('C' if state else 'A' for state in row) + '\n'
                for i, row in enumerate(X)))
            tree.write_text('(' + ','.join(f's{i}:1' for i in reversed(range(len(X)))) + ');')
            pd.DataFrame(dict(u=[0], v=[1], n11=[4], n10=[4], n01=[4], n00=[4],
                              status=['OK'], p_primary=[1e-8], distance=[20000], n_tests=[100],
                              n_samples=[16])).to_csv(score, sep='\t', index=False)
            prefix = folder / 'output'
            args = ['--results', str(score), '--fasta', str(fasta), '--tree', str(tree),
                    '--out', str(prefix), '--tree-missing-samples', 'drop', '--no-progress']
            with redirect_stdout(io.StringIO()):
                self.assertEqual(annotation_main(args), 0)
                with patch.object(postprocess, '_fit_pair_effect', side_effect=AssertionError('must reuse saved fit')):
                    self.assertEqual(annotation_main(args + ['--resume']), 0)
            provenance = json.loads(Path(str(prefix) + '.selection.json').read_text())
            self.assertEqual(provenance['samples']['n_samples'], 16)
            self.assertEqual(provenance['samples']['excluded_samples'], ['missing'])
            self.assertEqual(provenance['effect_checkpoint']['recovered_pairs'], 1)


if __name__ == '__main__':
    unittest.main()
