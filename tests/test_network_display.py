"""Focused graph-contract/export checks for the compact network display."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from ko_variation.network import build_gene_network, export_gene_network
from ko_variation.network_cli import main as redraw_main


def example_signals():
    beta = np.array([.1, -.2, 2, 4, np.nan])
    return pd.DataFrame(dict(u=[0, 2, 4, 6, 8], v=[1, 3, 5, 7, 9],
        u_gene=['A', 'A', 'A', 'C', 'D'], v_gene=['B', 'B', 'C', 'D', 'group_999'],
        u_label=['geneA', 'geneA', 'geneA', 'geneC', 'geneD'],
        v_label=['geneB', 'geneB', 'geneC', 'geneD', 'group_999'],
        p_primary=[1e-12, 1e-20, 1e-40, 1e-8, 1e-10], physical_distance=[20000]*5,
        adjusted_beta=beta, adjusted_beta_se=[.1]*5, adjusted_odds_ratio=np.exp(beta),
        effect_status=['OK', 'OK', 'OK', 'OK', 'PQL_NOT_CONVERGED']))


class NetworkDisplayTests(unittest.TestCase):
    def test_categories_preserve_edges_representative_and_small_pvalues(self):
        nodes, edges = build_gene_network(example_signals())
        with tempfile.TemporaryDirectory() as folder:
            paths = export_gene_network(nodes, edges, Path(folder)/'map', title='', dpi=72, strength_cutoffs=[.5, 3])
            payload = json.loads(paths['layout'].read_text())
            drawn = {(row['gene_a'], row['gene_b']): row for row in payload['edges']}
            self.assertEqual(set(drawn), set(zip(edges.gene_a, edges.gene_b)))
            self.assertEqual(drawn[('A','B')]['strength_category'], 'weak')
            self.assertEqual(drawn[('A','C')]['strength_category'], 'medium')
            self.assertEqual(drawn[('C','D')]['strength_category'], 'strong')
            self.assertEqual(drawn[('D','group_999')]['strength_category'], 'unestimated')
            self.assertEqual(drawn[('A','B')]['representative_adjusted_beta'], -.2)
            self.assertEqual(drawn[('A','B')]['adjusted_direction'], 'mixed')
            self.assertEqual(drawn[('A','C')]['min_significance'], 1e-40)
            self.assertEqual(drawn[('D','group_999')]['representative_adjusted_beta'], None)
            self.assertTrue(all(3 <= row['radius'] <= 5 for row in payload['nodes']))
            generic = next(row for row in payload['nodes'] if row['gene']=='group_999')
            self.assertEqual(generic['display_label'], 'Unannotated protein')
            document = paths['html'].read_text()
            self.assertNotIn('<script src=', document)
            self.assertNotIn('Group: Panaroo cluster', document)
            self.assertIn('d3.forceLink', document)
            self.assertTrue(paths['png'].read_bytes().startswith(b'\x89PNG'))
            self.assertIn('<svg', paths['svg'].read_text())

    def test_empty_network_and_tied_coefficients(self):
        nodes, edges = build_gene_network(example_signals().iloc[:0])
        with tempfile.TemporaryDirectory() as folder:
            paths = export_gene_network(nodes, edges, Path(folder)/'empty', dpi=72)
            self.assertEqual(json.loads(paths['layout'].read_text())['edges'], [])
            signals = example_signals().iloc[:4].copy()
            signals['adjusted_beta'] = 1
            nodes, edges = build_gene_network(signals)
            paths = export_gene_network(nodes, edges, Path(folder)/'tied', dpi=72)
            payload = json.loads(paths['layout'].read_text())
            self.assertEqual({row['strength_category'] for row in payload['edges']}, {'weak'})

    def test_redraw_does_not_touch_effect_table(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            effects = folder/'saved.distal.tsv'
            example_signals().to_csv(effects, sep='\t', index=False)
            before = effects.read_bytes()
            eggnog = folder/'network.emapper.annotations'
            eggnog.write_text('#query\tCOG_category\tGOs\nA\tM\tGO:0005886\nB\tS\t-\nC\tMT\t-\nD\tV\t-\n')
            result = redraw_main(['--effects', str(effects), '--out', str(folder/'redrawn'), '--dpi', '72', '--eggnog', str(eggnog)])
            self.assertEqual(result, 0)
            self.assertEqual(effects.read_bytes(), before)
            metadata = json.loads((folder/'redrawn.network.json').read_text())
            self.assertFalse(metadata['refitted'])
            payload = json.loads((folder/'redrawn.layout.json').read_text())
            displayed = {node['gene']: node for node in payload['nodes']}
            self.assertEqual(displayed['A']['function_category'], 'Cell envelope')
            self.assertEqual(displayed['C']['function_category'], 'Multiple categories')
            self.assertEqual(displayed['group_999']['function_category'], 'Unassigned')
            self.assertNotEqual(displayed['A']['color'], displayed['B']['color'])
            self.assertIn('eggnog', payload['meta']['annotation_sources'])


if __name__ == '__main__':
    unittest.main()
