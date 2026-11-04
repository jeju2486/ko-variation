"""Focused input-join checks for functional display; no external eggNOG run."""
import csv
from pathlib import Path
import tempfile
import unittest

import pandas as pd
from ko_variation.functional_annotation import annotate_nodes, prepare_proteins, read_eggnog


class FunctionalAnnotationTests(unittest.TestCase):
    def test_header_based_join_preserves_aliases_and_cluster_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            eggnog = root / 'annotations'
            # V2 and v3 put these columns in different positions; use names.
            eggnog.write_text('## emapper version\n#query\tGOs\tCOG_category\tPreferred_name\n'
                             'group_1\tGO:0005886\tM\tmprF\n'
                             'mixed\t-\tMT\tother\n')
            nodes = pd.DataFrame(dict(gene=['group_1', 'mixed', 'group_2'],
                label=['group_1', 'araJ; lmrS', 'group_2'], product=['', '', ''],
                n_coding_loci=[1, 1, 0], n_nearby_loci=[0, 0, 1]))
            catalogue = root / 'catalogue.tsv'
            catalogue.write_text('gene\tlabel\tproduct\ngroup_1\tmprF\tflippase\n')
            display = annotate_nodes(nodes, catalogue, eggnog).set_index('gene')
            self.assertEqual(display.loc['group_1', 'display_label'], 'mprF')
            self.assertEqual(display.loc['group_1', 'function_category'], 'Cell envelope')
            self.assertEqual(display.loc['mixed', 'display_label'], 'araJ; lmrS')
            self.assertEqual(display.loc['mixed', 'function_category'], 'Multiple categories')
            self.assertEqual(display.loc['group_2', 'display_label'], 'near Unannotated protein')
            self.assertEqual(set(display.index), set(nodes.gene))
            with self.assertRaisesRegex(ValueError, 'No eggNOG query IDs match'):
                annotate_nodes(nodes.assign(gene=['x', 'y', 'z']), eggnog=eggnog)
            eggnog.write_text('#query\tCOG_category\tGOs\nA\tM\t-\nA\tT\t-\n')
            with self.assertRaisesRegex(ValueError, 'Duplicate eggNOG'):
                read_eggnog(eggnog)

    def test_proteins_use_membership_not_gene_name_and_report_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            effects = root / 'effects.tsv'
            effects.write_text('u_gene\tv_gene\ngroup_1\tgroup_2\ngroup_1\tgroup_3\n')
            panaroo = root / 'panaroo.csv'
            with panaroo.open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerows([['Gene', 'Non-unique Gene name', 'sample'],
                    ['group_1', 'sameName', 'id1_len'],
                    ['group_2', 'sameName', 'missing'],
                    ['group_3', 'other', 'id3'],
                    ['unselected', 'other', 'id3']])
            bakta = root / 'bakta'
            (bakta / 'sample').mkdir(parents=True)
            (bakta / 'sample/sample.faa').write_text('>id1 description\nMKKL\n>id3\nMVV\n')
            out = root / 'functional'
            summary = prepare_proteins(effects, panaroo, bakta, out)
            self.assertEqual(summary['representative_proteins'], 1)
            self.assertEqual((out / 'network_proteins.faa').read_text(), '>group_1\nMKKL\n')
            manifest = pd.read_csv(out / 'protein_manifest.tsv', sep='\t').set_index('gene')
            self.assertEqual(manifest.loc['group_2', 'status'], 'NO_PROTEIN_MATCH')
            self.assertEqual(manifest.loc['group_3', 'status'], 'NO_PROTEIN_MATCH')


if __name__ == '__main__':
    unittest.main()
