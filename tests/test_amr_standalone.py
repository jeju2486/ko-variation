"""Fresh upstream inputs, row identity, target-only fits and checkpoint reuse."""
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from benchmarks import amr_checks as checks


class StandaloneChecksTests(unittest.TestCase):
    def test_pruned_tree_retry_drops_only_unmatched_rows_and_protects_completed_inputs(self):
        checks.load_kovar()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fasta, tree, unitigs = (root/n for n in ('samples.fasta','pruned.treefile','dna.unitigs'))
            fasta.write_text(''.join(f'>s{i}\n{row}\n' for i,row in enumerate(('AAA','CAA','ACA','CCA','CCC'))))
            tree.write_text('(s0:1,s1:1,s2:1,s3:1);')
            unitigs.write_text('0 ACGT\n1 CAGT\n2 GACT\n')
            original_fasta = checks.digest(fasta)
            args = SimpleNamespace(fasta=fasta, tree=tree, unitigs=unitigs, sample_ids=None,
                assemblies_list=None, gfa=None, cuttlefish_json=None, exclude_samples=None,
                tree_rooting='as-is', tree_missing_samples='error', work=None, out=root/'checks',
                min_maf=.05, alpha=.05, min_cell_count=20, bonferroni_tests=None, memory_gb=1, threads=1)
            with self.assertRaisesRegex(ValueError, 'TREE_MISSING_SAMPLES=drop'):
                checks.prepare_standalone(args)
            args.tree_missing_samples = 'drop'
            info, _ = checks.prepare_standalone(args)
            self.assertEqual(info['samples'], ['s0','s1','s2','s3'])
            self.assertEqual(info['excluded'], ['s4'])
            self.assertEqual(info['loci'], 3)  # Preserve original unitig numbering.
            mapping = checks.read_table(args.work/'sample_mapping.tsv')
            self.assertEqual(mapping[-1], {'row':'4','sample':'s4','retained':'0'})
            self.assertEqual(checks.digest(fasta), original_fasta)
            args.fasta, args.work = fasta, None
            args.tree_missing_samples = 'error'
            with self.assertRaisesRegex(ValueError, 'settings changed'):
                checks.prepare_standalone(args)

    def test_raw_gfa_row_order_is_verified_against_cuttlefish_inputs(self):
        checks.load_kovar()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fasta, tree, listing, gfa, metadata = (root/n for n in
                ('raw.fasta','tree.nwk','assemblies_list.txt','graph.gfa1','graph.json'))
            fasta.write_text('>contig00001\nAC\n'*4)
            tree.write_text('(s0:1,s1:1,s2:1,s3:1);')
            names = [f'/old/server/assemblies/s{i}.fna' for i in range(4)]
            listing.write_text('\n'.join(names)+'\n')
            metadata.write_text(json.dumps({'parameters info': {'input': names}}))
            gfa.write_text(''.join(f'P\tReference:{i}_Sequence:contig00001\t0+\t*\n' for i in range(4)))
            args = SimpleNamespace(fasta=fasta, tree=tree, sample_ids=None, assemblies_list=listing,
                gfa=gfa, cuttlefish_json=metadata, exclude_samples=None,
                tree_rooting='as-is', tree_missing_samples='error')
            work = root/'work'; work.mkdir()
            output, _ = checks.standalone_inputs(args, work)
            self.assertEqual([line[1:] for line in output.read_text().splitlines() if line.startswith('>')],
                             ['s0','s1','s2','s3'])
            metadata.write_text(json.dumps({'parameters info': {'input': names[::-1]}}))
            with self.assertRaisesRegex(ValueError, 'input order differs'):
                checks.standalone_inputs(args, work)

    def test_fresh_gff_only_inputs_resume_and_feed_pyseer_without_discovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a = np.array([0]*8+[1]*8, dtype='u1')
            b = np.array([0]*4+[1]*4+[0]*4+[1]*4, dtype='u1')
            c = np.roll(b, 2)
            matrix = np.column_stack((a, b, c))
            rng = random.Random(51)
            dna = [''.join(rng.choice('ACGT') for _ in range(100)) for _ in range(3)]
            fasta, tree, unitigs, ids = (root/name for name in ('raw.fasta','tree.nwk','dna.unitigs','ids.txt'))
            fasta.write_text(''.join('>contig00001\n'+''.join('C' if v else 'A' for v in row)+'\n' for row in matrix))
            ids.write_text(''.join(f's{i}\n' for i in range(16)))
            tree.write_text('('+','.join(f's{i}:1' for i in range(16))+');')
            unitigs.write_text(''.join(f'{i} {sequence[10:71]}\n' for i,sequence in enumerate(dna)))
            gffs = root/'gffs'; gffs.mkdir()
            genes = [('mecA','PBP2a transpeptidase'),('graR','response regulator'),('sen','enterotoxin')]
            for i, row in enumerate(matrix):
                records = ['##gff-version 3\n']
                for j, (gene, product) in enumerate(genes):
                    if row[j]:
                        records.append(f'ctg\ttest\tCDS\t{100*j+1}\t{100*j+100}\t.\t+\t0\t'
                                       f'ID=cds{j};gene={gene};product={product}\n')
                sequence = ''.join(dna[j] if row[j] else 'N'*100 for j in range(3))
                (gffs/f's{i}.gff').write_text(''.join(records)+'##FASTA\n>ctg\n'+sequence+'\n')
            source = {p:checks.digest(p) for p in root.rglob('*') if p.is_file()}
            out = root/'checks/amr_targets'
            args = ['kovar','--fasta',str(fasta),'--unitigs',str(unitigs),'--tree',str(tree),
                    '--sample-ids',str(ids),'--tree-rooting','as-is','--gff-dir',str(gffs),
                    '--out',str(out),'--threads','2','--memory-gb','1']
            checks.main(args)
            identity = json.loads((out/'diagnosis_identity.json').read_text())
            self.assertEqual(identity['source_identity']['kind'], 'standalone')
            self.assertEqual(identity['prepared']['patterns'], 3)
            self.assertEqual(identity['source_identity']['settings']['bonferroni_tests'], 3)
            rows = checks.read_table(out/'pair_comparison.tsv')
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(r['null_origin']=='fresh_diagnostic_fit' for r in rows))
            self.assertEqual(json.loads((out/'summary.json').read_text())['historical_comparison']['mode'], 'standalone')
            manifest_bytes = (out/'diagnosis_identity.json').read_bytes()
            pair_bytes = (out/'pair_comparison.tsv').read_bytes()
            with patch.object(checks,'calculate_pair',side_effect=AssertionError('Unexpected pair refit')):
                checks.main(args)
            self.assertEqual((out/'diagnosis_identity.json').read_bytes(), manifest_bytes)
            self.assertEqual((out/'pair_comparison.tsv').read_bytes(), pair_bytes)
            with patch.object(checks,'calculate_pair',side_effect=AssertionError('Unexpected report fit')):
                checks.main(['kovar','--out',str(out),'--report-only'])
            self.assertEqual((out/'pair_comparison.tsv').read_bytes(), pair_bytes)
            checks.main(['pyseer','--diagnosis',str(out),'--threads','2'])
            pyseer = root/'checks/amr_pyseer_lmm'
            self.assertTrue((pyseer/'pyseer_comparison.tsv').is_file())
            with patch.object(checks,'fit_response',side_effect=AssertionError('Unexpected LMM refit')):
                checks.main(['pyseer','--diagnosis',str(out),'--threads','2'])
            self.assertTrue(all(checks.digest(p)==sha for p,sha in source.items()))
            self.assertFalse((root/'kovar_output').exists())


if __name__ == '__main__':
    unittest.main()
