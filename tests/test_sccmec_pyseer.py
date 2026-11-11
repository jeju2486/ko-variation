"""Typing-call semantics, typing restart and actual fixed-response pyseer fit."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from benchmarks import sccmec_pyseer as script


class SccmecPyseerTests(unittest.TestCase):
    def test_uncertain_carriage_is_not_absence(self):
        for mec, cassette, targets, expected in [
                ('+', 'IV', 'mecA,ccrA2', 1), ('+', 'multiple', 'mecA,ccrA2', 1),
                ('+', '-', 'mecA', None), ('-', '-', '-', 0),
                ('-', 'III', '-', None), ('-', '-', 'ccrA1', None),
                ('+', 'untypable', 'mecA', None)]:
            r = script.interpret(dict(mecA=mec, type=cassette, subtype='-', regions='-', targets=targets), [])
            self.assertEqual(r['SCCmec_present'], expected)
            self.assertEqual(r['mecA_present'], int(mec == '+'))

    def test_embedded_fasta_typing_checkpoint_skips_subprocess(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            gff = root/'sample.gff'
            gff.write_text('##gff-version 3\n##FASTA\n>ctg\nACGTTGCA\n')
            original = script.common.digest(gff)
            job = (0, 'sample', gff, 'embedded_gff_fasta', root/'output', 'sccmec', {})
            def run(command, **kwargs):
                source = Path(command[command.index('--input')+1])
                self.assertEqual(source.read_text(), '>ctg\nACGTTGCA\n')
                folder = Path(command[command.index('--outdir')+1])
                script.common.table(folder/'isolate.tsv',
                    [dict(sample='isolate', mecA='+', type='IV', subtype='IVa', regions='IVa', targets='mecA,ccrA2')],
                    ['sample', 'mecA', 'type', 'subtype', 'regions', 'targets'])
                script.common.table(folder/'isolate.targets.details.tsv', [{'targets': 'mecA,ccrA2'}], ['targets'])
                return subprocess.CompletedProcess(command, 0)
            with patch.object(script.subprocess, 'run', side_effect=run) as process:
                first = script.classify(job)
                second = script.classify(job)
                self.assertEqual(process.call_count, 1)
            self.assertEqual(first['SCCmec_present'], 1)
            self.assertTrue(second['reused'])
            self.assertFalse((root/'output/isolates/000000/embedded_input.fna').exists())
            self.assertEqual(script.common.digest(gff), original)

    def test_fixed_response_sample_subsetting_actual_lmm_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            work, diag, assemblies = root/'work', root/'diagnosis', root/'assemblies'
            for path in (work, diag, assemblies):
                path.mkdir()
            rng = np.random.default_rng(71)
            matrix = rng.binomial(1, .4, size=(3, 40)).astype('u1')
            gene = np.array([0]*20+[1]*20)
            np.packbits(matrix, axis=1, bitorder='little').tofile(work/'patterns.bin')
            K = .2*np.eye(40) + .8*(np.arange(40)[:, None]//5 == np.arange(40)[None, :]//5)
            np.save(work/'K.npy', K)
            samples = [f's{i}' for i in range(40)]
            for sample in samples:
                (assemblies/(sample+'.fna')).write_text('>ctg\nACGTTGCA\n')
            info = dict(n=40, samples=samples, patterns=3, packed_width=5,
                        products={k: script.common.digest(work/k) for k in ('patterns.bin', 'K.npy')})
            script.common.atomic_json(diag/'diagnosis_identity.json',
                dict(prepared=info, source_identity=dict(settings={'alpha': .05}, work_directory=str(work))))
            rows = [dict(target=t, pattern=i, unitig=i, annotation_class='coding',
                         n_matching_references_binary_absent=0, clusters=t)
                    for i, t in enumerate(('mecA', 'graR', 'sen'))]
            script.common.table(diag/'target_unitigs.tsv', rows, list(rows[0]))
            original = {p: script.common.digest(p) for d in (work, diag, assemblies) for p in d.iterdir()}
            out = root/'result'
            args = ['--diagnosis', str(diag), '--work', str(work), '--assemblies', str(assemblies),
                    '--out', str(out), '--threads', '2']
            def calls(job):
                i, sample = job[:2]
                return dict(sample=sample, source=str(job[2]), source_kind='assembly', typing_status='OK',
                            mecA_present=int(gene[i]), SCCmec_present=None if i in (0, 21) else int(gene[i]),
                            carriage_status='UNCERTAIN_CASSETTE' if i in (0, 21) else 'SUPPORTED_PRESENT' if gene[i] else 'NOT_DETECTED_BY_PANEL',
                            source_sha256='fixture', error='')
            with patch.object(script.shutil, 'which', return_value=str(root/'bin/sccmec')), \
                    patch.object(script.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, 'mock-typing-v1', '')), \
                    patch.object(script, 'classify', side_effect=calls):
                script.main(args)
                main_hash = script.common.digest(out/'SCCmec/associations.tsv')
                with patch('pyseer.lmm.initialise_lmm', side_effect=AssertionError('Unexpected refit')):
                    script.main(args)
            self.assertEqual(script.common.digest(out/'SCCmec/associations.tsv'), main_hash)
            summary = json.loads((out/'summary.json').read_text())
            primary, sensitivity = summary['phenotypes']
            self.assertEqual((primary['n'], primary['positive'], primary['negative'], primary['excluded']), (38, 19, 19, 2))
            self.assertEqual((sensitivity['n'], sensitivity['positive']), (40, 20))
            results = script.common.read_table(out/'SCCmec/associations.tsv')
            self.assertEqual({r['response'] for r in results}, {'SCCmec'})
            self.assertEqual({r['targets'] for r in results}, {'graR', 'sen'})
            self.assertTrue(all(r['status'] == 'OK' for r in results))
            # Covariance rows must follow the same exclusions and original order.
            import pandas as pd
            kept = [i for i in range(40) if i not in (0, 21)]
            actual = pd.read_csv(out/'SCCmec/covariance.tsv', sep='\t', index_col=0)
            self.assertEqual(list(actual.index), [samples[i] for i in kept])
            np.testing.assert_allclose(actual.to_numpy(), K[np.ix_(kept, kept)])
            self.assertTrue(all(script.common.digest(p) == sha for p, sha in original.items()))


if __name__ == '__main__':
    unittest.main()
