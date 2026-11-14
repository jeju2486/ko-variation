"""Small end-to-end target audit: exact geometry, fitting and source preservation."""
import csv
import gzip
import json
from pathlib import Path
import random
import sqlite3
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from benchmarks import amr_checks as diagnostic
diagnostic.load_kovar()
from benchmarks.amr_checks import run_kovar as main, identify, identify_all, original_reuse, response_null
from ko_variation.discovery import main as discover
from ko_variation.discovery_store import digest


class TargetDiagnosisTests(unittest.TestCase):
    def test_unreadable_baseline_uses_only_positive_export_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            identity = dict(inputs={'fasta':'frozen'},settings={'alpha':.05},null_model_source_sha256='same-model')
            (root/'run_identity.json').write_text(json.dumps(identity))
            fields = ['u','v','predictor_pattern','response_pattern','significance','p_exact','p_lower','p_upper',
                      'beta','beta_status','min_distance','connected_genomes','distance_status']
            with gzip.open(root/'associations.tsv.gz','wt',newline='') as stream:
                writer = csv.DictWriter(stream,fields,delimiter='\t'); writer.writeheader()
                writer.writerow(dict(zip(fields,[10,20,0,1,'SIGNIFICANT',1e-12,1e-12,1e-12,2.5,'OK',15000,30,'DISTAL'])))
            class Broken:
                def execute(self,*args): raise sqlite3.DatabaseError('database disk image is malformed')
            old,distances,status = diagnostic.historical_comparison(SimpleNamespace(results=root,out=root,include_nearby=False),
                Broken(),{(0,1),(0,2)},[{'unitig':10},{'unitig':20}],identity)
            self.assertEqual(status['mode'],'significant_export')
            self.assertEqual(old[0,1]['status'],'SIGNIFICANT_FROM_EXPORT')
            self.assertEqual(old[0,1]['beta'],2.5)
            self.assertNotIn((0,2),old)
            self.assertEqual(distances[10,20],(15000,30,'DISTAL'))

    def test_require_original_ignores_library_versions_but_not_fitting_model(self):
        from ko_variation.null_identity import null_source_hash
        identity = dict(numpy='different',scipy='different',null_model_source_sha256=null_source_hash())
        self.assertFalse(original_reuse(identity,'auto')['enabled'])
        reuse = original_reuse(identity,'require')
        self.assertTrue(reuse['enabled'])
        self.assertFalse(reuse['library_versions_match'])
        identity['null_model_source_sha256'] = 'different fitting model'
        with self.assertRaisesRegex(ValueError,'null-fitting code differs'):
            original_reuse(identity,'require')

    def test_required_missing_original_never_fits_a_replacement(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db = sqlite3.connect(':memory:')
            db.executescript('CREATE TABLE null_aliases(pid,canonical); CREATE TABLE null_inventory(pid,source); '
                             'CREATE TABLE external_nulls(pid,path);')
            state = dict(work=root,out=root,source=db,response=None,reuse={'enabled':True,'policy':'require'})
            with patch.dict(diagnostic._KOVAR_WORKER,state,clear=True), patch.object(diagnostic,'fit_null_glmm') as fit:
                with self.assertRaisesRegex(ValueError,'no refit performed'):
                    response_null(5)
                fit.assert_not_called()
            db.close()

    def test_resistance_meca_is_not_the_adaptor(self):
        self.assertIsNone(identify('mecA', 'adaptor protein MecA'))
        self.assertEqual(identify('mecA', 'PBP2a family peptidoglycan transpeptidase MecA'), 'mecA')
        self.assertEqual(identify('group_25;tet(M)', 'ribosomal protection protein'), 'tetM')
        self.assertEqual(identify_all('mecA;graR','PBP2a transpeptidase;response regulator'), {'mecA','graR'})

    def test_full_pool_mapping_score_and_checkpoint_do_not_change_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = random.Random(740)
            slots = [''.join(rng.choice('ACGT') for _ in range(180)) for _ in range(4)]
            a = np.array([0]*45+[1]*35, dtype='u1')
            b = np.array([0]*35+[1]*10+[0]*10+[1]*25, dtype='u1')
            c = np.roll(b, 7)
            t = np.array([1]*4+[0]*76, dtype='u1')
            matrix = np.column_stack((a,a,b,c,t))
            fasta, tree, unitigs = root/'sample_ids.fasta', root/'tree.nwk', root/'unitigs'
            fasta.write_text(''.join(f'>s{i}\n'+''.join('C' if x else 'A' for x in row)+'\n' for i,row in enumerate(matrix)))
            tree.write_text('('+','.join(f's{i}:1' for i in range(80))+');')
            reverse = slots[2].translate(str.maketrans('ACGT','TGCA'))[::-1]
            unitigs.write_text('\n'.join(f'{i} {s}' for i,s in enumerate(
                [slots[0][40:101],slots[0][40:101],slots[1][40:101],reverse,slots[3][40:101]]))+'\n')
            source = root/'source'
            discover(['--fasta',str(fasta),'--tree',str(tree),'--unitigs',str(unitigs),'--out',str(source),
                '--min-cell-count','20','--screen-p','0.05','--prefit-nulls','all','--coarse-screen','none','--search','exhaustive',
                '--threads','1','--memory-gb','1'])
            gffs = root/'gffs'; gffs.mkdir()
            targets = ['mecA','graR','sen','tet(M)']
            products = ['PBP2a family peptidoglycan transpeptidase MecA','response regulator GraR',
                        'staphylococcal enterotoxin N','tetracycline ribosomal protection protein']
            with (root/'panaroo.csv').open('w',newline='') as stream:
                writer=csv.writer(stream)
                writer.writerow(['Gene','Non-unique Gene name','Annotation']+[f's{i}' for i in range(80)])
                for j, (target,product) in enumerate(zip(targets,products)):
                    states = [a,b,c,t][j]
                    writer.writerow([target,target,product]+[f'cds{j}' if states[i] else '' for i in range(80)])
            for i in range(80):
                dna = ''.join(slot if states[i] else 'N'*180 for slot,states in zip(slots,[a,b,c,t]))
                lines = ['##gff-version 3\n']
                for j, states in enumerate([a,b,c,t]):
                    if states[i]:
                        lines.append(f'ctg\tfixture\tCDS\t{j*180+31}\t{j*180+150}\t.\t+\t0\t'
                                     f'ID=cds{j};gene={targets[j]};product={products[j]}\n')
                (gffs/f's{i}.gff').write_text(''.join(lines)+'##FASTA\n>ctg\n'+dna+'\n')
            before = {p:digest(p) for p in source.rglob('*') if p.is_file() and '-shm' not in p.name}
            args = ['--results',str(source),'--fasta',str(fasta),'--unitigs',str(unitigs),
                '--panaroo',str(root/'panaroo.csv'),'--gff-dir',str(gffs),'--map-dir',str(root/'map'),
                '--out',str(root/'diagnosis'),'--nearby-bp','5','--threads','2']
            main(args)
            with (root/'diagnosis/target_unitigs.tsv').open() as stream:
                loci = list(csv.DictReader(stream,delimiter='\t'))
            self.assertEqual({int(r['unitig']) for r in loci if r['target']=='mecA'}, {0,1})
            self.assertTrue(any(r['target']=='sen' and r['unitig']=='3' for r in loci))
            with (root/'diagnosis/pair_comparison.tsv').open() as stream:
                pairs = list(csv.DictReader(stream,delimiter='\t'))
            self.assertEqual(len(pairs),2)
            gra = next(r for r in pairs if 'graR' in r['target_pairs'])
            self.assertEqual((gra['eligible_original'],gra['eligible_new']),('0','1'))
            self.assertTrue(gra['p_exact'])
            self.assertEqual(gra['beta_status'],'OK')
            self.assertEqual(gra['null_origin'],'original_null_cache')
            checkpoint = (root/'diagnosis/pair_comparison.tsv').read_bytes()
            # A pure script consolidation must preserve the old identity bytes:
            # the downstream pyseer checkpoint hashes this exact manifest.
            manifest = root/'diagnosis/diagnosis_identity.json'
            legacy = json.loads(manifest.read_text())
            legacy['script_sha256'] = '485d44a0616e1518d9d6233d09ba2db32a33c7a3672f039a175cb6109cbe5e14'
            manifest.write_text(json.dumps(legacy, indent=2))
            manifest_bytes = manifest.read_bytes()
            with patch.object(diagnostic,'calculate_pair',side_effect=AssertionError('Unexpected refit')):
                main(args)
            self.assertEqual((root/'diagnosis/pair_comparison.tsv').read_bytes(),checkpoint)
            self.assertEqual(manifest.read_bytes(),manifest_bytes)
            self.assertTrue(all(p.exists() and digest(p)==sha for p,sha in before.items()))

            # Simulate the preceding helper after it fitted diagnostic nulls.
            # An upgrade must retain mapping and beta while replacing scores.
            manifest = root/'diagnosis/diagnosis_identity.json'
            old_identity = json.loads(manifest.read_text())
            old_identity.pop('original_nulls')
            old_identity['script_sha256'] = '32713cfea041024439ba915315416df6ba9369f7942c63643ee072f752756391'
            manifest.write_text(json.dumps(old_identity))
            with sqlite3.connect(root/'diagnosis/diagnosis.sqlite') as db:
                for u,v,raw in db.execute('SELECT u,v,result FROM fits').fetchall():
                    result = json.loads(raw); result['null_origin']='fresh_diagnostic_fit'
                    db.execute('UPDATE fits SET result=? WHERE u=? AND v=?',(json.dumps(result),u,v))
            db.close()
            for metadata in (root/'diagnosis/null_models').glob('*/*.json'):
                record = json.loads(metadata.read_text()); record['origin']='fresh_diagnostic_fit'
                metadata.write_text(json.dumps(record))
            mapping_hash = digest(root/'diagnosis/target_unitigs.tsv')
            main(args+['--original-nulls','require'])
            with (root/'diagnosis/pair_comparison.tsv').open() as stream:
                required_pairs = list(csv.DictReader(stream,delimiter='\t'))
            self.assertTrue(all(r['null_origin']=='original_null_cache' for r in required_pairs))
            self.assertTrue(all(r['beta_reused_on_null_policy_change']=='True' for r in required_pairs))
            self.assertEqual([r['beta'] for r in required_pairs],[r['beta'] for r in pairs])
            self.assertEqual(digest(root/'diagnosis/target_unitigs.tsv'),mapping_hash)
            self.assertTrue(all(p.exists() and digest(p)==sha for p,sha in before.items()))

            # The user finished every fit but the source decisions DB failed.
            # A fully unreadable source checkpoint must not lose saved results.
            (source/'.kovar_work/work.sqlite').write_bytes(b'deliberately malformed original checkpoint')
            main(['--results',str(source),'--out',str(root/'diagnosis'),'--report-only'])
            with (root/'diagnosis/pair_comparison.tsv').open() as stream:
                recovered = list(csv.DictReader(stream,delimiter='\t'))
            self.assertEqual(len(recovered),2)
            self.assertTrue(all(r['old_query_status']=='UNAVAILABLE_NOT_IN_SIGNIFICANT_EXPORT' for r in recovered))
            self.assertTrue(all(r['old_comparison_source']=='UNAVAILABLE' for r in recovered))
            self.assertEqual([r['beta'] for r in recovered],[r['beta'] for r in pairs])
            self.assertEqual([r['p_exact'] for r in recovered],[r['p_exact'] for r in pairs])
            self.assertTrue(json.loads((root/'diagnosis/summary.json').read_text())['report_only'])
            self.assertEqual(json.loads((root/'diagnosis/summary.json').read_text())['samples'],80)


if __name__=='__main__':
    unittest.main()
