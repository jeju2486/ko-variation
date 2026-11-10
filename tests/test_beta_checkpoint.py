"""The normal pipeline preserves committed beta fits across interruptions."""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ko_variation.discovery import distance_and_beta, identity, parser, schema, set_meta
from ko_variation.discovery_store import connect
from ko_variation.null_identity import migrate_identity


class BetaCheckpointTests(unittest.TestCase):
    def test_cutoff_resume_reuses_distances_decisions_and_saved_betas(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary, paths, unitigs = root/'distance_tool', root/'graph.paths', root/'unitigs'
            binary.write_text('fixture')
            paths.write_text(str(binary)+'\n')
            unitigs.write_text('0 ACGT\n')
            args = SimpleNamespace(unitig_distance=str(binary), paths=paths, unitigs=unitigs,
                                   edges=None, kmer_length=61, distal_bp=10000,
                                   threads=2, no_beta=True, out=root, distance_batch_size=5000)
            db = connect(root/'work.sqlite')
            schema(db)
            rows = [(10,20,0,1,20000,1,'DISTAL'), (11,20,0,1,5000,1,'LOCAL'),
                    (20,30,1,2,3001,1,'LOCAL'), (30,40,2,3,3000,1,'LOCAL'),
                    (40,50,3,4,None,0,'UNKNOWN')]
            with db:
                set_meta(db,'distance_expanded',True)
                db.executemany('INSERT INTO distance VALUES (?,?,?,?,?,?,?)',rows)
                db.execute('INSERT INTO beta VALUES (0,1,2.5,"OK")')
                db.execute('INSERT INTO decisions VALUES (0,1,1,1,42,42,0.00001,"EXACT")')
            decisions = db.execute('SELECT * FROM decisions').fetchall()
            calls = []

            def beta(job):
                calls.append(tuple(job))
                return *job, 1.25, 'OK'

            help_result = subprocess.CompletedProcess([],0,stdout='',stderr='')
            with ThreadPoolExecutor(max_workers=2) as pool, \
                 patch('ko_variation.discovery.shutil.which',return_value=str(binary)), \
                 patch('ko_variation.discovery.subprocess.run',return_value=help_result) as run, \
                 patch('ko_variation.discovery.kernels.beta',side_effect=beta):
                distance_and_beta(db,root,pool,args)
                args.distal_bp, args.no_beta = 3000, False
                distance_and_beta(db,root,pool,args)
                self.assertEqual(calls,[(1,2)])
                self.assertEqual(db.execute('SELECT status FROM distance ORDER BY u,v').fetchall(),
                                 [('DISTAL',),('DISTAL',),('DISTAL',),('LOCAL',),('UNKNOWN',)])
                saved = db.execute('SELECT * FROM beta ORDER BY u,v').fetchall()
                self.assertEqual(saved,[(0,1,2.5,'OK'),(1,2,1.25,'OK')])
                args.distal_bp = 25000
                distance_and_beta(db,root,pool,args)
                args.distal_bp = 3000
                distance_and_beta(db,root,pool,args)
                self.assertEqual(calls,[(1,2)])
                self.assertEqual(db.execute('SELECT * FROM beta ORDER BY u,v').fetchall(),saved)
                self.assertEqual(db.execute('SELECT * FROM decisions').fetchall(),decisions)
                self.assertEqual(db.execute('SELECT u,v,pu,pv,distance,connected FROM distance ORDER BY u,v').fetchall(),
                                 [row[:6] for row in rows])
                # All invocations only inspect --help; no graph distances repeat.
                self.assertTrue(all(call.kwargs.get('capture_output') for call in run.call_args_list))
                history = json.loads(db.execute("SELECT value FROM meta WHERE key='distance_threshold_history'").fetchone()[0])
                self.assertEqual([entry['distal_bp'] for entry in history],[3000,25000,3000])
                self.assertEqual(history[0]['reclassified_unitig_pairs'],2)
                binary.write_text('changed distance executable')
                with self.assertRaisesRegex(ValueError,'Distance inputs changed'):
                    distance_and_beta(db,root,pool,args)
            db.close()

    def test_interruption_keeps_saved_fits_and_resume_reports_cumulative_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary, paths, unitigs = root/'distance_tool', root/'graph.paths', root/'unitigs'
            binary.write_text('fixture')
            paths.write_text(str(binary)+'\n')
            unitigs.write_text('0 ACGT\n')
            args = SimpleNamespace(unitig_distance=str(binary), paths=paths, unitigs=unitigs,
                                   edges=None, kmer_length=61, distal_bp=10000,
                                   threads=2, no_beta=False, out=root, distance_batch_size=5000)
            path = root/'work.sqlite'
            db = connect(path)
            schema(db)
            with db:
                set_meta(db, 'distance_expanded', True)
                db.executemany('INSERT INTO distance VALUES (?,?,?,?,?,1,?)', [
                    (10, 20, 0, 1, 20000, 'DISTAL'),
                    (11, 20, 0, 1, 22000, 'DISTAL'),
                    (10, 30, 0, 2, 25000, 'DISTAL'),
                    (20, 30, 1, 2, 25000, 'DISTAL'),
                    (20, 40, 1, 3, 5000, 'LOCAL'),
                    (30, 40, 2, 3, None, 'UNKNOWN')])
                db.execute('INSERT INTO beta VALUES (0,1,NULL,"UNSTABLE_EFFECT")')
            calls = []

            def beta(job):
                calls.append(tuple(job))
                return *job, 1.25, 'OK'

            def interrupted(pool, function, jobs, threads, **kwargs):
                yield function(next(iter(jobs)))
                raise RuntimeError('interrupted after beta commit')

            help_result = subprocess.CompletedProcess([], 0, stdout='', stderr='')
            with ThreadPoolExecutor(max_workers=2) as pool, \
                 patch('ko_variation.discovery.shutil.which', return_value=str(binary)), \
                 patch('ko_variation.discovery.subprocess.run', return_value=help_result), \
                 patch('ko_variation.discovery.kernels.beta', side_effect=beta):
                with patch('ko_variation.discovery.bounded_map', side_effect=interrupted):
                    with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                        distance_and_beta(db, root, pool, args)
                db.close()
                # A new database connection must see both the old attempt and
                # the fit committed immediately before interruption.
                db = connect(path)
                saved = db.execute('SELECT * FROM beta ORDER BY u,v').fetchall()
                self.assertEqual(len(saved), 2)
                calls.clear()
                distance_and_beta(db, root, pool, args)
                self.assertEqual(len(calls), 1)
                self.assertNotIn(calls[0], [(r[0], r[1]) for r in saved])
                self.assertTrue(all(db.execute('SELECT * FROM beta WHERE u=? AND v=?', r[:2]).fetchone()==r
                                    for r in saved))
                progress = json.loads((root/'beta_progress.json').read_text())
                self.assertEqual((progress['total_distal_pattern_pairs'], progress['saved'],
                                  progress['remaining'], progress['saved_before_invocation'],
                                  progress['committed_this_invocation']), (3, 3, 0, 2, 1))
                self.assertEqual(progress['status_counts'], {'OK': 2, 'UNSTABLE_EFFECT': 1})
                calls.clear()
                distance_and_beta(db, root, pool, args)
                self.assertEqual(calls, [])
                self.assertEqual(json.loads((root/'beta_progress.json').read_text())['saved'], 3)
            db.close()

    def test_previous_pc40_checkpoint_migrates_without_relaxing_scientific_checks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fasta, tree = root/'input.fa', root/'tree.nwk'
            fasta.write_text('>s\nAC\n')
            tree.write_text('(s:1);')
            args = parser().parse_args(['--fasta', str(fasta), '--tree', str(tree), '--out', str(root)])
            expected = identity(args)
            previous = copy.deepcopy(expected)
            previous['code'].update({
                'discovery.py': 'ed0f27ddee6a7b90a45d6002ffbe147a0787a7cef01520d78304bc25fae3b78e',
                'null_identity.py': 'dc344fa33bad4180ce81f4cd7e21f2cd900c789af2c558906bc2155cc4471133'})
            manifest = root/'run_identity.json'
            manifest.write_text(json.dumps(previous))
            self.assertTrue(migrate_identity(previous, expected, manifest))
            self.assertEqual(json.loads(manifest.read_text()), expected)
            # Immediately preceding release is accepted with numerical code unchanged.
            prior_cutoff = copy.deepcopy(expected)
            prior_cutoff['code'].update({
                'discovery.py':'03a470f36404bfad4521d5934d17c59b79dedeed16b2fccb0e525f934aa58a37',
                'null_identity.py':'d655a839c4d94928b2f1d8e476686192474a63fcf164e5b131a4bb037a581440'})
            self.assertTrue(migrate_identity(prior_cutoff,expected,manifest))
            self.assertTrue((root/'run_identity.v092.pc-screen-1.json').is_file())
            changed = copy.deepcopy(previous)
            changed['settings']['min_maf'] = .1
            with self.assertRaisesRegex(ValueError, 'Scientific settings changed'):
                migrate_identity(changed, expected, manifest)
            changed = copy.deepcopy(previous)
            changed['code']['glmm.py'] = 'unverified numerical source'
            with self.assertRaisesRegex(ValueError, 'not a verified release'):
                migrate_identity(changed, expected, manifest)


if __name__ == '__main__':
    unittest.main()
