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

    def test_restart_requires_same_numerical_release(self):
        expected=dict(settings={'alpha':.05},code={'glmm.py':'current'})
        with tempfile.TemporaryDirectory() as temporary:
            manifest=Path(temporary)/'identity.json'
            self.assertFalse(migrate_identity(expected,expected,manifest))
            with self.assertRaisesRegex(ValueError,'Scientific'):
                migrate_identity(dict(expected,settings={'alpha':.01}),expected,manifest)
            with self.assertRaisesRegex(ValueError,'verified'):
                migrate_identity(dict(expected,code={'glmm.py':'old'}),expected,manifest)

    def test_metadata_release_reuses_identical_numerical_state(self):
        expected=dict(version='0.9.4',settings={'null_backend':'tree'},
            execution_revision='tree-null-1',null_model_source_sha256='same-fitter',
            numpy='same-numpy',scipy='same-scipy',inputs={'tree':'same-tree'},
            code={'glmm.py':'same','tree_null.py':'same','null_identity.py':'new-helper'})
        previous=copy.deepcopy(expected)
        previous['version']='0.9.3'
        previous['code']['null_identity.py']='old-helper'
        with tempfile.TemporaryDirectory() as temporary:
            manifest=Path(temporary)/'run_identity.json'
            self.assertTrue(migrate_identity(previous,expected,manifest))
            self.assertEqual(json.loads(manifest.read_text()),expected)
            for key in ('glmm.py','tree_null.py'):
                changed=copy.deepcopy(previous);changed['code'][key]='changed'
                with self.assertRaisesRegex(ValueError,'verified'):
                    migrate_identity(changed,expected,manifest)
            changed=dict(previous,null_model_source_sha256='other-fitter')
            with self.assertRaisesRegex(ValueError,'verified'):
                migrate_identity(changed,expected,manifest)


if __name__ == '__main__':
    unittest.main()
