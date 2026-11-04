"""Focused interruption, deduplication and beta-only restart contracts."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ko_variation.beta_resume import fit_pending, main, queue_pairs
from ko_variation.discovery import identity, parser, schema, set_meta
from ko_variation.discovery_store import atomic_json, connect, prepare
from tests import test_discovery as fixtures


class BetaResumeTests(unittest.TestCase):
    @staticmethod
    def database(path):
        db = connect(path)
        schema(db)
        with db:
            set_meta(db, 'search_complete', True)
            set_meta(db, 'distance_expanded', True)
            db.executemany('INSERT INTO decisions VALUES (?,?,1,1,20,20,0.001,"OK")',
                           [(0, 1), (0, 2), (1, 2), (2, 3)])
            db.executemany('INSERT INTO distance VALUES (?,?,?,?,?,1,?)', [
                (10, 20, 0, 1, 20000, 'DISTAL'),
                (11, 20, 0, 1, 22000, 'DISTAL'),
                (10, 30, 0, 2, 25000, 'DISTAL'),
                (20, 30, 1, 2, 5000, 'LOCAL'),
                (30, 40, 2, 3, None, 'UNKNOWN')])
        return db

    def test_restart_keeps_committed_fit_and_deduplicates_unitigs(self):
        with tempfile.TemporaryDirectory() as folder:
            db = self.database(Path(folder)/'work.sqlite')
            try:
                self.assertEqual(queue_pairs(db)[0], 2)
                calls = []

                def beta(job):
                    calls.append(tuple(job))
                    return *job, 1.25, 'OK'

                def interrupt(status):
                    raise RuntimeError('interrupted after commit')

                with patch('ko_variation.beta_resume.kernels.beta', beta):
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                            fit_pending(db, pool, 1, interrupt)
                    committed = db.execute('SELECT u,v FROM beta').fetchall()
                    self.assertEqual(len(committed), 1)
                    calls.clear()
                    self.assertEqual(sum(queue_pairs(db)[1].values()), 1)
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        fit_pending(db, pool, 1, lambda status: None)
                    self.assertEqual(len(calls), 1)
                    self.assertNotIn(committed[0], calls)
                self.assertEqual(db.execute('SELECT count(*) FROM beta').fetchone()[0], 2)
            finally:
                db.close()

    def test_failed_fit_retry_is_explicit_and_pending_distances_block(self):
        with tempfile.TemporaryDirectory() as folder:
            db = self.database(Path(folder)/'work.sqlite')
            try:
                with db:
                    db.executemany('INSERT INTO beta VALUES (?,?,?,?)',
                                   [(0, 1, 1.25, 'OK'), (0, 2, None, 'UNSTABLE_EFFECT')])
                self.assertEqual(sum(queue_pairs(db)[1].values()), 2)
                self.assertEqual(db.execute('SELECT count(*) FROM beta_queue').fetchone()[0], 0)
                self.assertEqual(dict(queue_pairs(db, True)[1]), {'OK': 1})
                self.assertEqual(db.execute('SELECT u,v FROM beta_queue').fetchall(), [(0, 2)])
                with db:
                    db.execute("UPDATE distance SET status='PENDING' WHERE u=10 AND v=20")
                with self.assertRaisesRegex(ValueError, 'Distances are unfinished'):
                    queue_pairs(db)
            finally:
                db.close()

    def test_beta_only_exports_without_raw_inputs_and_reuses_same_kernel(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fasta, tree, unitigs = fixtures.DiscoveryTests.inputs(root)
            out = root/'results'
            work = out/'.kovar_work'
            work.mkdir(parents=True)
            args = parser().parse_args(['--fasta', str(fasta), '--tree', str(tree),
                                       '--out', str(out), '--memory-gb', '1', '--min-cell-count', '1'])
            frozen = identity(args)
            db = connect(work/'work.sqlite')
            schema(db)
            info = prepare(args, db, work)
            atomic_json(work/'run_identity.json', frozen)
            atomic_json(out/'run_identity.json', dict(frozen, work_directory=str(work)))
            patterns = db.execute('SELECT id,pattern FROM unitigs WHERE pattern IS NOT NULL').fetchall()
            u, pu = patterns[0]
            v, pv = next((v, pv) for v, pv in patterns if pv != pu)
            with db:
                set_meta(db, 'search_complete', True)
                set_meta(db, 'distance_expanded', True)
                db.execute('INSERT INTO decisions VALUES (?,?,1,1,20,20,0.001,"OK")', (pu, pv))
                db.execute('INSERT INTO distance VALUES (?,?,?,?,20000,1,"DISTAL")', (u, v, pu, pv))
            db.close()
            fasta.unlink()
            tree.unlink()
            unitigs.unlink()
            command = ['--out', str(root), '--threads', '1', '--memory-gb', '1']
            self.assertEqual(main(command), 0)
            progress = json.loads((out/'beta_progress.json').read_text())
            self.assertEqual(progress['committed_this_invocation'], 1)
            self.assertEqual(progress['remaining'], 0)
            self.assertTrue((out/'associations.tsv.gz').is_file())
            with patch('ko_variation.beta_resume.kernels.beta', side_effect=AssertionError('repeated fit')):
                self.assertEqual(main(command), 0)
            self.assertEqual(json.loads((out/'beta_progress.json').read_text())['committed_this_invocation'], 0)


if __name__ == '__main__':
    unittest.main()
