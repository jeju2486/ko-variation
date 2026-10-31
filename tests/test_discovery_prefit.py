"""Focused compact-cache, two-stage, migration and traversal regressions."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ko_variation import discovery_score as kernels
from ko_variation import discovery_null_cache as store
from ko_variation.discovery import schema,set_meta,get_meta,coarse_tasks,main
from ko_variation.discovery_store import connect,digest
from ko_variation.glmm import _weighted_eigen,prepare_kinship_eigensystem,score_predictor_block
from ko_variation.memory_budget import working_usage
from tests import test_discovery as fixtures


class PrefitTests(unittest.TestCase):
    def test_v1_pressure_uses_matching_hierarchical_inactive_counter(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            (folder/'usage').write_text('1000')
            (folder/'stat').write_text('inactive_file 20\ntotal_inactive_file 600\n')
            self.assertEqual(working_usage({'usage':folder/'usage','stat':folder/'stat'}),400)

    def cache(self, tau=.8, scalar=False):
        rng=np.random.default_rng(472)
        n=48
        G=rng.normal(size=(n,8))
        K=G@G.T/8+np.eye(n)*.2
        eigen=prepare_kinship_eigensystem(K)
        weights=np.full(n,.2) if scalar else rng.uniform(.05,.24,n)
        d=_weighted_eigen(K,weights,eigen)
        C=np.ones((n,1))
        Sinv_C=d.solve(C,tau)
        cache=dict(y=rng.integers(0,2,n).astype(float),mu=rng.uniform(.1,.9,n),
            weights=weights,tau=tau,C=C,Sinv_C=Sinv_C,M_inv=np.linalg.inv(C.T@Sinv_C),decomp=d)
        return cache,K,eigen,rng.integers(0,2,(n,19)).astype(float)

    def verify(self,path,expected): self.assertEqual(digest(path),expected)

    def test_sketch_bounds_reconstruct_exact_without_refitting(self):
        cache,K,eigen,X=self.cache()
        exact=np.array([r.score_z**2 for r in score_predictor_block(X,cache,compute_spa_adjustment=False)])
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            record=store.save(folder,0,cache,dict(status='OK',fit_seconds=1),K,eigen,rank=8)
            loaded=store.load(record,folder/'null_000000000.json',K,eigen,self.verify)
            self.assertEqual(record['mode'],'sketch')
            self.assertEqual(loaded['decomp'].eigenvectors.shape,(48,8))
            with patch.dict(kernels._STATE,dict(K=K,eigen=eigen,metrics={}),clear=True),\
                 patch('ko_variation.discovery_score.fit_null_glmm',side_effect=AssertionError('refitted')):
                low,high,_,_=kernels.adaptive_statistics(X,loaded,[exact],ranks=(0,8))
            np.testing.assert_allclose(low,exact,rtol=1e-9,atol=1e-9)
            np.testing.assert_allclose(high,exact,rtol=1e-9,atol=1e-9)
            self.assertEqual(loaded['decomp'].eigenvectors.shape,(48,48))

    def test_zero_and_shared_caches_omit_vectors_and_keep_scores(self):
        for tau,scalar,mode in [(0,False,'zero'),(.8,True,'shared')]:
            cache,K,eigen,X=self.cache(tau,scalar)
            expected=[r.score_z for r in score_predictor_block(X,cache,compute_spa_adjustment=False)]
            with tempfile.TemporaryDirectory() as temporary:
                folder=Path(temporary)
                record=store.save(folder,0,cache,dict(status='OK'),K,eigen)
                self.assertEqual(record['mode'],mode)
                with np.load(folder/record['array']['name']) as arrays:
                    self.assertNotIn('eigenvectors',arrays.files)
                loaded=store.load(record,folder/'null_000000000.json',K,eigen,self.verify)
                observed=[r.score_z for r in score_predictor_block(X,loaded,compute_spa_adjustment=False)]
                np.testing.assert_allclose(observed,expected,rtol=1e-10,atol=1e-10)

    def test_response_traversal_skips_legacy_prefix_on_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            db=connect(Path(temporary)/'work.sqlite');schema(db)
            roots=[1,2,3,4,5];ordered=[3,1,5,2,4]
            with db: set_meta(db,'root_cursor',[1,3])
            fresh=[]
            while True:
                batch,done=coarse_tasks(db,roots,ordered,2)
                fresh.extend(batch)
                if done: break
            legacy={(roots[i],roots[j]) for i in range(5) for j in range(i+1,5)
                    if i<1 or (i==1 and j<3)}
            all_pairs={(roots[i],roots[j]) for i in range(5) for j in range(i+1,5)}
            self.assertEqual(set(fresh),all_pairs-legacy)
            self.assertEqual(len(fresh),len(set(fresh)))
            self.assertEqual(get_meta(db,'coarse_scheduled_count'),10)
            self.assertEqual(coarse_tasks(db,roots,ordered,2),([],True))
            db.close()

    def test_multiple_response_columns_cover_all_pairs_without_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            db=connect(Path(temporary)/'work.sqlite');schema(db)
            roots=list(range(1,65));seen=[]
            while True:
                batch,done=coarse_tasks(db,roots,roots,128,workers=4)
                seen.extend(batch)
                if done: break
            self.assertEqual(len(seen),64*63//2)
            self.assertEqual(len(set(seen)),len(seen))
            self.assertEqual(get_meta(db,'coarse_scheduled_count'),len(seen))
            db.close()

    def test_prefit_only_then_cached_scoring_reuses_all_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            fasta,tree,_=fixtures.DiscoveryTests.inputs(folder)
            out=folder/'out'
            common=['--fasta',str(fasta),'--tree',str(tree),'--out',str(out),
                    '--threads','2','--memory-gb','1','--min-cell-count','1','--screen-p','.01','--search','exhaustive']
            self.assertEqual(main(common+['--prefit-only']),0)
            work=out/'.kovar_work'
            db=connect(work/'work.sqlite')
            self.assertEqual(db.execute('SELECT count(*) FROM decisions').fetchone()[0],0)
            before=get_meta(db,'prefit_runtime_totals')['new_null_fits']
            db.close()
            summary=json.loads((out/'null_stage_summary.json').read_text())
            self.assertEqual(summary['complement_aliases'],1)
            aliases=work/'null_canonical.npy'
            stamp=aliases.stat().st_mtime_ns
            (out/'null_stage_summary.json').unlink()
            with patch('ko_variation.discovery_prefit.ProcessPoolExecutor',side_effect=AssertionError('rechecked completed stage')):
                self.assertEqual(main(common+['--resume','--require-cached-nulls']),0)
            self.assertEqual(aliases.stat().st_mtime_ns,stamp)
            self.assertEqual(json.loads((out/'null_stage_summary.json').read_text()),summary)
            db=connect(work/'work.sqlite')
            self.assertEqual(get_meta(db,'prefit_runtime_totals')['new_null_fits'],before)
            self.assertEqual(get_meta(db,'runtime_totals').get('new_null_fits',0),0)
            self.assertTrue(get_meta(db,'search_complete'))
            db.close()

    def test_complement_view_retains_signed_score_and_squared_statistic(self):
        cache,K,eigen,X=self.cache()
        y=cache['y'].astype('u1')
        packed=np.packbits(np.array([y,1-y]),axis=1,bitorder='little')
        with patch.dict(kernels._STATE,dict(packed=packed,info={'n':len(y)}),clear=True):
            flipped,_=kernels._view((cache,'OK'),1,0)
        a=np.array([r.score_z for r in score_predictor_block(X,cache,compute_spa_adjustment=False)])
        b=np.array([r.score_z for r in score_predictor_block(X,flipped,compute_spa_adjustment=False)])
        np.testing.assert_allclose(a,-b,rtol=1e-10,atol=1e-10)



if __name__=='__main__': unittest.main()
