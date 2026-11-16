"""Tree nulls: dense agreement, compact restart, fallback and discovery integration."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ko_variation import discovery_nulls as store, discovery_score as kernels, glmm, kinship
from ko_variation.discovery import main
from ko_variation.discovery_store import digest
from ko_variation.tree_null import TreeAlgebra, TreeWorking


class ProductionTreeTests(unittest.TestCase):
    def test_nonzero_fit_scores_storage_and_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); path=root/'tree.nwk';n=64
            names=[f's{i}' for i in range(n)]
            path.write_text('('+','.join('('+','.join(f's{i}:.1' for i in range(j,j+8))+'):2'
                for j in range(0,n,8))+');')
            K=kinship.build_tree_covariance(path,names).K
            eigen=glmm.prepare_kinship_eigensystem(K)
            tree=TreeAlgebra.from_newick(path,names,engine='python')
            rng=np.random.default_rng(3)
            y=rng.binomial(1,np.repeat([.05,.1,.2,.3,.7,.8,.9,.95],8)).astype(float)
            dense,c0=glmm.fit_null_glmm(0,y,K,kinship_eigensystem=eigen)
            fitted,c1=glmm.fit_null_glmm(0,y,K,kinship_eigensystem=eigen,tree_algebra=tree)
            self.assertEqual((dense.status,fitted.status),('OK','OK'))
            self.assertGreater(fitted.tau,0)
            self.assertTrue(hasattr(c1['decomp'],'score_variance'))
            np.testing.assert_allclose(dense.fitted_probability,fitted.fitted_probability,atol=5e-7,rtol=0)
            X=rng.integers(0,2,(n,12)).astype(float)
            expected=glmm.score_predictor_block(X,c0,compute_spa_adjustment=False)
            actual=glmm.score_predictor_block(X,c1,compute_spa_adjustment=False)
            np.testing.assert_allclose([r.score_variance for r in actual],
                [r.score_variance for r in expected],rtol=1e-6)
            record=store.save(root,0,c1,dict(status='OK'),K,eigen)
            self.assertEqual(record['mode'],'tree')
            with np.load(root/record['array']['name']) as arrays:
                self.assertNotIn('eigenvectors',arrays.files)
            def verify(path,sha): self.assertEqual(digest(path),sha)
            loaded=store.load(record,root/'null_000000000.json',eigen,verify,tree)
            low,high,_,_=kernels.adaptive_statistics(X,loaded,[10.])
            np.testing.assert_allclose(low,[r.score_z**2 for r in actual],rtol=1e-12)
            np.testing.assert_array_equal(low,high)
            with patch.object(TreeWorking,'point',side_effect=np.linalg.LinAlgError('forced')):
                fallback,cache=glmm.fit_null_glmm(0,y,K,kinship_eigensystem=eigen,tree_algebra=tree)
            self.assertEqual(fallback.status,'OK')
            self.assertIn('_tree_fallback',cache)
            np.testing.assert_array_equal(fallback.fitted_probability,dense.fitted_probability)

    def test_parallel_tree_discovery_resumes_without_refitting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);rng=np.random.default_rng(12);n=32
            X=rng.integers(0,2,(n,5))
            fasta=root/'samples.fa';tree=root/'tree.nwk';out=root/'run'
            fasta.write_text(''.join(f'>s{i}\n'+''.join('C' if v else 'A' for v in row)+'\n' for i,row in enumerate(X)))
            tree.write_text('('+','.join('('+','.join(f's{i}:1' for i in range(j,j+8))+'):.5' for j in range(0,n,8))+');')
            args=['--fasta',str(fasta),'--tree',str(tree),'--out',str(out),'--search','exhaustive',
                '--screen-p','.05','--null-backend','tree','--min-cell-count','1','--threads','2','--memory-gb','1','--no-beta']
            self.assertEqual(main(args),0)
            work=out/'.kovar_work'
            products={p:digest(p) for p in work.glob('null_models/*/*.npz')}
            self.assertTrue(products)
            self.assertTrue(all(json.loads(p.read_text())['mode']=='tree' for p in work.glob('null_models/*/*.json')))
            self.assertEqual(main(args+['--resume','--require-cached-nulls']),0)
            self.assertEqual(products,{p:digest(p) for p in products})


if __name__=='__main__': unittest.main()
