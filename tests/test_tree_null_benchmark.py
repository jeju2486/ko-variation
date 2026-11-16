"""Tree algebra against dense covariance, and paired null benchmark restart."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import linalg

from benchmarks import tree_null_benchmark as bench
from ko_variation import glmm, kinship


class TreeNullTests(unittest.TestCase):
    def test_partial_checkpoint_is_ignored_for_plotting_and_repaired_only_on_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'checkpoint.jsonl'
            complete=b'{"index":0,"repeat":0}\n'
            path.write_bytes(complete+b'{"index":1')
            self.assertEqual(len(bench.checkpoint(path,repair=False)),1)
            self.assertEqual(path.read_bytes(),complete+b'{"index":1')
            self.assertEqual(len(bench.checkpoint(path)),1)
            self.assertEqual(path.read_bytes(),complete)

    def test_nonzero_phylogenetic_variance_and_scores_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'tree.nwk'
            path.write_text('('+','.join('('+','.join(f's{i}:.1' for i in range(j,j+8))+'):3' for j in range(0,64,8))+');')
            names=[f's{i}' for i in range(64)]
            K=kinship.build_tree_covariance(path,names).K
            y=np.array(([0]*7+[1])*4+([0]+[1]*7)*4,dtype=float)
            eigen=glmm.prepare_kinship_eigensystem(K)
            dense,c0=glmm.fit_null_glmm(0,y,K,kinship_eigensystem=eigen)
            tree=bench.TreeAlgebra.from_newick(path,names,'numba' if bench.numba else 'python')
            with patch.dict(bench.STATE,{'tree':tree}),patch.object(glmm,'_weighted_eigen',bench.weighted_tree),patch.object(glmm,'_profile_tau',bench.profile_tree):
                other,c1=glmm.fit_null_glmm(0,y,K,kinship_eigensystem=eigen)
            self.assertEqual(dense.status,'OK');self.assertEqual(other.status,'OK')
            self.assertGreater(dense.tau,.1)
            np.testing.assert_allclose(dense.fitted_probability,other.fitted_probability,atol=1e-7)
            self.assertAlmostEqual(dense.tau,other.tau,places=5)
            X=np.ascontiguousarray(np.random.default_rng(7).binomial(1,.4,size=(64,5)),dtype=float)
            expected=glmm.score_predictor_block(X,c0,compute_spa_adjustment=False)
            U,V=bench.tree_scores(X,c1,tree)
            np.testing.assert_allclose(U,[r.score_u for r in expected],atol=1e-7)
            np.testing.assert_allclose(V,[r.score_variance for r in expected],rtol=1e-6)

    def test_prepared_inputs_need_no_sqlite_or_saved_nulls(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temporary:
            work=Path(temporary)
            tree=work/'analysis.treefile';tree.write_text('((a:1,b:1):.5,(c:1,d:1):.5);')
            names=['a','b','c','d']
            matrix=np.array([[0,0,1,1],[0,1,0,1],[0,1,1,0]],dtype='u1')
            np.packbits(matrix,axis=1,bitorder='little').tofile(work/'patterns.bin')
            np.save(work/'K.npy',kinship.build_tree_covariance(tree,names).K)
            (work/'prepared.json').write_text(json.dumps(dict(samples=names,n=4,patterns=3,packed_width=1,
                products={name:bench.digest(work/name) for name in ('patterns.bin','K.npy')})))
            (work/'work.sqlite').write_bytes(b'not a readable database')
            original={p:bench.digest(p) for p in work.iterdir()}
            args=SimpleNamespace(work=work,patterns=2,seed=42,tree=None,engine='python',stress_cases=False)
            selected,K,samples,labels,_,family,error=bench.select_inputs(args)
            self.assertEqual(selected.shape,(2,4));self.assertEqual(samples,names)
            self.assertEqual(family,3);self.assertLessEqual(error,1e-10)
            self.assertTrue(all(bench.digest(p)==sha for p,sha in original.items()))

    def test_tree_solves_logdet_and_products_match_dense_with_zero_edges_and_pruned_tips(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'tree.nwk'
            path.write_text('(((a:0,b:.4,c:.3):.2,d:.7):.6,unused:.8);')
            names=['d','a','c','b']
            tree=bench.TreeAlgebra.from_newick(path,names,engine='python')
            K=kinship.build_tree_covariance(path,names).K
            rng=np.random.default_rng(3)
            rhs=rng.normal(size=(4,3)); weights=np.array([.1,.03,.22,.25])
            np.testing.assert_allclose(tree.apply('multiply',rhs),K@rhs,atol=1e-13)
            for tau in (0.,1e-8,.5,100.,1e6):
                S=np.diag(1/weights)+tau*K
                expected=linalg.solve(S,rhs,assume_a='pos')
                actual=tree.apply('solve',weights,tau,rhs)
                np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-10)
                white,logdet,_,_=tree.apply('white',weights,tau,rhs)
                np.testing.assert_allclose(white.T@white,rhs.T@expected,rtol=1e-9,atol=1e-10)
                self.assertAlmostEqual(logdet,np.linalg.slogdet(S)[1],places=8)
            if bench.numba is not None:
                tree.engine='numba'
                np.testing.assert_allclose(tree.apply('solve',weights,.5,rhs),linalg.solve(np.diag(1/weights)+.5*K,rhs),rtol=1e-12)

    def test_same_null_loop_and_checkpoint_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); inputs=root/'input';inputs.mkdir()
            rng=np.random.default_rng(51)
            matrix=rng.binomial(1,.4,size=(16,8))
            fasta=inputs/'samples.fasta';tree=inputs/'tree.nwk'
            fasta.write_text(''.join(f'>s{i}\n'+''.join('C' if v else 'A' for v in row)+'\n' for i,row in enumerate(matrix)))
            tree.write_text('('+','.join('('+','.join(f's{i}:1' for i in range(j,j+4))+'):0.5' for j in range(0,16,4))+');')
            out=root/'results'
            args=['--fasta',str(fasta),'--tree',str(tree),'--out',str(out),'--patterns','2',
                  '--threads','1','--engine','numba' if bench.numba else 'python','--stress-cases']
            before=bench.digest(Path(glmm.__file__))
            bench.main(args)
            summary=json.loads((out/'summary.json').read_text())
            self.assertEqual(summary['status_mismatches'],0)
            self.assertEqual(summary['bonferroni_decision_mismatches'],0)
            self.assertEqual(summary['jointly_successful_jobs'],summary['within_tolerances'])
            checkpoint=bench.digest(out/'checkpoint.jsonl')
            bench.main(args)
            self.assertEqual(bench.digest(out/'checkpoint.jsonl'),checkpoint)
            self.assertEqual(bench.digest(Path(glmm.__file__)),before)


if __name__=='__main__':unittest.main()
