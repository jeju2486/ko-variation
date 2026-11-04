"""Focused proof, external-association preservation and restart checks."""
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations, product
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ko_variation import discovery_score as kernels
from ko_variation.discovery import schema, search, get_meta
from ko_variation.discovery_eligibility import impossible_block
from ko_variation.discovery_store import connect


def eligible(a,b,c):
    return all(np.count_nonzero((a==x)&(b==y))>=c for x,y in product((0,1),repeat=2))


class EligibilityTests(unittest.TestCase):
    def test_hamming_proofs_against_exhaustive_binary_states(self):
        # All six-bit states, all anchor pairs and radii 0/1: covers identical,
        # complementary and overlapping balls, with equality at cell bounds.
        X=np.array(list(product((0,1),repeat=6)),dtype='u1')
        distances=np.count_nonzero(X[:,None,:]!=X[None,:,:],axis=2)
        valid=np.array([[eligible(a,b,1) for b in X] for a in X])
        for a,b in product(range(len(X)),repeat=2):
            for ra,rb in product((0,1),repeat=2):
                left=np.flatnonzero(distances[a]<=ra)
                right=np.flatnonzero(distances[b]<=rb)
                ca=sum(2<=int(X[i].sum())<=4 for i in left)
                cb=sum(2<=int(X[i].sum())<=4 for i in right)
                if impossible_block(6,1,ra,rb,int(distances[a,b]),ca,cb):
                    self.assertFalse(valid[np.ix_(left,right)].any())
        self.assertFalse(impossible_block(80,5,0,0,10,1,1))
        self.assertFalse(impossible_block(80,5,0,0,70,1,1))
        self.assertTrue(impossible_block(80,5,0,0,71,1,1))
        self.assertFalse(impossible_block(80,0,0,0,0,0,0))

    def test_pruned_internal_blocks_keep_all_eligible_external_pairs_after_restart(self):
        with TemporaryDirectory() as temporary:
            folder=Path(temporary)
            work=folder/'.kovar_work'
            work.mkdir()
            A=np.tile([0,1],40).astype('u1')
            B=np.tile([0,0,1,1],20).astype('u1')
            X=np.array([A,A.copy(),A.copy(),B,B.copy(),B.copy()])
            X[1,0]^=1
            X[2,1]^=1
            X[4,2]^=1
            X[5,3]^=1
            packed=np.packbits(X,axis=1,bitorder='little')
            packed.tofile(work/'patterns.bin')
            info=dict(n=80,patterns=6,packed_width=10)
            db=connect(work/'work.sqlite')
            schema(db)
            members={1:[0,1,2],2:[3,4,5],3:[0,1],4:[2],5:[3,4],6:[5]}
            parent={1:0,2:0,3:1,4:1,5:2,6:2}
            with db:
                for node,ids in members.items():
                    db.execute('INSERT INTO nodes(id,parent,anchor,size,ready) VALUES(?,?,?,?,1)',
                               (node,parent[node],ids[0],len(ids)))
                    db.execute('INSERT INTO reps VALUES(?,?)',(node,ids[0]))
                    db.executemany('INSERT INTO members VALUES(?,?)',((node,i) for i in ids))
            args=SimpleNamespace(search='hierarchy',threads=2,batch_size=2,memory_gb=1,
                min_cell_count=5,tmp_dir=None,out=folder,
                _plan=dict(frontier_tasks=4,query_buffer_pairs=3))
            state=dict(packed=packed,info=info)
            calls=[]
            def scorer(job):
                response,predictors,_,screening=job
                calls.extend((u,response) for u in predictors)
                rows=[(u,response,1,1 if eligible(X[u],X[response],5) else -2,
                       99.,99.,1e-20,'OK') for u in predictors]
                return dict(rows=rows,metrics={})
            # Fail AFTER proof summaries have been committed but before any
            # representative null/scoring work completes. Then resume the queue.
            with patch.dict(kernels._STATE,state,clear=True),ThreadPoolExecutor(max_workers=2) as pool:
                with patch('ko_variation.discovery_score.score_profiled',side_effect=RuntimeError('interrupted')):
                    with self.assertRaisesRegex(RuntimeError,'interrupted'):
                        search(db,info,pool,args)
                self.assertEqual(db.execute('SELECT count(*) FROM block_eligibility').fetchone()[0],2)
                with patch('ko_variation.discovery_score.score_profiled',side_effect=scorer):
                    search(db,info,pool,args)
            expected={(u,v) for u,v in combinations(range(6),2) if eligible(X[u],X[v],5)}
            observed=set(db.execute('SELECT u,v FROM decisions WHERE final=1'))
            self.assertEqual(observed,expected)
            self.assertEqual(len(expected),9)
            self.assertTrue(all(u<3<=v for u,v in calls))
            self.assertGreaterEqual(get_meta(db,'hierarchy_metrics')['eligibility_pruned_internal_blocks'],2)
            self.assertEqual(db.execute('SELECT count(*) FROM members').fetchone()[0],12)
            self.assertTrue(get_meta(db,'search_complete'))
            db.close()


if __name__=='__main__':
    unittest.main()
