"""Global top-fraction selection, restart and reuse of actual fitted nulls."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ko_variation import coarse_screen as screen
from ko_variation.discovery import schema,get_meta,set_meta,main
from ko_variation.discovery_store import connect
from tests import test_discovery as fixtures


class CoarseScreenTests(unittest.TestCase):
    def fixture(self, folder, force=False):
        rng=np.random.default_rng(22);n=64;m=20
        X=rng.integers(0,2,(m,n)).astype(float);X[1]=1-X[0]
        G=rng.normal(size=(n,6));K=G@G.T+np.eye(n)
        if force: K=np.outer(X[0]-.5,X[0]-.5)
        work=folder/'work';work.mkdir();np.save(work/'K.npy',K)
        np.packbits(X.astype('u1'),axis=1,bitorder='little').tofile(work/'patterns.bin')
        db=connect(work/'work.sqlite');schema(db)
        with db:
            db.execute('INSERT INTO nodes(id,parent,level,anchor,size,ready) VALUES (0,NULL,-1,0,20,1)')
            db.executemany('INSERT INTO nodes VALUES (?,0,0,?,1,1)',[(i+1,i) for i in range(m)])
            db.executemany('INSERT INTO reps VALUES (?,?)',[(i+1,i) for i in range(m)])
        info=dict(n=n,patterns=m,packed_width=8)
        args=SimpleNamespace(screen_block_size=3,screen_pcs=5,coarse_retain=.1,memory_gb=.25,threads=2,out=folder)
        Kc=K-K.mean(0)[None,:]-K.mean(1)[:,None]+K.mean()
        vals,vecs=np.linalg.eigh((Kc+Kc.T)*.5)
        used=min(5,int(np.count_nonzero(vals>max(1e-12,float(vals[-1])*1e-10))),n-2)
        Q=np.linalg.qr(np.column_stack((np.ones(n),vecs[:,-used:] if used else np.empty((n,0)))))[0]
        R=X-(X@Q)@Q.T;norm=np.linalg.norm(R,axis=1);bad=norm<=1e-10*np.sqrt(n)
        norm[bad]=1;R[bad]=0;R/=norm[:,None]
        candidates=[]
        for b in range(1,m):
            for a in range(b):
                value=np.float32(np.clip(abs(R[a]@R[b]),0,1))
                candidates.append((float('inf') if bad[a] or bad[b] else value,b,a))
        forced=sum(not np.isfinite(s) for s,_,_ in candidates)
        keep=forced+max(1,int((len(candidates)-forced)*.1)) if forced<len(candidates) else forced
        expected={(a,b) for _,b,a in sorted(candidates,key=lambda row:(-row[0],row[1],row[2]))[:keep]}
        return work,db,info,args,expected

    def test_global_absolute_top_fraction_and_each_stage_restart(self):
        for interrupted in ('correlations','selection','emit'):
            with self.subTest(stage=interrupted),tempfile.TemporaryDirectory() as temporary:
                folder=Path(temporary);work,db,info,args,expected=self.fixture(folder)
                actual_atomic=screen.atomic_json;stopped=False
                def interrupt(path,value):
                    nonlocal stopped
                    actual_atomic(path,value)
                    progressed=value.get('scored_pairs',0)>0 if interrupted=='correlations' else (
                        value.get('radix_pass',0)>0 if interrupted=='selection' else value.get('emitted',0)>0)
                    if Path(path).name=='progress.json' and value.get('stage')==interrupted and progressed and not stopped:
                        stopped=True;raise RuntimeError('interrupted stage')
                with patch('ko_variation.coarse_screen.atomic_json',side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError,'interrupted stage'):
                        screen.prepare_screen(db,work,info,args,get_meta,set_meta)
                result=screen.prepare_screen(db,work,info,args,get_meta,set_meta)
                pairs=np.fromfile(work/'coarse_screen'/'candidates.u32',dtype='<u4').reshape(-1,2)
                self.assertEqual({tuple(map(int,p)) for p in pairs},expected)
                self.assertEqual(result['selected_pairs'],len(expected));self.assertIn((0,1),expected)
                self.assertFalse((work/'coarse_screen'/'scores.f32').exists())
                arrays=screen.open_candidates(work,len(expected));seen=[]
                while True:
                    tasks,done=screen.candidate_tasks(db,arrays,4,get_meta,set_meta,workers=3)
                    if not seen:
                        self.assertGreater(len({b for _,b in tasks}),1)
                        del arrays;arrays=screen.open_candidates(work,len(expected))
                    seen+=tasks
                    if done:break
                self.assertEqual(len(seen),len(expected));self.assertEqual(len(set(seen)),len(seen))
                del arrays;db.close()

    def test_undefined_residuals_are_rescued(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);work,db,info,args,expected=self.fixture(folder,force=True)
            result=screen.prepare_screen(db,work,info,args,get_meta,set_meta)
            pairs=np.fromfile(work/'coarse_screen'/'candidates.u32',dtype='<u4').reshape(-1,2)
            self.assertEqual({tuple(map(int,p)) for p in pairs},expected)
            self.assertGreaterEqual(result['undefined_representatives'],2)
            self.assertGreater(result['selected_pairs'],19);db.close()

    def test_borrow_preparation_hierarchy_nulls_without_old_search_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);fasta,tree,_=fixtures.DiscoveryTests.inputs(folder)
            common=['--fasta',str(fasta),'--tree',str(tree),'--threads','2','--memory-gb','1',
                    '--min-cell-count','1','--leaf-size','2','--screen-p','.9']
            old=folder/'old';new=folder/'new'
            main(common+['--out',str(old),'--coarse-screen','none'])
            main(common+['--out',str(new),'--reuse-prepared-from',str(old),'--require-cached-nulls','--coarse-retain','1'])
            summary=json.loads((new/'run_summary.json').read_text())
            self.assertEqual(summary['bonferroni_tests'],21)
            self.assertEqual(summary['raw_p_threshold'],.05/21)
            self.assertEqual(summary['reused_prepared_from'],str(old.resolve()))
            self.assertEqual(summary['prefit_runtime_totals'].get('new_null_fits',0),0)
            self.assertFalse(list((new/'.kovar_work'/'null_models').glob('*/*.npz')))
            a=connect(old/'.kovar_work'/'work.sqlite');b=connect(new/'.kovar_work'/'work.sqlite')
            self.assertEqual(set(a.execute('SELECT u,v FROM decisions WHERE final=1')),
                             set(b.execute('SELECT u,v FROM decisions WHERE final=1')))
            a.close();b.close()


if __name__=='__main__': unittest.main()
