"""Focused resource/scheduling checks; no large-scale performance assertions."""
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ko_variation.memory_budget import (ByteLRU,cgroup_limits,detect_budget,plan_workers,GIB)
from ko_variation.discovery_execution import bounded_map,ResponsePool
from ko_variation.null_identity import migrate_identity,V090_CODE
from ko_variation.discovery_store import connect
from ko_variation.discovery import schema,search,get_meta
from ko_variation import discovery_score as kernels
from ko_variation.discovery_store import digest


class ResourceTests(unittest.TestCase):
    def test_adaptive_allocation_and_explicit_cap(self):
        for allocation in (8,32,240):
            budget=detect_budget(None,24,environ={'SLURM_MEM_PER_NODE':str(allocation*1024)},groups=[],physical=512*GIB)
            self.assertAlmostEqual(budget['working_gb'],allocation*.8,places=6)
        budget=detect_budget(4,24,environ={'SLURM_MEM_PER_NODE':'8192'},groups=[],physical=512*GIB)
        self.assertEqual(budget['working_gb'],4)
        low=detect_budget(192,24,environ={'SLURM_MEM_PER_NODE':'8192'},groups=[],physical=512*GIB)
        self.assertAlmostEqual(low['working_gb'],6.4)
        local=detect_budget(None,4,environ={},groups=[],physical=512*GIB)
        self.assertEqual(local['working_gb'],4)
        per_cpu=detect_budget(None,4,environ={'SLURM_MEM_PER_CPU':'1024','SLURM_CPUS_ON_NODE':'24'},groups=[],physical=512*GIB)
        self.assertAlmostEqual(per_cpu['working_gb'],19.2)

    def test_cgroup_parent_limit_and_memory_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'cgroup'
            (root/'slurm/job/task').mkdir(parents=True)
            (root/'memory.max').write_text('max')
            (root/'slurm/job/memory.max').write_text(str(8*GIB))
            member=Path(temporary)/'membership'
            member.write_text('0::/slurm/job/task\n')
            groups=cgroup_limits(root,member)
            budget=detect_budget(None,2,environ={'SLURM_MEM_PER_NODE':'32768'},groups=groups,physical=512*GIB)
            self.assertAlmostEqual(budget['working_gb'],6.4)
        big=detect_budget(None,24,environ={'SLURM_MEM_PER_NODE':str(240*1024)},groups=[],physical=512*GIB)
        plan=plan_workers(big,613,24,256)
        self.assertGreater(plan['cache_bytes_per_worker'],GIB)
        self.assertLessEqual(plan['total_cache_bytes'],big['working_bytes']*.6)
        self.assertLessEqual(plan['query_buffer_pairs'],65536)

    def test_lru_is_byte_bounded_and_shrinks(self):
        cache=ByteLRU(100)
        cache.put('a',1,40)
        cache.put('b',2,40)
        self.assertEqual(cache.get('a'),1)
        cache.put('c',3,40)
        self.assertNotIn('b',cache)
        self.assertLessEqual(cache.bytes,100)
        cache.trim(30)
        self.assertEqual(cache.bytes,0)
        cache.put('oversize',4,50)
        self.assertNotIn('oversize',cache)

    def test_unordered_consumes_completed_jobs(self):
        gate=threading.Event()
        def job(index):
            if index==0:
                gate.wait(3)
            return index
        with ThreadPoolExecutor(max_workers=2) as pool:
            stream=bounded_map(pool,job,range(5),2,ordered=False)
            try:
                first=next(stream)
                self.assertNotEqual(first,0)
            finally:
                gate.set()
            self.assertEqual(set([first]+list(stream)),set(range(5)))

    def test_response_affinity(self):
        class Executor:
            def __init__(self,index): self.index=index
            def submit(self,function,job): return self.index
        pool=object.__new__(ResponsePool)
        pool.workers,pool.next_worker=4,0
        pool.executors=[Executor(i) for i in range(4)]
        def score_profiled(job): return job
        self.assertEqual(pool.submit(score_profiled,(7,[])),3)
        self.assertEqual(pool.submit(score_profiled,(7,[])),3)

    def test_pressure_shrinks_cache_and_changed_files_are_rehashed(self):
        cache=ByteLRU(100)
        cache.put('a',1,40)
        cache.put('b',2,40)
        state=dict(nulls=cache,cache_limit=100,last_pressure=0.,metrics={},verified=OrderedDict(),
                   settings=dict(monitor={'test':'fixture'},total_working_bytes=1000))
        with patch.dict(kernels._STATE,state,clear=True),patch('ko_variation.discovery_score.working_usage',return_value=1200):
            kernels.pressure_guard()
            self.assertEqual(cache.limit,0)
            self.assertEqual(cache.bytes,0)
            self.assertEqual(kernels._STATE['metrics']['memory_pressure_events'],1)
            kernels._STATE['last_pressure']=0.
            with patch('ko_variation.discovery_score.working_usage',return_value=950):
                kernels.pressure_guard()
            self.assertEqual(cache.limit,50)
            cache.put('recovered',1,40)
            self.assertIn('recovered',cache)
            with tempfile.TemporaryDirectory() as temporary:
                product=Path(temporary)/'model.npz'
                product.write_bytes(b'abc')
                expected=digest(product)
                kernels.verify_product(product,expected)
                kernels.verify_product(product,expected)
                product.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'Corrupted'):
                    kernels.verify_product(product,expected)

    def test_migration_rejects_unknown_code_and_changed_science(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest=Path(temporary)/'identity.json'
            old=dict(version='0.9.0',code={k:next(iter(v)) for k,v in V090_CODE.items()},settings={'alpha':.05,'batch_size':256},
                     inputs={},numpy='x',scipy='y',null_model_source_sha256='fit')
            expected=dict(old,version='0.9.1',code={'new':'code'},settings={'alpha':.05})
            bad=dict(old,settings={'alpha':.01,'batch_size':256})
            with self.assertRaisesRegex(ValueError,'Scientific'):
                migrate_identity(bad,expected,manifest)
            bad=dict(old,code=dict(old['code'],**{'discovery.py':'unknown'}))
            with self.assertRaisesRegex(ValueError,'verified'):
                migrate_identity(bad,expected,manifest)

    def test_recursive_frontier_and_restart_do_not_lose_pairs(self):
        with tempfile.TemporaryDirectory() as temporary:
            db=connect(Path(temporary)/'work.sqlite')
            schema(db)
            nodes=[(0,None,8,0),(1,0,4,0),(2,0,4,4),(3,1,2,0),(4,1,2,2),(5,2,2,4),(6,2,2,6)]
            members={1:range(4),2:range(4,8),3:range(2),4:range(2,4),5:range(4,6),6:range(6,8)}
            with db:
                for node,parent,size,anchor in nodes:
                    db.execute('INSERT INTO nodes(id,parent,level,anchor,size,ready) VALUES(?,?,0,?,?,1)',(node,parent,anchor,size))
                    if node:
                        db.execute('INSERT INTO reps VALUES(?,?)',(node,anchor))
                        db.executemany('INSERT INTO members VALUES(?,?)',((node,pid) for pid in members[node]))
            args=SimpleNamespace(search='hierarchy',threads=2,batch_size=2,memory_gb=1,
                                 _plan=dict(frontier_tasks=4,query_buffer_pairs=3))
            calls=0
            observed=[]
            def score(job):
                nonlocal calls
                calls+=1
                if calls==2:
                    raise RuntimeError('interrupted worker')
                response,predictors,_,_=job
                observed.extend((u,response) for u in predictors)
                return dict(rows=[(u,response,1,1,99.,99.,1e-20,'OK') for u in predictors],metrics={'jobs':1,'pairs':len(predictors)})
            with ThreadPoolExecutor(max_workers=2) as pool,patch('ko_variation.discovery_score.score_profiled',side_effect=score):
                with self.assertRaisesRegex(RuntimeError,'interrupted'):
                    search(db,dict(patterns=8),pool,args)
            # A coarse external comparison starts before the internal queue is
            # exhausted; the old internal-first traversal scored (0,2) first.
            self.assertEqual(observed[0],(0,4))
            calls=100
            with ThreadPoolExecutor(max_workers=2) as pool,patch('ko_variation.discovery_score.score_profiled',side_effect=score):
                search(db,dict(patterns=8),pool,args)
            pairs=set(db.execute('SELECT u,v FROM decisions'))
            self.assertEqual(pairs,{(u,v) for v in range(8) for u in range(v)})
            self.assertEqual(get_meta(db,'queried_pairs_count'),28)
            self.assertTrue(get_meta(db,'search_complete'))
            db.close()


if __name__=='__main__':
    unittest.main()
