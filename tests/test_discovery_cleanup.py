"""Focused regressions for inventory reuse and unchanged-cache bookkeeping."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ko_variation import discovery_score as kernels, discovery_nulls as inventory
from ko_variation.discovery import schema,get_meta,set_meta,merge_metrics
from ko_variation.discovery_nulls import prefit,publish_aliases
from ko_variation.discovery_store import connect,atomic_json
from ko_variation.memory_budget import ByteLRU


class CleanupTests(unittest.TestCase):
    def test_interrupted_inventory_skips_committed_models_and_export_rescan(self):
        with tempfile.TemporaryDirectory() as temporary:
            work=Path(temporary)
            db=connect(work/'work.sqlite');schema(db)
            args=SimpleNamespace(prefit_nulls='all',null_cache_format='compact',null_cache_rank=128,
                                 threads=1,out=work)
            signature=json.dumps(['compact',128,3],separators=(',',':'))
            def entry(pid):
                path=work/f'null_{pid}.json'
                record=dict(status='OK',mode='zero',fit_seconds=1.,pattern=pid)
                atomic_json(path,record)
                return dict(pattern=pid,record=record,source=str(path),local=True)
            with db:
                db.executemany('INSERT INTO null_aliases VALUES (?,?)',[(i,i) for i in range(3)])
                inventory.record_models(db,[entry(0)],signature)
            publish_aliases(db,work,np.arange(3))
            seen=[]
            def fit(job):
                pid,_=job;seen.append(pid)
                return dict(entry(pid),status='OK',mode='zero',metrics={'new_null_fits':1},worker_state={'pid':1})
            def executor(**kwargs):
                return ThreadPoolExecutor(max_workers=kwargs['max_workers'])
            with patch('ko_variation.discovery_nulls.ProcessPoolExecutor',side_effect=executor),\
                 patch('ko_variation.discovery_score.prefit_profiled',side_effect=fit):
                prefit(db,work,{'patterns':3},args,{},get_meta,set_meta,merge_metrics)
            self.assertEqual(seen,[1,2])
            with patch.object(Path,'glob',side_effect=AssertionError('rescanned completed inventory')):
                self.assertEqual(inventory.local_summary(db,work),({'OK':3},3.))
            db.close()

    def test_unchanged_cached_model_is_not_recharged_each_job(self):
        base={'_pattern':0};cache=ByteLRU(10000);cache.put(0,(base,'OK'),100)
        state=dict(nulls=cache,settings={'batch_size':2})
        def chunk(job):
            kernels._STATE['active_null']=(0,(base,'OK'))
            return []
        with patch.dict(kernels._STATE,state,clear=True),\
             patch('ko_variation.discovery_score._score_chunk',side_effect=chunk),\
             patch('ko_variation.discovery_score.cache_charge',return_value=200) as charge:
            kernels.score((0,[1,2],None,False))
            charge.assert_not_called()
            base['_charge_dirty']=True
            kernels.score((0,[1],None,False))
            charge.assert_called_once_with(base)
            kernels.score((0,[1],None,False))
            self.assertEqual(charge.call_count,1)


if __name__=='__main__': unittest.main()
