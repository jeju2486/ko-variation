"""Focused v0.9 checks: mathematical bounds, restart and distance semantics."""
import csv
import gzip
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ko_variation.discovery import (main, parser, distance_and_beta, export, partition,
                                    schema, set_meta)
from ko_variation.discovery_store import connect, prepare, digest, atomic_json, open_patterns
from ko_variation import discovery_score as kernels
from ko_variation.glmm import _weighted_eigen, score_predictor_block


class DiscoveryTests(unittest.TestCase):
    def test_bounds_match_exact_with_covariates_and_nonzero_tau(self):
        rng = np.random.default_rng(483)
        n = 48
        G = rng.normal(size=(n,8))
        K = G @ G.T/8 + np.eye(n)*.2
        weights = rng.uniform(.03,.24,n)
        C = np.column_stack((np.ones(n),rng.normal(size=n)))
        X = rng.integers(0,2,(n,43)).astype(float)
        for tau in (0.,.5,500.):
            decomp = _weighted_eigen(K,weights)
            Sinv_C = decomp.solve(C,tau)
            cache = dict(y=rng.integers(0,2,n).astype(float),mu=rng.uniform(.05,.95,n),
                         weights=weights,C=C,tau=tau,Sinv_C=Sinv_C,
                         M_inv=np.linalg.inv(C.T@Sinv_C),decomp=decomp)
            exact = np.array([r.score_z**2 for r in score_predictor_block(X,cache,compute_spa_adjustment=False)])
            for critical in (.1,5.,50.):
                low,high,p,status = kernels.adaptive_statistics(X,cache,[critical],ranks=[0,8,16])
                self.assertTrue(np.all(low <= exact+1e-8))
                self.assertTrue(np.all(high >= exact-1e-8))
                self.assertTrue(np.array_equal(low>=critical,exact>=critical))

    @staticmethod
    def inputs(folder):
        rng = np.random.default_rng(54)
        n = 96
        A = rng.integers(0,2,n,dtype="u1")
        B = A.copy()
        B[[1,4,8,14,20,24]] ^= 1
        X = np.column_stack((A,A,B,1-B,rng.integers(0,2,(n,4)),np.zeros(n,dtype="u1")))
        fasta,tree,unitigs = folder/"sample_ids.fasta",folder/"tree.nwk",folder/"unitigs"
        fasta.write_text("".join(f">s{i}\n"+"".join("C" if v else "A" for v in row)+"\n" for i,row in enumerate(X)))
        tree.write_text("("+",".join(f"s{i}:1" for i in range(n))+");")
        unitigs.write_text("".join(f"{i} ACGTACGT\n" for i in range(X.shape[1])))
        return fasta,tree,unitigs

    def test_scanner_resume_parallel_exact_equivalence_and_local_retention(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            fasta,tree,unitigs = self.inputs(folder)
            common = ["--fasta",str(fasta),"--tree",str(tree),"--unitigs",str(unitigs),
                      "--min-cell-count","1","--screen-p","0.01","--batch-size","3","--threads","2","--memory-gb","1"]
            out = folder/"bounded"
            self.assertEqual(main(common+["--out",str(out),"--search","exhaustive"]),0)
            work = out/".kovar_work"
            db = connect(work/"work.sqlite")
            bounded = db.execute("SELECT u,v,final FROM decisions ORDER BY u,v").fetchall()
            self.assertTrue(any(r[2]==1 for r in bounded))
            self.assertEqual(db.execute("SELECT pattern FROM unitigs WHERE id=0").fetchone(),
                             db.execute("SELECT pattern FROM unitigs WHERE id=1").fetchone())
            self.assertIsNone(db.execute("SELECT pattern FROM unitigs WHERE id=8").fetchone()[0])
            before = {p:p.stat().st_mtime_ns for p in work.glob("null_models/*/*.npz")}
            # Simulate failure after committed pair decisions but before finishing
            # the response cursor; a resume must not refit/re-score those pairs.
            with db:
                set_meta(db,"search_complete",False)
                set_meta(db,"exhaustive_cursor",[1,0])
            db.close()
            kernels._STATE.clear()
            # Exercise migration from the exact released v0.9.0 identity while
            # preserving the completed pattern/model/decision products.
            from ko_variation.discovery_compat import V090_CODE
            legacy=json.loads((work/"run_identity.json").read_text())
            legacy['version']='0.9.0'
            legacy['code']={key:sorted(values)[0] for key,values in V090_CODE.items()}
            legacy['settings']['batch_size']=3
            atomic_json(work/'run_identity.json',legacy)
            self.assertEqual(main(common+["--out",str(out),"--search","exhaustive","--resume","--require-cached-nulls"]),0)
            self.assertTrue((work/'run_identity.v090.json').exists())
            self.assertEqual(before,{p:p.stat().st_mtime_ns for p in before})
            # The execution-only v0.9.1 migration also keeps every saved model.
            from ko_variation.discovery_compat import V091_CODE
            legacy=json.loads((work/'run_identity.json').read_text())
            legacy['version']='0.9.1'
            legacy['code']={key:sorted(values)[0] for key,values in V091_CODE.items()}
            atomic_json(work/'run_identity.json',legacy)
            self.assertEqual(main(common+["--out",str(out),"--search","exhaustive","--resume","--require-cached-nulls"]),0)
            self.assertTrue((work/'run_identity.v091.json').exists())
            self.assertEqual(before,{p:p.stat().st_mtime_ns for p in before})
            from ko_variation.discovery_compat import V092_CODE
            legacy=json.loads((work/'run_identity.json').read_text())
            legacy.pop('execution_revision',None)
            legacy['code']={key:sorted(values)[0] for key,values in V092_CODE.items()}
            atomic_json(work/'run_identity.json',legacy)
            self.assertEqual(main(common+["--out",str(out),"--search","exhaustive","--resume","--require-cached-nulls"]),0)
            self.assertTrue((work/'run_identity.v092.json').exists())
            self.assertEqual(before,{p:p.stat().st_mtime_ns for p in before})
            from ko_variation.discovery_compat import PREFIT1_CODE
            previous=json.loads((work/'run_identity.json').read_text())
            previous['execution_revision']='prefit-compact-1'
            previous['code']={key:sorted(values)[0] for key,values in PREFIT1_CODE.items()}
            atomic_json(work/'run_identity.json',previous)
            self.assertEqual(main(common+["--out",str(out),"--search","exhaustive","--resume","--require-cached-nulls"]),0)
            self.assertTrue((work/'run_identity.v092.prefit-compact-1.json').exists())
            self.assertEqual(before,{p:p.stat().st_mtime_ns for p in before})
            exact_out = folder/"exact"
            self.assertEqual(main(common+["--out",str(exact_out),"--search","exhaustive","--exact",
                                          "--reuse-nulls-from",str(out),"--require-cached-nulls"]),0)
            exact_db = connect(exact_out/".kovar_work"/"work.sqlite")
            self.assertEqual(bounded,exact_db.execute("SELECT u,v,final FROM decisions ORDER BY u,v").fetchall())
            exact_db.close()
            # The old combined benchmark may order exact patterns differently.
            # Reuse is by validated sample/covariance/bit-pattern identity.
            import shutil
            import scipy
            from ko_variation.null_identity import null_source_hash
            prior=folder/"old_benchmark"
            frozen=folder/"old_preparation"
            (prior/"null_models").mkdir(parents=True)
            frozen.mkdir()
            info=json.loads((work/"prepared.json").read_text())
            packed=np.array(open_patterns(work,info),copy=True)[::-1]
            np.savez(frozen/"preparation.npz",samples=info["samples"],K=np.load(work/"K.npy"),packed=packed)
            for meta in work.glob("null_models/*/*.json"):
                record=json.loads(meta.read_text())
                old_id=info["patterns"]-1-record["pattern"]
                old_meta=prior/"null_models"/f"null_{old_id:09d}.json"
                record["pattern"]=old_id
                if record["array"]:
                    source=meta.parent/record["array"]["name"]
                    target=old_meta.with_suffix(".npz")
                    shutil.copyfile(source,target)
                    record["array"]=dict(name=target.name,sha256=digest(target))
                atomic_json(old_meta,record)
            atomic_json(prior/"run_identity.json",dict(settings=dict(benchmark_dir=str(frozen),null_max_iter=100,null_tolerance=1e-7),
                        inputs=[dict(name="preparation.npz",sha256=digest(frozen/"preparation.npz"))],
                        null_model_source_sha256=null_source_hash(),versions=dict(numpy=np.__version__,scipy=scipy.__version__)))
            reused=folder/"reused"
            self.assertEqual(main(common+["--out",str(reused),"--search","exhaustive", "--reuse-nulls-from",str(prior),"--require-cached-nulls"]),0)
            self.assertFalse(list((reused/".kovar_work"/"null_models").glob("*/*.npz")))
            # Exercise the real hierarchy and compare its queried-pair decisions.
            hierarchy_out=folder/"hierarchy"
            self.assertEqual(main(common+["--out",str(hierarchy_out),"--leaf-size","2","--screen-p","0.9"]),0)
            hdb=connect(hierarchy_out/".kovar_work"/"work.sqlite")
            self.assertGreater(hdb.execute("SELECT count(*) FROM decisions").fetchone()[0],0,
                               str(hdb.execute("SELECT * FROM nodes").fetchall()))
            lookup={(u,v):state for u,v,state in bounded}
            self.assertEqual(set(hdb.execute('SELECT u,v FROM decisions WHERE final=1')),
                             {(u,v) for u,v,state in bounded if state==1})
            for u,v,state in hdb.execute("SELECT u,v,final FROM decisions"):
                self.assertEqual(state,lookup[u,v])
            hdb.close()
            # Distance adapter is mocked: no claim that the external Linux tool
            # ran here. Both sides of 10kb must survive export; only distal
            # pattern pairs are submitted for beta estimation.
            args=parser().parse_args(common+["--out",str(out),"--search","exhaustive"])
            paths=folder/"graph.paths"
            edge=folder/"sample.edges"
            edge.write_text("0 1 FF\n")
            paths.write_text(str(edge)+"\n")
            args.paths,args.unitig_distance=paths,"fake_distance"
            info=json.loads((work/"prepared.json").read_text())
            db=connect(work/"work.sqlite")
            beta_jobs=[]
            def fake_run(command,**kwargs):
                if "--help" in command:
                    return subprocess.CompletedProcess(command,0,stdout="--sgg-paths-file",stderr="")
                stem=Path(command[command.index("--output-stem")+1])
                query=Path(command[command.index("--queries-file")+1])
                lines=[]
                for i,line in enumerate(query.read_text().splitlines()):
                    u,v=map(int,line.split()[:2])
                    distance=10000 if i==0 else 12000
                    lines.append(f"{u} {v} {distance} 0 0 10 0 {distance} {distance}\n")
                Path(str(stem)+".ud_0_based").write_text("".join(lines))
                return subprocess.CompletedProcess(command,0)
            def fake_beta(job):
                beta_jobs.append(job)
                return *job,.7,"OK"
            with ThreadPoolExecutor(max_workers=2) as pool,patch("ko_variation.discovery.shutil.which",return_value=str(edge)),\
                    patch("ko_variation.discovery.subprocess.run",side_effect=fake_run),\
                    patch("ko_variation.discovery_score.beta",side_effect=fake_beta):
                distance_and_beta(db,work,pool,args)
            export(db,info,work,args,0)
            with gzip.open(out/"associations.tsv.gz","rt") as stream:
                rows=list(csv.DictReader(stream,delimiter="\t"))
            local=[r for r in rows if r["distance_status"]=="LOCAL"]
            distal=[r for r in rows if r["distance_status"]=="DISTAL"]
            self.assertTrue(local and distal and beta_jobs)
            self.assertTrue(all(r["beta"]=="" for r in local))
            self.assertTrue(all(r["beta"]!="" for r in distal))
            self.assertEqual(set(beta_jobs),set((int(r["predictor_pattern"]),int(r["response_pattern"])) for r in distal))
            import pandas as pd
            from ko_variation.network import discovery_network_input,build_gene_network
            adapted=discovery_network_input(pd.read_csv(out/"associations.tsv.gz",sep="\t"),
                                             json.loads((out/"run_summary.json").read_text()))
            annotation=pd.DataFrame(dict(locus=range(9),gene=[f"gene{i}" for i in range(9)]))
            nodes,edges=build_gene_network(adapted,annotation,significance_column="p_upper")
            self.assertGreater(len(edges),0)
            self.assertTrue(edges.significance_column.eq("p_upper").all())
            db.close()

    def test_interrupted_category_partition_resumes_same_anchor(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            fasta,tree,unitigs=self.inputs(folder)
            args=parser().parse_args(["--fasta",str(fasta),"--tree",str(tree),"--out",str(folder/"out"),"--leaf-size","2"])
            work=folder/"work"
            work.mkdir()
            db=connect(work/"work.sqlite")
            schema(db)
            with patch("ko_variation.discovery_store.prepare_kinship_eigensystem",side_effect=RuntimeError("interrupted after packing")):
                with self.assertRaisesRegex(RuntimeError,"interrupted"):
                    prepare(args,db,work)
            committed=(work/"patterns.bin").read_bytes()
            info=prepare(args,db,work)
            self.assertEqual(committed,(work/"patterns.bin").read_bytes())
            kernels.initialize(str(work),info,{})
            ids=list(range(info["patterns"]))
            total=kernels.standardized_sum(ids)
            _,central=kernels.central((ids,total))
            rho=np.corrcoef(kernels.bits(ids).astype(float))
            self.assertAlmostEqual(rho.mean(axis=1)[central],rho.mean(axis=1).max(),places=12)
            with db:
                db.execute("INSERT INTO nodes(id,parent,level,anchor,size) VALUES(0,NULL,-1,0,?)",(info["patterns"],))
                db.execute("INSERT INTO members SELECT 0,id FROM patterns")
                child=db.execute("INSERT INTO nodes(parent,level,anchor) VALUES(0,0,0)").lastrowid
                db.execute("INSERT INTO assigned VALUES(0,0,?)",(child,))
                set_meta(db,"partition:0",[0,child])
            with ThreadPoolExecutor(max_workers=1) as pool:
                partition(db,0,.8,pool,args,info["n"])
            assigned=db.execute("SELECT pid,child FROM assigned WHERE parent=0 ORDER BY pid").fetchall()
            self.assertEqual(len(assigned),info["patterns"])
            self.assertEqual(assigned[0][1],child)
            self.assertEqual(db.execute("SELECT count(*) FROM members WHERE node=?",(child,)).fetchone()[0],
                             sum(c==child for _,c in assigned))
            db.close()
            kernels._STATE.clear()


if __name__=="__main__":
    unittest.main()
