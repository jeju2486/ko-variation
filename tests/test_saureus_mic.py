"""Scientific contracts of the focal MIC workflow; no external AMRFinder run."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.optimize._numdiff import approx_derivative

from benchmarks import saureus_mic as mic


class FocalMicTests(unittest.TestCase):
    def test_database_version_paths_are_unquoted_and_uncoloured(self):
        for text,path in [
            ("Database directory: '/data/amrfinder/2026-11-01.1'\nDatabase version: x", "/data/amrfinder/2026-11-01.1"),
            ('Database directory: "/data/AMR database/latest"\n', "/data/AMR database/latest"),
            ("Database directory: /data/amrfinder/latest\n", "/data/amrfinder/latest"),
            ("Database directory: \x1b[32m'/data/amrfinder/latest'\x1b[0m\n", "/data/amrfinder/latest"),
            ("Database directory: '/data/O'\\''Brien/db'\n", "/data/O'Brien/db"),
        ]:
            self.assertEqual(mic.database_from_version(text),Path(path))
        self.assertIsNone(mic.database_from_version("No database installed"))

    def test_prepared_matrix_accepts_only_known_legacy_code_and_identical_inputs(self):
        current=dict(revision=mic.REVISION,code="new preparation code",sources={"fasta":{"sha256":"same"}},groups=10)
        previous={**current,"code":next(iter(mic.LEGACY_PREPARATION_CODES))}
        self.assertTrue(mic.compatible_preparation(previous,current))
        self.assertFalse(mic.compatible_preparation({**previous,"groups":11},current))
        self.assertFalse(mic.compatible_preparation({**previous,"sources":{"fasta":{"sha256":"changed"}}},current))
        self.assertFalse(mic.compatible_preparation({**previous,"code":"unknown code"},current))

    def test_source_search_ignores_other_species_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for name in ["target.fa","other.fa","other.fna"]: (root/name).write_text(">c\nACGT\n")
            found=mic.source_index(root,[".fna",".fa"],{"target"})
            self.assertEqual(set(found),{"target"})
            (root/"target.fna").write_text(">c\nACGT\n")
            with self.assertRaisesRegex(ValueError,"Multiple files"):
                mic.source_index(root,[".fna",".fa"],{"target"})

    def test_censored_likelihood_gradient_and_tail_stability(self):
        rng=np.random.default_rng(41)
        X=np.column_stack([np.ones(24),rng.normal(size=(24,3))])
        y=rng.normal(size=24)
        signs=np.tile([-1,0,1],8)
        theta=np.array([.2,-.3,.5,.1,-.1])
        penalty=np.array([0,.1,.1,.1])
        f=lambda t: mic.censored_loss(t,X,y,signs,penalty)[0]
        _,gradient=mic.censored_loss(theta,X,y,signs,penalty)
        np.testing.assert_allclose(gradient,approx_derivative(f,theta).ravel(),rtol=2e-5,atol=1e-6)
        self.assertTrue(np.isfinite(mic.prediction_loss(np.array([-1000.,1000.]),np.array([-1,1]),np.zeros(2),1.)).all())

    def test_censored_data_recovers_effect_without_imputing_limits(self):
        rng=np.random.default_rng(42)
        x=rng.integers(0,2,1200)
        X=np.column_stack([np.ones(len(x)),x])
        latent=-1.+1.4*x+rng.normal(0,.6,len(x))
        limits=np.clip(latent,-1.3,.9)
        signs=np.where(latent<-1.3,-1,np.where(latent>.9,1,0))
        beta,sigma=mic.fit_censored(X,limits,signs,.1,1)
        self.assertLess(abs(beta[1]-1.4),.12)
        self.assertLess(abs(sigma-.6),.08)
        naive=np.linalg.lstsq(X,limits,rcond=None)[0][1]
        self.assertGreater(beta[1],naive+.15)

    def test_unknown_or_duplicate_mutation_calls_are_not_wildtype(self):
        mutation={"Gene symbol":"gyrA_A3V","Element subtype":"POINT"}
        main={"mutant":[mutation],"wild":[],"missing":[],"duplicate":[]}
        calls={"mutant":[mutation],"wild":[{"Gene symbol":"gyrA_A3A"}],"missing":[],
               "duplicate":[{"Gene symbol":"gyrA_A3A"},{"Gene symbol":"gyrA_A3A"}]}
        G,names=mic.known_matrix(list(main),main,calls,90.)
        self.assertEqual(names,["blaZ_presence","mecA_presence","mecC_presence","gyrA_A3V"])
        np.testing.assert_equal(G[:,-1],[1,0,np.nan,np.nan])
        main["wild"]=[{"Gene symbol":"blaZ","% Coverage of reference sequence":"50","Method":"PARTIALX"}]
        G,_=mic.known_matrix(list(main),main,calls,90.)
        self.assertTrue(np.isnan(G[1,0]))

    def test_unitig_known_codon_overlap_on_both_strands(self):
        # Synthetic peptide/position; no claim this is a biological AMR allele.
        from Bio.Seq import Seq
        coding="ATGAAAGTTGGTTTT"  # MKVGF; reference residue at position 3 is A.
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/"sample.fa"
            for strand,sequence in [("+",coding),("-",str(Seq(coding).reverse_complement()))]:
                path.write_text(">ctg\n"+sequence+"\n")
                call={"Gene symbol":"gyrA_A3V","Contig id":"ctg","Start":"1","Stop":"15",
                      "Strand":strand,"Accession of closest sequence":"ref"}
                rows=mic.audit_sample(("sample",path,[call],{"2":"AAAGTTGGT"},{"2":"gyrA"},{"ref":"MKAGF"}))
                self.assertEqual(rows[0]["coordinate_resolved"],1)
                self.assertEqual(rows[0]["covers_known_codon"],1)

    def test_tree_projection_never_uses_test_test_covariance(self):
        rng=np.random.default_rng(12)
        U=rng.normal(size=(12,5)); K=U@U.T+np.eye(12)
        train=np.arange(8); test=np.arange(8,12)
        before=mic.tree_axes(K,train,test,3)
        K[8:,8:]+=1000*np.eye(4)
        after=mic.tree_axes(K,train,test,3)
        np.testing.assert_allclose(before[0],after[0])
        np.testing.assert_allclose(before[1],after[1])

    def test_primary_models_skip_the_tree_eigendecomposition(self):
        from unittest.mock import patch
        with patch.object(mic.np.linalg,"eigh",side_effect=AssertionError("No PCs requested")):
            train,test=mic.tree_axes(np.eye(8),np.arange(5),np.arange(5,8),0)
        self.assertEqual(train.shape,(5,0))
        self.assertEqual(test.shape,(3,0))

    def test_random_partner_counts_match_independent_added_patterns(self):
        rng=np.random.default_rng(7)
        F=rng.integers(0,2,(80,20)).astype(float)
        F[:,2]=F[:,0]  # A redundant neighbour must not increase the control budget.
        chosen,difference=mic.matched_partners(F,np.arange(60),[0],[2,3,4],list(range(5,20)),5,42)
        self.assertEqual(len(chosen),2)
        self.assertTrue(set(chosen).isdisjoint({0,2,3,4}))
        self.assertLess(difference,.1)

    def test_current_external_support_map_contract(self):
        nodes=[dict(gene=g,label=g) for g in ["blaZ","gyrA","parC","other"]]
        edges=[dict(gene_a="blaZ",gene_b="other",locus_pairs="external:gene_pair_support.tsv.gz"),
               dict(gene_a="gyrA",gene_b="parC",locus_pairs="external:gene_pair_support.tsv.gz")]
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/"map.html"
            path.write_text('<script id="data" type="application/json">'+json.dumps(dict(nodes=nodes,edges=edges))+'</script>')
            _,_,groups,_,rows=mic.load_map(path)
            self.assertEqual(groups["parC"],{"parC"})
            self.assertEqual(len(rows),3)

    def test_prediction_processes_share_cohort_and_resume_completed_partitions(self):
        # Simulated arrays exercise process workers and checkpoints, not AMRFinder.
        from ko_variation.phenotype import json_write, npz_write
        rng=np.random.default_rng(12)
        n=120
        X=rng.integers(0,2,(n,16)).astype(np.uint8)
        G=X[:,:3].astype(float)
        metadata=[]
        for i in range(n):
            y=-1+G[i,0]+rng.normal(0,.6)
            z=-.8+G[i,1]+.5*G[i,2]+rng.normal(0,.6)
            metadata.append(dict(benzylpenicillin_mic=str(2**max(-1.5,y)),
                benzylpenicillin_mic_sign="<" if y<-1.5 else "=",
                levofloxacin_mic=str(2**z),levofloxacin_mic_sign="="))
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            info=dict(samples=[f"s{i}" for i in range(n)],groups=[str(i//20) for i in range(n)],
                metadata=metadata,loci=list(range(16)),target_columns=dict(penicillin=[0],fluoroquinolone=[1,2]),
                neighbour_columns=dict(penicillin=[3],fluoroquinolone=[4]),
                second_columns=dict(penicillin=[5],fluoroquinolone=[6]),
                gene_columns=dict(blaZ=[0],gyrA=[1],parC=[2]),random_pool=list(range(7,16)))
            json_write(root/"prepared.json",info)
            npz_write(root/"prepared.npz",X=X,K=np.eye(n))
            npz_write(root/"genotypes.npz",G=G,names=np.array(["blaZ_presence","gyrA_A3V","parC_A3V"]))
            json_write(root/"audit_identity.json",dict(synthetic=True,revision=mic.AUDIT_REVISION))
            args=SimpleNamespace(out=root,quinolone="levofloxacin",penicillin_column="benzylpenicillin_mic",
                penicillin_sign_column="benzylpenicillin_mic_sign",folds=3,pcs=0,min_count=5,seed=42,
                alphas=[1.],threads=2,random_sets=1)
            mic.predict(args,info)
            prediction=Path(json.loads((root/"latest_prediction.json").read_text())["path"])
            rows=list(mic.records(prediction/"performance.tsv"))
            self.assertTrue(rows)
            self.assertTrue(all(r["status"]=="OK" for r in rows))
            self.assertTrue(all(int(r["n"])==n for r in rows))
            file=prediction/"checkpoints/penicillin/fold0_blaZ_gene_N1_N2_0.json"
            timestamp=file.stat().st_mtime_ns
            mic.predict(args,info)
            self.assertEqual(file.stat().st_mtime_ns,timestamp)
            self.assertTrue(any(r["model"]=="blaZ_unitigs_N1_N2" for r in rows))
            self.assertTrue(any(r["model"]=="gyrA_parC_alleles" for r in rows))
            self.assertEqual(len(list(mic.records(prediction/"neighbour_improvement.tsv"))),6)

    def test_catalog_offsets_and_prefixed_aliases_do_not_use_traditional_numbering(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/"catalog.tsv"
            path.write_text("Staphylococcus_aureus\tref\t3\tgrlA_A13V\tgrlA_A13V\n")
            catalog=mic.mutation_catalog(path)
            call={"Gene symbol":"Staphylococcus_aureus_grlA_A13V","Contig id":"ctg",
                "Start":"1","Stop":"15","Strand":"+","Accession of closest sequence":"ref"}
            calls=mic.parsed_calls([call,call],catalog)
            self.assertEqual(len(calls),1)
            self.assertEqual(calls[0]["Gene symbol"],"parC_A13V")
            self.assertEqual(calls[0]["Reference position"],3)
            genome=Path(temporary)/"genome.fa"; genome.write_text(">ctg\nATGAAAGTTGGTTTT\n")
            evidence=mic.audit_sample(("sample",genome,calls,{"0":"AAAGTTGGT"},{"0":"parC"},{"ref":"MKAGF"}))
            self.assertEqual(evidence[0]["covers_known_codon"],1)

    def test_reference_recovery_requires_one_gene_copy_and_resumes(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); source=root/"genome.fa"
            source.write_text(">ctg\n"+"ATGAAAGTT"+"GGT"*17+"\n")
            catalog=[dict(gene="gyrA",wild="A",position=13,allele="V",accession="ref",reference_position=3,marker="gyrA_A13V")]
            reference={"ref":"MKA"+"G"*17}
            def fake_blast(command,**kwargs):
                Path(command[command.index("-out")+1]).write_text("ref\tctg\t1\t60\t95\t1\t20\t20\t100\n")
            job=("s",source,{"gyrA"},catalog,reference,root/"recovery","tblastn",{"test":1})
            with patch.object(mic.subprocess,"run",side_effect=fake_blast) as run:
                calls,notes=mic.recover_target_calls(job)
                self.assertEqual(calls[0]["Gene symbol"],"gyrA_A13V")
                self.assertEqual(calls[0]["Reference position"],3)
                self.assertEqual(notes[0]["status"],"UNIQUE_REFERENCE_HIT")
                self.assertEqual(mic.recover_target_calls(job),(calls,notes))
                self.assertEqual(run.call_count,1)
            def duplicated(command,**kwargs):
                Path(command[command.index("-out")+1]).write_text(
                    "ref\tctg\t1\t60\t95\t1\t20\t20\t100\nref\tctg2\t1\t60\t95\t1\t20\t20\t100\n")
            with patch.object(mic.subprocess,"run",side_effect=duplicated):
                calls,notes=mic.recover_target_calls(("s2",*job[1:]))
                self.assertEqual(calls,[])
                self.assertEqual(notes[0]["status"],"MULTIPLE_REFERENCE_HITS")

    def test_full_audit_recovers_missing_calls_and_reuses_original_reports(self):
        import subprocess
        import sys
        from concurrent.futures import ThreadPoolExecutor
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); assemblies=root/"assemblies"; assemblies.mkdir()
            db=root/"db"; db.mkdir(); out=root/"out"; out.mkdir()
            (db/"AMRProt.fa").write_text(">refG\nMKA"+"G"*17+"\n>refP\nMKA"+"P"*17+"\n")
            (db/"AMRProt-mutation.tsv").write_text(
                "Staphylococcus_aureus\trefG\t3\tgyrA_A13V\tgyrA_A13V\n"
                "Staphylococcus_aureus\trefP\t3\tparC_A13V\tparC_A13V\n")
            version=f"Database directory: '{db}'\nDatabase version: synthetic\n"
            for sample,codon in [("mutant","GTT"),("wild","GCT")]:
                source=assemblies/f"{sample}.fa"
                source.write_text(">g\nATGAAA"+codon+"GGT"*17+"\n>p\nATGAAAGCT"+"CCT"*17+"\n")
                cached=out/"amrfinder"/sample; cached.mkdir(parents=True)
                mic.json_write(cached/"done.json",dict(sample=sample,source_sha256=mic.sha256(source),amrfinder=version))
                (cached/"mutations.tsv").write_text("Gene symbol\tElement subtype\n")
                (cached/"amr.tsv").write_text("Gene symbol\t% Coverage of reference sequence\tMethod\n"+
                    ("blaZ\t100\tEXACTX\n" if sample=="mutant" else ""))
            info=dict(identity={"synthetic":True},samples=["mutant","wild"],loci=[0,1,2],
                target_dna={"0":"AAAGTTGGT","1":"AAAGCTCCT","2":"ATGCGT"},
                target_gene={"0":"gyrA","1":"parC","2":"blaZ"},
                target_columns={"penicillin":[2],"fluoroquinolone":[0,1]},gene_columns={"blaZ":[2]})
            mic.npz_write(out/"prepared.npz",X=np.array([[1,1,1],[0,1,0]]))
            args=SimpleNamespace(out=out,assemblies=assemblies,gff_dir=None,amrfinder=sys.executable,
                amrfinder_db=None,reference_proteins=None,mutation_catalog=None,min_gene_coverage=90.,
                threads=2,tblastn=sys.executable)
            def run(command,**kwargs):
                if command[-1]=="-V": return subprocess.CompletedProcess(command,0,version,"")
                if "-query" not in command: raise AssertionError("Original AMRFinder reports should be reused")
                Path(command[command.index("-out")+1]).write_text(
                    "refG\tg\t1\t60\t95\t1\t20\t20\t100\nrefP\tp\t1\t60\t100\t1\t20\t20\t100\n")
            with patch.object(mic,"ProcessPoolExecutor",ThreadPoolExecutor),patch.object(mic.subprocess,"run",side_effect=run) as external:
                mic.allele_audit(args,info)
                initial=external.call_count
                self.assertEqual(initial,3)  # -V and two targeted recoveries; no AMRFinder rerun.
                with np.load(out/"genotypes.npz") as f:
                    names=f["names"].tolist(); G=f["G"]
                np.testing.assert_equal(G[:,names.index("gyrA_A13V")],[1,0])
                np.testing.assert_equal(G[:,names.index("parC_A13V")],[0,0])
                audit=list(mic.records(out/"unitig_allele_audit.tsv"))
                self.assertTrue(any(r.get("marker_interpretation")=="VERIFIED_MUTANT_MARKER" for r in audit))
                self.assertTrue(any(r.get("marker_interpretation")=="VERIFIED_WILDTYPE_MARKER" for r in audit))
                self.assertEqual(len(list(mic.records(out/"blaz_carriage_agreement.tsv"))),1)
                # Force derived-audit migration; both expensive call layers stay cached.
                mic.json_write(out/"audit_identity.json",dict(legacy=True))
                mic.allele_audit(args,info)
                self.assertEqual(external.call_count,initial+1)

    def test_second_neighbours_are_separate_and_anchor_genes_are_excluded(self):
        info=dict(loci=list(range(6)),locus_gene={str(i):g for i,g in enumerate(["blaZ","gyrA","parC","n1","n2","pool"])})
        groups={g:{g} for g in ["blaZ","gyrA","parC"]}
        adjacency={"blaZ":{"n1"},"n1":{"blaZ","n2"},"n2":{"n1"},"gyrA":{"parC"},"parC":{"gyrA"}}
        enriched=mic.neighbourhood_columns(info,(None,None,groups,adjacency))
        self.assertEqual(enriched["second_columns"]["penicillin"],[4])
        self.assertEqual(enriched["second_columns"]["fluoroquinolone"],[])
        enriched.update(target_columns={"penicillin":[0],"fluoroquinolone":[1,2]},
            neighbour_columns={"penicillin":[3],"fluoroquinolone":[]})
        models=mic.prediction_models("penicillin",enriched,["blaZ_presence"])
        self.assertEqual(models["blaZ_gene"]["base"],[0])
        self.assertEqual(models["blaZ_unitigs"]["base"],[1])
        self.assertEqual(models["blaZ_gene_N1_N2"]["extra"],[4,5])

    def test_failed_checkpoints_retry_and_keep_inner_error_details(self):
        from unittest.mock import patch
        rng=np.random.default_rng(51); n=90
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); prediction=root/"prediction"; prediction.mkdir()
            info=dict(samples=[str(i) for i in range(n)],groups=[str(i//15) for i in range(n)],
                metadata=[dict(benzylpenicillin_mic=str(2**v),benzylpenicillin_mic_sign="=") for v in rng.normal(size=n)],
                loci=list(range(5)),target_columns={"penicillin":[0]},neighbour_columns={"penicillin":[1]},
                second_columns={"penicillin":[2]},random_pool=[3,4])
            mic.json_write(root/"prepared.json",info)
            mic.npz_write(root/"prepared.npz",X=rng.integers(0,2,(n,5)),K=np.eye(n))
            mic.npz_write(root/"genotypes.npz",G=rng.integers(0,2,(n,1)),names=np.array(["blaZ_presence"]))
            mic.worker_init(dict(out=str(root),prediction_dir=str(prediction),quinolone="levofloxacin",
                penicillin_column="benzylpenicillin_mic",penicillin_sign_column="benzylpenicillin_mic_sign",
                folds=3,pcs=0,min_count=5,seed=42,alphas=[1.]))
            job=("penicillin",0,"blaZ_gene",0)
            with patch.object(mic,"fit_censored",side_effect=ValueError("specific numerical failure")):
                failed=mic.prediction_job(job)
            self.assertEqual(failed["status"],"FAILED")
            self.assertEqual(len(failed["inner_failures"]),3)
            self.assertIn("specific numerical failure",failed["reason"])
            passed=mic.prediction_job(job)
            self.assertEqual(passed["status"],"OK")

    def test_prepare_accepts_pruned_tree_without_renumbering_unitigs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            nodes=[dict(gene=g,label=g) for g in ["blaZ","gyrA","parC","other","pool"]]
            edges=[dict(gene_a="blaZ",gene_b="other"),dict(gene_a="gyrA",gene_b="parC"),
                   dict(gene_a="parC",gene_b="other")]
            network=root/"map.json"; network.write_text(json.dumps(dict(nodes=nodes,edges=edges)))
            annotation=root/"annotation.tsv"
            annotation.write_text("locus\tgene\tannotation_class\n"+"".join(
                f"{i}\t{g}\tcoding\n" for i,g in enumerate(["blaZ","gyrA","parC","other","pool"])))
            fasta=root/"samples.fasta"; fasta.write_text("".join(f">s{i}\nACACA\n" for i in range(32)))
            tree=root/"tree.nwk"; tree.write_text("("+",".join(f"s{i}:1" for i in range(31))+");")
            metadata=root/"metadata.tsv"; metadata.write_text("id\tisolate\tspecies\n"+"".join(
                f"s{i}\ts{i}\tStaphylococcus aureus\n" for i in range(32)))
            unitigs=root/"unitigs"; unitigs.write_text("0 ATGAAA\n1 ATGCCC\n2 ATGGGG\n")
            args=SimpleNamespace(out=root,network=network,annotation=annotation,fasta=fasta,tree=tree,
                metadata=metadata,unitigs=unitigs,sample_map=None,clusters=None,locus_map=None,groups=4,
                fasta_columns_are_locus_ids=True)
            data,ns,groups,adj,_=mic.load_map(network)
            prepared=mic.prepare(args,(data,ns,groups,adj))
            self.assertEqual(len(prepared["samples"]),31)
            self.assertNotIn("s31",prepared["samples"])
            self.assertEqual(prepared["loci"],[0,1,2,3,4])
            self.assertEqual(prepared["random_pool"],[4])
            # An installed initial helper already produced this matrix on ARC/
            # Linux. The database-path patch must not redo scientific preparation.
            from unittest.mock import patch
            prepared["identity"]["code"]=next(iter(mic.LEGACY_PREPARATION_CODES))
            mic.json_write(root/"prepared.json",prepared)
            before=(root/"prepared.json").read_bytes()
            with patch.object(mic.np,"stack",side_effect=AssertionError("Unexpected matrix rebuild")):
                reused=mic.prepare(args,(data,ns,groups,adj))
            self.assertEqual(reused,prepared)
            self.assertEqual((root/"prepared.json").read_bytes(),before)


if __name__=="__main__": unittest.main()
