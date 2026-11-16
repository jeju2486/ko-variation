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
                neighbour_columns=dict(penicillin=[3],fluoroquinolone=[4]),random_pool=list(range(5,16)))
            json_write(root/"prepared.json",info)
            npz_write(root/"prepared.npz",X=X,K=np.eye(n))
            npz_write(root/"genotypes.npz",G=G,names=np.array(["blaZ_presence","gyrA_A3V","parC_A3V"]))
            json_write(root/"audit_identity.json",dict(synthetic=True))
            args=SimpleNamespace(out=root,quinolone="levofloxacin",penicillin_column="benzylpenicillin_mic",
                penicillin_sign_column="benzylpenicillin_mic_sign",folds=3,pcs=0,min_count=5,seed=42,
                alphas=[1.],threads=2,random_sets=1)
            mic.predict(args,info)
            rows=list(mic.records(root/"performance.tsv"))
            self.assertTrue(rows)
            self.assertTrue(all(r["status"]=="OK" for r in rows))
            self.assertTrue(all(int(r["n"])==n for r in rows))
            file=root/"prediction_checkpoints/penicillin/fold0_Neighbours_0.json"
            timestamp=file.stat().st_mtime_ns
            mic.predict(args,info)
            self.assertEqual(file.stat().st_mtime_ns,timestamp)

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
