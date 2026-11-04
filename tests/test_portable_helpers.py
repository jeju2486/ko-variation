"""Focused checks for portable selection and annotation after wrapper removal."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from ko_variation.annotation_inputs import main as map_main
from ko_variation.postprocess import SelectionConfig
from ko_variation.selection_cli import select_file


class PortableHelperTests(unittest.TestCase):
    def test_streaming_preserves_full_denominator_and_strict_distance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "full.tsv", root / "selected.tsv"
            pd.DataFrame(dict(u=[0, 0, 0, 0], v=[1, 2, 3, 4],
                status=["OK"] * 4, p_primary=[.03, .001, 1e-8, 1e-8],
                n11=[25] * 4, n10=[25] * 4, n01=[25] * 4, n00=[25] * 4,
                min_distance=[20000, 20000, -1, 10000])).to_csv(source, sep="\t", index=False)
            config = SelectionConfig(distance_column="min_distance", ld_distance=10000)
            details = select_file(source, output, config, chunk_rows=1, pangwes_distances=True)
            selected = pd.read_csv(output, sep="\t")
            self.assertEqual(selected.v.tolist(), [2])
            self.assertEqual(selected.n_tests.tolist(), [4])
            self.assertEqual(details["undefined_distance_rows"], 1)
            self.assertEqual(details["input_rows"], 4)
            with patch("ko_variation.selection_cli.select_distal_signals", side_effect=AssertionError("must reuse cache")):
                self.assertEqual(select_file(source, output, config, resume=True,
                    pangwes_distances=True)["selected_rows"], 1)
            # Negative distances cannot be silently normalized in non-PAN-GWES data.
            with self.assertRaisesRegex(ValueError, "Distance must"):
                select_file(source, root / "strict.tsv", config, chunk_rows=1)

    def test_mapping_keeps_labels_and_ambiguous_gene_overlaps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = root / "bakta" / "S1"
            sample.mkdir(parents=True)
            (sample / "S1.fna").write_text(">contig_1\n" + "A" * 10 + "C" * 10 + "\n")
            (sample / "S1.gff3").write_text(
                "##gff-version 3\n"
                "contig_1\tbakta\tCDS\t1\t10\t.\t+\t0\tID=cds1;gene=tetM\n"
                "contig_1\tbakta\tCDS\t11\t20\t.\t+\t0\tID=cds2;gene=other\n")
            fasta, unitigs, panaroo, selected = [root / name for name in
                ("binary.fa", "unitigs.tsv", "panaroo.csv", "selected.tsv")]
            fasta.write_text(">S1\nAC\n")
            unitigs.write_text("0 AAAAA\n1 CCCCC\n")
            panaroo.write_text("Gene,Non-unique Gene name,Annotation,S1\n"
                "group_1,tet(M),resistance protein,cds1\nother,other,other protein,cds2\n")
            selected.write_text("u\tv\n0\t1\n")
            args = ["--selected", str(selected), "--fasta", str(fasta),
                "--unitigs", str(unitigs), "--panaroo", str(panaroo),
                "--bakta", str(root / "bakta"), "--out", str(root / "annotation")]

            def fake_pyseer(command, **kwargs):
                Path(command[5]).write_text(
                    "AAAAA\t0\tkvref0_ctg0:1-5;x;x;x\n"
                    "CCCCC\t1\tkvref0_ctg0:9-13;x;x;x\n")

            with patch("ko_variation.annotation_inputs.shutil.which", return_value="tool"), \
                    patch("ko_variation.annotation_inputs.subprocess.run", side_effect=fake_pyseer):
                self.assertEqual(map_main(args), 0)
            resolved = pd.read_csv(root / "annotation" / "locus_to_gene.tsv", sep="\t")
            self.assertEqual(resolved.gene.tolist(), ["group_1"])
            self.assertIn("tet(M)", resolved.label.iloc[0])
            audit = pd.read_csv(root / "annotation" / "annotation_status.tsv", sep="\t")
            self.assertEqual(audit.status.tolist(), ["RESOLVED", "AMBIGUOUS"])
            self.assertEqual({row["gene"] for row in json.loads(audit.candidate_evidence.iloc[1])},
                             {"group_1", "other"})
            with patch("ko_variation.annotation_inputs.subprocess.run", side_effect=AssertionError("must reuse annotation")):
                self.assertEqual(map_main(args + ["--resume"]), 0)


if __name__ == "__main__":
    unittest.main()
