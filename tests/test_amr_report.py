"""Focused reporting checks; no scanner, annotation executable or model fitting."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from ko_variation.amr_report import build_report, phenotype_inventory, read_tsv


SPECIES = "Staphylococcus aureus"


def write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return path


class AMRReportTests(unittest.TestCase):
    def test_metadata_deduplication_censoring_and_species(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write(Path(tmp) / "metadata.tsv", [
                dict(id="a", species=SPECIES, tetracycline_mic="8", tetracycline_mic_sign="=",
                     comments=json.dumps({"tetracycline": {"mic": "16", "mic_sign": "="},
                                          "gentamicin": {"mic": "4", "mic_sign": ">"}})),
                dict(id="b", species=SPECIES, tetracycline_mic="<=2", tetracycline_mic_sign="", comments=""),
                dict(id="c", species=SPECIES, tetracycline_mic="4", tetracycline_mic_sign="", comments=""),
                dict(id="d", species=SPECIES, tetracycline_mic="inf", tetracycline_mic_sign="=", comments=""),
                dict(id="e", species="Staphylococcus argenteus", tetracycline_mic="128", tetracycline_mic_sign="=", comments=""),
            ])
            rows, cohort = phenotype_inventory(path, SPECIES)
            tet = next(r for r in rows if r["antimicrobial"] == "tetracycline")
            self.assertEqual((tet["n_positive_numeric_mic"], tet["n_exact"], tet["n_censored"],
                              tet["n_missing_sign"], tet["n_source_conflicts"], tet["n_invalid_value"]), (3, 1, 1, 1, 1, 1))
            self.assertEqual(cohort["n_excluded_other_species"], 1)
            gent = next(r for r in rows if r["antimicrobial"] == "gentamicin")
            self.assertEqual((gent["n_censored"], gent["record_sources"]), (1, "comment_json"))

    def test_names_multiple_clusters_homonyms_and_neighbour_unions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = write(root / "metadata.tsv", [dict(id="a", species=SPECIES,
                tetracycline_mic="4", tetracycline_mic_sign="=", oxacillin_mic="8", oxacillin_mic_sign="=")])
            catalogue = write(root / "genes.tsv", [
                dict(gene="group_1", label="tet(M)", product="ribosomal protection"),
                dict(gene="group_2", label="tetM; other", product="ribosomal protection"),
                dict(gene="tetM", label="hypothetical protein", product="unknown"),
                dict(gene="x", label="x", product=""), dict(gene="y", label="group_44", product=""),
                dict(gene="z", label="z", product=""),
                dict(gene="good_mec", label="mecA", product="PBP2a transpeptidase"),
                dict(gene="bad_mec", label="mecA", product="adaptor protein MecA"),
            ])
            network = root / "network.html"
            payload = dict(nodes=read_tsv(catalogue), edges=[
                dict(gene_a="group_1", gene_b="group_2"),
                dict(gene_a="group_1", gene_b="x"), dict(gene_a="group_2", gene_b="x"),
                dict(gene_a="x", gene_b="y"), dict(gene_a="y", gene_b="z"),
                dict(gene_a="bad_mec", gene_b="z")], meta={})
            network.write_text('<script id="data" type="application/json">' + json.dumps(payload) + '</script>')
            audit = write(root / "audit.tsv", [dict(locus="0", status="RESOLVED", gene="group_1",
                candidate_evidence=json.dumps([dict(gene="group_1", annotation_class="upstream")]))])
            out = root / "report"
            build_report(species="saureus", metadata=metadata, out=out, gene_catalog=catalogue,
                         network=network, annotation_status=audit)
            rows = read_tsv(str(out) + ".determinants.tsv")
            tet = next(r for r in rows if r["determinant"] == "tetM")
            self.assertEqual((tet["n_annotation_clusters"], tet["n_direct_neighbours"], tet["n_second_order_neighbours"]), ("2", "1", "1"))
            mec = next(r for r in rows if r["determinant"] == "mecA")
            self.assertEqual(mec["n_annotation_clusters"], "1")
            matches = read_tsv(str(out) + ".matches.tsv")
            self.assertTrue(any(r["cluster_id"] == "bad_mec" and r["match_status"] == "rejected_product" for r in matches))
            self.assertFalse(any(r["cluster_id"] == "tetM" for r in matches))
            nearby = next(r for r in matches if r["cluster_id"] == "group_1")
            self.assertEqual(nearby["n_nearby_loci"], "1")
            second = read_tsv(str(out) + ".neighbours.tsv")
            self.assertEqual({r["annotation_label"] for r in second if r["determinant"] == "tetM" and r["order"] == "2"}, {"unannotated"})

    def test_missing_inputs_and_mutations_are_not_false_absence_or_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = write(root / "metadata.tsv", [dict(id="a", species=SPECIES,
                ciprofloxacin_mic="4", ciprofloxacin_mic_sign="=", tetracycline_mic="8",
                tetracycline_mic_sign="=", unknown_drug_SIR="R")])
            out = root / "report"
            build_report(species="saureus", metadata=metadata, out=out)
            rows = read_tsv(str(out) + ".determinants.tsv")
            gyr = next(r for r in rows if r["determinant"] == "gyrA")
            self.assertEqual((gyr["map_status"], gyr["screening_status"]), ("not_checked", "MUTATION_ANALYSIS_DEFERRED"))
            tet = next(r for r in rows if r["determinant"] == "tetM")
            self.assertEqual(tet["annotation_status"], "not_checked")
            self.assertEqual(tet["n_direct_neighbours"], "not_checked")
            unknown = next(r for r in rows if r["antimicrobial"] == "unknown_drug")
            self.assertEqual(unknown["screening_status"], "CATALOGUE_NOT_COVERED")
            self.assertEqual(unknown["n_positive_numeric_mic"], "0")

    def test_explicit_cohort_mapping_and_conflicting_signs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = write(root / "metadata.tsv", [
                dict(id="a", species=SPECIES, tetracycline_mic="<4", tetracycline_mic_sign=">"),
                dict(id="b", species=SPECIES, tetracycline_mic="8", tetracycline_mic_sign="="),
                dict(id="c", species=SPECIES, tetracycline_mic="16", tetracycline_mic_sign="="),
            ])
            mapping = write(root / "map.tsv", [dict(metadata_id="a", sample="isolate_a"), dict(metadata_id="b", sample="isolate_b")])
            inclusion = write(root / "inclusion.tsv", [dict(sample="isolate_a", status="included"),
                                                     dict(sample="isolate_b", status="excluded_missing_tree")])
            rows, cohort = phenotype_inventory(metadata, SPECIES, sample_map=mapping, sample_inclusion=inclusion)
            self.assertEqual((cohort["n_report_records"], cohort["n_metadata_without_sample_mapping"]), (1, 1))
            self.assertEqual((rows[0]["n_conflicting_sign"], rows[0]["n_positive_numeric_mic"]), (1, 0))
            with self.assertRaisesRegex(ValueError, "both"):
                phenotype_inventory(metadata, SPECIES, sample_map=mapping)


if __name__ == "__main__":
    unittest.main()
