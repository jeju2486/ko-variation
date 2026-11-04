"""Focused checks for annotation geometry, ambiguity and exported evidence."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from ko_variation.locus_annotation import create_geometry, qualify_reference, resolve_hits, annotate_work
from ko_variation.network import build_gene_network


class LocusAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.db = create_geometry(self.work / "gene_geometry.sqlite")
        self.addCleanup(self.db.close)
        self.db.execute("INSERT INTO contigs VALUES ('ref0', 'sample0', 'contig_1', 5000)")
        self.db.executemany("INSERT INTO cds VALUES (?, ?, ?, ?, ?, ?)", [
            ('ref0', 1000, 1200, '+', 'gyr_cluster', 'gyrA'),
            ('ref0', 3000, 3200, '-', 'tet_cluster', 'tetK')])
        self.metadata = {'gyr_cluster': {'label': 'gyrA'}, 'tet_cluster': {'label': 'tetK'}}

    def resolve(self, start, end, window=500):
        return resolve_hits(f"ref0:{start}-{end};;;", self.db, self.metadata, window)

    def test_overlap_boundaries_and_strand(self):
        status, rows, gene = self.resolve(999, 1000)
        self.assertEqual((status, gene, rows[0]['annotation_class'], rows[0]['distance_bp']),
                         ('RESOLVED', 'gyr_cluster', 'coding', 0))
        self.assertEqual(self.resolve(400, 500)[1][0]['distance_bp'], 500)
        self.assertEqual(self.resolve(400, 499)[0], 'INTERGENIC_OUTSIDE_WINDOW')
        self.assertEqual(self.resolve(400, 500, 0)[0], 'INTERGENIC_OUTSIDE_WINDOW')
        self.assertEqual(self.resolve(1201, 1210)[1][0]['annotation_class'], 'downstream')
        self.assertEqual(self.resolve(2900, 2950)[1][0]['annotation_class'], 'downstream')
        self.assertEqual(self.resolve(3201, 3250)[1][0]['annotation_class'], 'upstream')

    def test_ambiguity_and_unresolved_cds(self):
        self.db.execute("INSERT INTO cds VALUES ('ref0', 1400, 1500, '+', 'tet_cluster', 'copy')")
        self.assertEqual(self.resolve(1250, 1260)[0], 'AMBIGUOUS')
        self.db.execute("INSERT INTO cds VALUES ('ref0', 999, 1002, '+', '__UNRESOLVED_CDS', 'unresolved')")
        self.assertEqual(self.resolve(999, 1000)[0], 'AMBIGUOUS')
        raw = 'ref0:1100-1150;;;,ref0:3100-3150;;;'
        self.assertEqual(resolve_hits(raw, self.db, self.metadata, 500)[0], 'AMBIGUOUS')

    def test_qualified_references_and_aliases(self):
        fasta, gff = self.work/'raw.fna', self.work/'raw.gff'
        fasta.write_text('>contig_1\n' + 'A'*100 + '\n')
        gff.write_text('contig_1\tBakta\tCDS\t10\t20\t.\t-\t0\tID=cds1;gene=gyrA\n')
        metadata = {'cluster': {'label': 'cluster'}}
        for index in (1, 2):
            fa, mapped, count = qualify_reference(fasta, gff, self.work/str(index), f'sample{index}', index,
                                                 {'cds1': {'cluster'}}, metadata, self.db)
            self.assertEqual(count, 1)
            self.assertTrue(fa.read_text().startswith(f'>kvref{index}_ctg0\n'))
            self.assertTrue(mapped.read_text().splitlines()[1].startswith(f'kvref{index}_ctg0\t'))
        self.assertEqual(metadata['cluster']['label'], 'gyrA')

    def test_audit_mapping_and_network_evidence(self):
        self.db.commit()
        (self.work/'selected_loci.json').write_text(json.dumps([0, 1, 2]))
        (self.work/'gene_metadata.json').write_text(json.dumps(self.metadata))
        (self.work/'annotation_preparation.json').write_text('{}')
        (self.work/'pyseer_hits.tsv').write_text('AAA\t0\tref0:950-999;;;\nAAA\t1\tref0:3100-3150;;;\n')
        output = self.work/'mapping.tsv'
        summary = annotate_work(self.work, output)
        self.assertEqual(summary['annotation_status_counts'], {'RESOLVED': 2, 'UNMAPPED': 1})
        with output.open() as handle:
            rows = list(csv.DictReader(handle, delimiter='\t'))
        self.assertEqual(rows[0]['annotation_class'], 'upstream')
        self.assertEqual(rows[0]['annotation_distance_max_bp'], '1')
        signals = pd.DataFrame([dict(u=0, v=1, p_primary=1e-9, adjusted_beta=1.,
                                     adjusted_odds_ratio=2.718, effect_status='OK')])
        nodes, edges = build_gene_network(signals, pd.read_csv(output, sep='\t'))
        self.assertEqual(edges.iloc[0].n_nearby_locus_pairs, 1)
        self.assertEqual(edges.iloc[0].representative_u_annotation_class, 'upstream')
        self.assertEqual(nodes.set_index('gene').loc['gyr_cluster', 'n_nearby_loci'], 1)
        self.assertEqual(nodes.set_index('gene').loc['tet_cluster', 'n_coding_loci'], 1)


if __name__ == '__main__':
    unittest.main()
