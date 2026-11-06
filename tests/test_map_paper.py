"""Saved-effect aggregation, function assignment and publication plot contracts."""
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from benchmarks.functional_map import main as map_main,function_inputs
from benchmarks.paper_plots import main as paper_main
from ko_variation.network import discovery_network_input,build_gene_network


class MapPaperTests(unittest.TestCase):
    def test_empty_network_and_eggnog_incomplete_run_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bakta = root/'bakta/s1'
            bakta.mkdir(parents=True)
            (bakta/'s1.faa').write_text('>P1\nMKKL\n>P2\nMGGL\n')
            panaroo = root/'genes.csv'
            panaroo.write_text('Gene,Non-unique Gene name,s1\nA,geneA,P1\nB,geneB,P2\n')
            executable = root/'emapper.py'
            executable.write_text('fixture executable')
            database = root/'egg_data'
            database.mkdir()
            (database/'fixture.db').write_text('database inventory fixture')
            args = SimpleNamespace(eggnog=None,allow_unassigned=False,run_eggnog=True,panaroo=panaroo,
                                   bakta=bakta.parent,eggnog_data_dir=database,emapper=str(executable),
                                   out=root,threads=2)
            calls = []

            def fake(command, **kwargs):
                if '--version' in command:
                    return subprocess.CompletedProcess(command,0,stdout='emapper fixture version',stderr='')
                if '--help' in command:
                    return subprocess.CompletedProcess(command,0,stdout='--resume',stderr='')
                calls.append(command)
                if len(calls)==1:
                    raise subprocess.CalledProcessError(1,command)
                out = Path(command[command.index('--output_dir')+1])
                (out/'network.emapper.annotations').write_text('#query\tCOG_category\tGOs\nA\tM\t-\nB\tJ\t-\n')
                return subprocess.CompletedProcess(command,0)

            nodes = pd.DataFrame({'gene':['A','B']})
            edges = pd.DataFrame({'gene_a':['A'],'gene_b':['B']})
            with patch('benchmarks.functional_map.shutil.which',return_value=str(executable)), \
                 patch('benchmarks.functional_map.subprocess.run',side_effect=fake):
                with self.assertRaises(subprocess.CalledProcessError):
                    function_inputs(args,nodes,edges)
                saved = function_inputs(args,nodes,edges)
                self.assertIn('--resume',calls[-1])
                self.assertEqual(function_inputs(args,nodes,edges),saved)
                self.assertEqual(len(calls),2)
            self.assertIsNone(function_inputs(args,nodes.iloc[:0],edges.iloc[:0]))

    def test_streamed_map_preserves_representatives_and_paper_dimer_filter_is_display_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'discovery'
            source.mkdir()
            rows = []
            for u,v,distance,state,beta,status,p in [
                (0,1,20000,'DISTAL',.2,'OK',1e-8),
                (2,3,21000,'DISTAL',-.3,'OK',1e-10),
                (1,4,25000,'DISTAL',.6,'OK',1e-9),
                (4,5,27000,'DISTAL',None,'UNSTABLE_EFFECT',1e-8),
                (6,7,30000,'DISTAL',1.,'OK',1e-11),
                (0,6,10000,'LOCAL',None,'SKIPPED_LOCAL',1e-8),
                (1,7,None,'UNKNOWN',None,'SKIPPED_UNKNOWN',1e-8),
                (2,6,9000,'DISTAL',.5,'OK',1e-8),
                (0,8,21000,'DISTAL',.5,'OK',1e-9)]:
                rows.append(dict(u=u,v=v,predictor_pattern=u,response_pattern=v,
                                 significance='SIGNIFICANT',p_lower=p/2,p_upper=p,p_exact=np.nan,
                                 min_distance=distance,distance_status=state,beta=beta,beta_status=status))
            frame = pd.DataFrame(rows)
            effects = source/'associations.tsv.gz'
            frame.to_csv(effects,sep='\t',index=False)
            summary = dict(alpha=.05,bonferroni_tests=1000,raw_p_threshold=.00005)
            (source/'run_summary.json').write_text(json.dumps(summary))
            annotation = root/'locus_to_gene.tsv'
            pd.DataFrame(dict(locus=range(8),gene=['group_1','group_2','group_1','group_2','group_3','group_4','group_5','group_6'],
                              label=['mprF','rpoC','mprF','rpoC','lysP','group_4','tet(M)','group_6'],
                              product=['','','','', '', 'hypothetical protein','','hypothetical protein'],
                              annotation_class=['coding']*8,annotation_distance_max_bp=[0]*8)).to_csv(annotation,sep='\t',index=False)
            (root/'annotation_status.tsv').write_text('locus\tstatus\n8\tAMBIGUOUS\n')
            egg = root/'network.emapper.annotations'
            egg.write_text('#query\tCOG_category\tGOs\n'+'\n'.join([
                'group_1\tM\tGO:0005886','group_2\tJ\t-', 'group_3\tE\t-',
                'group_4\tS\t-','group_5\tV\t-','group_6\tS\t-'])+'\n')
            out = root/'functional_map'
            args = ['--results',str(source),'--annotation',str(annotation),'--eggnog',str(egg),
                    '--out',str(out),'--chunk-rows','2','--dpi','72']
            before = effects.read_bytes()
            with patch('ko_variation.glmm.fit_logistic_mixed',side_effect=AssertionError('plotting refit')):
                self.assertEqual(map_main(args),0)
                database_stamp = (out/'map_work.sqlite').stat().st_mtime_ns
                self.assertEqual(map_main(args),0)
                self.assertEqual((out/'map_work.sqlite').stat().st_mtime_ns,database_stamp)
            self.assertEqual(effects.read_bytes(),before)
            audit = json.loads((out/'map_audit.json').read_text())
            self.assertEqual(audit['aggregation']['pairs_missing_resolved_annotation'],1)
            self.assertEqual(audit['annotation_status_counts'],{'AMBIGUOUS':1})
            nodes = pd.read_csv(out/'map.nodes.tsv',sep='\t')
            edges = pd.read_csv(out/'map.edges.tsv',sep='\t')
            self.assertEqual((len(nodes),len(edges)),(6,4))
            self.assertEqual(nodes.loc[nodes.gene.eq('group_1'),'display_label'].iloc[0],'mprF')
            self.assertEqual(nodes.loc[nodes.gene.eq('group_4'),'function_category'].iloc[0],'Unassigned')
            native = discovery_network_input(frame,summary)
            native = native.loc[native.min_distance.gt(10000)]
            expected_nodes,expected_edges = build_gene_network(native,pd.read_csv(annotation,sep='\t'),
                                                               significance_column='p_upper',missing_genes='drop')
            columns = ['gene_a','gene_b','n_locus_pairs','representative_u','representative_v','adjusted_direction']
            pd.testing.assert_frame_equal(edges[columns],expected_edges[columns],check_dtype=False)
            np.testing.assert_allclose(edges.representative_adjusted_beta,expected_edges.representative_adjusted_beta,equal_nan=True)
            support = pd.read_csv(out/'gene_pair_support.tsv.gz',sep='\t')
            self.assertEqual(len(support),5)
            paper = root/'paper'
            self.assertEqual(paper_main(['--layout',str(out),'--out',str(paper),'--dpi','72',
                                         '--focus','mprF','--focus','gyrA']),0)
            report = json.loads((paper/'paper_figures.json').read_text())
            self.assertEqual(report['two_gene_components'],1)
            self.assertEqual(report['focus']['mprF']['neighbourhoods']['1']['genes'],2)
            self.assertEqual(report['focus']['mprF']['neighbourhoods']['2']['genes'],3)
            self.assertEqual(report['focus']['gyrA']['status'],'NOT_IN_SELECTED_NETWORK')
            self.assertFalse(report['qq_plot_generated'])
            svg = (paper/'network_overview.svg').read_text()
            self.assertIn('<text',svg)
            self.assertIn('+1 two-gene components',svg)
            self.assertNotIn('tet(M)',svg)
            self.assertTrue((paper/'focus_mprf_N2.png').read_bytes().startswith(b'\x89PNG'))


if __name__ == '__main__':
    unittest.main()
