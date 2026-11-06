"""Build a functionally annotated map from completed v0.9 discovery exports.

No score tests or beta refits run here. Large pair tables are streamed into
SQLite; the in-memory object is the collapsed gene network, not all unitig pairs.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

import numpy as np
import pandas as pd

from ko_variation.annotation_inputs import main as annotate_main
from ko_variation.checkpoint import fingerprint_file
from ko_variation.discovery_store import atomic_json, write_gzip
from ko_variation.functional_annotation import prepare_proteins
from ko_variation.network import discovery_network_input, NODE_COLUMNS, EDGE_COLUMNS
from ko_variation.network_display import export_gene_network
from ko_variation.workflow_cache import stage_identity, cache_matches, save_cache, file_signature


def extract(results, summary, out, distal_bp, chunk_rows):
    target, audit = out/'selected_distal.tsv', out/'selection_counts.json'
    manifest = out/'selection.cache.json'
    identity = stage_identity('map_distal_selection', [results, summary, Path(__file__)],
                              dict(distal_bp=distal_bp))
    if cache_matches(manifest, identity, [target, audit]):
        return target, json.loads(audit.read_text())
    provenance = json.loads(summary.read_text())
    counts, states, statuses = Counter(), Counter(), Counter()
    temporary = target.with_suffix('.partial')
    first = True
    for chunk in pd.read_csv(results, sep='\t', chunksize=chunk_rows):
        counts['input_unitig_pairs'] += len(chunk)
        states.update(chunk.distance_status.fillna('UNKNOWN'))
        selected = discovery_network_input(chunk, provenance)
        counts['certified_distal_before_distance_filter'] += len(selected)
        selected = selected.loc[selected.physical_distance.gt(distal_bp)].copy()
        for name in ('u','v'):
            values = pd.to_numeric(selected[name], errors='raise')
            if values.isna().any() or (values < 0).any() or (values % 1 != 0).any():
                raise ValueError('Unitig IDs must be nonnegative integers')
            selected[name] = values.astype('int64')
        if (selected.u >= selected.v).any():
            raise ValueError('Expected canonical unitig pairs u < v')
        counts['selected_distal_unitig_pairs'] += len(selected)
        statuses.update(selected.effect_status.fillna('NOT_ESTIMATED'))
        selected.to_csv(temporary, sep='\t', index=False, mode='w' if first else 'a', header=first)
        first = False
        print(f'[map] read {counts["input_unitig_pairs"]:,}; retained '
              f'{counts["selected_distal_unitig_pairs"]:,} distal pairs', flush=True)
    if first:
        discovery_network_input(pd.read_csv(results, sep='\t', nrows=0), provenance).to_csv(
            temporary, sep='\t', index=False)
    os.replace(temporary, target)
    record = dict(counts, distance_status_counts=dict(states), beta_status_counts=dict(statuses),
                  distance_rule=f'min_distance > {distal_bp} bp and source DISTAL',
                  alpha=provenance['alpha'], bonferroni_tests=provenance['bonferroni_tests'],
                  raw_p_threshold=provenance['raw_p_threshold'], exact_p_values_invented=False)
    atomic_json(audit, record)
    save_cache(manifest, identity, [target, audit])
    return target, record


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def aggregate(selected, annotation, out, chunk_rows):
    """One gene edge per unordered pair; keep every contributing pair on disk."""
    node_file, edge_file = out/'genes.nodes.tsv', out/'genes.edges.tsv'
    support = out/'gene_pair_support.tsv.gz'
    audit = out/'aggregation_counts.json'
    outputs = [node_file, edge_file, support, audit]
    identity = stage_identity('map_gene_aggregation', [selected, annotation, Path(__file__)], {})
    manifest = out/'aggregation.cache.json'
    if cache_matches(manifest, identity, outputs):
        return pd.read_csv(node_file, sep='\t', keep_default_na=False), \
            pd.read_csv(edge_file, sep='\t'), json.loads(audit.read_text())
    database = out/'map_work.sqlite'
    db = sqlite3.connect(database)
    db.execute('PRAGMA temp_store=FILE')
    db.execute('PRAGMA cache_size=-32768')
    try:
        db.executescript('''DROP TABLE IF EXISTS pairs; DROP TABLE IF EXISTS annotation;
          CREATE TABLE pairs(u INTEGER,v INTEGER,p REAL,beta REAL,status TEXT,distance REAL,
            PRIMARY KEY(u,v)) WITHOUT ROWID;
          CREATE TABLE annotation(locus INTEGER PRIMARY KEY,gene TEXT,label TEXT,product TEXT,
            class TEXT,coding INTEGER,nearby INTEGER,annotation_distance REAL);''')
        for frame in pd.read_csv(annotation, sep='\t', dtype=str, keep_default_na=False, chunksize=chunk_rows):
            if not {'locus','gene'}.issubset(frame.columns):
                raise ValueError('Annotation requires locus and gene columns')
            rows = []
            for r in frame.to_dict('records'):
                locus = number(r['locus'])
                if locus is None or locus < 0 or locus != int(locus) or not r['gene'].strip():
                    raise ValueError('Invalid annotation locus/gene')
                classes = set(r.get('annotation_class','').split('; '))
                rows.append((int(locus), r['gene'], r.get('label',''), r.get('product',''),
                             r.get('annotation_class',''), int('coding' in classes),
                             int(bool(classes & {'upstream','downstream'})),
                             number(r.get('annotation_distance_max_bp'))))
            with db:
                db.executemany('INSERT INTO annotation VALUES (?,?,?,?,?,?,?,?)', rows)
        for frame in pd.read_csv(selected, sep='\t', chunksize=chunk_rows):
            with db:
                db.executemany('INSERT INTO pairs VALUES (?,?,?,?,?,?)',
                    ((int(r.u), int(r.v), float(r.p_upper), number(r.adjusted_beta), str(r.effect_status),
                      float(r.physical_distance)) for r in frame.itertuples()))
        db.executescript('''CREATE TEMP VIEW mapped AS
          SELECT p.*,a.gene AS ug,b.gene AS vg,min(a.gene,b.gene) AS ga,max(a.gene,b.gene) AS gb,
            a.class AS uc,b.class AS vc,a.annotation_distance AS ud,b.annotation_distance AS vd,
            (a.nearby OR b.nearby) AS nearby
          FROM pairs p JOIN annotation a ON a.locus=p.u JOIN annotation b ON b.locus=p.v
          WHERE a.gene!=b.gene;
          CREATE TEMP TABLE ranked AS SELECT *,row_number() OVER
            (PARTITION BY ga,gb ORDER BY CASE WHEN status='OK' THEN 0 ELSE 1 END,p,u,v) AS rank
          FROM mapped;
          CREATE INDEX ranked_edge ON ranked(ga,gb,rank);''')
        groups = db.execute('''SELECT ga,gb,count(*),min(p),sum(status='OK'),
          sum(status='OK' AND beta>0),sum(status='OK' AND beta<0),sum(status='OK' AND beta=0),sum(nearby)
          FROM mapped GROUP BY ga,gb''')
        edges = []
        for a,b,count,p,nfit,pos,neg,zero,nearby in groups:
            r = db.execute('SELECT u,v,beta,status,distance,p,uc,vc,ud,vd FROM ranked '
                           'WHERE ga=? AND gb=? AND rank=1', (a,b)).fetchone()
            u,v,beta,status,distance,rp,uc,vc,ud,vd = r
            fitted = status == 'OK'
            direction = 'mixed' if pos and neg else 'positive' if pos else 'negative' if neg else 'neutral' if zero else 'unknown'
            ratio = math.exp(beta) if fitted and beta < 709 else np.nan
            edges.append([a,b,count,p,'p_upper',u,v,ratio,direction,nfit,
                          'external:gene_pair_support.tsv.gz',beta if fitted else np.nan,np.nan,
                          rp,distance,uc,vc,ud,vd,nearby])
        edge_frame = pd.DataFrame(edges, columns=EDGE_COLUMNS)
        attributes = {}
        for locus,gene,label,product,classes,coding,nearby,distance in db.execute('''
          WITH used AS (SELECT u AS locus FROM mapped UNION SELECT v FROM mapped)
          SELECT a.* FROM annotation a JOIN used ON used.locus=a.locus ORDER BY a.gene,a.locus'''):
            value = attributes.setdefault(gene, dict(labels=set(),products=set(),classes=set(),
                                                    loci=0,coding=0,nearby=0,distance=None))
            value['labels'].update(x.strip() for x in label.split(';') if x.strip())
            if product: value['products'].add(product)
            value['classes'].update(x for x in classes.split('; ') if x)
            value['loci'] += 1
            value['coding'] += coding
            value['nearby'] += nearby
            if distance is not None:
                value['distance'] = distance if value['distance'] is None else max(value['distance'],distance)
        neighbours, supports = defaultdict(set), Counter()
        for r in edge_frame.itertuples():
            neighbours[r.gene_a].add(r.gene_b)
            neighbours[r.gene_b].add(r.gene_a)
            supports[r.gene_a] += r.n_locus_pairs
            supports[r.gene_b] += r.n_locus_pairs
        nodes = []
        for gene,v in sorted(attributes.items()):
            nodes.append([gene,'; '.join(sorted(v['labels'])),'; '.join(sorted(v['products'])),'',
                          v['loci'],len(neighbours[gene]),supports[gene],'; '.join(sorted(v['classes'])),
                          v['coding'],v['nearby'],v['distance'] if v['distance'] is not None else np.nan])
        node_frame = pd.DataFrame(nodes, columns=NODE_COLUMNS)
        write_gzip(support, ['gene_a','gene_b','u','v','u_gene','v_gene','p_upper','beta','beta_status','min_distance'],
                   db.execute('SELECT ga,gb,u,v,ug,vg,p,beta,status,distance FROM mapped ORDER BY ga,gb,u,v'))
        total = db.execute('SELECT count(*) FROM pairs').fetchone()[0]
        missing = db.execute('SELECT count(*) FROM pairs p LEFT JOIN annotation a ON a.locus=p.u '
                             'LEFT JOIN annotation b ON b.locus=p.v WHERE a.locus IS NULL OR b.locus IS NULL').fetchone()[0]
        self_pairs = db.execute('SELECT count(*) FROM pairs p JOIN annotation a ON a.locus=p.u '
                                'JOIN annotation b ON b.locus=p.v WHERE a.gene=b.gene').fetchone()[0]
        unique = db.execute('SELECT count(*) FROM (SELECT u FROM pairs UNION SELECT v FROM pairs)').fetchone()[0]
        record = dict(selected_unitig_pairs=total, selected_unique_unitigs=unique,
                      pairs_missing_resolved_annotation=missing, same_gene_pairs_excluded=self_pairs,
                      mapped_intergene_unitig_pairs=total-missing-self_pairs,
                      gene_pairs=len(edge_frame), genes=len(node_frame),
                      representative_rule='smallest p_upper successful beta; ties u,v; otherwise smallest p_upper',
                      gene_level_significance_test=False, support_table=str(support.resolve()))
        node_frame.to_csv(node_file, sep='\t', index=False)
        edge_frame.to_csv(edge_file, sep='\t', index=False)
        atomic_json(audit, record)
        save_cache(manifest, identity, outputs)
        return node_frame, edge_frame, record
    finally:
        db.close()


def function_inputs(args, nodes, edges):
    if not len(nodes):
        return None
    if args.eggnog:
        return args.eggnog
    if not len(nodes) or args.allow_unassigned:
        return None
    if not args.run_eggnog or not args.panaroo or not args.bakta or not args.eggnog_data_dir:
        raise ValueError('Supply --eggnog, or --run-eggnog with --panaroo, --bakta and --eggnog-data-dir')
    folder = args.out/'functional'
    folder.mkdir(exist_ok=True)
    targets = folder/'protein_targets.tsv'
    target_data = edges[['gene_a','gene_b']].rename(columns={'gene_a':'u_gene','gene_b':'v_gene'})
    text = target_data.to_csv(sep='\t', index=False, lineterminator='\n')
    if not targets.exists() or targets.read_text() != text:
        targets.write_text(text, newline='\n')
    proteins = folder/'network_proteins.faa'
    products = [proteins, folder/'protein_manifest.tsv', folder/'protein_preparation.json']
    inventory = [file_signature(p) for p in sorted(args.bakta.glob('*/*.faa'))]
    identity = stage_identity('map_representative_proteins', [targets,args.panaroo,
        Path(sys.modules[prepare_proteins.__module__].__file__)], dict(protein_inventory=inventory))
    manifest = folder/'proteins.cache.json'
    if not cache_matches(manifest, identity, products):
        prepare_proteins(targets, args.panaroo, args.bakta, folder)
        save_cache(manifest, identity, products)
    executable = shutil.which(args.emapper)
    if not executable or not args.eggnog_data_dir.is_dir():
        raise ValueError('eggNOG executable or database directory is missing; no database download is attempted')
    env = dict(os.environ)
    env['PATH'] = str(Path(executable).resolve().parent)+os.pathsep+env.get('PATH','')
    env.pop('PYTHONPATH', None)
    version = subprocess.run([executable,'--version'],capture_output=True,text=True,env=env,check=True)
    identity = dict(protein=fingerprint_file(proteins), executable=fingerprint_file(executable),
                    tool_version=(version.stdout+version.stderr).strip(),
                    database=[file_signature(p) for p in sorted(args.eggnog_data_dir.iterdir()) if p.is_file()])
    key = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    run = folder/('eggnog_'+key)
    run.mkdir(exist_ok=True)
    annotation = run/'network.emapper.annotations'
    completed = run/'completed.json'
    if completed.exists() and annotation.exists():
        saved = json.loads(completed.read_text())
        if saved['identity'] == identity and saved['annotation'] == fingerprint_file(annotation):
            print('[map] reusing completed eggNOG annotations', flush=True)
            return annotation
    command = [executable,'-i',str(proteins.resolve()),'--itype','proteins','-o','network',
               '--output_dir',str(run.resolve()),'--data_dir',str(args.eggnog_data_dir.resolve()),
               '--cpu',str(args.threads)]
    started = run/'invocation.json'
    if started.exists():
        help_result = subprocess.run([executable,'--help'],capture_output=True,text=True,env=env,check=True)
        if '--resume' not in help_result.stdout+help_result.stderr:
            raise ValueError(f'Incomplete eggNOG run at {run}; this executable has no --resume option')
        command.append('--resume')
    atomic_json(started, dict(identity=identity, command=command))
    print(f'[map] eggNOG log: {run}/emapper.log', flush=True)
    with (run/'emapper.log').open('a') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
    if not annotation.is_file():
        raise ValueError('eggNOG finished without the expected annotation table')
    atomic_json(completed, dict(identity=identity, annotation=fingerprint_file(annotation),command=command))
    return annotation


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True, help='Discovery results directory or associations.tsv.gz')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--annotation', type=Path, help='Existing resolved locus_to_gene.tsv')
    p.add_argument('--gene-catalog', type=Path)
    p.add_argument('--fasta', type=Path)
    p.add_argument('--unitigs', type=Path)
    p.add_argument('--panaroo', type=Path)
    p.add_argument('--bakta', type=Path)
    p.add_argument('--pyseer-python', default=sys.executable)
    p.add_argument('--nearby-bp', type=int, default=500)
    p.add_argument('--distal-bp', type=int, default=10000)
    p.add_argument('--chunk-rows', type=int, default=50000)
    p.add_argument('--eggnog', type=Path, help='Completed .emapper.annotations, query IDs=cluster IDs')
    p.add_argument('--run-eggnog', action='store_true')
    p.add_argument('--emapper', default='emapper.py')
    p.add_argument('--eggnog-data-dir', type=Path)
    p.add_argument('--allow-unassigned', action='store_true', help='Explicitly skip functional assignment')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dpi', type=int, default=300)
    args = p.parse_args(argv)
    if min(args.threads,args.chunk_rows,args.dpi) < 1 or min(args.nearby_bp,args.distal_bp) < 0:
        p.error('Invalid CPU/chunk/display/distance setting')
    source = args.results.resolve()
    if source.is_dir():
        if not (source/'associations.tsv.gz').exists(): source /= 'results'
        source /= 'associations.tsv.gz'
    summary = source.parent/'run_summary.json'
    args.out = args.out.resolve()
    if args.out == source.parent or args.out.is_relative_to(source.parent/'.kovar_work'):
        p.error('Choose a separate downstream output, outside the discovery work directory')
    args.out.mkdir(parents=True,exist_ok=True)
    try:
        selected, selection = extract(source, summary, args.out, args.distal_bp, args.chunk_rows)
        annotation = args.annotation
        if annotation is None:
            if args.fasta is None:
                record = json.loads((source.parent/'run_identity.json').read_text())
                args.fasta = Path(record['settings']['fasta'])
            if not all([args.unitigs,args.panaroo,args.bakta]):
                raise ValueError('New gene mapping requires --unitigs DNA, --panaroo and --bakta')
            directory = args.out/'annotation'
            result = annotate_main(['--selected',str(selected),'--fasta',str(args.fasta),
                '--unitigs',str(args.unitigs),'--panaroo',str(args.panaroo),'--bakta',str(args.bakta),
                '--out',str(directory),'--nearby-bp',str(args.nearby_bp),
                '--pyseer-python',args.pyseer_python,'--resume'])
            if result: return result
            annotation = directory/'locus_to_gene.tsv'
        if args.gene_catalog is None and (annotation.parent/'gene_catalog.tsv').is_file():
            args.gene_catalog = annotation.parent/'gene_catalog.tsv'
        nodes, edges, aggregation = aggregate(selected,annotation,args.out,args.chunk_rows)
        print(f'[map] {aggregation["mapped_intergene_unitig_pairs"]:,} mapped intergene pairs; '
              f'{aggregation["pairs_missing_resolved_annotation"]:,} pairs lack resolved annotation; '
              f'{aggregation["same_gene_pairs_excluded"]:,} same-gene pairs excluded',flush=True)
        eggnog = function_inputs(args,nodes,edges)
        paths = export_gene_network(nodes,edges,args.out/'map',seed=args.seed,dpi=args.dpi,
                                    title='',labels='none',gene_catalog=args.gene_catalog,eggnog=eggnog)
        status = annotation.parent/'annotation_status.tsv'
        annotation_status = Counter()
        if status.is_file():
            for frame in pd.read_csv(status,sep='\t',usecols=['status'],chunksize=args.chunk_rows):
                annotation_status.update(frame.status)
        audit = dict(selection=selection,aggregation=aggregation,
                     annotation_status_counts=dict(annotation_status),
                     source=fingerprint_file(source),summary=fingerprint_file(summary),
                     annotation=fingerprint_file(annotation),
                     functional_annotation=str(eggnog.resolve()) if eggnog else None,
                     refitted=False, outputs={k:str(v.resolve()) for k,v in paths.items()})
        atomic_json(args.out/'map_audit.json',audit)
        print(f'[map] {len(nodes):,} genes; {len(edges):,} gene edges; {args.out}/map.html',flush=True)
        return 0
    except (ValueError,OSError,ImportError,sqlite3.Error,subprocess.CalledProcessError) as error:
        sys.stderr.write(f'Map workflow error: {error}\n')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
