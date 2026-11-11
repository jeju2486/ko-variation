#!/usr/bin/env python3
"""Read-only S. aureus target audit; all new state belongs to --out.

Find full-length exact unitig matches at annotated target CDSs or within a
bounded flank, including unitigs omitted from the discovery/map. Multiple
target/cluster matches are evidence, never silently resolved to one gene.
Direct tests use the production score statistic (not an LRT or MIC model).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import itertools
import json
import math
from pathlib import Path
import re
import sqlite3
import sys
import time
from urllib.parse import unquote

from ko_variation.memory_budget import guard_blas_environment
guard_blas_environment()
import numpy as np
import scipy
from Bio import SeqIO
from threadpoolctl import threadpool_limits

from ko_variation.discovery_execution import ResponsePool, bounded_map
from ko_variation.discovery_store import atomic_json, digest, open_patterns
from ko_variation.glmm import KinshipEigensystem, fit_logistic_mixed, fit_null_glmm, score_predictor_block
from ko_variation import discovery_nulls
from ko_variation.null_identity import null_source_hash

TARGETS = ('mecA', 'graR', 'sen', 'tetM', 'tetK')
COMPARISONS = (('mecA', 'graR'), ('mecA', 'sen'))
_WORKER = {}
PRE_REUSE_POLICY_HASHES = {'32713cfea041024439ba915315416df6ba9369f7942c63643ee072f752756391',
                         'da44d07dda016adf80e83f66fbcf8a3760d1c1d7cb8ed0e0655bde78027e85e3',
                         'c9d289ff3db52df7f743b1a652faa05937f070e2441df12022300aec4d226c64',
                         '07c5b79effdc787e92c253adbf90679be2ab92fcc734a4012c2800a721852383'}


def stamp(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def read_only(path):
    return sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True, timeout=60)


def log(out, message):
    text = '[AMR diagnosis] '+message
    print(text, flush=True)
    with (out/'progress.log').open('a', encoding='utf-8') as stream:
        stream.write(time.strftime('%Y-%m-%d %H:%M:%S ')+text+'\n')


def get(db, key, default=None):
    row = db.execute('SELECT value FROM meta WHERE key=?', (key,)).fetchone()
    return json.loads(row[0]) if row else default


def put(db, key, value):
    db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)', (key, json.dumps(value)))


def table(path, rows, fields):
    opener = gzip.open if str(path).endswith('.gz') else open
    temporary = Path(str(path)+'.partial')
    with opener(temporary, 'wt', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fields, delimiter='\t', extrasaction='ignore')
        writer.writeheader()
        writer.writerows({k: '' if r.get(k) is None else r.get(k, '') for k in fields} for r in rows)
    temporary.replace(path)


def identify_all(names, product):
    aliases = {re.sub('[^a-z0-9]', '', x.lower()) for x in re.split(r';|~~~|\|', names)}
    product = product.lower()
    # The unrelated proteolysis adaptor is also named MecA.
    found = {target for target in TARGETS[1:] if target.lower() in aliases}
    if 'meca' in aliases and 'adaptor' not in product and (
            'pbp2a' in product or 'transpeptidase' in product):
        found.add('mecA')
    return found


def identify(names, product):
    found = identify_all(names, product)
    return next((target for target in TARGETS if target in found), None)


def genome(gff, fasta=None):
    """Read one reference at a time; preserve original CDS IDs/coordinates."""
    features = []
    with Path(gff).open(encoding='utf-8') as stream:
        embedded = False
        for line in stream:
            if line.strip() == '##FASTA':
                embedded = True
                break
            if line.startswith('#') or not line.strip():
                continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) == 9 and fields[2] == 'CDS':
                attrs = {unquote(k): unquote(v) for k, v in
                         (x.split('=', 1) for x in fields[8].split(';') if '=' in x)}
                features.append((fields[0], int(fields[3])-1, int(fields[4]), fields[6], attrs))
        if fasta is None and not embedded:
            raise ValueError(f'No embedded FASTA in {gff}; provide original/recovered --bakta inputs')
        dna = {r.id: str(r.seq).upper() for r in SeqIO.parse(fasta if fasta else stream, 'fasta')}
    return features, dna


def references(args, info):
    refs, missing = {}, []
    for sample in info['samples']:
        gffs = [args.gff_dir/(sample+s) for s in ('.gff', '.gff3')] if args.gff_dir else []
        found = [p for p in gffs if p.is_file()]
        if len(found) > 1:
            raise ValueError(f'Multiple GFFs for {sample}')
        if found:
            refs[sample] = (found[0], None)
        elif args.bakta and (args.bakta/sample/(sample+'.gff3')).is_file():
            refs[sample] = (args.bakta/sample/(sample+'.gff3'), args.bakta/sample/(sample+'.fna'))
        else:
            missing.append(sample)
    if not refs:
        raise ValueError('No annotation reference filenames match the frozen discovery cohort')
    return refs, missing


def prepare_regions(args, db, info, refs):
    if get(db, 'regions_complete'):
        return
    members = defaultdict(lambda: defaultdict(list))
    with args.panaroo.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'Gene', 'Annotation'}.issubset(reader.fieldnames or []):
            raise ValueError('Use Panaroo gene_presence_absence_roary.csv with Gene and Annotation columns')
        for row in reader:
            targets = identify_all(row['Gene']+';'+row.get('Non-unique Gene name', ''), row['Annotation'])
            if targets:
                for sample in refs:
                    for feature in filter(None, re.split(r';|\t', row.get(sample, ''))):
                        members[sample][feature].extend((target, row['Gene']) for target in targets)
    done = set(get(db, 'reference_samples_done', []))
    for index, (sample, (gff, fasta)) in enumerate(refs.items(), 1):
        if sample in done:
            continue
        features, dna = genome(gff, fasta)
        records = []
        for contig, start, end, strand, attrs in features:
            feature = attrs.get('ID', '')
            labels = list(members[sample].get(feature, []))
            for suffix in ('_len', '_pseudo', '_len_pseudo'):
                labels += members[sample].get(feature+suffix, [])
            targets = identify_all(attrs.get('gene', ''), attrs.get('product', ''))
            if targets and not labels:
                labels = [(target, 'GFF:'+feature) for target in targets]
            if not labels:
                continue
            if contig not in dna or not 0 <= start < end <= len(dna[contig]):
                raise ValueError(f'Invalid target CDS coordinates: {sample}/{feature}')
            left, right = max(0, start-args.nearby_bp), min(len(dna[contig]), end+args.nearby_bp)
            for target, cluster in set(labels):
                records.append((sample, contig, start, end, strand, target, cluster, left, dna[contig][left:right]))
        with db:
            db.executemany('INSERT INTO regions(sample,contig,gs,ge,strand,target,cluster,left_pos,dna) '
                           'VALUES (?,?,?,?,?,?,?,?,?)', records)
            done.add(sample)
            put(db, 'reference_samples_done', sorted(done))
        if index % 25 == 0 or index == len(refs):
            log(args.out, f'target reference windows: {index:,}/{len(refs):,} samples')
    with db:
        put(db, 'regions_complete', True)


def scan_unitigs(args, db):
    if get(db, 'scan_complete'):
        return
    groups = defaultdict(list)
    for row in db.execute('SELECT sample,contig,gs,ge,strand,target,cluster,left_pos,dna FROM regions'):
        groups[(row[5], row[6], row[8], row[2]-row[7], row[3]-row[7])].append(row[:8])
    grouped = list(groups.items())
    seeds = defaultdict(list)
    for rid, ((target, cluster, dna, gs, ge), rows) in enumerate(grouped):
        # Sparse reference seeds + every query position guarantee an anchor
        # for any >=2k-1 bp exact overlap within the reference window.
        for offset in range(0, len(dna)-args.seed_k+1, args.seed_k):
            seed = dna[offset:offset+args.seed_k]
            if not set(seed)-set('ACGT'):
                seeds[seed].append((rid, offset))
    complement = str.maketrans('ACGT', 'TGCA')
    count, offset = get(db, 'scan_count', 0), get(db, 'scan_offset', 0)
    log(args.out, f'scanning full DNA unitig pool from byte {offset:,}; {len(grouped):,} distinct target windows')
    with args.unitigs.open('rb') as stream:
        stream.seek(offset)
        while line := stream.readline():
            fields = line.split()
            if len(fields) < 2 or fields[0].startswith(b'#'):
                continue
            locus, sequence = int(fields[0]), fields[1].decode('ascii').upper()
            if set(sequence)-set('ACGT'):
                raise ValueError(f'Invalid unitig DNA at ID {locus}')
            anchors = set()
            for reverse, query in enumerate((sequence, sequence.translate(complement)[::-1])):
                candidates = set()
                for q in range(len(query)-args.seed_k+1):
                    for rid, r in seeds.get(query[q:q+args.seed_k], ()):
                        candidates.add((rid, r-q))
                for rid, start in candidates:
                    (target, cluster, dna, gs, ge), rows = grouped[rid]
                    lo, hi = max(0, start), min(len(dna), start+len(query))
                    if hi-lo < 2*args.seed_k-1 or dna[lo:hi] != query[lo-start:hi-start]:
                        continue
                    for sample, contig, a, b, strand, _, _, left in rows:
                        anchors.add((locus, target, cluster, sample, contig, left+start, a, b, strand, reverse))
            if anchors:
                db.execute('INSERT OR REPLACE INTO sequences VALUES (?,?)', (locus, sequence))
                db.executemany('INSERT OR IGNORE INTO anchors VALUES (?,?,?,?,?,?,?,?,?,?,NULL)', anchors)
            count += 1
            if count % 25000 == 0:
                put(db, 'scan_offset', stream.tell()); put(db, 'scan_count', count)
                db.commit()
                log(args.out, f'unitigs scanned: {count:,}; candidate unitigs: '
                    f'{db.execute("SELECT count(*) FROM sequences").fetchone()[0]:,}')
        with db:
            put(db, 'scan_offset', stream.tell()); put(db, 'scan_count', count)
            put(db, 'scan_complete', True)


def validate_anchors(args, db, refs):
    """Confirm the ENTIRE unitig against its original assembly, including long unitigs."""
    complement = str.maketrans('ACGT', 'TGCA')
    for index, (sample, paths) in enumerate(refs.items(), 1):
        if not db.execute('SELECT 1 FROM anchors WHERE sample=? AND valid IS NULL LIMIT 1', (sample,)).fetchone():
            continue
        _, dna = genome(*paths)
        last = 0
        while rows := db.execute('SELECT a.rowid,a.contig,a.start_pos,a.reverse,s.dna FROM anchors a '
                'JOIN sequences s USING(locus) WHERE sample=? AND valid IS NULL AND a.rowid>? '
                'ORDER BY a.rowid LIMIT 2048', (sample,last)).fetchall():
            checked = []
            for rowid, contig, start, reverse, sequence in rows:
                if reverse: sequence = sequence.translate(complement)[::-1]
                reference = dna.get(contig, '')
                valid = start >= 0 and start+len(sequence) <= len(reference) and reference[start:start+len(sequence)] == sequence
                checked.append((int(valid), rowid))
            db.executemany('UPDATE anchors SET valid=? WHERE rowid=?', checked)
            last = rows[-1][0]
        db.commit()
        if index % 25 == 0 or index == len(refs):
            log(args.out, f'full-length exact-match validation: {index:,}/{len(refs):,} samples')


def locus_report(args, db, source, info):
    hits = defaultdict(lambda: defaultdict(lambda: dict(clusters=set(), samples=set(), coding=False, nearby=False)))
    for uid, target, cluster, sample, start, gs, ge, strand, length in db.execute(
            'SELECT a.locus,target,cluster,sample,start_pos,gs,ge,strand,length(s.dna) '
            'FROM anchors a JOIN sequences s USING(locus) WHERE valid=1'):
        end = start+length
        coding = start < ge and end > gs
        distance = 0 if coding else gs-end+1 if end <= gs else start-ge+1
        if distance > args.nearby_bp:
            continue
        record = hits[uid][target]
        record['clusters'].add(cluster); record['samples'].add(sample)
        record['coding'] |= coding; record['nearby'] |= not coding
    old_status = {}
    status_file = args.map_dir/'annotation/annotation_status.tsv'
    if status_file.is_file():
        with status_file.open(encoding='utf-8-sig') as stream:
            for row in csv.DictReader(stream, delimiter='\t'):
                uid = int(row['locus'])
                if uid in hits: old_status[uid] = row
    map_genes = set()
    nodes = args.map_dir/'map.nodes.tsv'
    if nodes.is_file():
        with nodes.open(encoding='utf-8-sig') as stream:
            map_genes = {row['gene'] for row in csv.DictReader(stream, delimiter='\t')}
    rows, coding = [], defaultdict(set)
    packed = open_patterns(args.work, info)
    sample_index = {sample:i for i,sample in enumerate(info['samples'])}
    for uid, targets in sorted(hits.items()):
        original = source.execute('SELECT pattern,present FROM unitigs WHERE id=?', (uid,)).fetchone()
        pid, present = original if original else (None,None)
        locus_status = ('NOT_IN_FROZEN_LOCUS_POOL' if original is None else
                        'MONOMORPHIC' if present in (0,info['n']) else
                        'MAF_FILTERED' if pid is None else 'RETAINED')
        binary = np.unpackbits(packed[pid], bitorder='little')[:info['n']] if pid is not None else None
        for target, evidence in sorted(targets.items()):
            disagreements = sum(not binary[sample_index[s]] for s in evidence['samples']) if binary is not None else None
            if pid is not None and not disagreements and (evidence['coding'] or args.include_nearby): coding[target].add(pid)
            status = old_status.get(uid, {})
            rows.append(dict(target=target, unitig=uid, pattern=pid, n_present=present, n_samples=info['n'],
                maf=min(present, info['n']-present)/info['n'] if present is not None else None,
                original_locus_status=locus_status, passes_original_locus_filter=int(pid is not None),
                annotation_class='coding' if evidence['coding'] else 'nearby',
                clusters=';'.join(sorted(evidence['clusters'])), n_matching_references=len(evidence['samples']),
                n_matching_references_binary_absent=disagreements,
                n_target_names_for_unitig=len(targets), match='FULL_LENGTH_EXACT_IN_TARGET_REFERENCE',
                uniqueness='NOT_A_GENOME_WIDE_UNIQUENESS_TEST',
                original_annotation_status=status.get('status', 'NOT_IN_SELECTED_ANNOTATION'),
                original_annotation_gene=status.get('gene', ''),
                any_cluster_in_map=int(bool(evidence['clusters'] & map_genes))))
    table(args.out/'target_unitigs.tsv', rows, ['target','unitig','pattern','n_present','n_samples','maf','original_locus_status',
        'passes_original_locus_filter','annotation_class','clusters','n_matching_references','n_matching_references_binary_absent',
        'n_target_names_for_unitig','match','uniqueness','original_annotation_status',
        'original_annotation_gene','any_cluster_in_map'])
    targets = []
    for target in TARGETS:
        clusters, carriers = set(), set()
        for cluster, sample in db.execute('SELECT DISTINCT cluster,sample FROM regions WHERE target=?', (target,)):
            clusters.add(cluster); carriers.add(sample)
        subset = [r for r in rows if r['target'] == target]
        targets.append(dict(target=target, source_clusters=';'.join(sorted(clusters)), n_source_carriers=len(carriers),
            n_coding_unitigs=sum(r['annotation_class']=='coding' for r in subset),
            n_nearby_unitigs=sum(r['annotation_class']=='nearby' for r in subset),
            n_retained_coding_patterns=len({r['pattern'] for r in subset if r['pattern'] is not None and r['annotation_class']=='coding'}),
            n_current_ambiguous=sum(r['original_annotation_status']=='AMBIGUOUS' for r in subset)))
    table(args.out/'targets.tsv', targets, ['target','source_clusters','n_source_carriers','n_coding_unitigs',
          'n_nearby_unitigs','n_retained_coding_patterns','n_current_ambiguous'])
    return rows, coding


def original_reuse(identity, policy):
    versions = {key:dict(original=identity.get(key), current=value) for key,value in
                (('numpy',np.__version__),('scipy',scipy.__version__))}
    same_model = identity.get('null_model_source_sha256') == null_source_hash()
    same_versions = all(v['original']==v['current'] for v in versions.values())
    if policy=='require' and not same_model:
        raise ValueError('Original-null reuse refused: null-fitting code differs; library-only override cannot bypass this')
    return dict(policy=policy, enabled=same_model and (same_versions or policy=='require'),
                fitting_code_matches=same_model, library_versions_match=same_versions, versions=versions)


def initialize_fits(work, info, out, reuse):
    threadpool_limits(limits=1)
    _WORKER.clear()
    _WORKER.update(work=Path(work), info=info, out=Path(out), source=read_only(Path(work)/'work.sqlite'),
        packed=open_patterns(work, info), K=np.load(Path(work)/'K.npy', mmap_mode='r'),
        eigen=KinshipEigensystem(np.load(Path(work)/'K_values.npy', mmap_mode='r'),
            np.load(Path(work)/'K_vectors.npy', mmap_mode='r'), info['shared_covariance']),
        reuse=reuse, response=None)


def vector(pid):
    return np.unpackbits(_WORKER['packed'][pid], bitorder='little')[:_WORKER['info']['n']].astype(float)


def response_null(response):
    s = _WORKER
    if s['response'] == response: return s['cached']
    folder = s['out']/'null_models'/str(response)
    local = folder/f'null_{response:09d}.json'
    cache, record, origin = None, None, 'fresh_diagnostic_fit'
    verify = lambda p, sha: None if digest(p)==sha else (_ for _ in ()).throw(ValueError(f'Corrupt null cache: {p}'))
    local_record = json.loads(local.read_text()) if local.is_file() else None
    required = s['reuse']['policy']=='require'
    save_local = False
    if local_record and (not required or local_record.get('origin')=='original_null_cache'):
        record = local_record; origin = record.get('origin', 'diagnostic_checkpoint')
        if record['status']=='OK': cache = discovery_nulls.load(record, local, s['eigen'], verify)
    elif s['reuse']['enabled']:
        row = s['source'].execute('SELECT canonical FROM null_aliases WHERE pid=?', (response,)).fetchone()
        canonical = row[0] if row else response
        candidates = [s['work']/'null_models'/f'{canonical//1000:06d}'/f'null_{canonical:09d}.json']
        for sql in ('SELECT source FROM null_inventory WHERE pid=?', 'SELECT path FROM external_nulls WHERE pid=?'):
            candidates.extend(Path(r[0]) for r in s['source'].execute(sql, (canonical,)) if r[0])
        old = next((p for p in candidates if p.is_file()), None)
        if old:
            record = json.loads(old.read_text()); origin = 'original_null_cache'
            if record.get('format')!=2:
                raise ValueError(f'Unsupported original null cache format for response {response}: {old}')
            if record['status']=='OK':
                cache = discovery_nulls.load(record, old, s['eigen'], verify)
                if not np.array_equal(cache['y'], vector(canonical)):
                    raise ValueError('Cached null response differs from frozen pattern')
                if response != canonical: cache = dict(cache, y=vector(response), mu=1-cache['mu'])
            save_local = True
        elif required:
            raise ValueError(f'Original null cache not found for response {response} (canonical {canonical}); '
                             f'no refit performed. Searched: {[str(p) for p in candidates]}')
    if record is None:
        reason = ('original cache not found' if s['reuse']['enabled'] else
                  'reuse disabled: null-fitting code differs' if not s['reuse']['fitting_code_matches'] else
                  'reuse disabled: NumPy/SciPy versions differ')
        print(f'[AMR diagnosis worker] fitting diagnostic null {response}; {reason}', flush=True)
        fit, cache = fit_null_glmm(response, vector(response), s['K'], kinship_eigensystem=s['eigen'])
        record = dict(status=fit.status, iterations=fit.iterations)
        if fit.status!='OK': cache = None
        save_local = True
    if cache is not None:
        n = s['info']['n']
        expected_shapes = dict(y=(n,),mu=(n,),weights=(n,),C=(n,1),Sinv_C=(n,1),M_inv=(1,1))
        if any(cache[k].shape!=shape or not np.isfinite(cache[k]).all() for k,shape in expected_shapes.items()):
            raise ValueError(f'Invalid cached null arrays for response {response}')
        if not np.array_equal(cache['y'],vector(response)) or np.any(cache['weights']<=0) or \
                not np.isfinite(cache['tau']) or cache['tau']<0 or np.any(cache['mu']<0) or np.any(cache['mu']>1):
            raise ValueError(f'Cached null does not match the response/valid probabilities for pattern {response}')
        discovery_nulls.complete(cache, s['K'], s['eigen'])
    if save_local:
        discovery_nulls.save(folder, response, cache, dict(record, origin=origin), s['K'], s['eigen'],
                             compact=False, rank=s['info']['n'], previous_record=local_record)
    value = (cache, record['status'], origin)
    s['response'], s['cached'] = response, value
    return value


def calculate_pair(job):
    u, v, previous = job
    x, y = vector(u), vector(v)
    n11 = int(x@y); n10 = int(x.sum())-n11; n01 = int(y.sum())-n11
    n00 = len(x)-n11-n10-n01
    result = dict(predictor_pattern=u, response_pattern=v, n00=n00, n10=n10, n01=n01, n11=n11,
                  min_cell=min(n00,n10,n01,n11), p_exact=None, score_status='SHARED_EXACT_PATTERN',
                  beta=None, beta_se=None, beta_status='SEPARATION_OR_MONOMORPHIC', null_origin='')
    if u==v: return result
    cache, status, origin = response_null(v)
    result.update(null_origin=origin, score_status=status)
    if cache is not None:
        score = score_predictor_block(x[:,None], cache, compute_spa_adjustment=False)[0]
        result['score_status'] = score.status
        if score.status=='OK' and np.isfinite(score.p_score): result['p_exact'] = float(score.p_score)
    if previous is not None:
        result.update({k:previous.get(k) for k in ('beta','beta_se','beta_status')})
        result['beta_reused_on_null_policy_change'] = True
    elif result['min_cell'] > 0:
        fit, alternative = fit_logistic_mixed(y, _WORKER['K'], covariates=x[:,None], kinship_eigensystem=_WORKER['eigen'])
        result['beta_status'] = fit.status
        if fit.status=='OK' and fit.converged:
            beta, variance = float(fit.beta[1]), float(alternative['M_inv'][1,1])
            if not fit.n_weights_clipped and np.isfinite([beta,variance]).all() and variance>0:
                result.update(beta=beta, beta_se=math.sqrt(variance))
            else: result['beta_status']='UNSTABLE_EFFECT'
    return result


class TargetPool(ResponsePool):
    """One owner per response: no concurrent writes to its diagnostic null."""
    def submit(self, function, job):
        return self.executors[job[1] % self.workers].submit(function, job)


def unitig_pair_rows(rows, include_nearby):
    by_target = defaultdict(list)
    for r in rows:
        if r['pattern'] is not None and not r['n_matching_references_binary_absent'] and (
                r['annotation_class']=='coding' or include_nearby):
            by_target[r['target']].append(r)
    for a,b in COMPARISONS:
        for x,y in itertools.product(by_target[a],by_target[b]):
            yield dict(target_a=a,target_b=b,unitig_a=x['unitig'],unitig_b=y['unitig'],
                pattern_a=x['pattern'],pattern_b=y['pattern'],class_a=x['annotation_class'],class_b=y['annotation_class'],
                clusters_a=x['clusters'],clusters_b=y['clusters'])


def historical_comparison(args, source, pairs, rows, identity):
    """All-or-nothing baseline DB reads; unreadable negatives stay unknown.

    The exported table contains significant pairs only. Absence from that table
    is never evidence of no query, no beta attempt or nonsignificance.
    """
    old, distances, errors = {}, {}, []
    try:
        current = json.loads((args.results/'run_identity.json').read_text())
        if any(current.get(k)!=identity.get(k) for k in ('inputs','settings','null_model_source_sha256')):
            raise ValueError('Original comparison inputs/settings differ from the frozen diagnostic')
    except (OSError,ValueError) as error:
        log(args.out,'WARNING: original comparison provenance unavailable: '+str(error))
        return old,distances,dict(mode='unavailable',errors=[str(error)])
    try:
        if source is None: raise sqlite3.DatabaseError('Original checkpoint unavailable')
        for u,v in pairs:
            row = source.execute('SELECT status,final,p,t_low,t_high FROM decisions WHERE u=? AND v=?',(u,v)).fetchone()
            beta = source.execute('SELECT value,status FROM beta WHERE u=? AND v=?',(u,v)).fetchone()
            old[u,v] = dict(status=row[0] if row else 'SHARED_EXACT_PATTERN' if u==v else 'NOT_QUERIED',
                final=row[1] if row else None,p=row[2] if row else None,
                lower=float(scipy.stats.chi2.sf(row[4],1)) if row and row[4] is not None else None,
                upper=float(scipy.stats.chi2.sf(row[3],1)) if row and row[3] is not None else None,
                beta=beta[0] if beta else None,beta_status=beta[1] if beta else 'NOT_FITTED',source='ORIGINAL_SQLITE')
        for r in unitig_pair_rows(rows,args.include_nearby):
            key = tuple(sorted((r['unitig_a'],r['unitig_b'])))
            if key not in distances:
                row = source.execute('SELECT distance,connected,status FROM distance WHERE u=? AND v=?',key).fetchone()
                if row: distances[key] = row
        return old,distances,dict(mode='original_sqlite',errors=[])
    except sqlite3.DatabaseError as error:
        errors.append(str(error))
        old.clear(); distances.clear()
        log(args.out,'WARNING: original checkpoint comparison unreadable: '+str(error)+
                    '; trying exported significant associations; no source repair attempted')
    exported = args.results/'associations.tsv.gz'
    wanted = set(pairs)
    loci = {int(r['unitig']) for r in rows}
    number = lambda text: float(text) if text and math.isfinite(float(text)) else None
    try:
        with gzip.open(exported,'rt',encoding='utf-8-sig',newline='') as stream:
            reader = csv.DictReader(stream,delimiter='\t')
            required = {'u','v','predictor_pattern','response_pattern','significance','p_exact','p_lower',
                        'p_upper','beta','beta_status','min_distance','connected_genomes','distance_status'}
            if not required.issubset(reader.fieldnames or []): raise ValueError('Original export schema incomplete')
            for r in reader:
                key = tuple(sorted((int(r['predictor_pattern']),int(r['response_pattern']))))
                if key not in wanted or r['significance']!='SIGNIFICANT': continue
                old.setdefault(key,dict(status='SIGNIFICANT_FROM_EXPORT',final=1,p=number(r['p_exact']),
                    lower=number(r['p_lower']),upper=number(r['p_upper']),beta=None,
                    beta_status=r['beta_status'],source='SIGNIFICANT_EXPORT'))
                if number(r['beta']) is not None:
                    old[key].update(beta=number(r['beta']),beta_status=r['beta_status'])
                a,b = int(r['u']),int(r['v'])
                if a in loci and b in loci:
                    distances[tuple(sorted((a,b)))] = (number(r['min_distance']),
                        int(r['connected_genomes']) if r['connected_genomes'] else None,r['distance_status'])
        log(args.out,f'export fallback matched {len(old):,}/{len(wanted):,} target pattern pairs; '
                    'other historical comparisons are unavailable, not nonsignificant')
        mode = 'significant_export'
    except (OSError,EOFError,ValueError,csv.Error) as error:
        errors.append(f'{exported}: {error}')
        old.clear(); distances.clear(); mode='unavailable'
        log(args.out,'WARNING: original export comparison unavailable: '+str(error))
    return old,distances,dict(mode=mode,errors=errors,export=str(exported))


def compare(args, db, source, info, identity, rows, patterns):
    pairs = defaultdict(set)
    for a,b in COMPARISONS:
        for u,v in itertools.product(patterns[a],patterns[b]): pairs[tuple(sorted((u,v)))].add(a+'--'+b)
    if len(pairs)>args.max_pairs:
        raise ValueError(f'{len(pairs):,} pattern pairs exceed --max-pairs {args.max_pairs:,}; '
                         'inspect target_unitigs.tsv before explicitly increasing the cap')
    reuse = original_reuse(identity,args.original_nulls)
    log(args.out, f'{len(pairs):,} distinct target pattern pairs; original null reuse enabled={reuse["enabled"]}; '
                 f'policy={reuse["policy"]}; fitting code matches={reuse["fitting_code_matches"]}; '
                 f'library versions match={reuse["library_versions_match"]}')
    if not reuse['library_versions_match']: log(args.out,'library version differences: '+json.dumps(reuse['versions']))
    pending = []
    for u,v in sorted(pairs,key=lambda p:(p[1],p[0])):
        row = db.execute('SELECT result FROM fits WHERE u=? AND v=?',(u,v)).fetchone()
        previous = json.loads(row[0]) if row else None
        if previous is None or (u!=v and args.original_nulls=='require' and previous.get('null_origin')!='original_null_cache'):
            pending.append((u,v,previous))
    if any(p[2] is not None for p in pending):
        log(args.out,'recalculating scores from original nulls; completed independent beta fits are retained')
    if pending:
        with TargetPool(args.threads, initialize_fits, (str(args.work),info,str(args.out),reuse)) as pool:
            for completed,result in enumerate(bounded_map(pool, calculate_pair, pending, args.threads, ordered=False),1):
                with db: db.execute('INSERT OR REPLACE INTO fits VALUES (?,?,?)',
                    (result['predictor_pattern'],result['response_pattern'],json.dumps(result,allow_nan=False)))
                saved = db.execute('SELECT count(*) FROM fits').fetchone()[0]
                if completed % 10 == 0 or completed==len(pending):
                    log(args.out, f'target fits saved: {saved:,}/{len(pairs):,}; jobs this invocation: {completed:,}/{len(pending):,}')
    return render_reports(args,db,source,info,identity,rows,pairs,reuse)


def render_reports(args, db, source, info, identity, rows, pairs, reuse):
    old_min = int(identity['settings'].get('min_cell_count',20))
    new_min = max(10, math.ceil(.02*info['n']))
    family = int(identity['settings'].get('bonferroni_tests') or math.comb(info['patterns'],2))
    cutoff = float(identity['settings']['alpha'])/family
    baseline,distances,comparison = historical_comparison(args,source,pairs,rows,identity)
    output = []
    for (u,v), labels in sorted(pairs.items()):
        saved = db.execute('SELECT result FROM fits WHERE u=? AND v=?',(u,v)).fetchone()
        if saved is None: raise ValueError(f'Diagnostic fit {u},{v} is not saved; report-only never calculates missing fits')
        r = json.loads(saved[0])
        old = baseline.get((u,v),dict(status='SHARED_EXACT_PATTERN' if u==v else
            'UNAVAILABLE_NOT_IN_SIGNIFICANT_EXPORT' if comparison['mode']=='significant_export' else
            'UNAVAILABLE_ORIGINAL_COMPARISON',final=None,p=None,lower=None,upper=None,beta=None,
            beta_status='UNAVAILABLE_ORIGINAL_COMPARISON',source='UNAVAILABLE'))
        lo,hi = old['lower'],old['upper']
        r.update(target_pairs=';'.join(sorted(labels)), original_min_cell=old_min, new_min_cell=new_min,
            eligible_original=int(u!=v and r['min_cell']>=old_min), eligible_new=int(u!=v and r['min_cell']>=new_min),
            old_query_status=old['status'],old_comparison_source=old['source'],
            old_final=old['final'],old_p_exact=old['p'],old_p_lower=lo,old_p_upper=hi,old_beta=old['beta'],
            old_beta_status=old['beta_status'], bonferroni_tests=family, p_cutoff=cutoff,
            significant_p_only=int(r['p_exact'] is not None and r['p_exact']<=cutoff),
            significant_and_eligible_new=int(u!=v and r['min_cell']>=new_min and r['p_exact'] is not None and r['p_exact']<=cutoff),
            beta_difference=r['beta']-old['beta'] if old['beta'] is not None and r['beta'] is not None else None,
            p_difference=r['p_exact']-old['p'] if old['p'] is not None and r['p_exact'] is not None else None,
            exact_p_within_old_bounds=int(lo*(1-1e-7) <= r['p_exact'] <= hi*(1+1e-7)) if lo is not None and hi is not None and r['p_exact'] is not None else None)
        output.append(r)
    fields = ['target_pairs','predictor_pattern','response_pattern','n00','n10','n01','n11','min_cell',
        'original_min_cell','new_min_cell','eligible_original','eligible_new','old_query_status','old_comparison_source','old_final',
        'old_p_exact','old_p_lower','old_p_upper','p_exact','score_status','exact_p_within_old_bounds','p_difference',
        'old_beta','old_beta_status','beta','beta_se','beta_status','beta_difference','null_origin','beta_reused_on_null_policy_change',
        'bonferroni_tests','p_cutoff','significant_p_only','significant_and_eligible_new']
    table(args.out/'pair_comparison.tsv',output,fields)
    def evidence():
        for r in unitig_pair_rows(rows,args.include_nearby):
            old = distances.get(tuple(sorted((r['unitig_a'],r['unitig_b']))))
            yield dict(r,distance=old[0] if old else None,connected_genomes=old[1] if old else None,
                distance_status=old[2] if old else 'NOT_CALCULATED' if comparison['mode']=='original_sqlite'
                else 'UNKNOWN_ORIGINAL_COMPARISON_UNAVAILABLE')
    table(args.out/'unitig_pair_evidence.tsv.gz',evidence(),['target_a','target_b','unitig_a','unitig_b',
        'pattern_a','pattern_b','class_a','class_b','clusters_a','clusters_b','distance','connected_genomes','distance_status'])
    return dict(samples=info['n'],original_min_cell=old_min,new_min_cell=new_min,pattern_pairs=len(pairs),
        significant_and_eligible_new=sum(r['significant_and_eligible_new'] for r in output),
        original_query_statuses={s:sum(r['old_query_status']==s for r in output) for s in {r['old_query_status'] for r in output}},
        bonferroni_tests=family,p_cutoff=cutoff,original_null_reuse_compatible=reuse['enabled'],original_null_reuse=reuse,
        historical_comparison=comparison,
        null_origin_counts={origin:sum(r['null_origin']==origin for r in output) for origin in {r['null_origin'] for r in output}},
        interpretation='Targeted unitig diagnostic, not gene-level significance or evidence of epistasis. '
        'Unqueried pairs are not labelled nonsignificant. Full matches at target references do not establish '
        'genome-wide mapping uniqueness. Unknown distances are not presumed distal.')


def report_saved(args):
    """Export committed fits even if the original SQLite checkpoint is corrupt."""
    state = json.loads((args.out/'diagnosis_identity.json').read_text())
    info,identity = state['prepared'],state['source_identity']
    args.include_nearby = state['include_nearby']
    lock = sqlite3.connect(args.out/'coordinator.lock.sqlite',timeout=0)
    source = db = None
    try:
        lock.execute('BEGIN EXCLUSIVE')
        db = read_only(args.out/'diagnosis.sqlite')
        with (args.out/'target_unitigs.tsv').open(encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream,delimiter='\t'))
        patterns = defaultdict(set)
        for r in rows:
            r['unitig'] = int(r['unitig'])
            r['pattern'] = int(r['pattern']) if r['pattern'] else None
            r['n_matching_references_binary_absent'] = int(r['n_matching_references_binary_absent']) if r['n_matching_references_binary_absent'] else None
            if r['pattern'] is not None and not r['n_matching_references_binary_absent'] and (
                    r['annotation_class']=='coding' or args.include_nearby): patterns[r['target']].add(r['pattern'])
        pairs = defaultdict(set)
        for a,b in COMPARISONS:
            for u,v in itertools.product(patterns[a],patterns[b]): pairs[tuple(sorted((u,v)))].add(a+'--'+b)
        # Validate every requested saved result BEFORE replacing any report.
        for u,v in pairs:
            row = db.execute('SELECT result FROM fits WHERE u=? AND v=?',(u,v)).fetchone()
            if row is None: raise ValueError(f'Fit {u},{v} is not saved; report-only cannot calculate it')
            json.loads(row[0])
        candidates = [args.work] if args.work else [args.results/'.kovar_work',Path(identity.get('work_directory','/not-found'))]
        args.work = next((p.resolve() for p in candidates if p and (p/'work.sqlite').is_file()),None)
        try:
            if args.work: source = read_only(args.work/'work.sqlite')
        except sqlite3.DatabaseError as error:
            log(args.out,'WARNING: cannot open original checkpoint: '+str(error))
        log(args.out,f'report-only: {len(pairs):,} saved target fits; no mapping, null fitting or beta fitting')
        reuse = dict(policy=state.get('original_nulls','auto'),enabled=None,
                     saved_fit_environment={k:state[k] for k in ('numpy','scipy','null_source_sha256')})
        summary = render_reports(args,db,source,info,identity,rows,pairs,reuse)
        summary.update(report_only=True,source_results=str(args.results),source_work=str(args.work),source_modified=False)
        atomic_json(args.out/'summary.json',summary)
        log(args.out,'reports recovered: '+str(args.out/'pair_comparison.tsv'))
    finally:
        if db: db.close()
        if source: source.close()
        lock.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--work',type=Path)
    p.add_argument('--fasta',type=Path)
    p.add_argument('--unitigs',type=Path)
    p.add_argument('--panaroo',type=Path)
    p.add_argument('--gff-dir',type=Path)
    p.add_argument('--bakta',type=Path)
    p.add_argument('--map-dir',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--nearby-bp',type=int,default=500)
    p.add_argument('--seed-k',type=int,default=31)
    p.add_argument('--include-nearby',action='store_true',help='Also test bounded-nearby candidates; default tests coding hits only')
    p.add_argument('--max-pairs',type=int,default=10000)
    p.add_argument('--original-nulls',choices=['auto','require'],default='auto',
                   help='require: reuse original nulls across library versions when fitting code matches; never refit missing nulls')
    p.add_argument('--report-only',action='store_true',help='Export saved diagnostic fits only; tolerate unreadable original comparisons, never fit')
    args = p.parse_args(argv)
    args.results,args.out = args.results.resolve(),args.out.resolve()
    if args.report_only:
        return report_saved(args)
    if not all((args.fasta,args.unitigs,args.panaroo,args.map_dir)):
        p.error('--fasta, --unitigs, --panaroo and --map-dir required unless --report-only')
    if min(args.threads,args.seed_k,args.max_pairs)<1 or args.nearby_bp<0:
        p.error('Positive threads/seed/max-pairs and nonnegative nearby-bp required')
    args.results,args.out = args.results.resolve(),args.out.resolve()
    if args.out==args.results or args.out.is_relative_to(args.results):
        p.error('Diagnostic output must be outside the discovery results directory')
    identity = json.loads((args.results/'run_identity.json').read_text())
    original_reuse(identity,args.original_nulls)
    candidates = [args.work] if args.work else [args.results/'.kovar_work',Path(identity.get('work_directory','/not-found'))]
    args.work = next((w.resolve() for w in candidates if w and (w/'prepared.json').is_file() and (w/'work.sqlite').is_file()),None)
    if args.work is None: p.error('Frozen .kovar_work required; provide --work if stored elsewhere')
    if args.out==args.work or args.out.is_relative_to(args.work):
        p.error('Diagnostic output must be outside the source work directory')
    info = json.loads((args.work/'prepared.json').read_text())
    source = read_only(args.work/'work.sqlite')
    work_identity = json.loads((args.work/'run_identity.json').read_text())
    if any(work_identity.get(k)!=identity.get(k) for k in ('inputs','null_model_source_sha256')):
        raise ValueError('Source work directory belongs to a different input cohort or null-fitting model')
    refs, missing = references(args,info)
    # Verify unitig numbering against the distance stage, and FASTA against the
    # frozen cohort input. These checks prevent scoring a different sequence pool.
    original_distance = source.execute("SELECT value FROM meta WHERE key='distance_identity'").fetchone()
    if original_distance and digest(args.unitigs)!=json.loads(original_distance[0])['unitigs']:
        raise ValueError('DNA unitig file differs from the original distance-stage unitig pool')
    if digest(args.fasta)!=identity['inputs']['fasta']:
        raise ValueError('Sample-ID FASTA differs from the frozen discovery input')
    for name in ('patterns.bin','K.npy','K_values.npy','K_vectors.npy'):
        if digest(args.work/name)!=info['products'][name]: raise ValueError(f'Frozen product changed: {name}')
    args.out.mkdir(parents=True,exist_ok=True)
    lock = sqlite3.connect(args.out/'coordinator.lock.sqlite', timeout=0)
    try:
        lock.execute('BEGIN EXCLUSIVE')
    except sqlite3.OperationalError as error:
        lock.close()
        raise ValueError('Another diagnostic coordinator is using this output') from error
    state = dict(source_identity=identity,prepared=info,unitigs=stamp(args.unitigs),fasta=stamp(args.fasta),
        panaroo=stamp(args.panaroo),references={s:[stamp(x) for x in paths if x] for s,paths in refs.items()},
        nearby_bp=args.nearby_bp,seed_k=args.seed_k,include_nearby=args.include_nearby,
        script_sha256=digest(Path(__file__)),null_source_sha256=null_source_hash(),numpy=np.__version__,scipy=scipy.__version__,
        original_nulls=args.original_nulls)
    manifest = args.out/'diagnosis_identity.json'
    if manifest.exists() and json.loads(manifest.read_text())!=state:
        previous = json.loads(manifest.read_text())
        ignored = {'script_sha256','original_nulls'}
        if previous.get('script_sha256') not in PRE_REUSE_POLICY_HASHES | {state['script_sha256']} or \
                {k:v for k,v in previous.items() if k not in ignored}!={k:v for k,v in state.items() if k not in ignored}:
            raise ValueError('Diagnostic inputs/settings changed; choose a separate diagnostic output')
        backup = manifest.with_name('diagnosis_identity.before_null_policy.json')
        if not backup.exists(): atomic_json(backup,previous)
    atomic_json(manifest,state)
    db = sqlite3.connect(args.out/'diagnosis.sqlite')
    db.execute('PRAGMA journal_mode=WAL'); db.execute('PRAGMA synchronous=FULL')
    db.executescript('''CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT);
      CREATE TABLE IF NOT EXISTS regions(sample TEXT,contig TEXT,gs INTEGER,ge INTEGER,strand TEXT,
        target TEXT,cluster TEXT,left_pos INTEGER,dna TEXT);
      CREATE TABLE IF NOT EXISTS sequences(locus INTEGER PRIMARY KEY,dna TEXT);
      CREATE TABLE IF NOT EXISTS anchors(locus INTEGER,target TEXT,cluster TEXT,sample TEXT,contig TEXT,
        start_pos INTEGER,gs INTEGER,ge INTEGER,strand TEXT,reverse INTEGER,valid INTEGER,
        UNIQUE(locus,target,cluster,sample,contig,start_pos,gs,ge,reverse));
      CREATE INDEX IF NOT EXISTS anchor_sample ON anchors(sample,valid);
      CREATE TABLE IF NOT EXISTS fits(u INTEGER,v INTEGER,result TEXT,PRIMARY KEY(u,v));''')
    log(args.out,f"frozen cohort {info['n']:,}; original min-cell={identity['settings']['min_cell_count']}; "
                 f"new min-cell={max(10,math.ceil(.02*info['n']))}; missing reference samples={len(missing)}")
    try:
        prepare_regions(args,db,info,refs)
        scan_unitigs(args,db)
        validate_anchors(args,db,refs)
        rows, patterns = locus_report(args,db,source,info)
        summary = compare(args,db,source,info,identity,rows,patterns)
        summary.update(missing_reference_samples=missing,scanned_unitigs=get(db,'scan_count'),
                       source_results=str(args.results),source_work=str(args.work),source_modified=False)
        atomic_json(args.out/'summary.json',summary)
        log(args.out,'complete: '+str(args.out/'pair_comparison.tsv'))
    finally:
        db.close(); source.close(); lock.close()


if __name__=='__main__':
    main()
