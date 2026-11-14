#!/usr/bin/env python3
"""Resumable KOVAR and pyseer checks in one standalone entry point.

kovar:   annotate and directly score the existing target-pair audit.
pyseer:  fit the same frozen target patterns using the installed pyseer LMM.
sccmec:  optionally compare assembly-level carriage with the saved patterns.

Numerical models, response orientation and output schemas are unchanged.
Each command can run in its own Python environment and reuses its saved output.
Source discovery results are read-only. See docs/amr_checks.md.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import csv
import gzip
import hashlib
import inspect
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from urllib.parse import unquote

# Parallelism is across models; prevent nested native-library oversubscription.
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np

_PYSEER_WORKER = {}


# Shared file helpers. Atomic replacement preserves committed checkpoints.
def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def table(path, rows, fields):
    opener = gzip.open if str(path).endswith('.gz') else open
    temporary = Path(str(path)+'.partial')
    with opener(temporary, 'wt', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fields, delimiter='\t', extrasaction='ignore')
        writer.writeheader()
        writer.writerows({k: '' if r.get(k) is None else r.get(k, '') for k in fields} for r in rows)
    temporary.replace(path)


# KOVAR target audit and checkpoint recovery.
def load_kovar():
    """Load KOVAR only for its stage, including spawned fitting workers."""
    global scipy, SeqIO, threadpool_limits, bounded_map, open_patterns
    global KinshipEigensystem, fit_logistic_mixed, fit_null_glmm, score_predictor_block
    global discovery_nulls, null_source_hash, TargetPool
    if 'fit_null_glmm' in globals():
        return
    import scipy
    from Bio import SeqIO
    from threadpoolctl import threadpool_limits
    from ko_variation.discovery_execution import ResponsePool, bounded_map
    from ko_variation.discovery_store import open_patterns
    from ko_variation.glmm import KinshipEigensystem, fit_logistic_mixed, fit_null_glmm, score_predictor_block
    from ko_variation import discovery_nulls
    from ko_variation.null_identity import null_source_hash

    class TargetPool(ResponsePool):
        """One owner per response: no concurrent writes to its diagnostic null."""
        def submit(self, function, job):
            return self.executors[job[1] % self.workers].submit(function, job)



TARGETS = ('mecA', 'graR', 'sen', 'tetM', 'tetK')
COMPARISONS = (('mecA', 'graR'), ('mecA', 'sen'))
_KOVAR_WORKER = {}
PRE_REUSE_POLICY_HASHES = {'485d44a0616e1518d9d6233d09ba2db32a33c7a3672f039a175cb6109cbe5e14',
                         '32713cfea041024439ba915315416df6ba9369f7942c63643ee072f752756391',
                         'da44d07dda016adf80e83f66fbcf8a3760d1c1d7cb8ed0e0655bde78027e85e3',
                         'c9d289ff3db52df7f743b1a652faa05937f070e2441df12022300aec4d226c64',
                         '07c5b79effdc787e92c253adbf90679be2ab92fcc734a4012c2800a721852383'}


def stamp(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def read_only(path):
    return sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True, timeout=60)


def log_kovar(out, message):
    text = '[AMR diagnosis] '+message
    print(text, flush=True)
    with (out/'progress.log').open('a', encoding='utf-8') as stream:
        stream.write(time.strftime('%Y-%m-%d %H:%M:%S ')+text+'\n')


def get(db, key, default=None):
    row = db.execute('SELECT value FROM meta WHERE key=?', (key,)).fetchone()
    return json.loads(row[0]) if row else default


def put(db, key, value):
    db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)', (key, json.dumps(value)))


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
            log_kovar(args.out, f'target reference windows: {index:,}/{len(refs):,} samples')
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
    log_kovar(args.out, f'scanning full DNA unitig pool from byte {offset:,}; {len(grouped):,} distinct target windows')
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
                log_kovar(args.out, f'unitigs scanned: {count:,}; candidate unitigs: '
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
            log_kovar(args.out, f'full-length exact-match validation: {index:,}/{len(refs):,} samples')


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
    load_kovar()
    threadpool_limits(limits=1)
    _KOVAR_WORKER.clear()
    _KOVAR_WORKER.update(work=Path(work), info=info, out=Path(out), source=read_only(Path(work)/'work.sqlite'),
        packed=open_patterns(work, info), K=np.load(Path(work)/'K.npy', mmap_mode='r'),
        eigen=KinshipEigensystem(np.load(Path(work)/'K_values.npy', mmap_mode='r'),
            np.load(Path(work)/'K_vectors.npy', mmap_mode='r'), info['shared_covariance']),
        reuse=reuse, response=None)


def vector(pid):
    return np.unpackbits(_KOVAR_WORKER['packed'][pid], bitorder='little')[:_KOVAR_WORKER['info']['n']].astype(float)


def response_null(response):
    s = _KOVAR_WORKER
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
        if not np.array_equal(cache['y'],vector(response)) or np.any(cache['weights']<=0) or\
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
        fit, alternative = fit_logistic_mixed(y, _KOVAR_WORKER['K'], covariates=x[:,None], kinship_eigensystem=_KOVAR_WORKER['eigen'])
        result['beta_status'] = fit.status
        if fit.status=='OK' and fit.converged:
            beta, variance = float(fit.beta[1]), float(alternative['M_inv'][1,1])
            if not fit.n_weights_clipped and np.isfinite([beta,variance]).all() and variance>0:
                result.update(beta=beta, beta_se=math.sqrt(variance))
            else: result['beta_status']='UNSTABLE_EFFECT'
    return result


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
        log_kovar(args.out,'WARNING: original comparison provenance unavailable: '+str(error))
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
        log_kovar(args.out,'WARNING: original checkpoint comparison unreadable: '+str(error)+
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
        log_kovar(args.out,f'export fallback matched {len(old):,}/{len(wanted):,} target pattern pairs; '
                    'other historical comparisons are unavailable, not nonsignificant')
        mode = 'significant_export'
    except (OSError,EOFError,ValueError,csv.Error) as error:
        errors.append(f'{exported}: {error}')
        old.clear(); distances.clear(); mode='unavailable'
        log_kovar(args.out,'WARNING: original export comparison unavailable: '+str(error))
    return old,distances,dict(mode=mode,errors=errors,export=str(exported))


def compare(args, db, source, info, identity, rows, patterns):
    pairs = defaultdict(set)
    for a,b in COMPARISONS:
        for u,v in itertools.product(patterns[a],patterns[b]): pairs[tuple(sorted((u,v)))].add(a+'--'+b)
    if len(pairs)>args.max_pairs:
        raise ValueError(f'{len(pairs):,} pattern pairs exceed --max-pairs {args.max_pairs:,}; '
                         'inspect target_unitigs.tsv before explicitly increasing the cap')
    reuse = original_reuse(identity,args.original_nulls)
    log_kovar(args.out, f'{len(pairs):,} distinct target pattern pairs; original null reuse enabled={reuse["enabled"]}; '
                 f'policy={reuse["policy"]}; fitting code matches={reuse["fitting_code_matches"]}; '
                 f'library versions match={reuse["library_versions_match"]}')
    if not reuse['library_versions_match']: log_kovar(args.out,'library version differences: '+json.dumps(reuse['versions']))
    pending = []
    for u,v in sorted(pairs,key=lambda p:(p[1],p[0])):
        row = db.execute('SELECT result FROM fits WHERE u=? AND v=?',(u,v)).fetchone()
        previous = json.loads(row[0]) if row else None
        if previous is None or (u!=v and args.original_nulls=='require' and previous.get('null_origin')!='original_null_cache'):
            pending.append((u,v,previous))
    if any(p[2] is not None for p in pending):
        log_kovar(args.out,'recalculating scores from original nulls; completed independent beta fits are retained')
    if pending:
        with TargetPool(args.threads, initialize_fits, (str(args.work),info,str(args.out),reuse)) as pool:
            for completed,result in enumerate(bounded_map(pool, calculate_pair, pending, args.threads, ordered=False),1):
                with db: db.execute('INSERT OR REPLACE INTO fits VALUES (?,?,?)',
                    (result['predictor_pattern'],result['response_pattern'],json.dumps(result,allow_nan=False)))
                saved = db.execute('SELECT count(*) FROM fits').fetchone()[0]
                if completed % 10 == 0 or completed==len(pending):
                    log_kovar(args.out, f'target fits saved: {saved:,}/{len(pairs):,}; jobs this invocation: {completed:,}/{len(pending):,}')
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
            log_kovar(args.out,'WARNING: cannot open original checkpoint: '+str(error))
        log_kovar(args.out,f'report-only: {len(pairs):,} saved target fits; no mapping, null fitting or beta fitting')
        reuse = dict(policy=state.get('original_nulls','auto'),enabled=None,
                     saved_fit_environment={k:state[k] for k in ('numpy','scipy','null_source_sha256')})
        summary = render_reports(args,db,source,info,identity,rows,pairs,reuse)
        summary.update(report_only=True,source_results=str(args.results),source_work=str(args.work),source_modified=False)
        atomic_json(args.out/'summary.json',summary)
        log_kovar(args.out,'reports recovered: '+str(args.out/'pair_comparison.tsv'))
    finally:
        if db: db.close()
        if source: source.close()
        lock.close()


def run_kovar(argv=None):
    p = argparse.ArgumentParser(prog="amr_checks.py kovar", description="Audit and score the existing KOVAR target pairs.")
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
    load_kovar()
    args.results,args.out = args.results.resolve(),args.out.resolve()
    if args.report_only:
        return report_saved(args)
    if not all((args.fasta,args.unitigs,args.panaroo,args.map_dir)):
        p.error('--fasta, --unitigs, --panaroo and --map-dir required unless --report-only')
    if min(args.threads,args.seed_k,args.max_pairs)<1 or args.nearby_bp<0:
        p.error('Positive threads/seed/max-pairs and nonnegative nearby-bp required')
    args.results,args.out = args.results.resolve(),args.out.resolve()
    if args.out==args.results or args.results in args.out.parents:
        p.error('Diagnostic output must be outside the discovery results directory')
    identity = json.loads((args.results/'run_identity.json').read_text())
    original_reuse(identity,args.original_nulls)
    candidates = [args.work] if args.work else [args.results/'.kovar_work',Path(identity.get('work_directory','/not-found'))]
    args.work = next((w.resolve() for w in candidates if w and (w/'prepared.json').is_file() and (w/'work.sqlite').is_file()),None)
    if args.work is None: p.error('Frozen .kovar_work required; provide --work if stored elsewhere')
    if args.out==args.work or args.work in args.out.parents:
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
    previous = json.loads(manifest.read_text()) if manifest.exists() else None
    if previous is not None and previous!=state:
        ignored = {'script_sha256','original_nulls'}
        if previous.get('script_sha256') not in PRE_REUSE_POLICY_HASHES | {state['script_sha256']} or\
                {k:v for k,v in previous.items() if k not in ignored}!={k:v for k,v in state.items() if k not in ignored}:
            raise ValueError('Diagnostic inputs/settings changed; choose a separate diagnostic output')
        backup = manifest.with_name('diagnosis_identity.before_null_policy.json')
        if not backup.exists(): atomic_json(backup,previous)
    if previous is None or any(
            previous.get(k) != v for k, v in state.items() if k != 'script_sha256'):
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
    log_kovar(args.out,f"frozen cohort {info['n']:,}; original min-cell={identity['settings']['min_cell_count']}; "
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
        log_kovar(args.out,'complete: '+str(args.out/'pair_comparison.tsv'))
    finally:
        db.close(); source.close(); lock.close()


# Matched pyseer LMM comparison.
def read_table(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def log_pyseer(out, message):
    text = '[AMR pyseer] ' + message
    print(text, flush=True)
    with (out/'progress.log').open('a', encoding='utf-8') as stream:
        stream.write(time.strftime('%Y-%m-%d %H:%M:%S ') + text + '\n')


def pyseer_environment():
    try:
        import pyseer
        import pyseer.lmm
        import pyseer.fastlmm.lmm_cov
        import pandas
        import scipy
    except ImportError as error:
        raise ValueError('Run with the Python environment containing pyseer, pandas and scipy. '
                         'KOVAR itself is not required in this environment: ' + str(error)) from error
    return dict(pyseer=getattr(pyseer, '__version__', 'unknown'), numpy=np.__version__,
                pandas=pandas.__version__, scipy=scipy.__version__,
                lmm_source_sha256=digest(inspect.getfile(pyseer.lmm)),
                fastlmm_source_sha256=digest(inspect.getfile(pyseer.fastlmm.lmm_cov)))


def prepare_pyseer(args, environment):
    diagnosis = json.loads((args.diagnosis/'diagnosis_identity.json').read_text())
    info, source = diagnosis['prepared'], diagnosis['source_identity']
    rows = read_table(args.diagnosis/'pair_comparison.tsv')
    if not rows:
        raise ValueError('pair_comparison.tsv has no saved diagnostic tests')
    if args.work is None:
        candidates = [args.diagnosis.parent.parent/'discovery/results/.kovar_work',
                      args.diagnosis.parent.parent/'discovery/.kovar_work',
                      Path(source.get('work_directory', '/not-found'))]
        args.work = next((p for p in candidates if (p/'patterns.bin').is_file() and (p/'K.npy').is_file()), None)
    if args.work is None:
        raise ValueError('Provide --work /path/to/the/original/.kovar_work (patterns.bin and K.npy required)')
    args.work = args.work.resolve()
    for path in (args.diagnosis, args.work):
        # Paths are resolved above; parents membership also works on Python 3.8.
        if args.out == path or args.out in path.parents or path in args.out.parents:
            raise ValueError('--out must be separate from the source diagnosis and work directories')
    # These are the only full frozen-product reads. No decisions/beta DB is needed.
    for name in ('patterns.bin', 'K.npy'):
        if digest(args.work/name) != info['products'][name]:
            raise ValueError(f'{name} differs from the frozen diagnostic input')
    ids = sorted({int(r[k]) for r in rows for k in ('predictor_pattern', 'response_pattern')})
    if ids[0] < 0 or ids[-1] >= info['patterns']:
        raise ValueError('Pattern ID is outside the frozen pattern pool')
    packed = np.memmap(args.work/'patterns.bin', mode='r', dtype='u1',
                       shape=(info['patterns'], info['packed_width']))
    selected = np.unpackbits(np.asarray(packed[ids]), axis=1, bitorder='little')[:, :info['n']]
    lookup = {pid: i for i, pid in enumerate(ids)}
    family = int(source['settings'].get('bonferroni_tests') or math.comb(info['patterns'], 2))
    cutoff = float(source['settings']['alpha']) / family
    seen = set()
    for r in rows:
        u, v = int(r['predictor_pattern']), int(r['response_pattern'])
        if (u, v) in seen:
            raise ValueError(f'Duplicate diagnostic pair {u},{v}')
        seen.add((u, v))
        x, y = selected[lookup[u]].astype(np.int64), selected[lookup[v]].astype(np.int64)
        n11 = int(x @ y)
        cells = [info['n'] - int(x.sum()) - int(y.sum()) + n11, int(x.sum()) - n11, int(y.sum()) - n11, n11]
        if cells != [int(r[k]) for k in ('n00', 'n10', 'n01', 'n11')]:
            raise ValueError(f'Frozen vectors do not reproduce the saved cells for {u},{v}')
        if int(r['bonferroni_tests']) != family or not math.isclose(float(r['p_cutoff']), cutoff, rel_tol=1e-12):
            raise ValueError('Saved diagnostic has a different testing family or threshold')
    K = np.load(args.work/'K.npy', allow_pickle=False)
    if K.shape != (info['n'], info['n']) or not np.isfinite(K).all():
        raise ValueError('Invalid frozen covariance dimensions/values')
    # Match pyseer initialise_lmm's mean-diagonal normalization, not a new kernel.
    factor = info['n'] / float(np.diag(K).sum())
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError('Frozen K must have a positive mean diagonal')
    identity = dict(revision=1, environment=environment,
                    diagnostic_identity_sha256=digest(args.diagnosis/'diagnosis_identity.json'),
                    comparison_sha256=digest(args.diagnosis/'pair_comparison.tsv'),
                    products={k: info['products'][k] for k in ('patterns.bin', 'K.npy')},
                    samples=info['samples'], pattern_ids=ids, bonferroni_tests=family,
                    p_cutoff=cutoff, pyseer_K_normalization_factor=factor,
                    orientation='saved predictor_pattern -> response_pattern',
                    model='pyseer initialise_lmm + fit_lmm_block; intercept only; no candidate/cell/prefilter gate')
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out/'run_identity.json'
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError('Comparison inputs or pyseer environment changed; use a separate --out')
    if not manifest.exists() and any(args.out.iterdir()):
        raise ValueError('Nonempty output has no comparison identity; choose an empty --out')
    atomic_json(manifest, identity)
    np.save(args.out/'selected_patterns.npy', selected, allow_pickle=False)
    table(args.out/'sample_order.tsv', ({'row': i, 'sample': s} for i, s in enumerate(info['samples'])), ['row', 'sample'])
    import pandas as pd
    pd.DataFrame(K, index=info['samples'], columns=info['samples']).to_csv(
        args.out/'covariance.tsv', sep='\t', float_format='%.17g')
    (args.out/'responses').mkdir(exist_ok=True)
    return rows, identity


def initialize_pyseer(out, identity):
    # Reuse the installed implementation, not a reimplementation of FaST-LMM.
    from pyseer.lmm import initialise_lmm, fit_lmm_block
    _PYSEER_WORKER.update(out=Path(out), identity=identity,
                   matrix=np.load(Path(out)/'selected_patterns.npy', mmap_mode='r'),
                   lookup={pid: i for i, pid in enumerate(identity['pattern_ids'])},
                   initialise=initialise_lmm, score=fit_lmm_block)


def fit_response(job):
    import pandas as pd
    response, predictors = job
    s = _PYSEER_WORKER
    started = time.monotonic()
    y = np.array(s['matrix'][s['lookup'][response]], dtype=float)
    phenotype = pd.Series(y, index=s['identity']['samples'])
    fitted_p, lmm, h2 = s['initialise'](phenotype, pd.DataFrame(), str(s['out']/'covariance.tsv'))
    if list(fitted_p.index) != s['identity']['samples'] or not np.array_equal(fitted_p.to_numpy(), y):
        raise ValueError('Pyseer changed sample order or dropped samples')
    if not math.isfinite(float(h2)) or not 0 <= h2 <= 1:
        raise ValueError(f'Invalid LMM null h2 for response {response}: {h2}')
    output = []
    # Each block shares a single fitted response null. No four-cell/MAF/MI filter.
    for first in range(0, len(predictors), 256):
        pids = predictors[first:first+256]
        X = np.array(s['matrix'][[s['lookup'][p] for p in pids]].T, dtype=float)
        result = s['score'](lmm, h2, X)
        if any(len(result[k]) != len(pids) for k in ('p_values', 'beta', 'bse')):
            raise ValueError('Unexpected pyseer LMM block dimensions')
        for i, predictor in enumerate(pids):
            values = [float(result[k][i]) for k in ('p_values', 'beta', 'bse')]
            good = all(math.isfinite(x) for x in values) and 0 <= values[0] <= 1 and values[2] > 0
            output.append(dict(predictor_pattern=predictor, response_pattern=response,
                               lmm_p=values[0] if good else None, lmm_beta=values[1] if math.isfinite(values[1]) else None,
                               lmm_se=values[2] if math.isfinite(values[2]) else None,
                               lmm_status='OK' if good else 'NONFINITE_OR_INVALID_LMM_RESULT', lmm_h2=float(h2)))
    record = dict(response=response, predictors=predictors, results=output, elapsed_seconds=time.monotonic()-started)
    atomic_json(s['out']/'responses'/f'{response:09d}.json', record)
    return response


def export_pyseer(args, rows, identity, resumed):
    results = {}
    for path in (args.out/'responses').glob('*.json'):
        record = json.loads(path.read_text())
        for r in record['results']:
            results[r['predictor_pattern'], r['response_pattern']] = r
    output = []
    for r in rows:
        u, v = int(r['predictor_pattern']), int(r['response_pattern'])
        value = results[u, v] if u != v else dict(lmm_p=None, lmm_beta=None, lmm_se=None,
                                                   lmm_h2=None, lmm_status='SHARED_EXACT_PATTERN')
        pv = value['lmm_p']
        kp = float(r['p_exact']) if r['p_exact'] and r['score_status'] == 'OK' else None
        cut = identity['p_cutoff']
        output.append(dict(r, **value, lmm_nominal=int(pv < .05) if pv is not None else None,
                           lmm_bonferroni=int(pv <= cut) if pv is not None else None,
                           lmm_bonferroni_eligible_original=int(pv <= cut and r['eligible_original'] == '1') if pv is not None else None,
                           lmm_bonferroni_eligible_new=int(pv <= cut and r['eligible_new'] == '1') if pv is not None else None,
                           nominal_comparison='UNAVAILABLE' if pv is None or kp is None else
                           'BOTH' if pv < .05 and kp < .05 else 'PYSEER_ONLY' if pv < .05 else
                           'KOVAR_ONLY' if kp < .05 else 'NEITHER'))
    extra = ['lmm_p', 'lmm_beta', 'lmm_se', 'lmm_h2', 'lmm_status', 'lmm_nominal', 'lmm_bonferroni',
             'lmm_bonferroni_eligible_original', 'lmm_bonferroni_eligible_new', 'nominal_comparison']
    table(args.out/'pyseer_comparison.tsv', output, list(rows[0]) + extra)
    groups = []
    for name in sorted({r['target_pairs'] for r in output}):
        rs = [r for r in output if r['target_pairs'] == name]
        ps = [r['lmm_p'] for r in rs if r['lmm_p'] is not None]
        eligible = [r['lmm_p'] for r in rs if r['lmm_p'] is not None and r['eligible_new'] == '1']
        groups.append(dict(target_pairs=name, pairs=len(rs), lmm_ok=len(ps),
                           lmm_nominal=sum(r['lmm_nominal'] or 0 for r in rs),
                           lmm_bonferroni=sum(r['lmm_bonferroni'] or 0 for r in rs),
                           lmm_bonferroni_eligible_original=sum(r['lmm_bonferroni_eligible_original'] or 0 for r in rs),
                           lmm_bonferroni_eligible_new=sum(r['lmm_bonferroni_eligible_new'] or 0 for r in rs),
                           lowest_lmm_p=min(ps) if ps else None,
                           lowest_lmm_p_eligible_new=min(eligible) if eligible else None,
                           pyseer_only_nominal=sum(r['nominal_comparison'] == 'PYSEER_ONLY' for r in rs)))
    table(args.out/'summary.tsv', groups, list(groups[0]))
    summary = dict(samples=len(identity['samples']), pairs=len(rows), resumed_response_models=resumed,
                   bonferroni_tests=identity['bonferroni_tests'], p_cutoff=identity['p_cutoff'], groups=groups,
                   environment=identity['environment'], source_modified=False,
                   interpretation='Matched unitig-pattern association, not a recreation of the 806-isolate SCCmec-carriage study. '
                   'p_exact and beta remain the saved KOVAR results; lmm_* come from pyseer. '
                   'No source SQL, KOVAR null fitting or beta fitting. Pyseer LMM nulls are newly fitted per response. '
                   'Unfiltered significance is distinguished from the original/new cell eligibility gates.')
    atomic_json(args.out/'summary.json', summary)
    for g in groups:
        log_pyseer(args.out, f"{g['target_pairs']}: {g['lmm_nominal']}/{g['pairs']} nominal; "
                      f"{g['lmm_bonferroni']} Bonferroni; {g['lmm_bonferroni_eligible_new']} Bonferroni + new cell gate")


def run_pyseer(argv=None):
    parser = argparse.ArgumentParser(prog="amr_checks.py pyseer", description="Compare saved KOVAR pairs with the installed pyseer LMM.")
    parser.add_argument('--diagnosis', type=Path, required=True, help='Completed AMR diagnostic output directory')
    parser.add_argument('--work', type=Path, help='Original .kovar_work; only patterns.bin and K.npy are read')
    parser.add_argument('--out', type=Path, help='Default: sibling amr_pyseer_lmm directory')
    parser.add_argument('--threads', type=int, default=4, help='Independent pyseer response workers; BLAS=1 each')
    args = parser.parse_args(argv)
    args.diagnosis = args.diagnosis.resolve()
    args.out = args.out.resolve() if args.out else args.diagnosis.parent/'amr_pyseer_lmm'
    if args.threads < 1:
        parser.error('--threads must be positive')
    environment = pyseer_environment()
    rows, identity = prepare_pyseer(args, environment)
    grouped = defaultdict(list)
    for r in rows:
        u, v = int(r['predictor_pattern']), int(r['response_pattern'])
        if u != v:
            grouped[v].append(u)
    pending, resumed = [], 0
    for response, predictors in sorted(grouped.items()):
        predictors.sort()
        path = args.out/'responses'/f'{response:09d}.json'
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['response'] != response or saved['predictors'] != predictors or len(saved['results']) != len(predictors):
                raise ValueError(f'Unexpected response checkpoint: {path}')
            resumed += 1
        else:
            pending.append((response, predictors))
    log_pyseer(args.out, f"{len(rows):,} same-orientation pairs; {len(identity['samples'])} isolates; "
                  f"{resumed}/{len(grouped)} response models reused; {len(pending)} to fit; workers={args.threads}")
    if pending:
        if args.threads == 1:
            initialize_pyseer(str(args.out), identity)
            for i, job in enumerate(pending, 1):
                fit_response(job)
                log_pyseer(args.out, f'response models saved: {resumed+i}/{len(grouped)}')
        else:
            with ProcessPoolExecutor(max_workers=min(args.threads, len(pending)), initializer=initialize_pyseer,
                                     initargs=(str(args.out), identity)) as pool:
                futures = [pool.submit(fit_response, job) for job in pending]
                for i, future in enumerate(as_completed(futures), 1):
                    future.result()
                    log_pyseer(args.out, f'response models saved: {resumed+i}/{len(grouped)}')
    export_pyseer(args, rows, identity, resumed)
    log_pyseer(args.out, 'complete: ' + str(args.out/'pyseer_comparison.tsv'))


# Optional assembly-level carriage comparison.
EMPTY = {'', '-', 'na', 'n/a', 'none', 'unknown', 'untypable', 'untypeable', 'unclassified', 'not determined'}
LIMITS = ['--min-targets-pident', '90', '--min-targets-coverage', '80',
          '--min-regions-pident', '85', '--min-regions-coverage', '83']


def interpret(report, details):
    """Panel-negative is distinct from a mec-positive, untyped cassette.

    Multiple supported cassette types can still establish presence. mec/ccr
    evidence without a supported complete type remains uncertain, not absent.
    """
    mec = report.get('mecA', '')
    if mec not in {'+', '-'}:
        raise ValueError('Unexpected sccmec mecA field: ' + mec)
    targets = {x.strip() for r in [report] + details for x in r.get('targets', '').split(',') if x.strip()}
    support = any(report.get(k, '').strip().lower() not in EMPTY for k in ('type', 'subtype', 'regions'))
    relevant = any(x.lower().startswith(('mec', 'ccr')) for x in targets)
    if mec == '+' and support:
        call, status = 1, 'SUPPORTED_PRESENT'
    elif mec == '+' or support or relevant:
        call, status = None, 'UNCERTAIN_CASSETTE'
    else:
        call, status = 0, 'NOT_DETECTED_BY_PANEL'
    return dict(mecA_present=int(mec == '+'), SCCmec_present=call, carriage_status=status,
                SCCmec_type=report.get('type', ''), SCCmec_subtype=report.get('subtype', ''),
                detected_targets=','.join(sorted(targets)), cassette_coverage=report.get('coverage', ''),
                tool_comment=report.get('comment', ''))


def input_index(root, extensions):
    index = defaultdict(list)
    if root and root.is_dir():
        for p in root.rglob('*'):
            if p.is_file():
                for suffix in extensions:
                    if p.name.lower().endswith(suffix):
                        index[p.name[:-len(suffix)]].append(p.resolve())
                        break
    return index


def genome_sources(samples, assemblies, gff_dir):
    fasta = input_index(assemblies, ('.fasta.gz', '.fna.gz', '.fas.gz', '.fa.gz', '.fasta', '.fna', '.fas', '.fa'))
    gffs = input_index(gff_dir, ('.gff3.gz', '.gff.gz', '.gff3', '.gff'))
    sources = {}
    for sample in samples:
        paths, kind = (fasta[sample], 'assembly') if fasta[sample] else (gffs[sample], 'embedded_gff_fasta')
        paths = sorted(set(paths))
        if len(paths) > 1:
            raise ValueError(f'Multiple input genomes for {sample}: {paths}; choose one source directory')
        sources[sample] = (paths[0], kind) if paths else (None, 'missing')
    return sources


def extract_fasta(gff, output):
    opener = gzip.open if str(gff).endswith('.gz') else open
    found = False
    with opener(gff, 'rt', encoding='utf-8') as source, output.open('w', encoding='utf-8') as dest:
        for line in source:
            if found:
                dest.write(line)
            elif line.strip() == '##FASTA':
                found = True
    if not found or output.stat().st_size == 0:
        raise ValueError(f'No embedded FASTA in {gff}; provide its original assembly')


def classify(job):
    index, sample, source, kind, out, executable, env = job
    folder = out/'isolates'/f'{index:06d}'
    folder.mkdir(parents=True, exist_ok=True)
    result = dict(sample=sample, source=str(source or ''), source_kind=kind,
                  mecA_present=None, SCCmec_present=None, typing_status='FAILED', carriage_status='UNKNOWN')
    if source is None:
        return dict(result, error='No exact sample-ID assembly/GFF filename found')
    signature = dict(source_sha256=digest(source), kind=kind)
    done = folder/'completed.json'
    if done.exists():
        previous = json.loads(done.read_text())
        if previous['input'] == signature:
            return dict(previous['result'], source=str(source), reused=True)
    temporary = folder/'embedded_input.fna'
    try:
        assembly = source
        if kind == 'embedded_gff_fasta':
            extract_fasta(source, temporary)
            assembly = temporary
        command = [executable, '--input', str(assembly), '--prefix', 'isolate',
                   '--outdir', str(folder), '--force', *LIMITS]
        with (folder/'sccmec.log').open('w', encoding='utf-8') as stream:
            subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        reports = read_table(folder/'isolate.tsv')
        if len(reports) != 1 or reports[0].get('sample') != 'isolate':
            raise ValueError('Expected one completed sccmec isolate report')
        details = read_table(folder/'isolate.targets.details.tsv')
        result.update(interpret(reports[0], details), typing_status='OK', error='',
                      source_sha256=signature['source_sha256'], report=str(folder/'isolate.tsv'))
        atomic_json(done, dict(input=signature, result=result))
        return dict(result, reused=False)
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        return dict(result, error=str(error) + f'; see {folder}/sccmec.log')
    finally:
        temporary.unlink(missing_ok=True)


def frozen_inputs(args):
    state = json.loads((args.diagnosis/'diagnosis_identity.json').read_text())
    info, source = state['prepared'], state['source_identity']
    if not args.work:
        paths = [args.diagnosis.parent.parent/'discovery/results/.kovar_work',
                 args.diagnosis.parent.parent/'discovery/.kovar_work', Path(source.get('work_directory', '/not-found'))]
        args.work = next((p for p in paths if (p/'patterns.bin').is_file() and (p/'K.npy').is_file()), None)
    if not args.work:
        raise ValueError('Set --work to the original .kovar_work containing patterns.bin and K.npy')
    args.work = args.work.resolve()
    for path in (args.diagnosis, args.work):
        # Paths are resolved above; parents membership also works on Python 3.8.
        if args.out == path or args.out in path.parents or path in args.out.parents:
            raise ValueError('--out must be separate from the source diagnostic and work directories')
    for name in ('patterns.bin', 'K.npy'):
        if digest(args.work/name) != info['products'][name]:
            raise ValueError(f'Frozen {name} differs from the completed diagnostic')
    mapping = read_table(args.diagnosis/'target_unitigs.tsv')
    evidence = defaultdict(lambda: dict(targets=set(), unitigs=set(), clusters=set()))
    mec_ids = set()
    for r in mapping:
        if r['annotation_class'] != 'coding' or not r['pattern'] or int(r['n_matching_references_binary_absent'] or 0):
            continue
        pid = int(r['pattern'])
        if r['target'] == 'mecA':
            mec_ids.add(pid)
        if r['target'] not in {'graR', 'sen'}:
            continue
        e = evidence[pid]
        e['targets'].add(r['target']); e['unitigs'].add(r['unitig'])
        e['clusters'].update(r['clusters'].split(';'))
    if not evidence:
        raise ValueError('No retained coding graR/sen patterns in target_unitigs.tsv')
    ids = sorted(set(evidence) | mec_ids)
    if ids[0] < 0 or ids[-1] >= info['patterns']:
        raise ValueError('Mapping pattern ID is outside the frozen pool')
    packed = np.memmap(args.work/'patterns.bin', mode='r', dtype='u1', shape=(info['patterns'], info['packed_width']))
    matrix = np.unpackbits(np.asarray(packed[ids]), axis=1, bitorder='little')[:, :info['n']]
    K = np.load(args.work/'K.npy', allow_pickle=False)
    if len(info['samples']) != info['n'] or K.shape != (info['n'], info['n']) or not np.isfinite(K).all():
        raise ValueError('Frozen cohort/covariance dimensions or values are invalid')
    return info, source, evidence, {pid: matrix[i] for i, pid in enumerate(ids)}, mec_ids, K


def phenotype_fit(name, calls, info, source, evidence, vectors, K, args, call_digest):
    """Fit one response null and score all eligible predictors with actual pyseer."""
    import pandas as pd
    from pyseer.lmm import initialise_lmm, fit_lmm_block
    field = 'SCCmec_present' if name == 'SCCmec' else 'mecA_present'
    indices = [i for i, r in enumerate(calls) if r[field] is not None]
    samples = [info['samples'][i] for i in indices]
    y = np.array([calls[i][field] for i in indices], dtype=float)
    folder = args.out/name
    folder.mkdir(exist_ok=True)
    table(folder/'phenotype.tsv', ({'sample': s, 'carriage': int(v)} for s, v in zip(samples, y)), ['sample', 'carriage'])
    table(folder/'excluded_samples.tsv',
                 ({'sample': r['sample'], 'reason': r['carriage_status'] if name == 'SCCmec' else r['typing_status'],
                   'error': r.get('error', '')} for r in calls if r[field] is None), ['sample', 'reason', 'error'])
    table(folder/'sample_order.tsv', ({'row': i, 'sample': s} for i, s in enumerate(samples)), ['row', 'sample'])
    single_cut = args.alpha / info['patterns']
    pair_family = int(source['settings'].get('bonferroni_tests') or math.comb(info['patterns'], 2))
    pair_cut = args.alpha / pair_family
    key = dict(calls_sha256=call_digest, indices=indices, response=name, predictor_ids=sorted(evidence))
    checkpoint = folder/'completed.json'
    if checkpoint.exists():
        record = json.loads(checkpoint.read_text())
        if record['input'] == key:
            log_pyseer(args.out, name + ': reusing completed LMM results')
            return record['summary']
    n = len(y)
    testable = n > 3 and len(np.unique(y)) == 2
    h2 = None
    if testable:
        covariance = K[np.ix_(indices, indices)]
        pd.DataFrame(covariance, index=samples, columns=samples).to_csv(folder/'covariance.tsv', sep='\t', float_format='%.17g')
        log_pyseer(args.out, f'{name}: fixed response; {n} isolates; {int(y.sum())} positive; {n-int(y.sum())} negative')
        p, model, h2 = initialise_lmm(pd.Series(y, index=samples), pd.DataFrame(), str(folder/'covariance.tsv'))
        if list(p.index) != samples or not np.array_equal(p.to_numpy(), y):
            raise ValueError('Pyseer changed sample order or dropped samples')
        h2 = float(h2)
        if not math.isfinite(h2) or not 0 <= h2 <= 1:
            raise ValueError('Invalid fitted pyseer LMM h2')
    output, eligible = [], []
    for pid, e in sorted(evidence.items()):
        x = np.array(vectors[pid][indices], dtype=float)
        n11 = int(x @ y); n10 = int(x.sum()) - n11; n01 = int(y.sum()) - n11
        n00 = n - n11 - n10 - n01
        af = float(x.mean()) if n else None
        status = 'RESPONSE_NOT_TESTABLE' if not testable else 'MONOMORPHIC_PREDICTOR' if len(np.unique(x)) < 2 else\
                 'AF_FILTERED' if not args.min_af <= af <= args.max_af else 'PENDING'
        row = dict(response=name, targets=';'.join(sorted(e['targets'])), pattern_id=pid,
                   unitigs=';'.join(sorted(e['unitigs'], key=int)), clusters=';'.join(sorted(e['clusters'])),
                   n=n, n00=n00, n10=n10, n01=n01, n11=n11, min_cell=min(n00, n10, n01, n11), af=af,
                   p_lmm=None, beta_lmm=None, se_lmm=None, h2=h2, status=status,
                   h2_at_upper=int(h2 >= .99999-1e-10) if h2 is not None else None,
                   p_cutoff_focal_genomewide=single_cut, p_cutoff_all_pairs=pair_cut,
                   nominal=None, significant_focal_genomewide=None, significant_all_pairs=None)
        output.append(row)
        if status == 'PENDING':
            eligible.append((row, x))
    for first in range(0, len(eligible), 256):
        block = eligible[first:first+256]
        result = fit_lmm_block(model, h2, np.column_stack([x for r, x in block]))
        for i, (r, _) in enumerate(block):
            pv, beta, se = (float(result[k][i]) for k in ('p_values', 'beta', 'bse'))
            if not all(math.isfinite(v) for v in (pv, beta, se)) or not 0 <= pv <= 1 or se <= 0:
                r['status'] = 'INVALID_LMM_RESULT'
            else:
                r.update(p_lmm=pv, beta_lmm=beta, se_lmm=se, status='OK', nominal=int(pv < .05),
                         significant_focal_genomewide=int(pv <= single_cut), significant_all_pairs=int(pv <= pair_cut))
    table(folder/'associations.tsv', output, list(output[0]))
    summary = dict(response=name, n=n, positive=int(y.sum()), negative=n-int(y.sum()), excluded=info['n']-n,
                   h2=h2, h2_at_upper=int(h2 >= .99999-1e-10) if h2 is not None else None,
                   model_status='FITTED' if testable else 'RESPONSE_NOT_TESTABLE',
                   status_counts=dict(Counter(r['status'] for r in output)),
                   focal_genomewide_family=info['patterns'], focal_genomewide_cutoff=single_cut,
                   all_pairs_family=pair_family, all_pairs_cutoff=pair_cut, targets={})
    for target in ('graR', 'sen'):
        rs = [r for r in output if target in r['targets'].split(';')]
        ps = [r['p_lmm'] for r in rs if r['p_lmm'] is not None]
        summary['targets'][target] = dict(patterns=len(rs), tested=len(ps), lowest_p=min(ps) if ps else None,
                                         nominal=sum(r['nominal'] or 0 for r in rs),
                                         significant_focal_genomewide=sum(r['significant_focal_genomewide'] or 0 for r in rs),
                                         significant_all_pairs=sum(r['significant_all_pairs'] or 0 for r in rs))
    atomic_json(checkpoint, dict(input=key, summary=summary))
    log_pyseer(args.out, name + ': ' + json.dumps(summary['targets']))
    return summary


def run_sccmec(argv=None):
    p = argparse.ArgumentParser(prog="amr_checks.py sccmec", description="Optional assembly-carriage pyseer comparison.")
    p.add_argument('--diagnosis', type=Path, required=True)
    p.add_argument('--work', type=Path)
    p.add_argument('--assemblies', type=Path, help='Recursively searched for exact sample-ID FASTA filenames')
    p.add_argument('--gff-dir', type=Path, help='Fallback: matching GFFs containing ##FASTA; no reannotation')
    p.add_argument('--sccmec', default='sccmec', help='Executable or absolute sccmec path from its Conda environment')
    p.add_argument('--out', type=Path, help='Default: sibling sccmec_pyseer directory')
    p.add_argument('--threads', type=int, default=4, help='Concurrent assembly typing jobs; BLAS=1')
    p.add_argument('--min-af', type=float, default=.05)
    p.add_argument('--max-af', type=float, default=.99)
    p.add_argument('--alpha', type=float, default=.05)
    args = p.parse_args(argv)
    args.diagnosis = args.diagnosis.resolve()
    args.out = args.out.resolve() if args.out else args.diagnosis.parent/'sccmec_pyseer'
    if args.threads < 1 or not 0 <= args.min_af < args.max_af <= 1 or not 0 < args.alpha < 1:
        p.error('Positive threads, ordered AF bounds in [0,1], and alpha in (0,1) required')
    environment = pyseer_environment()  # Fail before assembly typing if the LMM environment is missing.
    executable = shutil.which(args.sccmec)
    if not executable:
        p.error('sccmec not found; install it or set --sccmec /path/to/sccmec')
    executable = str(Path(executable).absolute())
    env = dict(os.environ, PATH=str(Path(executable).parent) + os.pathsep + os.environ.get('PATH', ''))
    if not shutil.which('blastn', path=env['PATH']):
        p.error('blastn is missing from the sccmec environment PATH')
    version = subprocess.run([executable, '--version'], env=env, capture_output=True, text=True, check=True)
    info, source, evidence, vectors, mec_ids, K = frozen_inputs(args)
    sources = genome_sources(info['samples'], args.assemblies, args.gff_dir)
    if not any(path for path, kind in sources.values()):
        p.error('No frozen sample IDs matched assemblies or embedded-FASTA GFFs')
    identity = dict(revision=1, environment=environment, sccmec_version=version.stdout.strip()+version.stderr.strip(),
                    sccmec_thresholds=LIMITS, diagnostic_identity_sha256=digest(args.diagnosis/'diagnosis_identity.json'),
                    mapping_sha256=digest(args.diagnosis/'target_unitigs.tsv'),
                    products={k: info['products'][k] for k in ('patterns.bin', 'K.npy')},
                    samples=info['samples'], predictors=sorted(evidence), alpha=args.alpha,
                    min_af=args.min_af, max_af=args.max_af)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out/'run_identity.json'
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError('Model, tool/database version, frozen inputs or thresholds changed; use a separate --out')
    if not manifest.exists() and any(args.out.iterdir()):
        raise ValueError('Nonempty output is not an existing SCCmec comparison')
    atomic_json(manifest, identity)
    log_pyseer(args.out, f'{info["n"]} frozen isolates; {len(evidence)} graR/sen predictors; typing workers={args.threads}')
    started = time.monotonic()
    jobs = [(i, s, *sources[s], args.out, executable, env) for i, s in enumerate(info['samples'])]
    calls_by_sample = {}
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        futures = [pool.submit(classify, job) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            r = future.result(); calls_by_sample[r['sample']] = r
            if i % 10 == 0 or i == len(jobs):
                log_pyseer(args.out, f'typing: {i}/{len(jobs)}; reused={sum(r.get("reused", False) for r in calls_by_sample.values())}; '
                                      f'failed={sum(r["typing_status"]!="OK" for r in calls_by_sample.values())}')
    calls = [calls_by_sample[s] for s in info['samples']]
    # Reuse flags are runtime metadata, not phenotype inputs, so exclude them from this stable table.
    table(args.out/'carriage.tsv', calls, ['sample', 'source', 'source_kind', 'source_sha256', 'typing_status',
                 'mecA_present', 'SCCmec_present', 'carriage_status', 'SCCmec_type', 'SCCmec_subtype',
                 'detected_targets', 'cassette_coverage', 'tool_comment', 'error', 'report'])
    if not any(r['typing_status'] == 'OK' for r in calls):
        raise ValueError('No successful SCCmec calls; inspect carriage.tsv and isolates/*/sccmec.log; no LMM was run')
    agreement = []
    known = [i for i, r in enumerate(calls) if r['mecA_present'] is not None]
    gene = np.array([calls[i]['mecA_present'] for i in known], dtype=int)
    for pid in sorted(mec_ids):
        x = vectors[pid][known]
        agreement.append(dict(pattern_id=pid, n=len(known), mecA_positive=int(gene.sum()), unitig_positive=int(x.sum()),
                              both_positive=int(x @ gene), gene_positive_unitig_absent=int(((gene == 1) & (x == 0)).sum()),
                              gene_negative_unitig_present=int(((gene == 0) & (x == 1)).sum()),
                              agreement_fraction=float((x == gene).mean()) if len(gene) else None))
    table(args.out/'meca_unitig_agreement.tsv', agreement, ['pattern_id', 'n', 'mecA_positive', 'unitig_positive',
                 'both_positive', 'gene_positive_unitig_absent', 'gene_negative_unitig_present', 'agreement_fraction'])
    summaries = [phenotype_fit(name, calls, info, source, evidence, vectors, K, args,
                              digest(args.out/'carriage.tsv')) for name in ('SCCmec', 'mecA')]
    flat = [dict(response=s['response'], target=target, n=s['n'], positive=s['positive'], negative=s['negative'],
                 excluded=s['excluded'], h2=s['h2'], h2_at_upper=s['h2_at_upper'], **values)
            for s in summaries for target, values in s['targets'].items()]
    table(args.out/'summary.tsv', flat, list(flat[0]))
    atomic_json(args.out/'summary.json', dict(typing_statuses=dict(Counter(r['typing_status'] for r in calls)),
                       carriage_statuses=dict(Counter(r['carriage_status'] for r in calls)), phenotypes=summaries,
                       typing_elapsed_and_analysis_seconds=time.monotonic()-started, source_modified=False,
                       interpretation='SCCmec is primary; mecA is a separate sensitivity response. Calls use the installed '
                       'SCCmec reference panel; negative means not detected at recorded thresholds, not proven absence. '
                       'Uncertain/failed calls are excluded with named reasons and matching covariance subsets. '
                       'Targets are unitig sequence variants, not whole-gene graR/sen absence. '
                       'Focal genome-wide correction uses all frozen patterns, not only these selected targets. '
                       'This is not a full reproduction of the original 806-isolate k=31 GOLD-GWAS analysis.'))
    log_pyseer(args.out, 'complete: ' + str(args.out/'summary.tsv'))



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name, description in (
            ('kovar', 'Existing target mapping, KOVAR scoring and report recovery'),
            ('pyseer', 'Matched pyseer LMM tests of the saved target patterns'),
            ('sccmec', 'Optional assembly-carriage pyseer tests')):
        sub.add_parser(name, help=description, add_help=False)
    args, remaining = parser.parse_known_args(argv)
    return {'kovar': run_kovar, 'pyseer': run_pyseer, 'sccmec': run_sccmec}[args.command](remaining)


if __name__ == '__main__':
    main()
