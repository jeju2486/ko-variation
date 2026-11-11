#!/usr/bin/env python3
"""Call assembly-level SCCmec carriage, then test graR/sen unitigs with pyseer.

SCCmec (primary) and mecA (sensitivity) are fixed responses, never unitig IDs.
The completed target diagnosis supplies coding annotations, sample order and
frozen KOVAR patterns/K. Source SQLite databases are never opened or modified.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

if __package__:
    from . import diagnose_amr_pyseer as common
else:
    import diagnose_amr_pyseer as common

import numpy as np

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
    signature = dict(source_sha256=common.digest(source), kind=kind)
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
        reports = common.read_table(folder/'isolate.tsv')
        if len(reports) != 1 or reports[0].get('sample') != 'isolate':
            raise ValueError('Expected one completed sccmec isolate report')
        details = common.read_table(folder/'isolate.targets.details.tsv')
        result.update(interpret(reports[0], details), typing_status='OK', error='',
                      source_sha256=signature['source_sha256'], report=str(folder/'isolate.tsv'))
        common.atomic_json(done, dict(input=signature, result=result))
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
        if common.digest(args.work/name) != info['products'][name]:
            raise ValueError(f'Frozen {name} differs from the completed diagnostic')
    mapping = common.read_table(args.diagnosis/'target_unitigs.tsv')
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
    common.table(folder/'phenotype.tsv', ({'sample': s, 'carriage': int(v)} for s, v in zip(samples, y)), ['sample', 'carriage'])
    common.table(folder/'excluded_samples.tsv',
                 ({'sample': r['sample'], 'reason': r['carriage_status'] if name == 'SCCmec' else r['typing_status'],
                   'error': r.get('error', '')} for r in calls if r[field] is None), ['sample', 'reason', 'error'])
    common.table(folder/'sample_order.tsv', ({'row': i, 'sample': s} for i, s in enumerate(samples)), ['row', 'sample'])
    single_cut = args.alpha / info['patterns']
    pair_family = int(source['settings'].get('bonferroni_tests') or math.comb(info['patterns'], 2))
    pair_cut = args.alpha / pair_family
    key = dict(calls_sha256=call_digest, indices=indices, response=name, predictor_ids=sorted(evidence))
    checkpoint = folder/'completed.json'
    if checkpoint.exists():
        record = json.loads(checkpoint.read_text())
        if record['input'] == key:
            common.log(args.out, name + ': reusing completed LMM results')
            return record['summary']
    n = len(y)
    testable = n > 3 and len(np.unique(y)) == 2
    h2 = None
    if testable:
        covariance = K[np.ix_(indices, indices)]
        pd.DataFrame(covariance, index=samples, columns=samples).to_csv(folder/'covariance.tsv', sep='\t', float_format='%.17g')
        common.log(args.out, f'{name}: fixed response; {n} isolates; {int(y.sum())} positive; {n-int(y.sum())} negative')
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
        status = 'RESPONSE_NOT_TESTABLE' if not testable else 'MONOMORPHIC_PREDICTOR' if len(np.unique(x)) < 2 else \
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
    common.table(folder/'associations.tsv', output, list(output[0]))
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
    common.atomic_json(checkpoint, dict(input=key, summary=summary))
    common.log(args.out, name + ': ' + json.dumps(summary['targets']))
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
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
    environment = common.pyseer_environment()  # Fail before assembly typing if the LMM environment is missing.
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
                    sccmec_thresholds=LIMITS, diagnostic_identity_sha256=common.digest(args.diagnosis/'diagnosis_identity.json'),
                    mapping_sha256=common.digest(args.diagnosis/'target_unitigs.tsv'),
                    products={k: info['products'][k] for k in ('patterns.bin', 'K.npy')},
                    samples=info['samples'], predictors=sorted(evidence), alpha=args.alpha,
                    min_af=args.min_af, max_af=args.max_af)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out/'run_identity.json'
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError('Model, tool/database version, frozen inputs or thresholds changed; use a separate --out')
    if not manifest.exists() and any(args.out.iterdir()):
        raise ValueError('Nonempty output is not an existing SCCmec comparison')
    common.atomic_json(manifest, identity)
    common.log(args.out, f'{info["n"]} frozen isolates; {len(evidence)} graR/sen predictors; typing workers={args.threads}')
    started = time.monotonic()
    jobs = [(i, s, *sources[s], args.out, executable, env) for i, s in enumerate(info['samples'])]
    calls_by_sample = {}
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        futures = [pool.submit(classify, job) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            r = future.result(); calls_by_sample[r['sample']] = r
            if i % 10 == 0 or i == len(jobs):
                common.log(args.out, f'typing: {i}/{len(jobs)}; reused={sum(r.get("reused", False) for r in calls_by_sample.values())}; '
                                      f'failed={sum(r["typing_status"]!="OK" for r in calls_by_sample.values())}')
    calls = [calls_by_sample[s] for s in info['samples']]
    # Reuse flags are runtime metadata, not phenotype inputs, so exclude them from this stable table.
    common.table(args.out/'carriage.tsv', calls, ['sample', 'source', 'source_kind', 'source_sha256', 'typing_status',
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
    common.table(args.out/'meca_unitig_agreement.tsv', agreement, ['pattern_id', 'n', 'mecA_positive', 'unitig_positive',
                 'both_positive', 'gene_positive_unitig_absent', 'gene_negative_unitig_present', 'agreement_fraction'])
    summaries = [phenotype_fit(name, calls, info, source, evidence, vectors, K, args,
                              common.digest(args.out/'carriage.tsv')) for name in ('SCCmec', 'mecA')]
    flat = [dict(response=s['response'], target=target, n=s['n'], positive=s['positive'], negative=s['negative'],
                 excluded=s['excluded'], h2=s['h2'], h2_at_upper=s['h2_at_upper'], **values)
            for s in summaries for target, values in s['targets'].items()]
    common.table(args.out/'summary.tsv', flat, list(flat[0]))
    common.atomic_json(args.out/'summary.json', dict(typing_statuses=dict(Counter(r['typing_status'] for r in calls)),
                       carriage_statuses=dict(Counter(r['carriage_status'] for r in calls)), phenotypes=summaries,
                       typing_elapsed_and_analysis_seconds=time.monotonic()-started, source_modified=False,
                       interpretation='SCCmec is primary; mecA is a separate sensitivity response. Calls use the installed '
                       'SCCmec reference panel; negative means not detected at recorded thresholds, not proven absence. '
                       'Uncertain/failed calls are excluded with named reasons and matching covariance subsets. '
                       'Targets are unitig sequence variants, not whole-gene graR/sen absence. '
                       'Focal genome-wide correction uses all frozen patterns, not only these selected targets. '
                       'This is not a full reproduction of the original 806-isolate k=31 GOLD-GWAS analysis.'))
    common.log(args.out, 'complete: ' + str(args.out/'summary.tsv'))


if __name__ == '__main__':
    main()
