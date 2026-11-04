"""Small-panel production benchmark; never enumerate a full genome by default.

Supply a small pseudo-FASTA panel with unique sample IDs and the matching tree.
This compares the release code paths, not a separate scoring implementation.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import sqlite3

from ko_variation.discovery import main,identity,parser as discovery_parser
from ko_variation.discovery_store import records, atomic_json, digest, open_patterns
from ko_variation.memory_budget import detect_budget


def prepare_panel(source, out, count, seed):
    """Freeze a genotype-only panel from existing immutable preparation.

    Selection does not inspect significance, phenotype or network membership.
    Sampling is over unique patterns, not over duplicated unitig columns.
    """
    import numpy as np
    source = source.resolve()
    if not (source/'run_identity.json').exists():
        source /= 'results'
    identity = json.loads((source/'run_identity.json').read_text())
    work = Path(identity.get('work_directory',source/'.kovar_work'))
    info = json.loads((work/'prepared.json').read_text())
    tree = Path(identity['settings']['tree'])
    if count < 3 or count > info['patterns']:
        raise ValueError('Panel size must be 3..number of prepared unique patterns')
    if digest(work/'patterns.bin') != info['products']['patterns.bin']:
        raise ValueError('Source prepared patterns changed')
    selected = np.sort(np.random.default_rng(seed).choice(info['patterns'],count,replace=False))
    frozen = dict(pattern_product=info['products']['patterns.bin'], tree_sha256=digest(tree),
                  selected_patterns=selected.tolist(), samples=info['samples'], seed=seed)
    directory = out/'panel_inputs'
    directory.mkdir(parents=True,exist_ok=True)
    fasta, target_tree = directory/'panel.fasta', directory/'tree.nwk'
    manifest = directory/'panel.json'
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved['identity'] != frozen or saved['fasta_sha256'] != digest(fasta) or \
                frozen['tree_sha256'] != digest(target_tree):
            raise ValueError('Frozen panel changed; use a new benchmark output')
    else:
        packed = open_patterns(work,info)
        X = np.unpackbits(packed[selected],axis=1,bitorder='little')[:,:info['n']].T
        temporary = fasta.with_suffix('.partial')
        with temporary.open('w') as stream:
            for sample, row in zip(info['samples'],X):
                stream.write('>'+sample+'\n'+''.join('C' if b else 'A' for b in row)+'\n')
        temporary.replace(fasta)
        shutil.copyfile(tree,target_tree)
        atomic_json(manifest,dict(identity=frozen,fasta_sha256=digest(fasta)))
    return fasta,target_tree,math.comb(info['patterns'],2)


def benchmark():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fasta",type=Path)
    p.add_argument("--tree",type=Path)
    p.add_argument('--source-run',type=Path,help='Existing discovery run; sample a frozen unique-pattern panel')
    p.add_argument('--patterns',type=int,default=1000,help='Unique patterns when using --source-run [1000]')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--memory-gb",type=float,default=None)
    p.add_argument("--min-cell-count",type=int,default=20)
    p.add_argument("--max-loci",type=int,default=2000)
    p.add_argument("--screen-p",type=float,default=.001)
    p.add_argument("--bonferroni-tests",type=int,help="Freeze the full study's hypothesis family for sampled panels")
    args=p.parse_args()
    args.out = args.out.resolve()
    if args.source_run:
        if args.fasta or args.tree or args.patterns > args.max_loci:
            p.error('--source-run replaces --fasta/--tree; --patterns cannot exceed --max-loci')
        args.fasta,args.tree,family = prepare_panel(args.source_run,args.out,args.patterns,args.seed)
        if args.bonferroni_tests is None:
            args.bonferroni_tests = family
    elif not args.fasta or not args.tree:
        p.error('Supply either --source-run or both --fasta and --tree')
    args.fasta,args.tree = args.fasta.resolve(),args.tree.resolve()
    _,sequence=next(records(args.fasta))
    if len(sequence)>args.max_loci:
        p.error("Input exceeds --max-loci; use a frozen small panel, not the full genome")
    args.out.mkdir(parents=True,exist_ok=True)
    common=["--fasta",str(args.fasta),"--tree",str(args.tree),"--threads",str(args.threads),
            "--min-cell-count",str(args.min_cell_count),"--screen-p",str(args.screen_p)]
    if args.memory_gb is not None:
        common += ['--memory-gb',str(args.memory_gb)]
    if args.bonferroni_tests is not None:
        common += ['--bonferroni-tests',str(args.bonferroni_tests)]
    results=[]
    reference=set()
    for name,options in (("exact",["--search","exhaustive","--exact"]),
                         ("bounds",["--search","exhaustive"]),
                         ("hierarchy_bounds",['--coarse-screen','none']),
                         ("hierarchy_pc10",['--coarse-screen','pc'])):
        out=args.out/name
        command=common+["--out",str(out)]+options
        if (out/".kovar_work"/"run_identity.json").exists():
            command += ["--resume"]
        if name!="exact":
            command += ["--reuse-nulls-from",str(args.out/"exact")]
        if name.startswith('hierarchy_'):
            command += ['--reuse-prepared-from',str(args.out/'exact')]
        saved=out/'run_summary.json'
        if saved.exists():
            summary=json.loads(saved.read_text())
            old_identity=json.loads((out/'.kovar_work'/'run_identity.json').read_text())
            current=identity(discovery_parser().parse_args(command))
            budget=detect_budget(args.memory_gb,args.threads)['working_gb']
            if old_identity!=current or summary['threads']!=args.threads or abs(summary['memory_gb']-budget)>1e-6:
                raise ValueError('Completed benchmark uses different code/settings/resources; choose a new --out')
            # Do not replace a completed method's measured time with the near-
            # zero time of simply reopening and exporting its checkpoint.
        else:
            main(command)
            summary=json.loads(saved.read_text())
        db=sqlite3.connect(out/".kovar_work"/"work.sqlite")
        hits=set(db.execute("SELECT u,v FROM decisions WHERE final=1"))
        db.close()
        if name=="exact":
            reference=hits
        metrics=summary.get('runtime_totals',{})
        prefit=summary.get('prefit_runtime_totals',{})
        screened=summary.get('coarse_screen') or {}
        gate_total=gate_retained=None
        if name=='hierarchy_pc10':
            import numpy as np
            db=sqlite3.connect(out/'.kovar_work'/'work.sqlite')
            lookup=dict(db.execute('SELECT node,pid FROM reps'));db.close()
            roots=np.load(out/'.kovar_work'/'coarse_screen'/'roots.npy')
            pairs=np.fromfile(out/'.kovar_work'/'coarse_screen'/'candidates.u32',dtype='<u4').reshape(-1,2)
            size=summary['unique_patterns']
            retained={min(lookup[int(roots[a])],lookup[int(roots[b])])*size+
                      max(lookup[int(roots[a])],lookup[int(roots[b])]) for a,b in pairs}
            db=sqlite3.connect(args.out/'hierarchy_bounds'/'.kovar_work'/'work.sqlite')
            gates=db.execute("WITH roots(pid) AS (SELECT r.pid FROM reps r JOIN nodes n ON n.id=r.node WHERE n.parent=0) "
                "SELECT u,v FROM decisions WHERE gate!=-1 AND status!='INELIGIBLE_NOT_SCREENED' "
                "AND u IN roots AND v IN roots")
            gate_total=gate_retained=0
            for u,v in gates:
                gate_total+=1;gate_retained+=int(u*size+v in retained)
            db.close()
        results.append(dict(method=name,seconds=summary["seconds_this_invocation"],
                            coarse_pairs_before_screen=screened.get('total_pairs'),
                            coarse_pairs_after_screen=screened.get('selected_pairs'),
                            coarse_screen_seconds=screened.get('elapsed_total_seconds'),
                            coarse_gate_or_unresolved_reference=gate_total,
                            coarse_gate_or_unresolved_recall=gate_retained/gate_total if gate_total else 'NA',
                            queries=summary["queried_pairs"],reference_hits=len(reference),hits=len(hits),
                            missed=len(reference-hits),extra=len(hits-reference),
                            recall=len(reference&hits)/len(reference) if reference else "NA",
                            null_fit_worker_seconds=metrics.get('null_fit_worker_seconds',0),
                            prefit_null_fit_worker_seconds=prefit.get('null_fit_worker_seconds',0),
                            prefit_cache_save_worker_seconds=prefit.get('cache_save_worker_seconds',0),
                            reconstructed_spectra=metrics.get('cache_reconstructions',0),
                            cache_load_worker_seconds=metrics.get('cache_load_worker_seconds',0),
                            bounds_worker_seconds=metrics.get('bounds_worker_seconds',0),
                            exact_worker_seconds=metrics.get('exact_worker_seconds',0),
                            coordinator_db_read_seconds=metrics.get('driver_db_read_seconds',0),
                            coordinator_db_write_seconds=metrics.get('driver_db_write_seconds',0),
                            jobs=metrics.get('jobs',0),
                            mean_pairs_per_job=metrics.get('pairs',0)/max(1,metrics.get('jobs',0))))
    with (args.out/"comparison.tsv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(results[0]),delimiter="\t",lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)


if __name__=="__main__":
    benchmark()
