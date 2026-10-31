#!/usr/bin/env python3
"""Read-only runtime audit of a running v0.9 discovery output; standard library only.

Does not import KOVAR, load large model arrays, alter checkpoints or fit models.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sqlite3
import statistics


def quantile(values,fraction):
    if not values:
        return None
    position=(len(values)-1)*fraction
    lo=int(position)
    hi=min(lo+1,len(values)-1)
    return values[lo]+(position-lo)*(values[hi]-values[lo])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--log',type=Path,help='Current Slurm .out log, preferably the live server copy')
    parser.add_argument('--cpus',type=int,default=24)
    parser.add_argument('--detailed',action='store_true',help='Scan decision/task tables; can be expensive on very large runs')
    parser.add_argument('--hierarchy-depth',action='store_true',help='Report node levels and pending blocks by depth; reads pending tasks, not all decisions')
    parser.add_argument('--model-history',action='store_true',help='Read every saved model JSON for historical fitting costs (automatic for v0.9.0)')
    args=parser.parse_args()
    manifest=json.loads((args.out/'run_identity.json').read_text())
    work=Path(manifest.get('work_directory',args.out/'.kovar_work'))
    report=dict(output=str(args.out.resolve()),work_directory=str(work),cpus=args.cpus,
                execution_revision=manifest.get('execution_revision','original'),
                note='Snapshot; saved fit durations include earlier invocations if this directory was resumed.')
    prepared=work/'prepared.json'
    if prepared.exists():
        info=json.loads(prepared.read_text())
        report.update(samples=info['n'],original_loci=info['loci'],patterns=info['patterns'])
    durations=[]
    statuses=Counter()
    history=args.model_history or manifest.get('version')=='0.9.0'
    for path in (work/'null_models').glob('*/null_*.json') if history else ():
        # Files are atomically renamed by KOVAR; incomplete .partial files are ignored.
        record=json.loads(path.read_text())
        statuses[record['status']]+=1
        durations.append(float(record['fit_seconds']))
    durations.sort()
    report['saved_null_models']=len(durations) if history else None
    report['saved_null_statuses']=dict(statuses)
    report['null_fitting']=dict(sum_worker_wall_seconds=sum(durations),
        median_seconds=statistics.median(durations) if durations else None,
        p90_seconds=quantile(durations,.9),p99_seconds=quantile(durations,.99),
        max_seconds=max(durations) if durations else None,
        ideal_parallel_wall_lower_bound_seconds=sum(durations)/args.cpus)
    if not history:
        report['null_fitting']={'note':'Historical model files were not scanned; use runtime_totals for new completed batches, or --model-history.'}
    db=sqlite3.connect((work/'work.sqlite').resolve().as_uri()+'?mode=ro',uri=True,timeout=30)
    db.execute('PRAGMA query_only=ON')
    try:
        db.execute('BEGIN')
        report['meta']={key:json.loads(value) for key,value in db.execute(
            "SELECT key,value FROM meta WHERE key IN ('hierarchy_complete','search_initialized','root_cursor','response_coarse_columns','response_coarse_next',"
            "'coarse_scheduled_count','search_complete','distance_expanded','runtime_totals','prefit_runtime_totals',"
            "'active_stage','null_stage_progress','null_stage_summary','worker_state','hierarchy_metrics','queried_pairs_count','memory_budget','resource_plan')")}
        roots=db.execute('SELECT count(*) FROM nodes WHERE parent=0').fetchone()[0]
        report['coarse_categories']=roots
        report['coarse_category_pairs']=roots*(roots-1)//2
        report['queried_pairs']=report['meta'].get('queried_pairs_count')
        report['runtime_totals']=report['meta'].pop('runtime_totals',{})
        report['hierarchy_metrics']=report['meta'].pop('hierarchy_metrics',{})
        for key in ('prefit_runtime_totals','null_stage_summary','worker_state'):
            report[key]=report['meta'].pop(key,{})
        if args.detailed:
            report['queried_pairs']=db.execute('SELECT count(*) FROM decisions').fetchone()[0]
            report['pair_counts_by_final_state']=dict(db.execute('SELECT final,count(*) FROM decisions GROUP BY final'))
            report['pair_states_note']='1 significant; -1 nonsignificant; 0 unresolved; -2 final-ineligible (may still be scored as a proxy).'
            report['pair_counts_by_status']=dict(db.execute('SELECT status,count(*) FROM decisions GROUP BY status'))
            report['traversal_tasks_by_done']=dict(db.execute('SELECT done,count(*) FROM tasks GROUP BY done'))
        cursor=report['meta'].get('root_cursor')
        if cursor:
            i,j=cursor
            report['coarse_pairs_scheduled']=i*(2*roots-i-1)//2+j-i-1
            report['still_processing_within_root_categories']=cursor==[0,1]
        if 'coarse_scheduled_count' in report['meta']:
            report['coarse_pairs_scheduled']=report['meta']['coarse_scheduled_count']
            report['still_processing_within_root_categories']=report['coarse_pairs_scheduled']==0
        if args.hierarchy_depth:
            scheduled=report.get('coarse_pairs_scheduled',0)
            report['coarse_pairs_not_yet_scheduled']=max(0,report['coarse_category_pairs']-scheduled)
            report['nodes_by_depth']=[dict(depth=level,nodes=count) for level,count in db.execute(
                'SELECT level,count(*) FROM nodes WHERE level>=0 GROUP BY level ORDER BY level')]
            # A cross-block can have different depths on its two sides. Group
            # them explicitly rather than inventing a single current depth.
            report['pending_blocks_by_depth']=[dict(depth_a=a,depth_b=b,pending_blocks=count,
                member_pair_upper_bound=pairs) for a,b,count,pairs in db.execute(
                'SELECT min(a.level,b.level) AS da,max(a.level,b.level) AS db,count(*),'
                'sum(CASE WHEN t.a=t.b THEN a.size*(a.size-1)/2 ELSE a.size*b.size END) '
                'FROM tasks t JOIN nodes a ON a.id=t.a JOIN nodes b ON b.id=t.b '
                'WHERE t.done=0 GROUP BY da,db ORDER BY da,db')]
            report['depth_note']=('Depth 0 is the coarse category level. Coarse comparisons and refinements '
                'interleave. Pending blocks include in-flight work; member-pair upper bounds are before '
                'pruning/cache reuse and exclude coarse comparisons not yet scheduled. Scheduled is not completed.')
    finally:
        db.close()
    if args.log:
        text=args.log.read_text(errors='replace')
        updates=[(int(q.replace(',','')),int(t)) for q,t in re.findall(
            r'search: ([\d,]+) queried pattern pairs; (\d+)s',text)]
        if updates:
            q,t=updates[-1]
            initial_q,initial_t=updates[0]
            elapsed=t-initial_t
            report['latest_log']=dict(queried_pairs=q,search_elapsed_seconds=t,
                observed_new_queries=q-initial_q,observed_window_seconds=elapsed,
                average_queries_per_second=(q-initial_q)/elapsed if elapsed>0 else None)
            # Fit duration sum is only comparable when cache and log cover the
            # same fresh invocation. This occupancy estimate is not CPU profiling.
            fresh=text.startswith('Starting new work:') and 'External null-cache import: none' in text
            if fresh and t and history:
                report['null_fitting']['approx_allocated_worker_wall_fraction']=sum(durations)/(args.cpus*t)
                report['null_fitting']['occupancy_note']='Approximate: live metadata and log snapshots may differ in time; unfinished fits are not included.'
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
