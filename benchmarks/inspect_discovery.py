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
import time


def quantile(values,fraction):
    if not values:
        return None
    position=(len(values)-1)*fraction
    lo=int(position)
    hi=min(lo+1,len(values)-1)
    return values[lo]+(position-lo)*(values[hi]-values[lo])


def search_progress(report, internal):
    """Bound finished root branches using only scheduled and pending work.

    Every unfinished opened branch has at least one pending task, including
    in-flight tasks. Parent completion and child scheduling are atomic in KOVAR.
    Multiple pending descendants can share a branch, making the lower bound
    conservative. This measures the search policy's work, not elapsed time.
    """
    meta=report['meta']
    total=report['coarse_category_pairs']+internal
    pending=sum(row['pending_blocks'] for row in report['pending_blocks_by_depth'])
    opened=min(total,report.get('coarse_pairs_scheduled',0)+internal) if meta.get('search_initialized') else 0
    lower=max(0,opened-pending)
    upper=max(0,opened-int(pending>0))
    if meta.get('search_complete'):
        lower=upper=total
    return dict(total_branches=total,opened_branches=opened,pending_blocks=pending,
                finished_branches_lower=lower,finished_branches_upper=upper,
                percent_lower=100*lower/total if total else None,
                percent_upper=100*upper/total if total else None,
                ready=bool(meta.get('hierarchy_complete') and meta.get('search_initialized')),
                note='Branch-count coverage under the current pruning policy, not time completion or exactly tested pair coverage.')


def estimate_eta(report, previous):
    if report['meta'].get('search_complete'):
        return dict(seconds_lower=0.,seconds_upper=0.,note='Search is complete; later stages are excluded.')
    current=report['search_progress'];old=previous.get('search_progress',{})
    if not old:
        return dict(note='Previous snapshot lacks branch progress; save a new-format snapshot first.')
    same=all(report.get(key)==previous.get(key) for key in ('work_directory','run_inputs','patterns'))
    elapsed=report['snapshot_time_unix']-previous.get('snapshot_time_unix',report['snapshot_time_unix'])
    if not same or current['total_branches']!=old.get('total_branches'):
        return dict(note='Previous snapshot belongs to a different input/run or hierarchy size.')
    if not current['ready'] or not old.get('ready') or elapsed<=0:
        return dict(note='Two snapshots from initialized hierarchy search, separated in time, are required.')
    delta_low=max(0,current['finished_branches_lower']-old['finished_branches_upper'])
    delta_high=max(0,current['finished_branches_upper']-old['finished_branches_lower'])
    if delta_low==0:
        return dict(note='Completion intervals overlap; take another snapshot later for an ETA.')
    total=current['total_branches']
    return dict(seconds_lower=(total-current['finished_branches_upper'])*elapsed/delta_high,
                seconds_upper=(total-current['finished_branches_lower'])*elapsed/delta_low,
                finished_branches_per_hour_lower=delta_low*3600/elapsed,
                finished_branches_per_hour_upper=delta_high*3600/elapsed,
                note='Rough search-only projection assuming recent branch throughput continues; branch costs can differ.')


def print_summary(report):
    meta=report['meta'];progress=report['search_progress']
    stage=meta.get('active_stage','not recorded')
    print(f'Stage: {stage}')
    if report.get('search_mode')=='exhaustive':
        print('This run is exhaustive; hierarchy branch progress does not apply.')
        print(f"Queried pattern pairs: {report.get('queried_pairs')}")
        return
    if meta.get('search_complete'):
        print('Hierarchy search: 100% complete (distance/beta/export may still remain).')
    elif progress['ready'] and progress['total_branches']:
        lo=min(99.99,progress['percent_lower']);hi=min(99.99,progress['percent_upper'])
        print(f'Estimated search coverage: {lo:.2f}% to {hi:.2f}% of top-level branches finished')
    else:
        print('Search percentage: unavailable until hierarchy construction and search initialization finish.')
    print(f"Coarse comparisons opened: {report.get('coarse_pairs_scheduled',0):,} / {report['coarse_category_pairs']:,}")
    print(f"Coarse comparisons not yet opened: {report['coarse_pairs_not_yet_scheduled']:,}")
    print(f"Pending blocks (including in-flight): {progress['pending_blocks']:,}")
    print(f"Remaining top-level branches: {progress['total_branches']-progress['finished_branches_upper']:,}"
          f" to {progress['total_branches']-progress['finished_branches_lower']:,}")
    for row in report['pending_blocks_by_depth']:
        print(f"  Depth ({row['depth_a']},{row['depth_b']}): {row['pending_blocks']:,} blocks")
    queried=report.get('queried_pairs')
    if queried is not None: print(f'Queried pattern pairs, including earlier runs: {queried:,}')
    rate=report.get('latest_log',{}).get('recent_queries_per_hour')
    if rate is not None: print(f'Recent new queries/hour: {rate:,.0f}')
    eta=report.get('eta',{})
    if meta.get('search_complete'): print('Search remaining time: 0 (later stages excluded).')
    elif 'seconds_lower' in eta:
        print(f"Rough search ETA: {eta['seconds_lower']/3600:.1f} to {eta['seconds_upper']/3600:.1f} hours")
        print(eta['note'])
    else: print('Search ETA: '+eta.get('note','save a snapshot and compare another snapshot later.'))
    print('Depths interleave; coverage is not elapsed-time completion. 100% requires the completion flag.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--log',type=Path,help='Current Slurm .out log, preferably the live server copy')
    parser.add_argument('--cpus',type=int,default=24)
    parser.add_argument('--detailed',action='store_true',help='Scan decision/task tables; can be expensive on very large runs')
    parser.add_argument('--hierarchy-depth',action='store_true',help='Report node levels and pending blocks by depth; reads pending tasks, not all decisions')
    parser.add_argument('--summary',action='store_true',help='Print readable progress instead of the default JSON')
    parser.add_argument('--json-out',type=Path,help='Also save a diagnosis snapshot outside the KOVAR checkpoint')
    parser.add_argument('--previous',type=Path,help='Previous diagnosis JSON for a rough branch-throughput ETA')
    parser.add_argument('--model-history',action='store_true',help='Read every saved model JSON for historical fitting costs (automatic for v0.9.0)')
    args=parser.parse_args()
    manifest=json.loads((args.out/'run_identity.json').read_text())
    work=Path(manifest.get('work_directory',args.out/'.kovar_work'))
    report=dict(output=str(args.out.resolve()),work_directory=str(work),cpus=args.cpus,
                execution_revision=manifest.get('execution_revision','original'),
                run_inputs=manifest.get('inputs'),
                search_mode=manifest.get('settings',{}).get('search','hierarchy'),
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
        report['snapshot_time_unix']=time.time()
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
        if args.hierarchy_depth or args.summary or args.previous:
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
            internal=db.execute('SELECT count(*) FROM nodes WHERE parent=0 AND size>1').fetchone()[0]
            report['search_progress']=search_progress(report,internal)
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
            recent_q,recent_t=min(updates,key=lambda row:abs(row[1]-max(initial_t,t-3600)))
            report['latest_log']=dict(queried_pairs=q,search_elapsed_seconds=t,
                observed_new_queries=q-initial_q,observed_window_seconds=elapsed,
                average_queries_per_second=(q-initial_q)/elapsed if elapsed>0 else None,
                recent_queries_per_hour=(q-recent_q)*3600/(t-recent_t) if t>recent_t else None)
            # Fit duration sum is only comparable when cache and log cover the
            # same fresh invocation. This occupancy estimate is not CPU profiling.
            fresh=text.startswith('Starting new work:') and 'External null-cache import: none' in text
            if fresh and t and history:
                report['null_fitting']['approx_allocated_worker_wall_fraction']=sum(durations)/(args.cpus*t)
                report['null_fitting']['occupancy_note']='Approximate: live metadata and log snapshots may differ in time; unfinished fits are not included.'
    if args.previous:
        report['eta']=estimate_eta(report,json.loads(args.previous.read_text()))
    output=json.dumps(report,indent=2)
    if args.json_out: args.json_out.write_text(output+'\n',encoding='utf-8')
    if args.summary: print_summary(report)
    else: print(output)


if __name__=='__main__':
    main()
