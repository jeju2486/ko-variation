"""Resume full beta refits from a discovery run's saved distal distances.

Uses the existing numerical kernel and SQLite beta table. No pair search,
distance calculation, raw-input preparation or null fitting is repeated.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import time

from .memory_budget import guard_blas_environment

guard_blas_environment(default_threads=1)

import numpy as np
import scipy

from . import discovery_score as kernels
from .discovery import export, get_meta, parser as discovery_parser, set_meta
from .discovery_execution import ResponsePool, bounded_map
from .discovery_store import atomic_json, connect, digest
from .memory_budget import detect_budget, plan_workers


def queue_pairs(db, retry_failed=False):
    """Freeze a disk-backed queue; never materialize all pairs in Python RAM."""
    if not get_meta(db, 'search_complete') or not get_meta(db, 'distance_expanded'):
        raise ValueError('Finish discovery and distance expansion before beta-only restart')
    if db.execute("SELECT 1 FROM distance WHERE status='PENDING' LIMIT 1").fetchone():
        raise ValueError('Distances are unfinished; resume the normal discovery run first')
    db.execute('DROP TABLE IF EXISTS temp.beta_queue')
    db.execute('CREATE TEMP TABLE beta_queue(u INTEGER,v INTEGER,PRIMARY KEY(v,u)) WITHOUT ROWID')
    db.execute("INSERT OR IGNORE INTO beta_queue SELECT x.pu,x.pv FROM distance x "
               "JOIN decisions d ON d.u=x.pu AND d.v=x.pv "
               "WHERE x.status='DISTAL' AND d.final=1")
    total = db.execute('SELECT count(*) FROM beta_queue').fetchone()[0]
    if retry_failed:
        with db:
            db.execute("DELETE FROM beta WHERE coalesce(status,'')!='OK' AND EXISTS "
                       "(SELECT 1 FROM beta_queue q WHERE q.u=beta.u AND q.v=beta.v)")
    statuses = Counter(dict(db.execute('SELECT b.status,count(*) FROM beta b JOIN beta_queue q '
                                      'ON b.u=q.u AND b.v=q.v GROUP BY b.status')))
    db.execute('DELETE FROM beta_queue WHERE EXISTS '
               '(SELECT 1 FROM beta b WHERE b.u=beta_queue.u AND b.v=beta_queue.v)')
    db.commit()
    return total, statuses


def fit_pending(db, pool, threads, report):
    """Commit every completed fit before reporting it; interrupted fits retry."""
    jobs = db.execute('SELECT u,v FROM beta_queue ORDER BY v,u')
    for row in bounded_map(pool, kernels.beta, jobs, threads, ordered=False):
        with db:
            db.execute('INSERT OR IGNORE INTO beta VALUES (?,?,?,?)', row)
        report(row[3])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True, help='Existing discovery results, or parent containing results/')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--memory-gb', type=float)
    p.add_argument('--tmp-dir', type=Path, help='Original persistent work directory, if used')
    p.add_argument('--unitigs', type=Path, help='Original sequence source label for the regenerated unitig table')
    p.add_argument('--retry-failed', action='store_true', help='Explicitly retry saved non-OK fits; OK fits are retained')
    p.add_argument('--progress-seconds', type=float, default=30)
    options = p.parse_args(argv)
    if options.threads < 1 or not np.isfinite(options.progress_seconds) or options.progress_seconds <= 0:
        p.error('--threads and --progress-seconds must be positive')
    out = options.out.resolve()
    if not (out/'run_identity.json').is_file() and (out/'results/run_identity.json').is_file():
        out /= 'results'
    identity = json.loads((out/'run_identity.json').read_text())
    work = Path(identity.get('work_directory', out/'.kovar_work')).resolve()
    if options.tmp_dir and options.tmp_dir.resolve() != work:
        raise ValueError('Use the original persistent work directory recorded in run_identity.json')
    if not (work/'work.sqlite').is_file():
        raise ValueError(f'Missing discovery work database: {work}/work.sqlite')
    # Match numerical code/versions without requiring raw FASTA/graph inputs.
    saved = json.loads((work/'run_identity.json').read_text())
    if {k: v for k, v in identity.items() if k != 'work_directory'} != saved:
        raise ValueError('Output and work-directory identities differ')
    if saved['numpy'] != np.__version__ or saved['scipy'] != scipy.__version__:
        raise ValueError('Use the original NumPy/SciPy versions for this checkpoint')
    for name, expected in saved['code'].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError(f'Checkpoint code differs: {name}; use the matching discovery checkout')
    info = json.loads((work/'prepared.json').read_text())
    for name, expected in info['products'].items():
        if digest(work/name) != expected:
            raise ValueError(f'Prepared product changed: {name}')
    args = discovery_parser().parse_args(['--fasta', saved['settings']['fasta'],
                                          '--tree', saved['settings']['tree'], '--out', str(out)])
    vars(args).update(saved['settings'])
    args.out, args.threads, args.unitigs = out, options.threads, options.unitigs
    # Keep an existing sequence-source label when no raw path was supplied.
    if args.unitigs is None and (out/'unitigs.tsv.gz').is_file():
        import csv
        import gzip
        with gzip.open(out/'unitigs.tsv.gz', 'rt', newline='') as stream:
            first = next(csv.DictReader(stream, delimiter='\t'), {})
        if first.get('sequence_source'):
            args.unitigs = Path(first['sequence_source'])
    args._memory_info = detect_budget(options.memory_gb, args.threads)
    args.memory_gb = args._memory_info['working_gb']
    args._plan = plan_workers(args._memory_info, info['n'], args.threads, args.batch_size)
    settings = dict(args._plan, cache_bytes_per_worker=0)
    lock = sqlite3.connect(work/'coordinator.lock.sqlite', timeout=0)
    db = None
    started = last_report = time.monotonic()
    completed = 0
    try:
        lock.execute('CREATE TABLE IF NOT EXISTS lock(id INTEGER)')
        try:
            lock.execute('BEGIN EXCLUSIVE')
        except sqlite3.OperationalError as error:
            raise ValueError('Another coordinator is using this work directory') from error
        db = connect(work/'work.sqlite', min(64, args.memory_gb*1024*.05))
        total, statuses = queue_pairs(db, options.retry_failed)
        initial = sum(statuses.values())
        with db:
            set_meta(db, 'active_stage', 'beta')

        def publish(force=False, finished=False):
            nonlocal last_report
            now = time.monotonic()
            if not force and now-last_report < options.progress_seconds:
                return
            elapsed = now-started
            saved_count = initial+completed
            remaining = total-saved_count
            eta = elapsed/completed*remaining if completed else None
            record = dict(stage='complete' if finished else 'beta', total_distal_pattern_pairs=total,
                          saved_before_invocation=initial, committed_this_invocation=completed,
                          saved=saved_count, remaining=remaining, status_counts=dict(statuses),
                          seconds_this_invocation=elapsed, eta_seconds=eta, threads=args.threads,
                          retry_failed=options.retry_failed, work_directory=str(work))
            atomic_json(out/'beta_progress.json', record)
            message = f'; rough ETA {eta/3600:.2f} h' if eta is not None else ''
            print(f'[KOVAR] beta: {saved_count:,}/{total:,} saved; {remaining:,} remaining; '
                  f'{statuses.get("OK", 0):,} OK; {saved_count-statuses.get("OK", 0):,} non-OK'
                  f'{message}', flush=True)
            last_report = now

        def report(status):
            nonlocal completed
            completed += 1
            statuses[status] += 1
            publish()

        publish(force=True)
        with ResponsePool(args.threads, kernels.initialize, (str(work), info, settings)) as pool:
            fit_pending(db, pool, args.threads, report)
        # Regenerate public tables atomically, including saved pre-restart fits.
        export(db, info, work, args, time.monotonic()-started)
        with db:
            set_meta(db, 'active_stage', 'complete')
        publish(force=True, finished=True)
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    finally:
        if db is not None:
            db.close()
        lock.rollback()
        lock.close()
    return 0


if __name__ == '__main__':
    main()
