"""Coordinator-owned null inventory; model manifests remain crash recovery records."""
import json


def initialize(db):
    db.execute('''CREATE TABLE IF NOT EXISTS null_inventory(
        pid INTEGER PRIMARY KEY,status TEXT,mode TEXT,fit_seconds REAL,
        source TEXT,local INTEGER,signature TEXT)''')


def record_models(db, rows, signature=None):
    """Called inside the same coordinator transaction as progress/decisions."""
    db.executemany('INSERT OR REPLACE INTO null_inventory VALUES (?,?,?,?,?,?,?)',
        ((r['pattern'],r['record']['status'],r['record'].get('mode','failed'),
          r['record'].get('fit_seconds',0.),r['source'],int(r['local']),
          signature) for r in rows))


def local_summary(db, work):
    """Migrate old/on-demand manifests once, then aggregate without filesystem scans."""
    initialized=db.execute("SELECT value FROM meta WHERE key='null_inventory_seeded'").fetchone()
    if not initialized:
        pending=[]
        for source in (work/'null_models').glob('*/null_*.json'):
            record=json.loads(source.read_text())
            pending.append((record['pattern'],record['status'],record.get('mode','failed'),
                record.get('fit_seconds',0.),str(source),1,None))
            if len(pending)>=1000:
                with db:
                    db.executemany('INSERT OR IGNORE INTO null_inventory VALUES (?,?,?,?,?,?,?)',pending)
                pending.clear()
        with db:
            db.executemany('INSERT OR IGNORE INTO null_inventory VALUES (?,?,?,?,?,?,?)',pending)
            db.execute("INSERT OR REPLACE INTO meta VALUES ('null_inventory_seeded','true')")
    counts=dict(db.execute('SELECT status,count(*) FROM null_inventory WHERE local=1 GROUP BY status'))
    seconds=db.execute('SELECT coalesce(sum(fit_seconds),0.) FROM null_inventory WHERE local=1').fetchone()[0]
    return counts,seconds
