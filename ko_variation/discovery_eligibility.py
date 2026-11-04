"""Necessary joint-cell conditions for entire hierarchy blocks.

These bounds prune final-search tasks, never categories or their external
associations. Representatives still use the unrestricted exploration gate.
"""
from pathlib import Path

import numpy as np

from . import discovery_score as kernels
from .discovery_execution import bounded_map
from .discovery_store import open_patterns
from .memory_budget import ByteLRU, MIB


def impossible_block(n, minimum, radius_a, radius_b, distance, count_a, count_b, same=False):
    """Return a proof of ineligibility, not a covariation significance claim."""
    if minimum <= 0:
        return False
    if count_a < (2 if same else 1) or count_b < 1:
        return True
    lower = max(0, distance-radius_a-radius_b)
    upper = min(n, distance+radius_a+radius_b)
    # All four cells >= c requires 2c <= Hamming(A,B) <= n-2c.
    return upper < 2*minimum or lower > n-2*minimum


class Eligibility:
    def __init__(self, db, pool, args, info):
        self.db, self.pool, self.args = db, pool, args
        self.n, self.minimum = info['n'], args.min_cell_count
        self.cache = ByteLRU(min(8*MIB,max(MIB,int(args.memory_gb*(1<<30)*.005))))
        work = Path(args.tmp_dir) if args.tmp_dir else args.out/'.kovar_work'
        self.packed = open_patterns(work, info)

    def summaries(self, tasks):
        """Compute missing summaries in bounded worker batches and save on disk.

        Radius is the actual maximum distance to the fixed node anchor over ALL
        members, including members with insufficient marginal counts. No dense
        pair matrix or per-node member list is held by the coordinator.
        """
        result, missing = {}, []
        for node in sorted({node for task in tasks for node in task}):
            if node in self.cache:
                result[node] = self.cache.get(node)
                continue
            row = self.db.execute('SELECT center,radius,eligible_members FROM block_eligibility '
                                  'WHERE node=? AND min_cell=?', (node,self.minimum)).fetchone()
            if row is not None:
                result[node] = row
            else:
                center = self.db.execute('SELECT anchor FROM nodes WHERE id=?',(node,)).fetchone()[0]
                missing.append((node,center))
                result[node] = [center,0,0]
        def jobs():
            for node,center in missing:
                cursor = self.db.execute('SELECT pid FROM members WHERE node=? ORDER BY pid',(node,))
                while True:
                    rows = cursor.fetchmany(2048)
                    if not rows:
                        break
                    yield node,center,[row[0] for row in rows],self.minimum
        for node,radius,count in bounded_map(self.pool,kernels.eligibility_summary,jobs(),
                                              self.args.threads,ordered=False):
            result[node][1] = max(result[node][1],radius)
            result[node][2] += count
        if missing:
            with self.db:
                self.db.executemany('INSERT OR REPLACE INTO block_eligibility VALUES (?,?,?,?,?)',
                    ((node,self.minimum,*result[node]) for node,_ in missing))
        for node,summary in result.items():
            self.cache.put(node,tuple(summary),192)
        return result

    def impossible(self, a, b, summaries):
        center_a,radius_a,count_a = summaries[a]
        center_b,radius_b,count_b = summaries[b]
        distance = int(kernels._POPCOUNT[np.bitwise_xor(self.packed[center_a],
                                                       self.packed[center_b])].sum())
        return impossible_block(self.n,self.minimum,radius_a,radius_b,distance,
                                count_a,count_b,same=a==b)
