"""Bounded worker scheduling with response affinity and immediate completion."""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import multiprocessing


class ResponsePool:
    """Pin a response to one long-lived worker so its null stays reusable.

    Independent category/refit jobs are distributed round-robin. Every process
    initializer restricts native libraries to one thread; total CPUs stay bounded.
    """
    def __init__(self, workers, initializer, initargs):
        self.workers = workers
        self.next_worker = 0
        self.response_aliases=None
        if initargs:
            from pathlib import Path
            import numpy as np
            path=Path(initargs[0])/'null_canonical.npy'
            if path.exists(): self.response_aliases=np.load(path,mmap_mode='r')
        context = multiprocessing.get_context("spawn")
        self.executors = [ProcessPoolExecutor(max_workers=1, mp_context=context,
                           initializer=initializer, initargs=initargs) for _ in range(workers)]

    def submit(self, function, job):
        if function.__name__ in {"score", "score_profiled"}:
            mapping=getattr(self,'response_aliases',None)
            response=int(mapping[job[0]]) if mapping is not None else int(job[0])
            index = response % self.workers
        else:
            index = self.next_worker % self.workers
            self.next_worker += 1
        return self.executors[index].submit(function, job)

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        for executor in self.executors:
            executor.shutdown(wait=True, cancel_futures=kind is not None)


def bounded_map(pool, function, jobs, threads, *, ordered=True):
    """Bound queued work; scoring consumes completions without head-of-line waits.

    Numerical category reductions retain input order for deterministic floating
    point sums. Their completed futures can still refill the queue immediately.
    """
    source, pending, completed = iter(jobs), {}, {}
    submitted = yielded = 0

    def submit_one():
        nonlocal submitted
        job = next(source, None)
        if job is None:
            return False
        pending[pool.submit(function, job)] = submitted
        submitted += 1
        return True

    for _ in range(max(1, threads*2)):
        if not submit_one():
            break
    while pending or completed:
        if ordered and yielded in completed:
            yield completed.pop(yielded)
            yielded += 1
            # A bounded completed buffer avoids accumulating all results when
            # the first numerical-reduction task happens to run slowly.
            while len(pending)+len(completed) < max(1,threads*2) and submit_one():
                pass
            continue
        if not pending:
            break
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            index = pending.pop(future)
            result = future.result()
            if ordered:
                completed[index] = result
            else:
                yield result
                yielded += 1
        while len(pending)+len(completed) < max(1,threads*2) and submit_one():
            pass
