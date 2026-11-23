"""Bounded worker scheduling with response affinity and immediate completion."""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import itertools
import multiprocessing


def process_pool(workers, initializer, initargs):
    return ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                               initializer=initializer, initargs=initargs)


class ResponsePool:
    """Pin a response to one long-lived worker so its null stays reusable.

    Independent category/refit jobs are distributed round-robin. Every process
    initializer restricts native libraries to one thread; total CPUs stay bounded.
    """
    def __init__(self, workers, initializer, initargs, *, canonical_responses=None):
        self.workers = workers
        self.next_worker = 0
        self.canonical_responses = canonical_responses
        self.executors = [process_pool(1, initializer, initargs) for _ in range(workers)]

    def submit_response(self, function, job, response):
        if self.canonical_responses is not None:
            response = int(self.canonical_responses[response])
        return self.executors[response % self.workers].submit(function, job)

    def submit(self, function, job):
        index = self.next_worker % self.workers
        self.next_worker += 1
        return self.executors[index].submit(function, job)

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        for executor in self.executors:
            executor.shutdown(wait=True, cancel_futures=kind is not None)


def bounded_map(pool, function, jobs, threads, *, ordered=True, response_index=None):
    """Bound queued work; scoring consumes completions without head-of-line waits.

    Numerical category reductions retain input order for deterministic floating
    point sums. Their completed futures can still refill the queue immediately.
    """
    source, pending, completed = iter(jobs), {}, {}
    submitted = yielded = 0
    submit_response = getattr(pool, 'submit_response', None) if response_index is not None else None

    def submit_one():
        nonlocal submitted
        job = next(source, None)
        if job is None:
            return False
        future = (submit_response(function, job, job[response_index]) if submit_response else
                  pool.submit(function, job))
        pending[future] = submitted
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


def batches(values, size):
    source = iter(values)
    while True:
        batch = list(itertools.islice(source, size))
        if not batch:
            break
        yield batch


def merge_metrics(target, values):
    for key,value in values.items():
        target[key]=max(target.get(key,0),value) if key=="cache_peak_bytes_worker" else target.get(key,0)+value
