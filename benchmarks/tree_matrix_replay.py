"""Experimental tree score kernels and response tiles for the replay benchmark."""
from itertools import chain
import math
import time

from numba import njit
import numpy as np

from ko_variation import discovery_score as kernels, tree_score
from ko_variation.discovery_execution import bounded_map, merge_metrics


ORIGINAL_ROWS = kernels._fast_tree_rows
ORIGINAL_GET_NULL = kernels.get_null
RESPONSES_PER_TILE = 4


@njit(cache=True)
def raw_contrasts_until(left, right, tip, a_fraction, b_fraction, scale,
                       data, limit, interval, prefix, sibling_nodes):
    """Use raw predictors and omit the intercept-only root mean."""
    width = data.shape[1]
    norms = np.zeros(width)
    stopped = np.zeros(width, dtype=np.bool_)
    active = np.empty(width, dtype=np.int64)
    remaining = 0
    has_prefix = prefix.size != 0
    for col in range(width):
        if has_prefix:
            norms[col] = prefix[col]
        if math.isfinite(norms[col]) and norms[col] > limit[col]:
            stopped[col] = True
        else:
            active[remaining] = col
            remaining += 1
    if remaining == 0:
        return norms, stopped, 0

    means = np.empty((len(tip), width))
    terms, since_check = 0, 0
    for node in range(len(tip)):
        if tip[node] >= 0:
            for pos in range(remaining):
                col = active[pos]
                means[node, col] = data[tip[node], col]
            continue
        a, b = left[node], right[node]
        count_contrast = not has_prefix or not sibling_nodes[node]
        for pos in range(remaining):
            col = active[pos]
            if count_contrast:
                z = (means[a, col] - means[b, col]) / scale[node]
                norms[col] += z * z
            means[node, col] = a_fraction[node] * means[a, col] + b_fraction[node] * means[b, col]
        if count_contrast:
            terms += remaining
            since_check += 1
        if since_check == interval:
            kept = 0
            for pos in range(remaining):
                col = active[pos]
                if math.isfinite(norms[col]) and norms[col] > limit[col]:
                    stopped[col] = True
                else:
                    active[kept] = col
                    kept += 1
            remaining = kept
            since_check = 0
            if remaining == 0:
                break
    return norms, stopped, terms


def raw_early_stop(self, X, cutoff, interval):
    state = kernels._STATE
    u = X.T @ self.residual
    upper_u = np.abs(u) + 1e-9 * np.maximum(1., np.abs(u))
    limit = np.maximum(upper_u**2 / cutoff, 1e-12)
    limit += 1e-9 * np.maximum(1., limit)
    prefix = state.get("sibling_prefix", state["empty_prefix"])
    if prefix.size:
        kernels.metric("sibling_early_rejections", int(np.count_nonzero(np.isfinite(prefix) & (prefix > limit))))
    v, stopped, terms = raw_contrasts_until(
        *self.tree_indices, *self.coefficients[:3], np.ascontiguousarray(X), limit,
        interval, prefix, state["sibling_nodes"])
    v[stopped] = np.nan
    return u, v, stopped, terms


def score_with_prefix(X, cache, response, predictors, eligible, gate, final):
    state = kernels._STATE
    entry = state.get("sibling_partials", {}).get(response)
    if entry is None:
        return ORIGINAL_ROWS(X, cache, response, predictors, eligible, gate, final)
    ids, bounds = entry
    state["sibling_prefix"] = bounds[np.searchsorted(ids, predictors)]
    try:
        return ORIGINAL_ROWS(X, cache, response, predictors, eligible, gate, final)
    finally:
        del state["sibling_prefix"]


def get_preloaded_null(response, external=None, *, manifest=None):
    state = kernels._STATE
    saved = state.get("preloaded_nulls", {})
    if manifest is None and response in saved:
        value = saved[response]
        if "active_null" in state:
            if state["active_null"] is not None:
                kernels.metric("null_job_reuse_hits")
            state["active_null"] = response, value
        return value
    return ORIGINAL_GET_NULL(response, external, manifest=manifest)


def sibling_indices(left, right, tip):
    internal = np.flatnonzero(tip < 0)
    tip_pairs = (tip[left[internal]] >= 0) & (tip[right[internal]] >= 0)
    return internal[tip_pairs]


def initialize(matrix):
    state = kernels._STATE
    tree = state["tree"]
    sibling_nodes = np.zeros(len(tree.tip), dtype=bool)
    if matrix:
        siblings = sibling_indices(tree.left, tree.right, tree.tip)
        sibling_nodes[siblings] = True
        left, right = tree.left[siblings], tree.right[siblings]
        state["sibling_tips"] = tree.tip[left], tree.tip[right]
        state["sibling_lengths"] = tree.length[left] + tree.length[right]
        kernels._fast_tree_rows = score_with_prefix
        kernels.get_null = get_preloaded_null
    state.update(sibling_nodes=sibling_nodes, empty_prefix=np.empty(0))
    tree_score.TreeScore.early_stop = raw_early_stop


def sibling_partials(jobs, preloaded):
    state = kernels._STATE
    left, right = state["sibling_tips"]
    if not len(left):
        return {}
    lengths = state["sibling_lengths"]
    weights, predictors, responses = [], [], []
    for response, ids, external, _ in jobs:
        cache, status = ORIGINAL_GET_NULL(response, external)
        preloaded[response] = cache, status
        if cache is None or cache["tau"] == 0. or not hasattr(cache["decomp"], "score_variance"):
            continue
        variances = 1. / cache["weights"]
        weights.append(1. / (variances[left] + variances[right] + cache["tau"] * lengths))
        predictors.append(np.array(sorted(ids), dtype=np.int64))
        responses.append(response)
    if not responses:
        return {}

    distinct = np.unique(np.concatenate(predictors))
    raw = kernels.bits(distinct)
    differences = (raw[:, left] != raw[:, right]).astype(float)
    positions = [np.searchsorted(distinct, ids) for ids in predictors]
    selected = sum(len(ids) for ids in predictors)
    full_tile = len(distinct) * len(responses)
    kernels.metric("sibling_screened_pairs", selected)
    kernels.metric("sibling_tiles", 1)
    if len(responses) > 1:
        kernels.metric("sibling_multi_response_tiles", 1)
    if len(responses) > 1 and full_tile <= 2 * selected:
        tile = differences @ np.column_stack(weights)
        bounds = [tile[pos, col] for col, pos in enumerate(positions)]
        kernels.metric("sibling_dense_tiles", 1)
        kernels.metric("sibling_matrix_entries", full_tile)
    else:
        bounds = [differences[pos] @ weight for pos, weight in zip(positions, weights)]
        kernels.metric("sibling_matrix_entries", selected)
    return {response: (ids, values) for response, ids, values in zip(responses, predictors, bounds)}


def score_response_tile(job):
    _, jobs = job
    state = kernels._STATE
    preloaded = {}
    before = dict(state["metrics"])
    evictions = state["nulls"].evictions
    started = time.perf_counter()
    if max(len(job[1]) for job in jobs) <= state["settings"]["batch_size"]:
        state["sibling_partials"] = sibling_partials(jobs, preloaded)
    state["preloaded_nulls"] = preloaded
    elapsed = time.perf_counter() - started
    metrics = {key: value - before.get(key, 0) for key, value in state["metrics"].items()}
    metrics.update(sibling_prepass_worker_seconds=elapsed, worker_job_seconds=elapsed,
                   null_cache_evictions=state["nulls"].evictions - evictions,
                   sibling_worker_dispatches=1)
    rows, inventory, worker_state = [], [], None
    try:
        for child in jobs:
            result = kernels.score_profiled(child)
            rows.extend(result["rows"])
            inventory.extend(result["inventory"])
            merge_metrics(metrics, result["metrics"])
            if result["worker_state"]:
                worker_state = result["worker_state"]
    finally:
        state.pop("sibling_partials", None)
        del state["preloaded_nulls"]
    return dict(rows=rows, metrics=metrics, inventory=inventory, worker_state=worker_state)


def response_tiles(pool, jobs, width):
    groups = [[] for _ in range(pool.workers)]
    for job in jobs:
        response = job[0]
        canonical = int(pool.canonical_responses[response]) if pool.canonical_responses is not None else response
        group = groups[canonical % pool.workers]
        if len(job[1]) > width:
            yield response, [job]
            continue
        group.append(job)
        if len(group) == RESPONSES_PER_TILE:
            yield group[0][0], group[:]
            group.clear()
    for group in groups:
        if group:
            yield group[0][0], group


def map_with_sibling_tiles(pool, function, jobs, threads, *, width, ordered=True, response_index=None):
    if function is kernels.score_profiled:
        jobs = iter(jobs)
        first = next(jobs, None)
        if first is None:
            return
        jobs = chain((first,), jobs)
        if first[3]:
            yield from bounded_map(pool, score_response_tile, response_tiles(pool, jobs, width), threads,
                                   ordered=ordered, response_index=0)
            return
    yield from bounded_map(pool, function, jobs, threads, ordered=ordered, response_index=response_index)
