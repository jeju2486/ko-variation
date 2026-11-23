"""Batched score variances for saved tree null models."""
import math

from numba import njit
import numpy as np


@njit(cache=True)
def contrast_coefficients(left, right, tip, length, weights, tau):
    variance = np.empty(len(tip))
    left_fraction = np.zeros(len(tip))
    right_fraction = np.zeros(len(tip))
    scale = np.zeros(len(tip))
    for node in range(len(tip)):
        if tip[node] >= 0:
            variance[node] = 1.0 / weights[tip[node]] + tau * length[node]
        else:
            a, b = left[node], right[node]
            total = variance[a] + variance[b]
            left_fraction[node] = variance[b] / total
            right_fraction[node] = variance[a] / total
            variance[node] = variance[a] * left_fraction[node] + tau * length[node]
            scale[node] = math.sqrt(total)
    return left_fraction, right_fraction, scale, math.sqrt(variance[-1])


@njit(cache=True)
def contrast_norms(left, right, tip, fractions_a, fractions_b, scale, root_scale, data):
    width = data.shape[1]
    means = np.empty((len(tip), width))
    norms = np.zeros(width)
    for node in range(len(tip)):
        if tip[node] >= 0:
            means[node, :] = data[tip[node], :]
        else:
            a, b = left[node], right[node]
            for j in range(width):
                z = (means[a, j] - means[b, j]) / scale[node]
                norms[j] += z * z
                means[node, j] = fractions_a[node] * means[a, j] + fractions_b[node] * means[b, j]
    for j in range(width):
        z = means[-1, j] / root_scale
        norms[j] += z * z
    return norms


@njit(cache=True)
def contrast_norms_until(left, right, tip, fractions_a, fractions_b, scale, root_scale,
                         data, limit, check_interval):
    width = data.shape[1]
    means = np.empty((len(tip), width))
    norms = np.zeros(width)
    stopped = np.zeros(width, dtype=np.bool_)
    active = np.arange(width)
    remaining, terms, since_check = width, 0, 0
    for node in range(len(tip)):
        if tip[node] >= 0:
            for pos in range(remaining):
                j = active[pos]
                means[node, j] = data[tip[node], j]
            continue
        a, b = left[node], right[node]
        for pos in range(remaining):
            j = active[pos]
            z = (means[a, j] - means[b, j]) / scale[node]
            norms[j] += z * z
            means[node, j] = fractions_a[node] * means[a, j] + fractions_b[node] * means[b, j]
        terms += remaining
        since_check += 1
        if since_check == check_interval:
            kept = 0
            for pos in range(remaining):
                j = active[pos]
                if math.isfinite(norms[j]) and norms[j] > limit[j]:
                    stopped[j] = True
                else:
                    active[kept] = j
                    kept += 1
            remaining = kept
            if not remaining:
                return norms, stopped, terms
            since_check = 0
    for pos in range(remaining):
        j = active[pos]
        z = means[-1, j] / root_scale
        norms[j] += z * z
    return norms, stopped, terms + remaining


class TreeScore:
    def __init__(self, cache):
        tree = cache["decomp"].tree
        self.tree_indices = tree.left, tree.right, tree.tip
        self.coefficients = contrast_coefficients(*self.tree_indices, tree.length,
                                                   cache["weights"], cache["tau"])
        self.design, self.inverse_design, self.nuisance = (cache[k] for k in ("C", "Sinv_C", "M_inv"))
        self.residual = cache["y"] - cache["mu"]

    def project(self, X):
        coefficient = self.nuisance @ (self.inverse_design.T @ X)
        return np.ascontiguousarray(X - self.design @ coefficient)

    def __call__(self, X):
        variance = contrast_norms(*self.tree_indices, *self.coefficients, self.project(X))
        return X.T @ self.residual, variance

    def early_stop(self, X, cutoff, interval):
        u = X.T @ self.residual
        # A partial norm lower-bounds the final variance. Leave slack around
        # the classification boundary and the minimum usable variance.
        upper_u = np.abs(u) + 1e-9 * np.maximum(1., np.abs(u))
        limit = np.maximum(upper_u ** 2 / cutoff, 1e-12)
        limit += 1e-9 * np.maximum(1., limit)
        variance, stopped, terms = contrast_norms_until(
            *self.tree_indices, *self.coefficients, self.project(X), limit, interval)
        variance[stopped] = np.nan
        return u, variance, stopped, terms


class ZeroTauScore:
    def __init__(self, cache):
        self.weights = cache["weights"]
        self.total_weight = self.weights.sum()
        self.vectors = np.column_stack((cache["y"] - cache["mu"], self.weights))

    def __call__(self, X):
        u, present_weight = (X.T @ self.vectors).T
        variance = present_weight * (1. - present_weight / self.total_weight)
        return u, variance


def classify(u, v, eligible, gate, final, skipped):
    valid = np.isfinite(u) & np.isfinite(v) & (v > 1e-12)
    statistic = np.full(u.size, np.nan)
    statistic[valid] = (u[valid] * (1. / np.sqrt(v[valid]))) ** 2
    valid &= np.isfinite(statistic)
    route = np.where(skipped, -1, np.where(valid, np.where(statistic >= gate, 1, -1), 0))
    decided = np.where(skipped, -1, np.where(valid, np.where(statistic >= final, 1, -1), 0))
    return route, np.where(eligible, decided, -2), statistic
