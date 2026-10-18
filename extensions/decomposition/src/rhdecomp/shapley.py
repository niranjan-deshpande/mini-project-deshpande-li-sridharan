"""Exact Shapley decomposition of R^2 over k candidates (all 2^k subsets; fine for k <= ~12).

Both R^2 notions reduce to gram_r2 on second moments, so every subset costs a k x k solve.
"""
from math import factorial

import numpy as np

from .functional import group_moments
from .geometry import gram_r2


def subset_r2_table(G, c, yy):
    """R^2 for every subset of regressors, indexed by bitmask (bit i set = candidate i included)."""
    k = len(c)
    return np.array([gram_r2(G, c, yy, [i for i in range(k) if m >> i & 1]) for m in range(1 << k)])


def shapley_from_table(table, k):
    """Shapley values phi_i = sum_S |S|!(k-|S|-1)!/k! [R2(S + i) - R2(S)] over S not containing i."""
    w = [factorial(s) * factorial(k - s - 1) / factorial(k) for s in range(k)]
    phi = np.zeros(k)
    for m in range(1 << k):
        s = bin(m).count("1")
        for i in range(k):
            if not m >> i & 1:
                phi[i] += w[s] * (table[m | 1 << i] - table[m])
    return phi


def _decompose(G, c, yy):
    k = len(c)
    table = subset_r2_table(G, c, yy)
    full = (1 << k) - 1
    return {"full_r2": float(table[full]),
            "shapley": shapley_from_table(table, k),
            "unique": np.array([table[full] - table[full & ~(1 << i)] for i in range(k)]),
            "single_r2": np.array([table[1 << i] for i in range(k)])}


def shapley_geo(C, v):
    """Shapley/unique decomposition of R^2_geo of v on the columns of C [d, k]."""
    C, v = np.asarray(C, float), np.asarray(v, float)
    return _decompose(C.T @ C, C.T @ v, v @ v)


def shapley_functional(s_v, S):
    """Shapley/unique decomposition of the functional R^2 of s_v ~ 1 + S, S [T, k]."""
    G, c, vy = group_moments(s_v, S, np.zeros(len(s_v))).covariances()
    return _decompose(G, c, vy)
