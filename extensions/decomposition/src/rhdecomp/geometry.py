"""Geometric decomposition: cosines, the span fit v = C beta + v_res, noise ceilings, and same-k nulls.

Convention: a candidate matrix C to fit against is [d, k] (columns are directions, as in the math);
a pool of vectors (null DoMs, baselines) is [m, d] (rows are vectors).
"""
from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np


def cosine(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def stack_columns(directions, names):
    """[d, k] matrix from a {name: vector} dict, in the order of `names`."""
    return np.column_stack([np.asarray(directions[n], float) for n in names])


@dataclass
class SpanFit:
    beta: np.ndarray      # [k] least-squares coefficients
    r2: float             # ||C beta||^2 / ||v||^2
    fitted: np.ndarray    # C beta, [d]
    residual: np.ndarray  # v - C beta, [d]; orthogonal to every column of C


def span_fit(C, v):
    """Project v onto the column span of C [d, k]. A projection, not a statistical regression: no p-values."""
    C, v = np.asarray(C, float), np.asarray(v, float)
    assert C.shape[0] == v.shape[0], "C must be [d, k] with candidates as columns"
    beta, *_ = np.linalg.lstsq(C, v, rcond=None)
    fitted = C @ beta
    return SpanFit(beta=beta, r2=float(fitted @ fitted / (v @ v)), fitted=fitted, residual=v - fitted)


def disattenuated_cosine(cos, rho_v, rho_c):
    """cos / sqrt(rho_v rho_c). Can exceed 1 in absolute value when reliabilities are underestimated."""
    return cos / np.sqrt(rho_v * rho_c)


def ceiling_r2(r2, rho_v):
    """R^2_geo relative to the noise ceiling of v: R^2 / rho_v."""
    return r2 / rho_v


def null_span_r2(v, pool, k, n_draws=1000, seed=0):
    """R^2_geo of v on k-subsets of null vectors (pool [m, d]).

    Enumerates every k-subset when there are at most n_draws of them; otherwise samples n_draws subsets uniformly.
    """
    pool = np.asarray(pool, float)
    m = len(pool)
    if comb(m, k) <= n_draws:
        subsets = [list(s) for s in combinations(range(m), k)]
    else:
        rng = np.random.default_rng(seed)
        subsets = [rng.choice(m, k, replace=False) for _ in range(n_draws)]
    return np.array([span_fit(pool[s].T, v).r2 for s in subsets])


def gram_r2(G, c, yy, idx=None):
    """Fraction of y explained by its projection onto a subset of regressors, from second moments.

    G: [k, k] Gram (or covariance) matrix of the regressors, c: [k] cross-moments with y, yy: y's second moment.
    Geometric: G = C^T C, c = C^T v, yy = v.v. Functional: covariances (OLS with intercept).
    A pseudo-inverse handles collinear (e.g. duplicated) regressors.
    """
    idx = np.arange(len(c)) if idx is None else np.asarray(idx, int)
    if len(idx) == 0:
        return 0.0
    cs = c[idx]
    return float(cs @ np.linalg.pinv(G[np.ix_(idx, idx)], rtol=1e-10, hermitian=True) @ cs / yy)
