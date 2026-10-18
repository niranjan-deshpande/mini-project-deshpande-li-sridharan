"""Functional decomposition: regress token-level v scores on candidate scores (with intercept).

Tokens are the samples, but tokens within a rollout are correlated, so uncertainty comes from a cluster bootstrap
over rollouts. To make that cheap on millions of tokens we reduce each rollout to its sufficient statistics once
and re-weight them per resample.
"""
from dataclasses import dataclass

import numpy as np

from .geometry import gram_r2
from .stats import percentile_ci


@dataclass
class GroupMoments:
    """Per-group sums of (globally centered) y and X: n [g], sx [g,k], sy [g], sxx [g,k,k], sxy [g,k], syy [g]."""
    n: np.ndarray
    sx: np.ndarray
    sy: np.ndarray
    sxx: np.ndarray
    sxy: np.ndarray
    syy: np.ndarray

    def covariances(self, w=None):
        """(cov(X), cov(X, y), var(y)) of the data with group g repeated w[g] times (default: once)."""
        w = np.ones(len(self.n)) if w is None else np.asarray(w, float)
        n = w @ self.n
        mx, my = w @ self.sx / n, w @ self.sy / n
        G = np.tensordot(w, self.sxx, 1) / n - np.outer(mx, mx)
        c = w @ self.sxy / n - mx * my
        vy = w @ self.syy / n - my ** 2
        return G, c, vy


def group_moments(y, X, groups):
    y = np.asarray(y, float)
    X = np.asarray(X, float).reshape(len(y), -1)
    # Center globally first so the raw-moment formulas don't lose precision.
    y = y - y.mean()
    X = X - X.mean(0)
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.flatnonzero(np.r_[True, np.diff(inv[order]) != 0])
    ends = np.r_[starts[1:], len(order)]
    g, k = len(starts), X.shape[1]
    M = GroupMoments(np.empty(g), np.empty((g, k)), np.empty(g), np.empty((g, k, k)), np.empty((g, k)), np.empty(g))
    for j, (a, b) in enumerate(zip(starts, ends)):
        ii = order[a:b]
        Xg, yg = X[ii], y[ii]
        M.n[j], M.sx[j], M.sy[j] = len(ii), Xg.sum(0), yg.sum()
        M.sxx[j], M.sxy[j], M.syy[j] = Xg.T @ Xg, Xg.T @ yg, yg @ yg
    return M


def functional_r2(s_v, S):
    """In-sample R^2 of OLS s_v ~ 1 + S, with S [T, k] candidate token scores."""
    G, c, vy = group_moments(s_v, S, np.zeros(len(s_v))).covariances()
    return gram_r2(G, c, vy)


def functional_r2_bootstrap(s_v, S, groups, n_boot=1000, seed=0, level=0.95):
    """Point R^2 plus a cluster-bootstrap CI that resamples rollouts (groups) with replacement."""
    M = group_moments(s_v, S, groups)
    rng = np.random.default_rng(seed)
    g = len(M.n)
    samples = np.array([gram_r2(*M.covariances(np.bincount(rng.integers(0, g, g), minlength=g)))
                        for _ in range(n_boot)])
    return {"r2": gram_r2(*M.covariances()), "ci": percentile_ci(samples, level), "samples": samples}
