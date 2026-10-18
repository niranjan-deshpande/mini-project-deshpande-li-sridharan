"""AUROC, the task split, the paired task-level bootstrap, and length control.

`auroc` and `task_half` are copied from reproduction/multi-model/t2_analyze.py so numbers match his exactly.
"""
import hashlib
from itertools import combinations

import numpy as np


def auroc(y, s):
    """Mann-Whitney AUROC with average ranks for ties. NaN if a class is empty."""
    y = np.asarray(y, bool); s = np.asarray(s, float)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, r); r = (sums / cnt)[inv]
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def task_half(sample_id):
    """md5(sample_id) % 2: 0 = selection half, 1 = held-out half."""
    return int(hashlib.md5(str(sample_id).encode()).hexdigest(), 16) % 2


def split_masks(sample_ids):
    """Boolean masks (selection, held_out) over an array of sample_ids."""
    half = np.array([task_half(t) for t in sample_ids])
    return half == 0, half == 1


def aurocs(scores, y, mask=None):
    """AUROC of every probe in a {name: passage scores} dict, optionally restricted to `mask`."""
    y = np.asarray(y, bool)
    m = np.ones(len(y), bool) if mask is None else np.asarray(mask, bool)
    return {k: auroc(y[m], np.asarray(s)[m]) for k, s in scores.items()}


def percentile_ci(x, level=0.95):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (float("nan"), float("nan"))
    a = (1 - level) / 2 * 100
    return (float(np.percentile(x, a)), float(np.percentile(x, 100 - a)))


def paired_task_bootstrap(scores, y, tasks, n_boot=1000, seed=0, metric=auroc, level=0.95):
    """Resample tasks with replacement; score every probe on the same resample.

    scores: {name: [n] passage scores}, y: [n] labels, tasks: [n] sample_ids (pass only the evaluation passages).
    Resamples where y has a single class are skipped (counted in `n_skipped`).
    Returns point estimates, percentile CIs, and point/CI for every pairwise difference a - b.
    """
    names = list(scores)
    S = np.column_stack([np.asarray(scores[k], float) for k in names])
    y = np.asarray(y, bool)
    tasks = np.asarray(tasks)
    uniq, inv = np.unique(tasks, return_inverse=True)
    members = [np.flatnonzero(inv == t) for t in range(len(uniq))]
    rng = np.random.default_rng(seed)

    point = np.array([metric(y, S[:, j]) for j in range(len(names))])
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([members[t] for t in rng.integers(0, len(uniq), len(uniq))])
        yb = y[idx]
        if yb.all() or not yb.any():
            continue
        draws.append([metric(yb, S[idx, j]) for j in range(len(names))])
    draws = np.array(draws).reshape(-1, len(names))

    out = {"names": names, "n_boot": n_boot, "n_skipped": n_boot - len(draws), "samples": draws,
           "point": dict(zip(names, point.tolist())),
           "ci": {k: percentile_ci(draws[:, j], level) for j, k in enumerate(names)},
           "diff": {}}
    for i, j in combinations(range(len(names)), 2):
        out["diff"][(names[i], names[j])] = {"point": float(point[i] - point[j]),
                                             "ci": percentile_ci(draws[:, i] - draws[:, j], level)}
    return out


def length_residualize(scores, n_tokens):
    """Residuals of each score on [1, log(n_tokens)].

    Fit by OLS on exactly the passages passed in (the evaluation set, hacks and clean together). This removes any
    length-linked part of the score, including the part of a real hack signal that happens to covary with length,
    so the residual AUROC is a conservative lower bound.
    """
    X = np.column_stack([np.ones(len(n_tokens)), np.log(np.asarray(n_tokens, float))])
    out = {}
    for k, s in scores.items():
        s = np.asarray(s, float)
        beta, *_ = np.linalg.lstsq(X, s, rcond=None)
        out[k] = s - X @ beta
    return out
