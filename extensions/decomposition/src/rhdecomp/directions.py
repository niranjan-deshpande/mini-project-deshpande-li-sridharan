"""Difference-of-means directions, split-half reliability, and the label-shuffle null.

A, B are per-pair activations [n_pairs, d]: row i of A is the branch with the concept, row i of B the branch without.
"""
from dataclasses import dataclass

import numpy as np

from .geometry import cosine


def dom(A, B, idx=None):
    """DoM = mean(A) - mean(B), optionally over the pairs in `idx`."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    if idx is not None:
        A, B = A[idx], B[idx]
    return A.mean(0) - B.mean(0)


def spearman_brown(r):
    """Full-data reliability from the mean half-half correlation: 2r / (1 + r)."""
    return 2 * r / (1 + r)


@dataclass
class Reliability:
    r: float              # mean cosine between half-sample DoMs
    rho: float            # Spearman-Brown full-data reliability
    r_samples: np.ndarray  # one cosine per random split


def split_half_reliability(A, B, n_splits=100, seed=0):
    """Split the pairs into two random halves n_splits times; cosine between the half DoMs.

    With an odd number of pairs one pair is left out of each split so both halves are the same size.
    """
    n = len(A)
    h = n // 2
    rng = np.random.default_rng(seed)
    rs = np.empty(n_splits)
    for i in range(n_splits):
        p = rng.permutation(n)
        rs[i] = cosine(dom(A, B, p[:h]), dom(A, B, p[h:2 * h]))
    r = float(rs.mean())
    return Reliability(r=r, rho=float(spearman_brown(r)), r_samples=rs)


def label_shuffled_doms(A, B, k, seed=0):
    """k label-shuffled DoMs [k, d]: the null for paired data.

    Shuffling labels in paired data means swapping A and B within a pair, i.e. multiplying the pair difference
    A_i - B_i by eps_i in {+1, -1}. We use *balanced* flips: exactly floor(n/2) randomly chosen pairs are swapped, so
    sum(eps) is 0 (n even) or 1 (n odd). Independent coin flips would leave sum(eps) ~ +-sqrt(n), and because every
    pair difference shares the true direction v, the null DoM would then contain (sum(eps)/n) * v -- on real data that
    leak gave |cos(null, v)| ~ 0.4. Balanced flips keep only the per-pair noise.
    """
    D = np.asarray(A, float) - np.asarray(B, float)
    n = len(D)
    rng = np.random.default_rng(seed)
    out = np.empty((k, D.shape[1]))
    for i in range(k):
        eps = np.ones(n)
        eps[rng.choice(n, n // 2, replace=False)] = -1
        out[i] = eps @ D / n
    return out
