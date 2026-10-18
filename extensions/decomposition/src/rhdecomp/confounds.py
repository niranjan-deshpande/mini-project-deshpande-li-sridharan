"""Confound checks on real rollouts using the Phase 2 passage labels.

Confound labels c are 0/1 with NaN (or any value other than 0/1) meaning "no unanimous label"; those passages are
dropped from every cell. Every result carries its n.
"""
import numpy as np

from .stats import auroc


def _valid(c):
    c = np.asarray(c, float)
    return (c == 0) | (c == 1), c == 1


def confound_crosstab(y, c):
    """Counts n[hack][confound] and P(confound | hack) vs. P(confound | clean)."""
    y = np.asarray(y, bool)
    ok, c1 = _valid(c)
    n = {h: {k: int((ok & (y == h) & (c1 == k)).sum()) for k in (0, 1)} for h in (0, 1)}
    p = {h: (n[h][1] / (n[h][0] + n[h][1]) if n[h][0] + n[h][1] else float("nan")) for h in (0, 1)}
    return {"n": n, "p_confound_given_hack": p[1], "p_confound_given_clean": p[0]}


def stratified_auroc(y, s, c):
    """Hack-vs-clean AUROC of s computed separately within confound = 0 and confound = 1."""
    y, s = np.asarray(y, bool), np.asarray(s, float)
    ok, c1 = _valid(c)
    out = {}
    for k in (0, 1):
        m = ok & (c1 == k)
        out[k] = {"auroc": auroc(y[m], s[m]), "n": int(m.sum()), "n_hack": int(y[m].sum()),
                  "n_clean": int((~y[m]).sum())}
    return out


def confound_response_auroc(y, s, c):
    """AUROC of s for confound = 1 vs. 0 on clean passages only (also: candidate validity, impossible vs. solvable)."""
    y, s = np.asarray(y, bool), np.asarray(s, float)
    ok, c1 = _valid(c)
    m = ok & ~y
    return {"auroc": auroc(c1[m], s[m]), "n": int(m.sum()), "n_confound": int(c1[m].sum()),
            "n_no_confound": int((~c1[m]).sum())}
