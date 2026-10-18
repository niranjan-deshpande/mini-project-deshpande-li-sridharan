"""Decision-level analyses: token projections, passage max-pooling, the max detector, and peak decomposition.

Token data model: token scores are a {name: [T]} dict over all tokens (rollouts concatenated), with
passage_idx [T] giving each token's row in the passage table (-1 = token not in any passage).
Passage scores are max over a passage's tokens of <v, x_t>. Max-pooling is nonlinear, so the passage score of
C beta must be pooled from its token scores sum_i beta_i s_i(t), never combined from per-candidate passage maxes.
"""
from collections import Counter, defaultdict

import numpy as np

from .stats import auroc, paired_task_bootstrap


# ---------- token projections and pooling ----------

def project(X, directions, chunk=8192):
    """{name: X @ v} for X [T, d] (e.g. one rollout's fp16 layer-40 activations).

    Computed in float32, chunk rows at a time, so a long fp16 rollout is never upcast whole.
    """
    names = list(directions)
    V = np.column_stack([np.asarray(directions[n], np.float32) for n in names])
    out = np.empty((len(X), len(names)))
    for a in range(0, len(X), chunk):
        out[a:a + chunk] = np.asarray(X[a:a + chunk], np.float32) @ V
    return {n: out[:, j] for j, n in enumerate(names)}


def project_rollouts(rollouts, directions):
    """Project an iterable of (X [T_r, d], passage_idx [T_r]) and concatenate.

    Returns (token_scores {name: [T]}, passage_idx [T], rollout_idx [T]). X may be a memmap; only one rollout is
    in memory at a time.
    """
    parts, pidx, ridx = defaultdict(list), [], []
    for r, (X, p) in enumerate(rollouts):
        for k, s in project(X, directions).items():
            parts[k].append(np.asarray(s, float))
        pidx.append(np.asarray(p, int))
        ridx.append(np.full(len(p), r))
    return {k: np.concatenate(v) for k, v in parts.items()}, np.concatenate(pidx), np.concatenate(ridx)


def add_decomposition(token_scores, v_name, cand_names, beta, fit_name="fit", res_name="res"):
    """Add token scores for C beta (sum_i beta_i s_i) and v_res (s_v - C beta) to a copy of token_scores.

    Uses linearity of the projection at the token level, so it works with precomputed projections.
    """
    out = dict(token_scores)
    fit = sum(b * np.asarray(token_scores[n], float) for b, n in zip(beta, cand_names))
    out[fit_name] = fit
    out[res_name] = np.asarray(token_scores[v_name], float) - fit
    return out


def pool_max(token_scores, passage_idx, n_passages):
    """Max-pool each [T] token score array into [n_passages]; passages with no tokens get NaN."""
    p = np.asarray(passage_idx, int)
    keep = p >= 0
    out = {}
    for k, s in token_scores.items():
        m = np.full(n_passages, -np.inf)
        np.maximum.at(m, p[keep], np.asarray(s, float)[keep])
        m[np.isneginf(m)] = np.nan
        out[k] = m
    return out


# ---------- max detector ----------

def clean_z_params(passage_scores, names, y, fit_mask):
    """(mu, sigma) of each candidate's passage scores over the clean (y == 0) passages in fit_mask."""
    clean = np.asarray(fit_mask, bool) & ~np.asarray(y, bool)
    mu = np.array([np.mean(passage_scores[n][clean]) for n in names])
    sd = np.array([np.std(passage_scores[n][clean], ddof=1) for n in names])
    return mu, sd


def max_detector(passage_scores, names, mu, sd):
    """s_max = max_i (s_i - mu_i) / sd_i over candidates."""
    Z = np.column_stack([(passage_scores[n] - m) / s for n, m, s in zip(names, mu, sd)])
    return Z.max(1)


def compare_max_detector(passage_scores, y, tasks, cand_names, fit_name, fit_mask, eval_mask,
                         n_boot=1000, seed=0):
    """Max detector vs. best single candidate vs. the linear combination C beta, paired bootstrap on eval_mask.

    z-score parameters and the choice of best single candidate both come from fit_mask (the selection half),
    so the evaluation half is untouched by any choice.
    """
    y = np.asarray(y, bool)
    fit_mask, eval_mask = np.asarray(fit_mask, bool), np.asarray(eval_mask, bool)
    mu, sd = clean_z_params(passage_scores, cand_names, y, fit_mask)
    s_max = max_detector(passage_scores, cand_names, mu, sd)
    best = max(cand_names, key=lambda n: auroc(y[fit_mask], passage_scores[n][fit_mask]))
    probes = {"max": s_max[eval_mask], "best_single": passage_scores[best][eval_mask],
              fit_name: passage_scores[fit_name][eval_mask]}
    res = paired_task_bootstrap(probes, y[eval_mask], np.asarray(tasks)[eval_mask], n_boot=n_boot, seed=seed)
    res["best_single_name"] = best
    res["z_mu"], res["z_sd"] = mu, sd
    return res


# ---------- peak decomposition ----------

def peak_decomposition(token_scores, v_name, cand_names, beta, passage_idx, y, category=None, cancel_frac=0.5):
    """Split each hack passage's peak v score into per-candidate contributions plus a residual.

    At t* = argmax_t s_v(t) within the passage:
        s_v(t*) - mu_v = sum_i beta_i (s_i(t*) - mu_i) + (s_res(t*) - mu_res)
    exactly, with s_res = s_v - sum_i beta_i s_i. The centers mu are means over all tokens of clean passages
    (the token distribution the functional regression uses), not means of clean passage peaks.

    Shares use absolute contributions: share_i = |c_i| / (sum_j |c_j| + |residual|).
    A passage is flagged for cancellation when two candidates of opposite sign each contribute at least
    cancel_frac * |s_v(t*) - mu_v|; `cluster` lists every candidate above that size.
    """
    y = np.asarray(y, bool)
    p = np.asarray(passage_idx, int)
    s_v = np.asarray(token_scores[v_name], float)
    S = np.column_stack([np.asarray(token_scores[n], float) for n in cand_names])
    beta = np.asarray(beta, float)
    s_res = s_v - S @ beta

    clean_tok = (p >= 0) & ~y[np.clip(p, 0, None)]
    mu_S, mu_v, mu_res = S[clean_tok].mean(0), s_v[clean_tok].mean(), s_res[clean_tok].mean()

    order = np.argsort(p, kind="stable")
    ps = p[order]
    records = []
    for q in np.flatnonzero(y):
        a, b = np.searchsorted(ps, q), np.searchsorted(ps, q, side="right")
        if a == b:
            continue
        t = order[a + np.argmax(s_v[order[a:b]])]
        contrib = beta * (S[t] - mu_S)
        resid = s_res[t] - mu_res
        elev = s_v[t] - mu_v
        mag = np.abs(contrib)
        total = mag.sum() + abs(resid)
        top = np.argsort(-mag)
        big = mag >= cancel_frac * abs(elev)
        records.append({
            "passage": int(q), "category": None if category is None else category[q], "token": int(t),
            "elevation": float(elev), "contrib": contrib, "residual": float(resid),
            "dominant": cand_names[top[0]],
            "dominant_share": float(mag[top[0]] / total),
            "top2_share": float(mag[top[:2]].sum() / total),
            "residual_share": float(abs(resid) / total),
            "opposite_sign": bool((contrib[big] > 0).any() and (contrib[big] < 0).any()),
            "cluster": tuple(n for n, f in zip(cand_names, big) if f),
        })
    return records


def summarize_peaks(records):
    """Per hack category (plus 'all'): n, mean shares, dominant-candidate counts, and flagged clusters."""
    groups = defaultdict(list)
    for r in records:
        groups["all"].append(r)
        if r["category"] is not None:
            groups[r["category"]].append(r)
    out = {}
    for g, rs in groups.items():
        flagged = [r for r in rs if r["opposite_sign"]]
        out[g] = {"n": len(rs),
                  "dominant_share": float(np.mean([r["dominant_share"] for r in rs])),
                  "top2_share": float(np.mean([r["top2_share"] for r in rs])),
                  "residual_share": float(np.mean([r["residual_share"] for r in rs])),
                  "dominant_counts": dict(Counter(r["dominant"] for r in rs)),
                  "n_opposite_sign": len(flagged),
                  "opposite_sign_clusters": dict(Counter(r["cluster"] for r in flagged))}
    return out
