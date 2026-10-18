"""T2 step 2 (paper Fig 8): DoM probe vs generic LLM monitor at passage level, within the impossible splits.

Ground truth = unanimous 3-run judge label (passages without consensus are dropped). Tasks (problem ids) are split
50/50 by hash: the selection half picks (layer, vector, aggregation) by AUROC; everything is reported on the held-out
half. The monitor's flag gives one operating point (FPR_m, TPR_m); the probe is thresholded at the same FPR.
Combos: probe-or-monitor (flag if either), and probe->monitor (monitor only sees probe-flagged passages; the probe
threshold is set at a looser FPR so the cascade's FPR can be compared).

Usage: python3 t2_analyze.py --scores DIR [DIR ...] --labels labels.jsonl --out t2_report.json
"""
import argparse, glob, hashlib, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--scores", nargs="+", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--splits", default="conflicting,oneoff")
ap.add_argument("--boot", type=int, default=1000)
a = ap.parse_args()


def auroc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, r); r = (sums / cnt)[inv]
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def tpr_at_fpr(y, s, fpr):
    neg = np.sort(s[~y])[::-1]
    k = int(np.floor(fpr * len(neg)))
    thr = neg[k] if k < len(neg) else -np.inf  # flag strictly above the (k+1)-th highest negative
    return float((s[y] > thr).mean()), float((s[~y] > thr).mean()), thr


# --- load probe scores ---
MX, MN, IDX, keys = [], [], [], None
for d in a.scores:
    for f in sorted(glob.glob(f"{d}/scores_*.npz")):
        z = np.load(f); keys = list(z["keys"])
        MX.append(z["max"]); MN.append(z["mean"])
        IDX += [json.loads(l) for l in open(f.replace("scores_", "index_").replace(".npz", ".jsonl"))]
MX, MN = np.concatenate(MX), np.concatenate(MN)
pos = {(x["uid"], x["n"]): i for i, x in enumerate(IDX)}

# --- join labels ---
splits = set(a.splits.split(","))
L = [json.loads(l) for l in open(a.labels)]
rows = [x for x in L if x["split"] in splits and x["gt"] is not None and (x["uid"], x["n"]) in pos
        and x["monitor"] is not None and x["uid"].startswith(IDX[0]["uid"].split("|")[0])]
ii = np.array([pos[(x["uid"], x["n"])] for x in rows])
y = np.array([x["gt"] for x in rows])
mon = np.array([x["monitor"] for x in rows])
task = np.array([x["sample_id"] for x in rows])
half = np.array([int(hashlib.md5(t.encode()).hexdigest(), 16) % 2 for t in task])
sel, test = half == 0, half == 1
print(f"passages: {len(rows)} (pos {y.sum()}), rollouts {len({x['uid'] for x in rows})}, tasks {len(set(task))}; "
      f"selection {sel.sum()} (pos {y[sel].sum()}), held-out {test.sum()} (pos {y[test].sum()})")

# --- selection ---
grid = []
for agg, A in (("max", MX), ("mean", MN)):
    for L_ in range(A.shape[1]):
        for k, key in enumerate(keys):
            grid.append((auroc(y[sel], A[ii[sel], L_, k]), agg, L_, key))
grid.sort(key=lambda g: -np.nan_to_num(g[0], nan=-1))
best = grid[0]
agg, Lb, key = best[1], best[2], best[3]
A = MX if agg == "max" else MN
s = A[ii, Lb, keys.index(key)]
print("top-5 on selection half:", [(round(g[0], 3), g[1], g[2], g[3]) for g in grid[:5]])

# --- held-out evaluation ---
yt, st, mt = y[test], s[test], mon[test]
rng = np.random.default_rng(0)
tasks_t = np.unique(task[test])


def boot(fn):
    vals = []
    for _ in range(a.boot):
        pick = rng.choice(tasks_t, len(tasks_t))
        m = np.concatenate([np.where(task[test] == t)[0] for t in pick])
        v = fn(m)
        if not np.isnan(v):
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


res = {"selected": {"agg": agg, "layer": Lb, "key": key, "sel_auroc": best[0]},
       "n_test": int(test.sum()), "n_test_pos": int(yt.sum()), "n_test_tasks": len(tasks_t)}
res["probe_auroc"] = auroc(yt, st)
res["probe_auroc_ci"] = boot(lambda m: auroc(yt[m], st[m]))
m_tpr, m_fpr = float(mt[yt].mean()), float(mt[~yt].mean())
res["monitor"] = {"tpr": m_tpr, "fpr": m_fpr}
p_tpr, p_fpr, thr = tpr_at_fpr(yt, st, m_fpr)
res["probe_at_monitor_fpr"] = {"tpr": p_tpr, "fpr": p_fpr}
pf = st > thr
res["probe_or_monitor"] = {"tpr": float((pf | mt)[yt].mean()), "fpr": float((pf | mt)[~yt].mean())}
# cascade: probe at a looser FPR (2x monitor's, capped) gates the monitor
_, _, thr2 = tpr_at_fpr(yt, st, min(0.5, 2 * m_fpr + 0.05))
casc = (st > thr2) & mt
res["probe_then_monitor"] = {"tpr": float(casc[yt].mean()), "fpr": float(casc[~yt].mean()),
                             "monitor_calls_frac": float((st > thr2).mean())}
# by channel
res["by_channel"] = {}
for ch in ("reasoning", "content", "tool_calls"):
    m = np.array([x["channel"] == ch for x in rows])[test]
    res["by_channel"][ch] = {"n": int(m.sum()), "pos": int(yt[m].sum()), "probe_auroc": auroc(yt[m], st[m]),
                             "monitor_tpr": float(mt[m][yt[m]].mean()) if yt[m].any() else None}
# rollout level: max over passages vs any-positive
R = {}
for j in np.where(test)[0]:
    u = rows[j]["uid"]; R.setdefault(u, [False, -np.inf, False])
    R[u][0] |= bool(y[j]); R[u][1] = max(R[u][1], s[j]); R[u][2] |= bool(mon[j])
ry = np.array([v[0] for v in R.values()]); rs = np.array([v[1] for v in R.values()]); rm = np.array([v[2] for v in R.values()])
res["rollout_level"] = {"n": len(R), "pos": int(ry.sum()), "probe_auroc": auroc(ry, rs),
                        "monitor_tpr": float(rm[ry].mean()) if ry.any() else None,
                        "monitor_fpr": float(rm[~ry].mean()) if (~ry).any() else None}
# layer profile on held-out for the selected key/agg
res["layer_profile_heldout"] = [auroc(yt, A[ii[test], L_, keys.index(key)]) for L_ in range(A.shape[1])]
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "layer_profile_heldout"}, indent=1))
print("held-out AUROC by layer (every 4th):", [round(x, 2) for x in res["layer_profile_heldout"][3::4]])
