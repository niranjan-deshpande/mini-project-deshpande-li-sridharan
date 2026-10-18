"""Phase 5: how much of v_hack's token-level output is explained by Qwen's emotion directions?

Emotion vectors: GPU session 2 (Sofroniew et al. recipe, 40-emotion subset, L40, denoised primary; notes/emotions_method.md).
Targets: token-level scores of v_hack (hack__turn), the 6 candidates, the 10 baselines (the null: "a subspace of many
emotion directions can explain a lot of almost anything") and 3 balanced-shuffle DoMs (diagnostic only; they keep part of
v_hack, see DECISIONS.md). All assistant tokens of impossible-split rollouts.

  1. In-sample OLS R^2 on the held-out half (all tokens; comparable to Phase 4's functional R^2), rollout-bootstrap CI,
     and cluster-level Shapley shares over the 10 emotion clusters.
  2. Penalized fits (brief's spec): Lasso and Ridge on standardized emotion scores, lambda by GroupKFold CV over
     rollouts on the SELECTION half (token subsample), R^2 on every held-out-half token. lambda_min and lambda_1se.
  3. Lasso selection frequencies (rollout bootstrap) for v_hack, reported per emotion and per cluster; lasso entry order.
  4. Geometry: cosines of each direction with the unit emotion vectors; R^2 of each direction in their span.
  5. Desperate / frustrated / anxious / calm (Sofroniew's reward-hacking emotions): passage-level hack AUROC (max and mean
     pooled, length-controlled), response to contradiction reasoning on clean passages, and drift over the rollout.
Every moment is accumulated per rollout (within-rollout scatter + mean), so all R^2s use every token in one pass.
Output: results/phase5/phase5.json
"""
import glob, hashlib, json, sys, time
from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import lasso_path
from sklearn.model_selection import GridSearchCV, GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rhdecomp.emotions import DEFAULT_ALPHAS, _model, selection_frequencies  # noqa: E402
from rhdecomp.geometry import cosine, span_fit  # noqa: E402
from rhdecomp.stats import auroc, length_residualize as _lr  # noqa: E402
from rhdecomp.confounds import confound_response_auroc  # noqa: E402
length_residualize = lambda s, n: _lr({"x": s}, n)["x"]

S1 = ROOT / "data/gpu/s1"
OUT = ROOT / "results/phase5"; OUT.mkdir(parents=True, exist_ok=True)
V_NAME = "hack__turn"
CANDS = ["testvocab_swe__turn", "testvocab_noncode__turn", "task_failure__turn", "legit_shortcut_plain__turn",
         "eval_awareness__turn", "grader_modeling__turn"]
BASES = [f"{b}__turn" for b in ["base_python_rust", "base_formal_casual", "base_british_american", "base_snake_camel",
         "base_prose_bullets", "base_loop_comprehension", "base_print_logging", "base_json_yaml", "base_metric_imperial", "base_i_we"]]
SHUF = [f"null_hack_shuffle{i:02d}" for i in range(3)]
TARGETS = [V_NAME] + CANDS + BASES + SHUF + ["claude_rebuilt__turn"]
FOCUS = ["desperate", "frustrated", "anxious", "calm"]
N_FIT = 300_000          # selection-half token subsample for the penalized fits
N_BOOT_R2, N_BOOT_SEL = 1000, 200
rng = np.random.default_rng(0)

EZ = np.load(S1 / "emotions/emotion_vectors.npz")
EMO = [str(x) for x in EZ["names"]]
ecols = json.load(open(S1 / "emotions/rollouts/columns.json"))["columns"]
assert ecols[:40] == [f"{e}__raw" for e in EMO] and ecols[40:] == [f"{e}__denoised" for e in EMO]
CLUSTERS = json.load(open(ROOT / "prompts/emotions/emotions.json"))["subset_40_by_cluster"]
CL_NAMES = list(CLUSTERS)
cl_of = {e: c for c, es in CLUSTERS.items() for e in es}
assert set(cl_of) == set(EMO)

nT, nE = len(TARGETS), len(EMO)
ZCOLS = TARGETS + [f"{e}__den" for e in EMO] + [f"{e}__raw" for e in EMO]
iT = np.arange(nT); iDen = nT + np.arange(nE); iRaw = nT + nE + np.arange(nE)

# ---------------------------------------------------------------- labels
gt = {(x["uid"], x["n"]): x for x in map(json.loads, open(ROOT.parents[1] / "reproduction/multi-model/results/labels/judge_full.jsonl"))
      if x["uid"].startswith("qwen")}
p2 = {(x["uid"], x["n"]): x for x in map(json.loads, open(ROOT / "data/final/phase2_labels.jsonl"))}

# ---------------------------------------------------------------- one pass over rollouts
names = list(np.load(S1 / "directions.npz")["names"])
extra_cols = json.load(open(S1 / "extra/columns.json"))["columns"]
frac = {0: N_FIT / 3.28e6}
mom = {0: [], 1: []}            # per rollout: (n, mean [p], within scatter [p, p])
sub = []                         # selection-half subsample rows of Z, with rollout id
sub_g = []
PASS = []                        # held-out labelled passages: pooled emotion/baseline scores + labels
t0 = time.time()
for f in sorted(glob.glob(str(S1 / "rollouts/*.npz"))):
    meta = json.load(open(f.replace(".npz", ".passages.json")))
    if meta["split"] not in ("conflicting", "oneoff"):
        continue
    h = int(hashlib.md5(meta["sample_id"].encode()).hexdigest(), 16) % 2
    z = np.load(f); proj = z["proj"]; stem = Path(f).stem
    ex = np.load(S1 / "extra" / f"{stem}.npy"); em = np.load(S1 / "emotions/rollouts" / f"{stem}.npy")
    get = lambda c: proj[:, names.index(c)] if c in names else ex[:, extra_cols.index(c)]
    Z = np.column_stack([get(c) for c in TARGETS] + [em[:, 40:], em[:, :40]]).astype(np.float64)
    m = Z.mean(0); Zc = Z - m
    mom[h].append((len(Z), m, Zc.T @ Zc, meta["uid"]))
    if h == 0:
        keep = rng.random(len(Z)) < frac[0]
        sub.append(Z[keep]); sub_g.append(np.full(keep.sum(), len(mom[0]) - 1))
        continue
    pos = z["pos"]; where = {int(p): i for i, p in enumerate(pos)}
    nmax = max(p["n"] for p in meta["passages"])
    for p in meta["passages"]:
        idx = [where[t] for t in range(p["tok0"], p["tok1"]) if t in where]
        g = gt.get((meta["uid"], p["n"]), {})
        if not idx:
            continue
        q = p2.get((meta["uid"], p["n"]), {})
        X = Z[idx]
        PASS.append({"uid": meta["uid"], "rollout": len(mom[1]) - 1, "n": p["n"], "rel_pos": p["n"] / max(nmax, 1),
                     "sample_id": meta["sample_id"], "n_tokens": len(idx), "hack": g.get("gt"),
                     "contradiction": q.get("contradiction_failure"),
                     "max": X.max(0), "mean": X.mean(0)})
print(f"loaded {len(mom[0])} selection + {len(mom[1])} held-out rollouts in {time.time() - t0:.0f}s")


def pooled(ms, w=None):
    """Pooled (mean, covariance) over rollouts, rollout g weighted w[g] (bootstrap counts); stable parallel formula."""
    n = np.array([x[0] for x in ms], float); M = np.stack([x[1] for x in ms]); W = np.stack([x[2] for x in ms])
    w = np.ones(len(ms)) if w is None else np.asarray(w, float)
    N = w @ n; mu = (w * n) @ M / N; D = M - mu
    S = np.tensordot(w, W, 1) + np.einsum("g,gi,gj->ij", w * n, D, D)
    return mu, S / N


def ols_r2(C, y, X):
    """R^2 of OLS (with intercept) of column y on columns X, from a covariance matrix C."""
    G, c = C[np.ix_(X, X)], C[X, y]
    return float(c @ np.linalg.pinv(G, rtol=1e-10, hermitian=True) @ c / C[y, y])


def heldout_r2(ms, y, X, a, b, w=None):
    """R^2 on held-out rollouts of the fixed prediction a + Z[:, X] b for column y (SSE per rollout from moments)."""
    w = np.ones(len(ms)) if w is None else np.asarray(w, float)
    u = np.zeros(len(ZCOLS)); u[y] = 1; u[X] -= b         # residual r = u.Z - a
    sse = sum(wi * (u @ W @ u + n * (u @ m - a) ** 2) for wi, (n, m, W, _) in zip(w, ms))
    mu, C = pooled(ms, w); N = w @ np.array([x[0] for x in ms])
    return float(1 - sse / (C[y, y] * N))


res = {"n": {"selection_rollouts": len(mom[0]), "heldout_rollouts": len(mom[1]),
             "heldout_tokens": int(sum(x[0] for x in mom[1])), "fit_subsample_tokens": int(sum(len(s) for s in sub))},
       "emotions": EMO, "clusters": CLUSTERS}

# ---------------------------------------------------------------- 1. in-sample OLS R^2 on held-out tokens (+ cluster Shapley)
muH, CH = pooled(mom[1])
G1 = len(mom[1])
boot_w = [np.bincount(rng.integers(0, G1, G1), minlength=G1) for _ in range(N_BOOT_R2)]
boot_C = [pooled(mom[1], w)[1] for w in boot_w]
ols = {}
for j, t in enumerate(TARGETS):
    den = ols_r2(CH, j, iDen); raw = ols_r2(CH, j, iRaw)
    bs = np.array([ols_r2(C, j, iDen) for C in boot_C])
    ols[t] = {"r2_denoised": den, "ci_denoised": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
              "r2_raw": raw, "r2_on_10_baselines": ols_r2(CH, j, [TARGETS.index(b) for b in BASES if b != t])}
base_den = np.array([ols[b]["r2_denoised"] for b in BASES])
res["ols_heldout"] = {"per_target": ols,
                      "null_baselines_denoised": {"mean": float(base_den.mean()), "min": float(base_den.min()),
                                                  "max": float(base_den.max()), "values": dict(zip(BASES, base_den.tolist()))},
                      "vhack_rank_among_baselines": int((base_den >= ols[V_NAME]["r2_denoised"]).sum())}


def cluster_shapley(C, y):
    k = len(CL_NAMES)
    cols = [[iDen[EMO.index(e)] for e in CLUSTERS[c]] for c in CL_NAMES]
    tab = np.array([ols_r2(C, y, sum((cols[i] for i in range(k) if mk >> i & 1), [])) if mk else 0.0 for mk in range(1 << k)])
    wts = [factorial(s) * factorial(k - s - 1) / factorial(k) for s in range(k)]
    phi = np.zeros(k)
    for mk in range(1 << k):
        s = bin(mk).count("1")
        for i in range(k):
            if not mk >> i & 1:
                phi[i] += wts[s] * (tab[mk | 1 << i] - tab[mk])
    single = {CL_NAMES[i]: float(tab[1 << i]) for i in range(k)}
    return dict(zip(CL_NAMES, phi.tolist())), single


shap = {}
for t in [V_NAME] + CANDS + BASES:
    phi, single = cluster_shapley(CH, TARGETS.index(t))
    shap[t] = {"shapley": phi, "single_cluster_r2": single}
res["cluster_shapley_heldout"] = shap
res["single_emotion_r2_vhack"] = {e: ols_r2(CH, 0, [iDen[i]]) for i, e in enumerate(EMO)}
# token-level correlation of every target with every denoised emotion; baselines give the null for single emotions
sd = np.sqrt(np.diag(CH))
corr = CH[np.ix_(iT, iDen)] / np.outer(sd[iT], sd[iDen])
res["token_corr"] = {t: dict(zip(EMO, corr[j].tolist())) for j, t in enumerate(TARGETS)}
nb = np.abs(corr[[TARGETS.index(b) for b in BASES]])
res["token_corr_null"] = {"abs_p95": float(np.percentile(nb, 95)), "abs_max": float(nb.max()),
                          "per_emotion_abs_max": dict(zip(EMO, nb.max(0).tolist()))}
# does emotion space add anything beyond the 6 candidates (and vice versa)? in-sample OLS on held-out tokens
iC = [TARGETS.index(c) for c in CANDS]
both = ols_r2(CH, 0, iC + list(iDen))
res["joint_candidates_emotions_vhack"] = {"r2_candidates": ols_r2(CH, 0, iC), "r2_emotions": ols_r2(CH, 0, iDen), "r2_both": both,
                                          "unique_emotions": both - ols_r2(CH, 0, iC), "unique_candidates": both - ols_r2(CH, 0, iDen)}
res["joint_null_baselines"] = {b: {"unique_emotions": ols_r2(CH, TARGETS.index(b), iC + list(iDen)) - ols_r2(CH, TARGETS.index(b), iC)}
                               for b in BASES}
np.savez(OUT / "moments_heldout.npz", mean=muH, cov=CH, cols=np.array(ZCOLS))
print(f"OLS done {time.time() - t0:.0f}s: v_hack R2 den {ols[V_NAME]['r2_denoised']:.3f}, baselines mean {base_den.mean():.3f}")

# ---------------------------------------------------------------- 2. penalized fits (CV on selection half), held-out R^2
Zs = np.concatenate(sub); gs = np.concatenate(sub_g)
mu0, C0 = pooled(mom[0])


def penalized(j, X, kind):
    y = (Zs[:, j] - mu0[j]) / np.sqrt(C0[j, j])            # standardized target: one alpha grid for every target
    alphas = DEFAULT_ALPHAS[kind]
    gsr = GridSearchCV(_model(kind), {f"{kind}__alpha": list(alphas)}, cv=GroupKFold(5), scoring="r2", n_jobs=-1)
    gsr.fit(Zs[:, X], y, groups=gs)
    mean, se = gsr.cv_results_["mean_test_score"], gsr.cv_results_["std_test_score"] / np.sqrt(5)
    best = int(np.argmax(mean))
    ok = np.flatnonzero(mean >= mean[best] - se[best])
    a1se = float(alphas[ok.max()])                         # most regularized alpha within 1 SE of the best
    out = {"cv_r2": float(mean[best]), "cv_curve": dict(zip(map(float, alphas), mean.tolist()))}
    for tag, al in (("min", float(alphas[best])), ("1se", a1se)):
        pipe = _model(kind, al).fit(Zs[:, X], y)
        sc, est = pipe[0], pipe[-1]
        b_std = est.coef_ * np.sqrt(C0[j, j])              # back to target units
        b = b_std / sc.scale_; a = mu0[j] + est.intercept_ * np.sqrt(C0[j, j]) - b @ sc.mean_
        out[tag] = {"alpha": al, "heldout_r2": heldout_r2(mom[1], j, X, a, b), "n_nonzero": int((est.coef_ != 0).sum()),
                    "coef_std": dict(zip(EMO, est.coef_.tolist()))}
    return out


pen = {}
for t in [V_NAME] + CANDS + BASES + SHUF:
    j = TARGETS.index(t)
    pen[t] = {"lasso": penalized(j, iDen, "lasso"), "ridge": penalized(j, iDen, "ridge")}
    print(f"  {t}: lasso {pen[t]['lasso']['min']['heldout_r2']:.3f} ridge {pen[t]['ridge']['min']['heldout_r2']:.3f} "
          f"({time.time() - t0:.0f}s)", flush=True)
pen["hack__turn__raw"] = {"lasso": penalized(0, iRaw, "lasso"), "ridge": penalized(0, iRaw, "ridge")}
res["penalized"] = pen
bl = np.array([pen[b]["lasso"]["min"]["heldout_r2"] for b in BASES]); br = np.array([pen[b]["ridge"]["min"]["heldout_r2"] for b in BASES])
res["penalized_null_baselines"] = {"lasso_mean": float(bl.mean()), "lasso_max": float(bl.max()),
                                   "ridge_mean": float(br.mean()), "ridge_max": float(br.max())}

# ---------------------------------------------------------------- 3. lasso selection frequencies + entry order (v_hack)
yv = (Zs[:, 0] - mu0[0]) / np.sqrt(C0[0, 0])
sel = {}
for tag in ("min", "1se"):
    sf = selection_frequencies(yv, Zs[:, iDen], gs, pen[V_NAME]["lasso"][tag]["alpha"], n_boot=N_BOOT_SEL, seed=0)
    nz = sf["coefs"] != 0
    sel[tag] = {"emotion": {e: {"freq": float(sf["freq"][i]), "coef_mean": float(sf["coef_mean"][i]),
                                "coef_sd": float(sf["coef_sd"][i])} for i, e in enumerate(EMO)},
                "cluster_any_selected": {c: float(nz[:, [EMO.index(e) for e in CLUSTERS[c]]].any(1).mean()) for c in CL_NAMES},
                "cluster_net_coef_mean": {c: float(sf["coefs"][:, [EMO.index(e) for e in CLUSTERS[c]]].sum(1).mean())
                                          for c in CL_NAMES}}
res["lasso_selection_vhack"] = sel
Xs = (Zs[:, iDen] - Zs[:, iDen].mean(0)) / Zs[:, iDen].std(0)
alphas_p, coefs_p, _ = lasso_path(Xs, yv - yv.mean(), n_alphas=200, eps=1e-4)
entry = []
for i in np.argsort([np.argmax(coefs_p[k] != 0) if (coefs_p[k] != 0).any() else 10 ** 9 for k in range(nE)]):
    k0 = np.argmax(coefs_p[i] != 0)
    if (coefs_p[i] != 0).any():
        entry.append({"emotion": EMO[i], "alpha": float(alphas_p[k0]), "sign": int(np.sign(coefs_p[i][min(k0 + 5, len(alphas_p) - 1)]))})
res["lasso_entry_order_vhack"] = entry[:12]
print(f"selection done {time.time() - t0:.0f}s")

# ---------------------------------------------------------------- 4. geometry
D = np.load(ROOT / "data/analysis/dom_turn_L40.npz")
vec = lambda n: D[n.replace("__turn", "")]
Eden = EZ["denoised"]; Eraw = EZ["raw"]
geo = {}
for t in [V_NAME] + CANDS + BASES:
    v = vec(t)
    geo[t] = {"cos_denoised": {e: cosine(v, Eden[i]) for i, e in enumerate(EMO)},
              "span_r2_denoised": span_fit(Eden.T, v).r2, "span_r2_raw": span_fit(Eraw.T, v).r2}
null_cos = np.abs([geo[b]["cos_denoised"][e] for b in BASES for e in EMO])
res["geometry"] = {"per_direction": geo, "null_abs_cos_p95": float(np.percentile(null_cos, 95)),
                   "null_abs_cos_max": float(null_cos.max()),
                   "null_span_r2_denoised": {"mean": float(np.mean([geo[b]["span_r2_denoised"] for b in BASES])),
                                             "max": float(np.max([geo[b]["span_r2_denoised"] for b in BASES]))},
                   "chance_span_r2_isotropic": (nE - 1) / Eden.shape[1]}

# ---------------------------------------------------------------- 5. focus emotions on passages (held-out, labelled)
lab = np.array([p["hack"] is not None for p in PASS])
P = [p for p, l in zip(PASS, lab) if l]
y = np.array([bool(p["hack"]) for p in P]); ntok = np.array([p["n_tokens"] for p in P])
contra = np.array([-1 if p["contradiction"] is None else int(p["contradiction"]) for p in P])
foc = {}
for name in FOCUS + [V_NAME] + BASES:
    jj = iDen[EMO.index(name)] if name in EMO else TARGETS.index(name)
    d = {}
    for pool in ("max", "mean"):
        s = np.array([p[pool][jj] for p in P])
        d[pool] = {"hack_auroc": auroc(y, s), "hack_auroc_len": auroc(y, length_residualize(s, ntok)),
                   "contradiction_response_clean": confound_response_auroc(y, s, contra)["auroc"]}
    # drift over the rollout: Spearman of passage-mean score with passage index, per held-out rollout (all passages)
    rhos = []
    for r in sorted({p["rollout"] for p in PASS}):
        ps = [p for p in PASS if p["rollout"] == r]
        if len(ps) >= 5:
            rhos.append(spearmanr([p["n"] for p in ps], [p["mean"][jj] for p in ps])[0])
    d["drift_spearman_median"] = float(np.nanmedian(rhos)); d["drift_frac_positive"] = float(np.nanmean(np.array(rhos) > 0))
    foc[name] = d
for pool in ("max", "mean"):
    nb = np.array([max(foc[b][pool]["hack_auroc"], 1 - foc[b][pool]["hack_auroc"]) for b in BASES])
    nbl = np.array([max(foc[b][pool]["hack_auroc_len"], 1 - foc[b][pool]["hack_auroc_len"]) for b in BASES])
    foc[f"null_baselines_{pool}"] = {"abs_auroc_max": float(nb.max()), "abs_auroc_len_max": float(nbl.max())}
all_emo_auc = {e: {pool: auroc(y, length_residualize(np.array([p[pool][iDen[i]] for p in P]), ntok)) for pool in ("max", "mean")}
               for i, e in enumerate(EMO)}
res["focus_passages"] = {"n_passages": int(len(P)), "n_hack": int(y.sum()), "per_score": foc,
                         "all_emotions_hack_auroc_len": all_emo_auc,
                         "corr_vhack_tokens": {e: float(CH[0, iDen[EMO.index(e)]] / np.sqrt(CH[0, 0] * CH[iDen[EMO.index(e)], iDen[EMO.index(e)]]))
                                               for e in EMO}}

json.dump(res, open(OUT / "phase5.json", "w"), indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
print(f"done {time.time() - t0:.0f}s")
print(json.dumps({"ols_vhack": ols[V_NAME], "ols_null": res["ols_heldout"]["null_baselines_denoised"],
                  "pen_vhack": {k: {t: pen[V_NAME][k][t]["heldout_r2"] for t in ("min", "1se")} for k in ("lasso", "ridge")},
                  "pen_null": res["penalized_null_baselines"]}, indent=1, default=float))
