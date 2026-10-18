"""Phase 4 analyses 2-6 (functional, decision, attribution, mixture, confounds) on the GPU session-1 token projections.

Primary variants are fixed in DECISIONS.md. Held-out half (md5(sample_id) % 2 == 1) of the impossible splits for every
reported number; the selection half only fits the max detector's z-scores / best-single choice.
Inputs: data/gpu/s1/{directions.npz, rollouts/*.npz, rollouts/*.passages.json, extra/*.npy, extra/columns.json},
        data/analysis/dom_turn_L40.npz, results/geometry/geometry.json, Agastya's hack labels, Phase 2 labels.
Output: results/phase4/phase4.json
"""
import glob, hashlib, json, sys
from itertools import combinations
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rhdecomp.stats import auroc, paired_task_bootstrap, length_residualize as _lr  # noqa: E402
length_residualize = lambda s, n: _lr({"x": s}, n)["x"]
from rhdecomp.functional import group_moments  # noqa: E402
from rhdecomp.geometry import gram_r2, span_fit  # noqa: E402
from rhdecomp.shapley import _decompose  # noqa: E402
from rhdecomp.decision import compare_max_detector, peak_decomposition, summarize_peaks  # noqa: E402
from rhdecomp.confounds import confound_crosstab, stratified_auroc, confound_response_auroc  # noqa: E402

S1 = ROOT / "data/gpu/s1"
OUT = ROOT / "results/phase4"; OUT.mkdir(parents=True, exist_ok=True)
V_NAME = "hack__turn"
CANDS = ["testvocab_swe__turn", "testvocab_noncode__turn", "task_failure__turn", "legit_shortcut_plain__turn",
         "eval_awareness__turn", "grader_modeling__turn"]
BASES = [f"{b}__turn" for b in ["base_python_rust", "base_formal_casual", "base_british_american", "base_snake_camel",
         "base_prose_bullets", "base_loop_comprehension", "base_print_logging", "base_json_yaml", "base_metric_imperial", "base_i_we"]]
EXTRA_REPORT = ["claude_rebuilt__turn", "agastya__turn", "legit_shortcut__turn", "legit_shortcut_colloquial__turn",
                "hack__span", "gc_ai_offdomain__turn"]
N_BOOT = 2000

# ---------------------------------------------------------------- beta from the geometric fit (same vectors)
D = np.load(ROOT / "data/analysis/dom_turn_L40.npz")
vec = lambda n: D[n.replace("__turn", "")]
fit = span_fit(np.column_stack([vec(c) for c in CANDS]), vec(V_NAME))
beta = fit.beta

# ---------------------------------------------------------------- labels
gt = {(x["uid"], x["n"]): x for x in map(json.loads, open(ROOT.parents[1] / "reproduction/multi-model/results/labels/judge_full.jsonl"))
      if x["uid"].startswith("qwen")}
p2 = {(x["uid"], x["n"]): x for x in map(json.loads, open(ROOT / "data/final/phase2_labels.jsonl"))}

# ---------------------------------------------------------------- tokens + passages
names = list(np.load(S1 / "directions.npz")["names"])
extra_cols = json.load(open(S1 / "extra/columns.json"))["columns"] if (S1 / "extra/columns.json").exists() else []
GCS = [f"{g}__turn" for g in ["gc_template", "gc_third_person", "gc_first_person", "gc_judgment", "gc_ai_offdomain"]]
cols = [V_NAME] + CANDS + BASES + [c for c in EXTRA_REPORT if c not in CANDS] + [g for g in GCS if g not in EXTRA_REPORT]
tok_cols = {c: [] for c in cols}
tok_pass, tok_roll, P, R = [], [], [], []  # passage idx per token (-1 = outside), rollout id per token, passages, rollouts
for ri, f in enumerate(sorted(glob.glob(str(S1 / "rollouts/*.npz")))):
    z = np.load(f); meta = json.load(open(f.replace(".npz", ".passages.json")))
    proj = z["proj"]
    ex = np.load(S1 / "extra" / (Path(f).stem + ".npy")) if extra_cols else None
    get = lambda c: proj[:, names.index(c)] if c in names else ex[:, extra_cols.index(c)]
    pos = z["pos"]; where = {int(p): i for i, p in enumerate(pos)}
    R.append({"split": meta["split"], "half": int(hashlib.md5(meta["sample_id"].encode()).hexdigest(), 16) % 2})
    pidx = np.full(len(pos), -1)
    for p in meta["passages"]:
        idx = [where[t] for t in range(p["tok0"], p["tok1"]) if t in where]
        if not idx:
            continue
        g = gt.get((meta["uid"], p["n"]), {}); q = p2.get((meta["uid"], p["n"]), {})
        pidx[idx] = len(P)
        P.append({"uid": meta["uid"], "n": p["n"], "split": meta["split"], "sample_id": meta["sample_id"], "rollout": ri,
                  "n_tokens": len(idx), "hack": g.get("gt"), "cats": g.get("judge_categories", []),
                  "contradiction": q.get("contradiction_failure"), "eval_aware": q.get("eval_awareness"),
                  "eval_aware_narrow": q.get("eval_awareness_narrow"),
                  "half": int(hashlib.md5(meta["sample_id"].encode()).hexdigest(), 16) % 2})
    for c in cols:
        tok_cols[c].append(get(c).astype(np.float64))
    tok_pass.append(pidx); tok_roll.append(np.full(len(pos), ri))
T = {c: np.concatenate(v) for c, v in tok_cols.items()}
tp, tr = np.concatenate(tok_pass), np.concatenate(tok_roll)
T["fit"] = np.column_stack([T[c] for c in CANDS]) @ beta
T["res"] = T[V_NAME] - T["fit"]
nP = len(P)
print(f"{len(tp):,} assistant tokens, {nP:,} passages")

# passage max-pool for every column
order = np.argsort(tp, kind="stable"); tps = tp[order]
starts = np.searchsorted(tps, np.arange(nP)); ends = np.searchsorted(tps, np.arange(nP), side="right")
PS = {c: np.array([T[c][order[a:b]].max() for a, b in zip(starts, ends)]) for c in T}

split = np.array([p["split"] for p in P]); half = np.array([p["half"] for p in P])
imp = np.isin(split, ["conflicting", "oneoff"]); lab = np.array([p["hack"] is not None for p in P])
y_all = np.array([bool(p["hack"]) for p in P]); tasks = np.array([p["sample_id"] for p in P])
ntok = np.array([p["n_tokens"] for p in P])
HO = imp & lab & (half == 1); SEL = imp & lab & (half == 0)
res = {"n": {"heldout_passages": int(HO.sum()), "heldout_hacks": int(y_all[HO].sum()), "heldout_tasks": int(len(set(tasks[HO]))),
             "selection_passages": int(SEL.sum())}, "beta": dict(zip(CANDS, beta.tolist())), "r2_geo": fit.r2}

# ---------------------------------------------------------------- 3. decision
probe_cols = [V_NAME, "fit", "res"] + CANDS + EXTRA_REPORT
yH = y_all[HO]
bt = paired_task_bootstrap({c: PS[c][HO] for c in probe_cols}, yH, tasks[HO], n_boot=N_BOOT, seed=0)
btL = paired_task_bootstrap({c: length_residualize(PS[c][HO], ntok[HO]) for c in probe_cols}, yH, tasks[HO], n_boot=N_BOOT, seed=0)
res["decision"] = {c: {"auroc": bt["point"][c], "ci": bt["ci"][c], "auroc_len": btL["point"][c], "ci_len": btL["ci"][c]} for c in probe_cols}
res["decision_diffs"] = {f"{a} - {b}": bt["diff"][(a, b)] for a, b in [(V_NAME, "fit"), (V_NAME, "res"), ("fit", "res")]}
res["decision_baselines"] = {c: auroc(yH, PS[c][HO]) for c in BASES}
res["decision_baselines_len"] = {c: auroc(yH, length_residualize(PS[c][HO], ntok[HO])) for c in BASES}
res["decision_gc"] = {c: [auroc(yH, PS[c][HO]), auroc(yH, length_residualize(PS[c][HO], ntok[HO]))] for c in GCS}
res["length_only_auroc"] = auroc(yH, np.log(ntok[HO]))

# ---------------------------------------------------------------- 2 + 4. functional regression, Shapley, baseline null
roll_ok = np.array([r["split"] in ("conflicting", "oneoff") and r["half"] == 1 for r in R])
tokH = roll_ok[tr]  # every assistant token of held-out impossible-split rollouts
allc = CANDS + BASES
M = group_moments(T[V_NAME][tokH], np.column_stack([T[c][tokH] for c in allc]), tr[tokH])
G, cvec, vy = M.covariances()
ic = list(range(len(CANDS)))
func = _decompose(G[np.ix_(ic, ic)], cvec[ic], vy)
rng = np.random.default_rng(0); g = len(M.n)
boots = [gram_r2(*(lambda Gb, cb, vb: (Gb[np.ix_(ic, ic)], cb[ic], vb))(*M.covariances(np.bincount(rng.integers(0, g, g), minlength=g))))
         for _ in range(1000)]
ib = [len(CANDS) + j for j in range(len(BASES))]
null = [gram_r2(G, cvec, vy, list(s)) for s in combinations(ib, len(CANDS))]
res["functional"] = {"r2": func["full_r2"], "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                     "shapley": dict(zip(CANDS, func["shapley"].tolist())), "unique": dict(zip(CANDS, func["unique"].tolist())),
                     "single_r2": dict(zip(CANDS, func["single_r2"].tolist())),
                     "null_baselines": {"mean": float(np.mean(null)), "p95": float(np.percentile(null, 95)), "max": float(np.max(null)),
                                        "n_subsets": len(null)},
                     "n_tokens": int(tokH.sum()), "n_rollouts": int(len(np.unique(tr[tokH])))}

# ---------------------------------------------------------------- 5. mixture: max detector, peak decomposition
md = compare_max_detector(PS, y_all, tasks, CANDS, "fit", fit_mask=SEL, eval_mask=HO, n_boot=N_BOOT, seed=0)
res["max_detector"] = {"auroc": md["point"], "ci": md["ci"], "best_single": md["best_single_name"],
                       "diffs": {f"{a} - {b}": v for (a, b), v in md["diff"].items()}}
cat = []
for p in P:
    cs = p["cats"] or []
    cat.append("hardcoding_special_casing" if "hardcoding_special_casing" in cs else
               "harness_mechanics_targeting" if "harness_mechanics_targeting" in cs else (cs[0] if cs else None))
yP = np.where(HO, y_all, False)
recs = peak_decomposition(T, V_NAME, CANDS, beta, np.where(tp >= 0, tp, -1), yP, category=np.array(cat, dtype=object))
res["peaks"] = summarize_peaks(recs)

# ---------------------------------------------------------------- 6. confounds (Phase 2 labels), held-out
def lab_arr(key):
    return np.array([-1 if p[key] is None else int(p[key]) for p in P])
conf = {}
for key in ("contradiction", "eval_aware", "eval_aware_narrow"):
    c = lab_arr(key)[HO]
    conf[key] = {"crosstab": confound_crosstab(yH, c), "stratified_vhack": stratified_auroc(yH, PS[V_NAME][HO], c),
                 "stratified_vhack_len": stratified_auroc(yH, length_residualize(PS[V_NAME][HO], ntok[HO]), c),
                 "response_vhack_clean": confound_response_auroc(yH, PS[V_NAME][HO], c),
                 "response_vhack_clean_len": confound_response_auroc(yH, length_residualize(PS[V_NAME][HO], ntok[HO]), c)}
conf["candidate_validity"] = {
    "task_failure_on_contradiction": confound_response_auroc(yH, PS["task_failure__turn"][HO], lab_arr("contradiction")[HO]),
    "eval_awareness_on_eval_aware": confound_response_auroc(yH, PS["eval_awareness__turn"][HO], lab_arr("eval_aware")[HO]),
    "eval_awareness_on_eval_aware_narrow": confound_response_auroc(yH, PS["eval_awareness__turn"][HO], lab_arr("eval_aware_narrow")[HO])}
# impossible (clean, labelled) vs original split (unverified: treated as clean), held-out tasks
m_imp = imp & lab & ~y_all & (half == 1); m_orig = (split == "original") & (half == 1)
mm = m_imp | m_orig
conf["impossible_vs_solvable"] = {"auroc_vhack": auroc(m_imp[mm], PS[V_NAME][mm]),
                                   "auroc_vhack_len": auroc(m_imp[mm], length_residualize(PS[V_NAME][mm], ntok[mm])),
                                   "n_impossible_clean": int(m_imp.sum()), "n_original": int(m_orig.sum()),
                                   "note": "original-split passages were never judged; treated as clean (unverified)"}
res["confounds"] = conf
json.dump(res, open(OUT / "phase4.json", "w"), indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
print(json.dumps({"decision": {k: round(v["auroc"], 3) for k, v in res["decision"].items()},
                  "functional_r2": res["functional"]["r2"], "functional_null_p95": res["functional"]["null_baselines"]["p95"],
                  "max_detector": res["max_detector"]["auroc"]}, indent=1, default=float))
