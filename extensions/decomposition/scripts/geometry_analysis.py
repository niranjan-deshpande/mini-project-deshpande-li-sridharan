"""Phase 4 item 1 (geometric) + item 7 (general cheating) + the CP C geometry, from synthetic-pair activations only.

Builds every DoM at L40 (turn primary, span secondary), split-half reliabilities, cosines with v_hack (raw and
disattenuated), R^2_geo of v_hack in span(C) with the baseline and shuffle nulls and the noise ceiling, Shapley shares
and unique contributions. Writes results/geometry/geometry.json and data/analysis/extra_directions.npz (directions that
need session-2 projection: legit_shortcut plain/colloquial).
"""
import json, sys
from itertools import combinations
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rhdecomp.directions import split_half_reliability, label_shuffled_doms  # noqa: E402
from rhdecomp.geometry import cosine, span_fit, null_span_r2  # noqa: E402
from rhdecomp.shapley import shapley_geo  # noqa: E402

S1 = ROOT / "data/gpu/s1"
OUT = ROOT / "results/geometry"; OUT.mkdir(parents=True, exist_ok=True)
CANDS = ["testvocab_swe", "testvocab_noncode", "task_failure", "legit_shortcut_plain", "eval_awareness", "grader_modeling"]
BASES = ["base_python_rust", "base_formal_casual", "base_british_american", "base_snake_camel", "base_prose_bullets",
         "base_loop_comprehension", "base_print_logging", "base_json_yaml", "base_metric_imperial", "base_i_we"]
GC = ["gc_template", "gc_third_person", "gc_first_person", "gc_judgment", "gc_ai_offdomain"]

variant = {}
for f in (ROOT / "data/final").glob("pairs_*.jsonl"):
    for l in open(f):
        p = json.loads(l)
        variant[f"{p['family']}_{p['meta']['i']:03d}"] = p.get("variant") or p["meta"].get("variant")


def load(d, harvest="turn"):
    z = np.load(S1 / "pairs" / f"{d}.npz")
    li = list(z["layers"]).index(40)
    A, B = z[f"{harvest}_hack"][:, li].astype(np.float64), z[f"{harvest}_honest"][:, li].astype(np.float64)
    return A, B, list(z["ids"])


def pairs_for(name, harvest="turn"):
    if name.startswith("legit_shortcut_"):
        A, B, ids = load("legit_shortcut", harvest)
        m = np.array([variant[i] == name.split("_")[-1] for i in ids])
        return A[m], B[m]
    if name == "claude":
        z = np.load(S1 / "claude_pair_acts.npz"); li = list(z["layers"]).index(40)
        return z[f"{harvest}_hack"][:, li].astype(np.float64), z[f"{harvest}_honest"][:, li].astype(np.float64)
    A, B, _ = load(name, harvest)
    return A, B


res = {}
for harvest in ("turn", "span"):
    names = ["hack", "claude"] + CANDS + ["legit_shortcut", "legit_shortcut_colloquial"] + BASES + GC
    V, rel, n = {}, {}, {}
    for nm in names:
        A, B = pairs_for(nm, harvest)
        V[nm] = (A - B).mean(0); n[nm] = len(A)
        rel[nm] = split_half_reliability(A, B, n_splits=100, seed=0).rho
    v = V["hack"]
    C = np.column_stack([V[c] for c in CANDS])
    fit = span_fit(C, v)
    base_pool = np.stack([V[b] for b in BASES])
    null_base = null_span_r2(v, base_pool, k=len(CANDS), n_draws=1000)
    Ah, Bh = pairs_for("hack", harvest)
    shuf = label_shuffled_doms(Ah, Bh, k=40, seed=0)
    null_shuf = null_span_r2(v, np.asarray(shuf), k=len(CANDS), n_draws=1000)
    sh = shapley_geo(C, v)
    cos = {nm: cosine(V[nm], v) for nm in names if nm != "hack"}
    out = {
        "n_pairs": n, "reliability": rel,
        "cos_with_vhack": cos,
        "cos_disattenuated": {nm: cos[nm] / np.sqrt(rel["hack"] * rel[nm]) for nm in cos},
        "baseline_cos_abs": {"mean": float(np.mean([abs(cos[b]) for b in BASES])), "max": float(np.max([abs(cos[b]) for b in BASES]))},
        "r2_geo": fit.r2, "r2_geo_over_rho": fit.r2 / rel["hack"], "beta": dict(zip(CANDS, fit.beta.tolist())),
        "null_baselines_r2": {"mean": float(null_base.mean()), "p95": float(np.percentile(null_base, 95)), "max": float(null_base.max()),
                              "n_subsets": int(len(null_base))},
        "null_shuffled_r2": {"mean": float(null_shuf.mean()), "p95": float(np.percentile(null_shuf, 95))},
        "random_kd": len(CANDS) / len(v),
        "shapley": dict(zip(CANDS, np.asarray(sh["shapley"]).tolist())) if isinstance(sh, dict) else None,
        "unique": dict(zip(CANDS, np.asarray(sh["unique"]).tolist())) if isinstance(sh, dict) else None,
        "candidate_cos_matrix": {a: {b: cosine(V[a], V[b]) for b in CANDS} for a in CANDS},
        "cos_luna_vs_claude": cosine(V["hack"], V["claude"]),
    }
    res[harvest] = out
    if harvest == "turn":
        np.savez(ROOT / "data/analysis/extra_directions.npz", V=np.stack([V["legit_shortcut_plain"], V["legit_shortcut_colloquial"]]),
                 names=np.array(["legit_shortcut_plain__turn", "legit_shortcut_colloquial__turn"]))
        np.savez(ROOT / "data/analysis/dom_turn_L40.npz", **{k: V[k] for k in V})
json.dump(res, open(OUT / "geometry.json", "w"), indent=1, default=float)
t = res["turn"]
print(f"reliability v_hack {t['reliability']['hack']:.3f}; cos(Luna, Claude) {t['cos_luna_vs_claude']:.3f}")
print(f"R2_geo {t['r2_geo']:.3f} (ceiling-adjusted {t['r2_geo_over_rho']:.3f}); baseline null mean {t['null_baselines_r2']['mean']:.3f} "
      f"p95 {t['null_baselines_r2']['p95']:.3f}; shuffled null p95 {t['null_shuffled_r2']['p95']:.3f}; k/d {t['random_kd']:.4f}")
for nm in CANDS + ["legit_shortcut", "legit_shortcut_colloquial"] + GC:
    print(f"  {nm:26s} n={t['n_pairs'][nm]:3d} rho={t['reliability'][nm]:.3f} cos={t['cos_with_vhack'][nm]:+.3f} "
          f"disatt={t['cos_disattenuated'][nm]:+.3f}" + (f" shapley={t['shapley'][nm]:.3f} unique={t['unique'][nm]:.3f}" if nm in CANDS and t['shapley'] else ""))
print(f"  baselines |cos| mean {t['baseline_cos_abs']['mean']:.3f} max {t['baseline_cos_abs']['max']:.3f}")
