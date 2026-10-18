"""Paper figures for the writeup: (a) cosines with v_hack, (b) geometric vs functional R^2 against the control null.

Sources: results/geometry/geometry.json (candidate/control cosines, geometric R^2), results/phase4/phase4.json (functional R^2),
results/phase5/phase5.json (emotion cosines, emotion R^2), data/analysis/dom_turn_L40.npz and
results/phase5/moments_heldout.npz (per-subset null values, recomputed here; their summaries match the JSONs).
Output: results/paper/fig_cosines.pdf, fig_r2.pdf (+ .png previews).
"""
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/paper"; OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True, "pdf.fonttype": 42, "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
RED, BLUE, GRAY = "#d62728", "#1f77b4", "#7f7f7f"
VH = r"$v_{\mathrm{hack}}$"

geo = json.load(open(ROOT / "results/geometry/geometry.json"))["turn"]
p4 = json.load(open(ROOT / "results/phase4/phase4.json"))
p5 = json.load(open(ROOT / "results/phase5/phase5.json"))

CANDS = [("legit_shortcut_plain", "Legitimate shortcut"), ("eval_awareness", "Eval awareness"),
         ("testvocab_swe", "Test vocab (code)"), ("task_failure", "Task failure"),
         ("grader_modeling", "Grader modeling"), ("testvocab_noncode", "Test vocab (non-code)")]
BASES = ["base_python_rust", "base_formal_casual", "base_british_american", "base_snake_camel", "base_prose_bullets",
         "base_loop_comprehension", "base_print_logging", "base_json_yaml", "base_metric_imperial", "base_i_we"]

# ------------------------------------------------------------------ (a) cosines
cos = geo["cos_with_vhack"]
cand_cos = [cos[k] for k, _ in CANDS]
base_cos = [cos[b] for b in BASES]
emo_cos = list(p5["geometry"]["per_direction"]["hack__turn"]["cos_denoised"].values())

fig, ax = plt.subplots(figsize=(3.75, 2.45))
ys = np.arange(len(CANDS))[::-1] + 2.4          # candidates on top, two summary rows below
ax.barh(ys, cand_cos, height=0.62, color=RED, label="Candidate confounders")
ax.scatter(base_cos, np.full(len(base_cos), 1.0), marker="|", s=90, linewidths=1.3, color=GRAY, label="Controls")
ax.scatter(emo_cos, np.full(len(emo_cos), 0.0), marker="|", s=90, linewidths=1.0, color=BLUE, label="Emotions")
ax.axvline(0, color="black", lw=0.6)
ax.axhline(1.85, color="0.85", lw=0.6)
ax.set_yticks(list(ys) + [1.0, 0.0], [n for _, n in CANDS] + ["Controls (10)", "Emotions (40)"])
ax.tick_params(axis="y", length=0)
ax.set_xlim(-0.3, 0.3)
ax.set_xticks(np.arange(-0.3, 0.31, 0.1), [r"$0$" if abs(t) < 1e-9 else rf"${t:+.1f}$".replace("-", "-") for t in np.arange(-0.3, 0.31, 0.1)])
ax.set_ylim(-0.6, ys[0] + 0.6)
ax.set_xlabel(f"Cosine similarity with {VH}")
ax.set_title("Cosine similarities", fontsize=9)
ax.grid(axis="x", color="0.92", lw=0.5); ax.set_axisbelow(True)
fig.savefig(OUT / "fig_cosines.pdf"); fig.savefig(OUT / "fig_cosines.png", dpi=300)
plt.close(fig)

# ------------------------------------------------------------------ (b) R^2: v_hack vs control null
d = np.load(ROOT / "data/analysis/dom_turn_L40.npz")


def geo_r2(v, Xs):
    X = np.stack(Xs, 1); b, *_ = np.linalg.lstsq(X, v, rcond=None)
    return float(np.sum((X @ b) ** 2) / np.sum(v ** 2))


m = np.load(ROOT / "results/phase5/moments_heldout.npz", allow_pickle=True)
S, cols = m["cov"], list(m["cols"])


def func_r2(y, X):
    y, X = cols.index(y), [cols.index(x) for x in X]
    return float(S[y, X] @ np.linalg.solve(S[np.ix_(X, X)], S[X, y]) / S[y, y])


subsets = list(itertools.combinations(BASES, 6))
null_geo_c = [geo_r2(d["hack"], [d[b] for b in s]) for s in subsets]
null_fun_c = [func_r2("hack__turn", [b + "__turn" for b in s]) for s in subsets]
assert abs(np.mean(null_geo_c) - geo["null_baselines_r2"]["mean"]) < 1e-6
assert abs(np.mean(null_fun_c) - p4["functional"]["null_baselines"]["mean"]) < 1e-6
pd = p5["geometry"]["per_direction"]
null_geo_e = [pd[b + "__turn"]["span_r2_denoised"] for b in BASES]
null_fun_e = [p5["ols_heldout"]["per_target"][b + "__turn"]["r2_denoised"] for b in BASES]

groups = [  # (x, v_hack value, null values, color)
    (0.0, geo["r2_geo"], null_geo_c, RED), (1.0, pd["hack__turn"]["span_r2_denoised"], null_geo_e, BLUE),
    (2.6, p4["functional"]["r2"], null_fun_c, RED), (3.6, p5["ols_heldout"]["per_target"]["hack__turn"]["r2_denoised"], null_fun_e, BLUE),
]
rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(2.6, 2.45))
w = 0.36
for x, v, null, c in groups:
    ax.bar(x - w / 2, v, width=w, color=c)
    ax.scatter(x + w / 2 + rng.uniform(-0.09, 0.09, len(null)), null, s=4 if len(null) > 20 else 9,
               color=GRAY, alpha=0.5 if len(null) > 20 else 0.9, linewidths=0)
ax.set_xticks([0.5, 3.1], ["Geometric", "Functional"])
ax.tick_params(axis="x", length=0)
ax.set_ylim(0, 1); ax.set_ylabel(r"$R^2$")
ax.set_title(r"$R^2_{\mathrm{geo}}$ and $R^2_{\mathrm{func}}$", fontsize=9)
ax.grid(axis="y", color="0.92", lw=0.5); ax.set_axisbelow(True)
h = [plt.Rectangle((0, 0), 1, 1, color=RED), plt.Rectangle((0, 0), 1, 1, color=BLUE),
     plt.Line2D([], [], ls="", marker="o", ms=3, color=GRAY)]
ax.legend(h, ["On 6 candidates", "On 40 emotions", "Controls (null)"], loc="upper left", frameon=False, handlelength=1.0)
fig.savefig(OUT / "fig_r2.pdf"); fig.savefig(OUT / "fig_r2.png", dpi=300)
plt.close(fig)
print("wrote", OUT)
