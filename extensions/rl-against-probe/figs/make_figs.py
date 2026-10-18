"""RL figures (held-out = 309 Impossible-LCB tasks, lenient prompt; impossible splits; hack = GPT-5.6 Sol judge).
fig1 training curves, fig2 held-out ROC, fig3 held-out score distributions, fig4 held-out outcomes,
fig5 probe vs reasoning monitor, fig6 text vs weights (cross-scoring)."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                     "savefig.bbox": "tight", "axes.titlesize": 10.5, "legend.frameon": False})
ARMS = ["armA", "armB", "armC", "armD"]
save = lambda fig, n: [fig.savefig(f"{OUT}/{n}.{e}") for e in ("png", "pdf")]
rng = np.random.default_rng(0)

# ---------- fig1: training trajectories, 10-step bins, impossible tasks ----------
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.4))
B = 10
for arm in ARMS:
    T = [r for r in training(arm) if r["imp"]]
    xs, hk, rc, ud, ci = [], [], [], [], []
    for b in range(0, 100, B):
        S = [r for r in T if b <= r["step"] < b + B]; H = [r for r in S if r["hack"]]
        f = np.array([r["score"] > TAU for r in H])
        xs.append(b + B / 2); hk.append(np.mean([r["hack"] for r in S])); rc.append(f.mean())
        ud.append(np.mean([r["hack"] and r["score"] <= TAU for r in S]))
        ci.append(np.sqrt(f.mean() * (1 - f.mean()) / len(f)) * 1.96)
    kw = dict(color=COLOR[arm], label=LABEL[arm], marker="o", ms=3.5, lw=1.8)
    ax[0].plot(xs, hk, **kw); ax[1].plot(xs, rc, **kw); ax[2].plot(xs, ud, **kw)
    ax[1].fill_between(xs, np.array(rc) - ci, np.array(rc) + ci, color=COLOR[arm], alpha=0.12, lw=0)
for a_, t in zip(ax, ["(a) Hack rate (judge)", "(b) Probe recall at τ (of hacks)", "(c) Undetected hacks (of all samples)"]):
    a_.set_title(t, loc="left"); a_.set_xlabel("RL step"); a_.set_ylim(0, 1); a_.set_xlim(0, 100)
    a_.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); a_.grid(axis="y", alpha=0.25)
ax[0].legend(loc="lower right", fontsize=8.5)
fig.suptitle("Training: impossible tasks, 10-step bins (16 tasks × 8 samples per step)", x=0.01, ha="left", fontsize=10.5, y=1.02)
save(fig, "fig1_training_curves")

# ---------- fig2: held-out ROC, paper Fig 8 style ----------
fig, ax = plt.subplots(figsize=(4.6, 4.3))
for m in MODELS:
    R = held_out(m); H = np.array([r["probe"] for r in R if r["verdict"]["hack"]]); N = np.array([r["probe"] for r in R if not r["verdict"]["hack"]])
    fpr, tpr = roc(H, N)
    ax.plot(fpr, tpr, color=COLOR[m], lw=1.8, label=f"{LABEL[m]}  (AUROC {auroc(H, N):.2f})")
    ax.plot((N > TAU).mean(), (H > TAU).mean(), "o", color=COLOR[m], ms=7, mec="white", mew=1.2, zorder=5)
ax.plot([0, 1], [0, 1], ":", color="#bbb", lw=1)
ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate (recall)"); ax.set_aspect("equal")
ax.set_title("Held-out ROC, layer-13 probe\n(dots = fixed threshold τ)", loc="left")
ax.legend(loc="lower right", fontsize=8)
save(fig, "fig2_heldout_roc")

# ---------- fig3: held-out score distributions ----------
fig, ax = plt.subplots(1, 5, figsize=(14, 2.9), sharex=True, sharey=True)
bins = np.linspace(0, 1.5e6, 61)
for a_, m in zip(ax, MODELS):
    R = held_out(m); H = np.array([r["probe"] for r in R if r["verdict"]["hack"]]); N = np.array([r["probe"] for r in R if not r["verdict"]["hack"]])
    a_.hist(N, bins, density=True, color="#9ecae1", alpha=0.9, label="not hack")
    a_.hist(H, bins, density=True, histtype="step", color=COLOR[m] if m != "base" else "#333", lw=1.8, label="hack")
    a_.axvline(TAU, color="k", ls="--", lw=1)
    a_.set_title(f"{LABEL[m]}\nrecall {np.mean(H > TAU):.0%} · FPR {np.mean(N > TAU):.0%}", loc="left")
    a_.set_xlabel("probe score (×10⁶)"); a_.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:.1f}"))
    a_.set_yticks([])
ax[0].legend(fontsize=8, loc="upper left"); ax[0].text(TAU, ax[0].get_ylim()[1] * 0.97, " τ", va="top", fontsize=9)
fig.suptitle("Held-out probe scores on impossible tasks: hacks vs not (dashed line = τ, fixed before RL)", x=0.01, ha="left", fontsize=10.5, y=1.08)
save(fig, "fig3_heldout_score_dists")

# ---------- fig4: held-out outcomes on impossible tasks ----------
fig, ax = plt.subplots(figsize=(6.4, 3.6))
x = np.arange(len(MODELS))
caught = []; missed = []
for m in MODELS:
    R = held_out(m); caught.append(np.mean([r["verdict"]["hack"] and r["probe"] > TAU for r in R]))
    missed.append(np.mean([r["verdict"]["hack"] and r["probe"] <= TAU for r in R]))
ax.bar(x, caught, color="#4c72b0", label="hack, caught by probe")
ax.bar(x, missed, bottom=caught, color="#dd8452", label="hack, missed by probe")
for i in range(len(MODELS)):
    ax.text(i, caught[i] / 2, f"{caught[i]:.0%}", ha="center", va="center", color="white", fontsize=9)
    ax.text(i, caught[i] + missed[i] / 2, f"{missed[i]:.0%}", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    ax.text(i, caught[i] + missed[i] + 0.015, f"{caught[i] + missed[i]:.0%}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x, [LABEL[m].replace(": ", ":\n") for m in MODELS], fontsize=8.5)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); ax.set_ylim(0, 0.88)
ax.set_ylabel("share of impossible-task samples"); ax.legend(loc="upper right", fontsize=8.5)
ax.set_title("Held-out: how often each model hacks, and whether the probe catches it", loc="left")
save(fig, "fig4_heldout_outcomes")

# ---------- fig5: probe vs reasoning-only monitor ----------
MON = {"base": ["base_len"], "armA": ["armA_len", "armA_len2"], "armB": ["armB_len", "armB_len2"],
       "armC": ["armC_len", "armC_len2", "armC_len3"], "armD": ["armD_len", "armD_len2", "armD_len3"]}
pr, mr = [], []
for m, ds in MON.items():
    rows = []
    for d in ds:
        V = [json.loads(l) for l in open(f"{ROOT}/data/judge_eval/{d}_lenient.jsonl")]
        mon = {r["i"]: r["score"] for r in map(json.loads, open(f"{ROOT}/data/cot_monitor/{d}.jsonl"))}
        rows += [(mon[i], v["probe"]) for i, v in enumerate(V) if i in mon and v["verdict"].get("hack") and v["probe"] == v["probe"]]
    ms, ps = np.array(rows).T; pr.append((ps > TAU).mean()); mr.append((ms >= 50).mean())
fig, ax = plt.subplots(figsize=(6.4, 3.4)); w = 0.38
ax.bar(x - w / 2, pr, w, color="#4c72b0", label="layer-13 probe (score > τ)")
ax.bar(x + w / 2, mr, w, color="#8172b3", label="reasoning-only LLM monitor (score ≥ 50)")
for i in range(len(MODELS)):
    ax.text(i - w / 2, pr[i] + 0.01, f"{pr[i]:.0%}", ha="center", va="bottom", fontsize=8.5)
    ax.text(i + w / 2, mr[i] + 0.01, f"{mr[i]:.0%}", ha="center", va="bottom", fontsize=8.5)
ax.set_xticks(x, [LABEL[m].replace(": ", ":\n") for m in MODELS], fontsize=8.5); ax.set_ylim(0, 1.08)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); ax.set_ylabel("recall on judged hacks")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8.5, ncol=2); ax.set_title("Held-out: B evades the probe, not a reader of its reasoning", loc="left")
save(fig, "fig5_probe_vs_monitor")

# ---------- fig6: text vs weights ----------
C = json.load(open(f"{ROOT}/data/score_analysis.json"))["cells"]
names = {"base": "base", "A": "A", "B": "B"}
mods = [k for k in ["base", "A", "B"] if any(c.startswith(k + "|") for c in C)]
fig, ax = plt.subplots(figsize=(6.0, 3.4)); w = 0.26
cols = {"base": "#7f7f7f", "A": "#1f77b4", "B": "#d62728"}
for j, mdl in enumerate(mods):
    vals = [C.get(f"{mdl}|{t}", {}).get("rec_tau", np.nan) for t in mods]
    ax.bar(np.arange(len(mods)) + (j - 1) * w, vals, w, color=cols[mdl], label=f"{names[mdl]} weights")
    for i, v in enumerate(vals):
        ax.text(i + (j - 1) * w, v + 0.01, f"{v:.0%}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(np.arange(len(mods)), [f"{names[t]}'s text" for t in mods]); ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); ax.set_ylabel("probe recall at τ")
ax.legend(loc="upper left", fontsize=8.5, ncol=3)
ax.set_title("Same transcripts re-scored under each model's weights:\nthe drop travels with B's text", loc="left")
save(fig, "fig6_text_vs_weights")
print("done")
