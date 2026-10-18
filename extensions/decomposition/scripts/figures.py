"""CHECKPOINT D figures (static PNG) from results/geometry/geometry.json and results/phase4/phase4.json."""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
G = json.load(open(ROOT / "results/geometry/geometry.json"))["turn"]
R = json.load(open(ROOT / "results/phase4/phase4.json"))
OUT = ROOT / "results/figures"
BLUE, ORANGE, VIOLET, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#4a3aa7", "#a3a29d", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": "white"})
CANDS = ["testvocab_swe", "testvocab_noncode", "task_failure", "legit_shortcut_plain", "eval_awareness", "grader_modeling"]
LABEL = {"testvocab_swe": "test vocab (SWE)", "testvocab_noncode": "test vocab (non-code)", "task_failure": "task failure",
         "legit_shortcut_plain": "legit shortcut (plain)", "eval_awareness": "eval awareness", "grader_modeling": "grader modeling",
         "gc_template": "gc: template", "gc_third_person": "gc: 3rd person", "gc_first_person": "gc: 1st person",
         "gc_judgment": "gc: judgment", "gc_ai_offdomain": "gc: AI off-domain"}
BASES = [k for k in G["cos_with_vhack"] if k.startswith("base_")]
GCS = [k for k in G["cos_with_vhack"] if k.startswith("gc_")]

# 1. cosine with v_hack (disattenuated)
fig, ax = plt.subplots(figsize=(7, 6))
rows = [(LABEL[c], G["cos_disattenuated"][c], BLUE) for c in CANDS] + [(LABEL[g], G["cos_disattenuated"][g], ORANGE) for g in GCS] + \
       [(b.replace("base_", "baseline: "), G["cos_disattenuated"][b], GRAY) for b in BASES]
y = np.arange(len(rows))[::-1]
ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.7)
p95 = np.percentile(np.abs([G["cos_with_vhack"][b] for b in BASES]), 95)
ax.axvspan(-p95, p95, color=GRAY, alpha=0.15, lw=0)
ax.axvline(0, color=INK2, lw=0.8)
ax.set_yticks(y, [r[0] for r in rows]); ax.set_xlabel("cosine with v_hack (disattenuated), L40 turn mean")
ax.set_title("Which directions point along v_hack?\nblue = candidates, orange = general cheating, gray = baselines (band: baseline |cos| p95)",
             fontsize=10, loc="left", color=INK)
fig.tight_layout(); fig.savefig(OUT / "fig1_cosines.png", dpi=160, bbox_inches="tight"); plt.close(fig)

# 2. detection AUROC (held-out, length-controlled)
D = R["decision"]
rows = [("v_hack", D["hack__turn"], VIOLET), ("Cβ̂ (candidate part)", D["fit"], VIOLET), ("v_res (residual)", D["res"], VIOLET),
        ("Claude v_hack", D["claude_rebuilt__turn"], VIOLET)] + [(LABEL[c], D[f"{c}__turn"], BLUE) for c in CANDS]
fig, ax = plt.subplots(figsize=(7, 5.5))
y = np.arange(len(rows) + len(BASES))[::-1]
for i, (lab, d, col) in enumerate(rows):
    ax.barh(y[i], d["auroc_len"] - 0.5, left=0.5, color=col, height=0.7)
    ax.plot(d["ci_len"], [y[i]] * 2, color=INK, lw=1)
for j, b in enumerate(BASES):
    a = R["decision_baselines_len"][f"{b}__turn"]
    ax.barh(y[len(rows) + j], max(a, 1 - a) - 0.5, left=0.5, color=GRAY, height=0.7)
ax.set_yticks(y, [r[0] for r in rows] + [b.replace("base_", "baseline: ") + " (either sign)" for b in BASES])
ax.axvline(0.5, color=INK2, lw=0.8); ax.set_xlim(0.4, 1.0)
ax.set_xlabel("held-out AUROC, length-controlled\n(95% task-bootstrap CI)")
ax.set_title("Detection: many DoM directions detect hacks", fontsize=10, loc="left", color=INK)
fig.tight_layout(); fig.savefig(OUT / "fig2_detection.png", dpi=160, bbox_inches="tight"); plt.close(fig)

# 3. functional R2 + Shapley
F = R["functional"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [1, 2]})
a1.bar([0], [F["r2"]], color=BLUE, width=0.6); a1.errorbar([0], [F["r2"]], yerr=[[F["r2"] - F["ci"][0]], [F["ci"][1] - F["r2"]]], color=INK)
a1.bar([1], [F["null_baselines"]["mean"]], color=GRAY, width=0.6)
a1.errorbar([1], [F["null_baselines"]["mean"]], yerr=[[0], [F["null_baselines"]["p95"] - F["null_baselines"]["mean"]]], color=INK2)
a1.set_xticks([0, 1], ["6 candidates", "6 baselines\n(mean, p95)"]); a1.set_ylim(0, 1); a1.set_ylabel("token-level R² of v_hack scores")
sh = F["shapley"]; ks = [c for c in CANDS]
a2.barh(range(len(ks))[::-1], [sh[f"{k}__turn"] for k in ks], color=BLUE, height=0.7)
a2.set_yticks(range(len(ks))[::-1], [LABEL[k] for k in ks]); a2.set_xlabel("Shapley share of R²")
fig.suptitle("Functional: candidates reproduce the probe's token scores only modestly beyond unrelated contrasts", fontsize=10, x=0.02, ha="left")
fig.tight_layout(); fig.savefig(OUT / "fig3_functional.png", dpi=160, bbox_inches="tight"); plt.close(fig)

# 4. confound strata
C = R["confounds"]
rows = [("all held-out passages", R["decision"]["hack__turn"]["auroc"], R["n"]["heldout_passages"], R["n"]["heldout_hacks"])]
for key, lab in (("contradiction", "contradiction"), ("eval_aware", "eval-aware (broad)"), ("eval_aware_narrow", "eval-aware (narrow)")):
    for k, v in C[key]["stratified_vhack"].items():
        rows.append((f"{lab} = {k}", v["auroc"], v["n"], v["n_hack"]))
fig, ax = plt.subplots(figsize=(7, 3.8))
y = np.arange(len(rows))[::-1]
ax.barh(y, [r[1] - 0.5 for r in rows], left=0.5, color=[VIOLET] + [BLUE if "= 0" in r[0] else ORANGE for r in rows[1:]], height=0.7)
for yy, r in zip(y, rows):
    ax.text(r[1] + 0.005, yy, f"n={r[2]}, hacks={r[3]}", va="center", fontsize=8, color=INK2)
ax.set_yticks(y, [r[0] for r in rows]); ax.axvline(0.5, color=INK2, lw=0.8); ax.set_xlim(0.45, 1.05)
ax.set_xlabel("v_hack AUROC, hack vs clean, within stratum (held-out)")
ax.set_title("v_hack AUROC within confound strata", fontsize=10, loc="left", color=INK)
fig.tight_layout(); fig.savefig(OUT / "fig4_confounds.png", dpi=160, bbox_inches="tight"); plt.close(fig)
print("figures written")
