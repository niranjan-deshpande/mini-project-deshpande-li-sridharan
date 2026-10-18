"""Per-arm training curves: judge hack rate and probe recall at tau vs RL step (impossible splits, 10-step bins).

Reads results/judge_rl/arm{A,B,C,D}.jsonl. Writes figs/fig_training_panel.{pdf,png}, sized for \\textwidth (6.5in).
Usage: python figs/fig_training_panel.py   (from rh_rl/)
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "results", "judge_rl")
TAU = 770537
BIN = 10
ARMS = [("armA", "A: no probe"), ("armB", "B: threshold penalty"),
        ("armC", "C: linear penalty"), ("armD", "D: drop flagged")]
C_HACK, C_REC = "#222222", "#c0392b"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "axes.unicode_minus": False, "pdf.fonttype": 42, "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})
pct = PercentFormatter(1.0, decimals=0)


def load(arm):
    rows = []
    for line in open(os.path.join(DATA, f"{arm}.jsonl")):
        d = json.loads(line)
        for r in d["rows"]:
            if r["split"] == "original" or r["verdict"].get("hack") is None or r["probe"] != r["probe"]:
                continue
            rows.append((d["step"], bool(r["verdict"]["hack"]), r["probe"] > TAU))
    return np.array(rows, dtype=float)


def binned(R):
    """Per bin: (center, hack rate, its 95% half-width, recall, its 95% half-width)."""
    out = []
    for b in range(0, 100, BIN):
        S = R[(R[:, 0] >= b) & (R[:, 0] < b + BIN)]
        H = S[S[:, 1] == 1]
        h, rc = S[:, 1].mean(), H[:, 2].mean()
        out.append((b + BIN / 2, h, 1.96 * np.sqrt(h * (1 - h) / len(S)),
                    rc, 1.96 * np.sqrt(rc * (1 - rc) / len(H))))
    return np.array(out)


fig, axes = plt.subplots(1, 4, figsize=(6.5, 2.0), sharey=True)
for ax, (arm, title) in zip(axes, ARMS):
    x, h, hci, rc, rci = binned(load(arm)).T
    for y, ci, c, lab in [(h, hci, C_HACK, "Hack rate (judge)"), (rc, rci, C_REC, r"Probe recall at $\tau$")]:
        ax.fill_between(x, y - ci, y + ci, color=c, alpha=0.12, lw=0)
        ax.plot(x, y, color=c, lw=1.3, marker="o", ms=2.4, label=lab)
    ax.set_title(title)
    ax.set_xlim(0, 100); ax.set_xticks([0, 50, 100]); ax.set_xlabel("RL step")
    ax.set_ylim(0, 1); ax.set_yticks([0, 0.25, 0.5, 0.75, 1]); ax.yaxis.set_major_formatter(pct)
    ax.grid(axis="y", lw=0.4, alpha=0.35)
axes[0].set_ylabel("Rate")
handles, labels = axes[0].get_legend_handles_labels()
fig.tight_layout(pad=0.3, w_pad=1.6, rect=(0, 0, 1, 0.88))
fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0), handlelength=1.8,
           columnspacing=2.0)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, f"fig_training_panel.{ext}"), bbox_inches="tight", pad_inches=0.02, dpi=300)
print("first/last bin (hack, recall):")
for arm, _ in ARMS:
    B = binned(load(arm))
    print(f"  {arm}: {B[0,1]:.2f}/{B[0,3]:.2f} -> {B[-1,1]:.2f}/{B[-1,3]:.2f}")
