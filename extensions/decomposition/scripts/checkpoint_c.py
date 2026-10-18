"""CHECKPOINT C (selection half only): is the Luna-built v_hack still the probe Agastya validated?

Reports passage AUROC (impossible splits, consensus labels, L40, turn, max-pool) of the Luna v_hack next to the rebuilt
Claude v_hack and Agastya's saved vector on the same passages, with paired task-level bootstrap CIs and length-controlled
versions; cos(Luna, Claude) next to each vector's split-half reliability.
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rhdecomp.stats import auroc, paired_task_bootstrap, length_residualize as _lr  # noqa: E402
length_residualize = lambda s, n: _lr({"x": s}, n)["x"]
from rhdecomp.directions import split_half_reliability  # noqa: E402
from rhdecomp.geometry import cosine  # noqa: E402

S1 = ROOT / "data/gpu/s1"
z = np.load(ROOT / "data/analysis/passages.npz"); names = list(z["names"])
rows = [json.loads(l) for l in open(ROOT / "data/analysis/passages.jsonl")]
m = np.array([r["split"] in ("conflicting", "oneoff") and r["hack"] is not None and r["half"] == 0 for r in rows])
y = np.array([bool(r["hack"]) for r in rows])[m]
tasks = np.array([r["sample_id"] for r in rows])[m]
ntok = np.array([r["n_tokens"] for r in rows])[m]
probes = {"luna_vhack": "hack__turn", "claude_rebuilt": "claude_rebuilt__turn", "agastya_saved": "agastya__turn"}
S = {k: z["max"][m, names.index(v)] for k, v in probes.items()}
boot = paired_task_bootstrap(S, y, tasks, n_boot=2000, seed=0)
boot_len = paired_task_bootstrap({k: length_residualize(v, ntok) for k, v in S.items()}, y, tasks, n_boot=2000, seed=0)

D = np.load(S1 / "directions.npz"); dn = list(D["names"])
v_luna, v_cl = D["V"][dn.index("hack__turn")], D["V"][dn.index("claude_rebuilt__turn")]
li = list(np.load(S1 / "pairs/hack.npz")["layers"]).index(40)
hz = np.load(S1 / "pairs/hack.npz"); cz = np.load(S1 / "claude_pair_acts.npz")
rel_l = split_half_reliability(hz["turn_hack"][:, li].astype(np.float32), hz["turn_honest"][:, li].astype(np.float32), n_splits=100, seed=0)
rel_c = split_half_reliability(cz["turn_hack"][:, li].astype(np.float32), cz["turn_honest"][:, li].astype(np.float32), n_splits=100, seed=0)
rl, rc = (rel_l["rho"], rel_c["rho"]) if isinstance(rel_l, dict) else (rel_l.rho, rel_c.rho)
c = cosine(v_luna, v_cl)
res = {"selection_half": {"n_passages": int(m.sum()), "n_hack": int(y.sum()), "n_tasks": int(len(set(tasks)))},
       "auroc": {k: {"point": boot["point"][k], "ci": boot["ci"][k]} for k in probes},
       "auroc_length_controlled": {k: {"point": boot_len["point"][k], "ci": boot_len["ci"][k]} for k in probes},
       "diff_luna_minus_claude": boot["diff"][("luna_vhack", "claude_rebuilt")],
       "cos_luna_claude": c, "reliability_luna": rl, "reliability_claude": rc,
       "cos_disattenuated": c / np.sqrt(rl * rc), "n_pairs_luna": int(len(hz["turn_hack"])), "n_pairs_claude": int(len(cz["turn_hack"]))}
res["go"] = abs(res["diff_luna_minus_claude"]["point"]) <= 0.03 or res["diff_luna_minus_claude"]["point"] > 0
(ROOT / "results/cpC").mkdir(parents=True, exist_ok=True)
json.dump(res, open(ROOT / "results/cpC/checkpoint_c.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
