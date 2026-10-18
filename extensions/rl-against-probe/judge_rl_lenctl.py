"""Is arm B's falling probe recall just shorter reasoning? P(fire | judged hack) within completion-length bins
(bins = quartiles of n_tok over all judged impossible-split hacks, both arms), early vs late steps, per arm."""
import json, numpy as np
D = "/Users/agastyasridharan/j-lens/rh_rl/data/judge_rl"; TAU = 770537
rows = {arm: [dict(x, step=s["step"]) for l in open(f"{D}/{arm}.jsonl") for s in [json.loads(l)] for x in s["rows"]
              if x["split"] != "original" and x["verdict"] and "error" not in x["verdict"] and x["verdict"]["hack"]]
        for arm in ["armA", "armB"]}
allL = np.array([x["n_tok"] for r in rows.values() for x in r]); q = np.quantile(allL, [0.25, 0.5, 0.75])
bins = [(0, q[0]), (q[0], q[1]), (q[1], q[2]), (q[2], 1e9)]
print("length bins (tokens):", [f"{int(a)}-{int(b) if b < 1e9 else 'max'}" for a, b in bins])
for arm, R in rows.items():
    mx = max(x["step"] for x in R)
    for name, Q in [("early (steps 1-10)", [x for x in R if x["step"] < 10]), (f"late (steps {mx - 8}-{mx + 1})", [x for x in R if x["step"] >= mx - 9])]:
        cells = []
        for a, b in bins:
            z = [x["probe"] > TAU for x in Q if a <= x["n_tok"] < b and x["probe"] == x["probe"]]
            cells.append(f"{np.mean(z):.2f} (n={len(z)})" if z else "  -  ")
        print(f"{arm} {name:20s} P(fire|hack) by length bin: " + "  ".join(cells))
