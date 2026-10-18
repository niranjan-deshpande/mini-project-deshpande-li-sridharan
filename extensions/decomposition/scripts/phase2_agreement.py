"""Agreement between two Phase 2 judges: Cohen's kappa per label (and per split), prevalence, disagreements."""
import argparse, json, sys
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="pilot")
ap.add_argument("--judges", nargs=2, default=["luna", "haiku"])
a = ap.parse_args()
base = Path(__file__).resolve().parents[1] / "data" / "phase2" / a.run
recs = {j: {json.loads(f.read_text())["uid"]: json.loads(f.read_text()) for f in (base / j).glob("*.json")} for j in a.judges}
uids = sorted(set.intersection(*[{u for u, r in recs[j].items() if "labels" in r} for j in a.judges]))


def kappa(x, y):
    x, y = np.asarray(x, bool), np.asarray(y, bool)
    po = (x == y).mean(); pe = x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


out = {"n_rollouts": len(uids)}
for L in ("contradiction_failure", "eval_awareness"):
    X, Y, S, dis = [], [], [], []
    for u in uids:
        la, lb = recs[a.judges[0]][u]["labels"], recs[a.judges[1]][u]["labels"]
        for n in la:
            X.append(la[n][L]); Y.append(lb[n][L]); S.append(recs[a.judges[0]][u]["split"])
            if la[n][L] != lb[n][L]:
                dis.append((u, n, a.judges[0] if la[n][L] else a.judges[1]))
    X, Y, S = map(np.array, (X, Y, S))
    out[L] = {"n_passages": len(X), "kappa": kappa(X, Y), "agree_pct": float((X == Y).mean()),
              f"prev_{a.judges[0]}": float(X.mean()), f"prev_{a.judges[1]}": float(Y.mean()),
              "both_positive": int((X & Y).sum()), "only_" + a.judges[0]: int((X & ~Y).sum()),
              "only_" + a.judges[1]: int((~X & Y).sum()),
              "by_split": {s: {"kappa": kappa(X[S == s], Y[S == s]), "n": int((S == s).sum()),
                               "pos_both": int((X & Y)[S == s].sum())} for s in sorted(set(S))}}
    out[L]["disagreements_sample"] = dis[:40]
for j in a.judges:
    rs = [recs[j][u] for u in uids]
    out[f"{j}_cost"] = sum(r["cost"] for r in rs)
    out[f"{j}_withdrawn_flags"] = sum(r["withdrawn"] for r in rs)
json.dump(out, open(base / "agreement.json", "w"), indent=1)
print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "disagreements_sample"})
                  for k, v in out.items()}, indent=1))
