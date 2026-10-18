"""Turn GPU session-1 rollout outputs into a passage table: max-pooled (and mean) score per direction per passage,
joined with hack labels (unanimous judge consensus), Phase 2 labels, split half, and passage length.

Usage: python scripts/passage_scores.py --out data/gpu/s1 --dest data/analysis/passages.npz
"""
import argparse, glob, hashlib, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(ROOT / "data/gpu/s1"))
ap.add_argument("--dest", default=str(ROOT / "data/analysis/passages.npz"))
ap.add_argument("--labels", default=str(ROOT.parents[1] / "reproduction/multi-model/results/labels/judge_full.jsonl"))
ap.add_argument("--phase2", default=str(ROOT / "data/final/phase2_labels.jsonl"))
a = ap.parse_args()

gt = {(x["uid"], x["n"]): x for x in map(json.loads, open(a.labels)) if x["uid"].startswith("qwen")}
p2 = {(x["uid"], x["n"]): x for x in map(json.loads, open(a.phase2))}
rows, MX, MN, names = [], [], [], None
for f in sorted(glob.glob(f"{a.out}/rollouts/*.npz")):
    z = np.load(f)
    meta = json.load(open(f.replace(".npz", ".passages.json")))
    names = list(z["names"]) if names is None else names
    where = {int(p): i for i, p in enumerate(z["pos"])}
    P = z["proj"]
    for p in meta["passages"]:
        idx = [where[t] for t in range(p["tok0"], p["tok1"]) if t in where]
        if not idx:
            continue
        g = gt.get((meta["uid"], p["n"]), {}); q = p2.get((meta["uid"], p["n"]), {})
        rows.append({"uid": meta["uid"], "n": p["n"], "split": meta["split"], "sample_id": meta["sample_id"],
                     "channel": p["channel"], "n_tokens": len(idx),
                     "hack": g.get("gt"), "hack_categories": g.get("judge_categories", []),
                     "hack_statuses": g.get("judge_statuses", []),
                     "contradiction": q.get("contradiction_failure"), "eval_aware": q.get("eval_awareness"),
                     "eval_aware_narrow": q.get("eval_awareness_narrow"),
                     "half": int(hashlib.md5(meta["sample_id"].encode()).hexdigest(), 16) % 2})
        MX.append(P[idx].max(0)); MN.append(P[idx].mean(0))
Path(a.dest).parent.mkdir(parents=True, exist_ok=True)
np.savez(a.dest, max=np.stack(MX).astype(np.float32), mean=np.stack(MN).astype(np.float32), names=np.array(names))
with open(Path(a.dest).with_suffix(".jsonl"), "w") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")
print(f"{len(rows)} passages from {len({r['uid'] for r in rows})} rollouts; {len(names)} directions")
