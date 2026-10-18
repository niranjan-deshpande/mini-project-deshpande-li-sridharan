"""T3 detail: per-condition generation health (length, cap hits, repetition, script mix) and example snippets."""
import json, re, sys, glob, collections, os
import numpy as np

def health(r):
    t = r["response"]
    w = re.findall(r"\S+", t)
    uniq = len(set(w)) / max(1, len(w))
    # longest run of an identical repeated 3-gram window, crude degeneration signal
    tri = [" ".join(w[i:i+3]) for i in range(max(0, len(w)-2))]
    top3 = collections.Counter(tri).most_common(1)[0][1] / max(1, len(tri)) if tri else 0
    cjk = sum('一' <= c <= '鿿' for c in t) / max(1, len(t))
    return dict(tok=r["n_tokens"], uniq=uniq, top3=top3, cjk=cjk)

out = {}
for model in ("qwen", "gptoss", "cal", "gptoss_cal", "test"):
    for f in sorted(glob.glob(f"data/t3_raw/{model}/*.jsonl")):
        rows = [json.loads(l) for l in open(f)]
        cond = os.path.basename(f)[:-6]
        H = [health(r) for r in rows]
        cap = max(r["n_tokens"] for r in rows)
        out[f"{model}/{cond}"] = dict(n=len(rows), cap=cap,
            mean_tok=float(np.mean([h["tok"] for h in H])),
            at_cap=float(np.mean([h["tok"] >= cap - 1 for h in H])),
            uniq=float(np.median([h["uniq"] for h in H])),
            top3=float(np.median([h["top3"] for h in H])),
            degen=float(np.mean([h["top3"] > 0.05 or h["uniq"] < 0.15 for h in H])),
            cjk=float(np.mean([h["cjk"] > 0.05 for h in H])),
            invoked=float(np.mean([r["invoked"] for r in rows])),
            mentioned=float(np.mean([r["mentioned"] for r in rows])),
            has_code=float(np.mean([r["has_code"] for r in rows])))
for k, v in out.items():
    print(f"{k:16} n={v['n']:3} cap={v['cap']:5} tok={v['mean_tok']:6.0f} atcap={v['at_cap']:.2f} uniq={v['uniq']:.2f} top3={v['top3']:.3f} degen={v['degen']:.2f} cjk={v['cjk']:.2f} inv={v['invoked']:.2f} men={v['mentioned']:.2f} code={v['has_code']:.2f}")
json.dump(out, open("data/t3_detail.json", "w"), indent=1)
