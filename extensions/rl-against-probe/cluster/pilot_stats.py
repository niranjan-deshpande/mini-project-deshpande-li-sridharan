"""Pass / truncation / length stats per split for sampler outputs. Usage: python pilot_stats.py A.jsonl [B.jsonl ...]"""
import json, sys, statistics as st
for p in sys.argv[1:]:
    R = [json.loads(l) for l in open(p)]
    by = {}
    for r in R:
        d = by.setdefault(r["split"], [0, 0, 0])
        d[0] += sum(r["passed"]); d[1] += len(r["passed"]); d[2] += any(r["passed"])
    fin = [f for r in R for f in r["finish"]]
    rl = [len(c.split("</think>")[0]) for r in R for c in r["completions"]]
    print(f"{p}: n_tasks={len(R)} trunc={sum(f == 'length' for f in fin) / len(fin):.1%} "
          f"median_reasoning_chars={st.median(rl):.0f}")
    for k, v in sorted(by.items()):
        print(f"   {k:12s} pass {v[0]}/{v[1]} = {v[0] / v[1]:.1%}   tasks with >=1 pass {v[2]}")
