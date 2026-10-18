import json, sys, statistics as st
for eff in sys.argv[1:]:
    R = [json.loads(l) for l in open(f"/data/agastyas/rh_rl/p0_effort_{eff}.jsonl")]
    by = {}
    for r in R:
        d = by.setdefault(r["split"], [0, 0])
        d[0] += sum(r["passed"]); d[1] += len(r["passed"])
    fin = [f for r in R for f in r["finish"]]
    trunc = sum(f == "length" for f in fin) / len(fin)
    lens = [len(c) for r in R for c in r["completions"]]
    print(eff, f"trunc={trunc:.1%}", f"median_chars={st.median(lens):.0f}",
          " ".join(f"{k}={v[0]}/{v[1]}" for k, v in sorted(by.items())))
