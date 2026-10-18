import json, numpy as np
from vllm.entrypoints.openai.parser.harmony_utils import get_encoding
enc = get_encoding()
for f in ["low", "medium"]:
    L, fin = [], []
    for l in open(f"/data/agastyas/rh_rl/p1_gptoss_{f}.jsonl"):
        r = json.loads(l)
        for c, fr in zip(r["completions"], r["finish"]):
            a, _, b = c.partition("</think>")
            L.append(len(enc.encode(a)) + len(enc.encode(b)) + 8); fin.append(fr)
    L = np.array(L)
    print(f, "n", len(L), "mean", int(L.mean()), "p50", int(np.median(L)), "p90", int(np.percentile(L, 90)), "p99", int(np.percentile(L, 99)), "max", L.max(), "sum", L.sum(), "trunc", sum(x == "length" for x in fin))
