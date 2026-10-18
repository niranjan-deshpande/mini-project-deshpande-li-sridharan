import json, sys, numpy as np
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/data/agastyas/hf_cache_latekey/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a")
for fn in sys.argv[1:]:
    L, P, F = [], [], []
    for l in open(fn):
        r = json.loads(l); P.append(r["prompt_len"])
        for c, f in zip(r["completions"], r["finish"]):
            L.append(len(tok(c, add_special_tokens=False)["input_ids"])); F.append(f)
    L = np.array(L)
    print(fn, "n", len(L), "tot_tok", L.sum(), "mean", L.mean().round(), "p50", np.median(L), "p90", np.percentile(L,90), "max", L.max(), "trunc", sum(f=="length" for f in F), "prompt_mean", np.mean(P).round())
