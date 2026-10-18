"""T4 (paper Fig 10): does the probe peak mark a point where hacking is about to become likely?

For judge-negative reasoning passages with a high probe score, cut the rollout right after the probe's argmax token
("peak") and, as a control, after a random token of the same passage at least 64 tokens from the peak. Resample n
continuations from each cut with the served model (raw token prompt via /v1/completions) and save them for judging.

Usage (athena02): python t4_resample.py --model <snapshot> --rollouts R.jsonl --index t2/qwen_part1/index_0.jsonl
    --scores t2/qwen_part1/scores_0.npz --cands cands.json --out t4/qwen.jsonl
`cands.json` = list of {uid, n} chosen on the Mac from labels + scores.
"""
import argparse, json, random, urllib.request
import concurrent.futures as cf
import numpy as np
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--rollouts", nargs="+", required=True)
ap.add_argument("--index", nargs="+", required=True)
ap.add_argument("--cands", required=True)
ap.add_argument("--layer", type=int, default=40)
ap.add_argument("--key", type=int, default=1, help="index of the probe key in the argmax lists (turn__all = 1)")
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--max-new", type=int, default=1024)
ap.add_argument("--port", type=int, default=8200)
ap.add_argument("--served", default="qwen3.8-27b")
ap.add_argument("--out", required=True)
a = ap.parse_args()

src = open("/data/agastyas/rh_probe/t2_acts.py").read()
ns = {}
exec(src[src.index("def to_chat"):src.index("rows = [json.loads")], ns)
tok = AutoTokenizer.from_pretrained(a.model)
KEY = open("/data/agastyas/rh_probe/api_key").read().strip()
R = {}
for f in a.rollouts:
    for l in open(f):
        r = json.loads(l); R[r["uid"]] = r
IDX = {}
for f in a.index:
    for l in open(f):
        x = json.loads(l); IDX[(x["uid"], x["n"])] = x
cands = json.load(open(a.cands))
rng = random.Random(0)


def complete(ids):
    body = {"model": a.served, "prompt": ids, "max_tokens": a.max_new, "n": a.n, "temperature": 0.6, "top_p": 0.95,
            "skip_special_tokens": False}
    req = urllib.request.Request(f"http://localhost:{a.port}/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    return [c["text"] for c in json.load(urllib.request.urlopen(req, timeout=3600))["choices"]]


jobs = []
for c in cands:
    r, x = R[c["uid"]], IDX[(c["uid"], c["n"])]
    text = tok.apply_chat_template(ns["to_chat"](r), tools=r.get("tools") or None, tokenize=False,
                                   add_generation_prompt=False, preserve_thinking=True)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    peak = x["argmax"][a.layer][a.key]
    pool = [t for t in range(x["tok0"], x["tok1"]) if abs(t - peak) >= 64]
    if not pool:
        continue
    ctrl = rng.choice(pool)
    for cond, pos in (("peak", peak), ("random", ctrl)):
        jobs.append({"uid": c["uid"], "n": c["n"], "cond": cond, "pos": pos, "rel": (pos - x["tok0"]) / (x["tok1"] - x["tok0"]),
                     "prefix_tail": tok.decode(ids[max(x["tok0"], pos - 200): pos + 1]), "ids": ids[: pos + 1]})
print(f"{len(cands)} candidates -> {len(jobs)} cuts x {a.n} samples", flush=True)
with cf.ThreadPoolExecutor(16) as ex, open(a.out, "w") as f:
    futs = {ex.submit(complete, j["ids"]): j for j in jobs}
    for i, fu in enumerate(cf.as_completed(futs)):
        j = futs[fu]
        try:
            outs = fu.result()
        except Exception as e:
            print("error", j["uid"], e); continue
        f.write(json.dumps({k: v for k, v in j.items() if k != "ids"} | {"continuations": outs}) + "\n"); f.flush()
        if i % 10 == 0:
            print(f"  {i + 1}/{len(jobs)}", flush=True)
print("DONE")
