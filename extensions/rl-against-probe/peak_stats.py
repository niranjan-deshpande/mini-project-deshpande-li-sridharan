"""Where the layer-13 probe peaks: top-peak tokens and windows in judged hacks / non-hacks, for base|base, base|B, B|B, B|base
(runs/score_trace/*_peaks13.jsonl, rsynced to data/score_trace/)."""
import json, glob, collections, re, sys
ROOT = "/Users/agastyasridharan/j-lens/rh_rl/data"
CH = {"base": ["base_len_0", "base_len_1236"], "B": ["armB_len_0", "armB_len2_0"]}
def lab(chunk):
    d = chunk.rsplit("_", 1)[0]; V = [json.loads(l) for l in open(f"{ROOT}/judge_eval/{d}_lenient.jsonl")]
    return V
for m, t in (("base", "base"), ("base", "B"), ("B", "B"), ("B", "base")):
    P = []
    for c in CH[t]:
        f = f"{ROOT}/score_trace/{m}__{c}_peaks13.jsonl"
        try: V = lab(c); P += [(r, V[r["idx"]]) for r in map(json.loads, open(f))]
        except FileNotFoundError: pass
    P = [(r, v) for r, v in P if v["split"] != "original" and "error" not in v["verdict"] and r["peaks"]]
    if not P: continue
    for h in (True, False):
        Q = [r for r, v in P if v["verdict"]["hack"] == h]
        tok = collections.Counter(re.search(r"⟦(.*?)⟧", r["peaks"][0][2], re.S).group(1).strip().lower() for r in Q)
        pos = sorted(r["peaks"][0][0] / max(1, r["n"]) for r in Q)
        print(f"{m}|{t} {'hacks' if h else 'non-hacks'} n={len(Q)} median rel. position of top peak {pos[len(pos)//2]:.2f}; top-peak tokens:",
              ", ".join(f"{k!r}:{v}" for k, v in tok.most_common(18)))
    if len(sys.argv) > 1:
        for r, v in [x for x in P if x[1]["verdict"]["hack"]][:int(sys.argv[1])]:
            print("   ", f"{r['peaks'][0][1] / 1e3:.0f}k", repr(r["peaks"][0][2][-160:])[:220])
