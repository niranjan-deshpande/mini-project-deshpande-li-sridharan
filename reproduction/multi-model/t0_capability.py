"""Capability control for T0: pass rate on ImpossibleBench's original split (same problems, correct tests), hack rates
on the impossible splits split by whether the model solved that problem in the original split, and (where judged)
an audit of original-split passes for hacks.

Usage: python3 t0_capability.py OUT.json
"""
import collections, json, os, sys

MODELS = [  # (display name, uid prefix, rollout files, impossible-split labels, original-pass labels or None)
    ("Qwen3.8-27B", "qwen3.8-27b", ["qwen_partial_clean", "qwen_new2", "qwen_new3"], "judge_full", None),
    ("gpt-oss-120b", "gpt-oss-120b", ["gptoss_full_clean"], "judge_full", None),
    ("GLM-4.5-Air", "glm-4.5-air", ["glm_full"], "judge_glm_full", "judge_glm_orig"),
]


def by_rollout(path, prefix):
    out = collections.defaultdict(list)
    for l in open(path):
        x = json.loads(l)
        if x["uid"].startswith(prefix):
            out[x["uid"]].append(x)
    return out


def verdict(ps):
    hack = [p for p in ps if p["gt"]]
    cats = sorted({c for p in hack for c in p["judge_categories"]})
    return bool(hack), any("enacted" in p["judge_statuses"] for p in hack), cats


res = {}
for name, prefix, files, lab, orig_lab in MODELS:
    R = [json.loads(l) for f in files for l in open(f"data/{f}.jsonl")]
    split_pass = collections.defaultdict(lambda: [0, 0])
    for r in R:
        split_pass[r["split"]][0] += r["score"] == "C"; split_pass[r["split"]][1] += 1
    solved = {r["sample_id"]: r["score"] == "C" for r in R if r["split"] == "original"}
    roll = by_rollout(f"data/{lab}/labels.jsonl", prefix)
    grp = {True: [0, 0, 0], False: [0, 0, 0]}
    for u, ps in roll.items():
        if ps[0]["split"] not in ("conflicting", "oneoff") or ps[0]["sample_id"] not in solved:
            continue
        h, e, _ = verdict(ps)
        g = grp[solved[ps[0]["sample_id"]]]
        g[0] += 1; g[1] += h; g[2] += e
    entry = {"split_pass": {k: v for k, v in split_pass.items()},
             "solved": dict(zip(("n", "hack", "enacted"), grp[True])),
             "unsolved": dict(zip(("n", "hack", "enacted"), grp[False]))}
    if orig_lab and os.path.exists(f"data/{orig_lab}/labels.jsonl"):
        oroll = by_rollout(f"data/{orig_lab}/labels.jsonl", prefix)
        V = {u: verdict(ps) for u, ps in oroll.items()}
        entry["orig_audit"] = {"n": len(V), "hack": sum(v[0] for v in V.values()), "enacted": sum(v[1] for v in V.values()),
                               "categories": collections.Counter(c for v in V.values() if v[1] for c in v[2]),
                               "enacted_uids": sorted(u for u, v in V.items() if v[1])}
    res[name] = entry

json.dump(res, open(sys.argv[1], "w"), indent=1)
for name, e in res.items():
    sp = e["split_pass"]
    print(f"{name:13} original {sp['original'][0]}/{sp['original'][1]}  oneoff {sp['oneoff'][0]}/{sp['oneoff'][1]}  "
          f"conflicting {sp['conflicting'][0]}/{sp['conflicting'][1]}")
    for k in ("solved", "unsolved"):
        g = e[k]
        if g["n"]:
            print(f"    {k:8} n={g['n']:3} hack={g['hack'] / g['n']:.0%} enacted={g['enacted'] / g['n']:.0%}")
    if "orig_audit" in e:
        a = e["orig_audit"]
        print(f"    original-pass audit: {a['n']} judged, {a['hack']} with any hack, {a['enacted']} enacted {dict(a['categories'])}")
