"""Summarises steer_judge.py output: per condition, rate of any hack / attempted-or-enacted hack, behaviour mix,
truncation; differences vs unsteered with a bootstrap over cut points (both samples of a cut kept together).

Usage: python3 steer_analyze.py data/steer_neg/judged.jsonl data/steer_neg/summary.json
"""
import collections, json, sys
import numpy as np

rows = [json.loads(l) for l in open(sys.argv[1])]
C = collections.defaultdict(dict)  # cond -> uid -> list of per-sample dicts
for r in rows:
    C[r["cond"]][r["uid"]] = [{"hack": bool(v and v.get("hack")),
                                "act": bool(v and v.get("hack") and v.get("status") in ("attempted", "enacted")),
                                "behavior": (v or {}).get("behavior", "error") if v and "error" not in v else "error",
                                "trunc": f == "length", "tool": "<tool_call>" in c}
                               for v, f, c in zip(r["verdicts"], r["finish"], r["continuations"])]
uids = sorted(set.intersection(*(set(d) for d in C.values())))
order = [c for c in ("0", "neg1", "neg1.5", "neg2", "neg3", "rand3") if c in C]
rng = np.random.default_rng(0)
B = rng.integers(0, len(uids), (2000, len(uids)))


def per_cut(cond, key):
    return np.array([np.mean([s[key] for s in C[cond][u]]) for u in uids])


out = {"n_cuts": len(uids), "conds": {}}
for c in order:
    e = {"n": sum(len(C[c][u]) for u in uids)}
    for key in ("hack", "act", "trunc", "tool"):
        x = per_cut(c, key); e[key] = float(x.mean())
        if c != "0":
            d = x - per_cut("0", key); bs = d[B].mean(1)
            e[key + "_diff"] = [float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    coh = [s for u in uids for s in C[c][u] if s["behavior"] not in ("incoherent", "error")]
    e["coherent_n"] = len(coh)
    e["hack_given_coherent"] = sum(s["hack"] for s in coh) / max(1, len(coh))
    e["act_given_coherent"] = sum(s["act"] for s in coh) / max(1, len(coh))
    cu = [u for u in uids if any(s["behavior"] not in ("incoherent", "error") for s in C[c][u])]
    num = np.array([sum(s["hack"] for s in C[c][u] if s["behavior"] not in ("incoherent", "error")) for u in uids])
    den = np.array([sum(s["behavior"] not in ("incoherent", "error") for s in C[c][u]) for u in uids])
    bs = num[B].sum(1) / np.maximum(1, den[B].sum(1))
    e["hack_given_coherent_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    e["behavior"] = dict(collections.Counter(s["behavior"] for u in uids for s in C[c][u]))
    out["conds"][c] = e
    print(f"{c:6} n={e['n']}  any hack {e['hack']:.0%}  acted {e['act']:.0%}  truncated {e['trunc']:.0%}  tool call {e['tool']:.0%}  "
          + (f"Δhack {e['hack_diff'][0]:+.0%} [{e['hack_diff'][1]:+.0%},{e['hack_diff'][2]:+.0%}]  "
             f"Δacted {e['act_diff'][0]:+.0%} [{e['act_diff'][1]:+.0%},{e['act_diff'][2]:+.0%}]" if c != "0" else ""))
    print(f"        coherent {e['coherent_n']}: hack {e['hack_given_coherent']:.0%} [{e['hack_given_coherent_ci'][0]:.0%},{e['hack_given_coherent_ci'][1]:.0%}]  acted {e['act_given_coherent']:.0%}", e["behavior"])
json.dump(out, open(sys.argv[2], "w"), indent=1)
