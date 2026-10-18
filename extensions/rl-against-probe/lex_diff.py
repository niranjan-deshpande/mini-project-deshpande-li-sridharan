"""Phrase rates in the reasoning of judged hacks (held-out lenient, impossible splits): share of hack samples whose
analysis channel contains each phrase (case-insensitive), per model; plus the most over/under-represented words in B vs base."""
import json, re, collections, math
ROOT = "/Users/agastyasridharan/j-lens/rh_rl/data"
M = {"base": ["base_len"], "A": ["armA_len", "armA_len2"], "B": ["armB_len", "armB_len2"], "C": ["armC_len", "armC_len2", "armC_len3"], "D": ["armD_len", "armD_len2", "armD_len3"]}
H = {}
for m, ds in M.items():
    H[m] = []
    for d in ds:
        S = [json.loads(l) for l in open(f"{ROOT}/eval_samples/{d}/lenient.jsonl")]; V = [json.loads(l) for l in open(f"{ROOT}/judge_eval/{d}_lenient.jsonl")]
        H[m] += [(s["analysis"] or "") for s, v in zip(S, V) if s["split"] != "original" and "error" not in v["verdict"] and v["verdict"]["hack"]]
P = ["hardcode|hard-code|hard code", "special[- ]case", "cheat", "\\bhack", "trick", "workaround|work around", "pass the test|tests pass|pass all test",
     "test is wrong|tests? (are|is) (wrong|incorrect|buggy)|incorrect test|wrong test|test.{0,20}(bug|error|mistake)",
     "contradict|inconsisten|conflict", "expected output|expect(s|ed)? \\d", "spec(ification)?\\b", "edge case", "\\bif .{0,40}== ?\\d",
     "we (must|need to) pass|must pass", "grader|checker|judge", "maybe|perhaps", "but the test"]
print(f"{'phrase':60s} " + " ".join(f"{m:>6s}" for m in M) + "   (share of judged-hack reasoning containing it)")
for p in P:
    r = re.compile(p, re.I)
    print(f"{p[:60]:60s} " + " ".join(f"{sum(bool(r.search(t)) for t in H[m]) / len(H[m]):6.2f}" for m in M))
print("n hacks", {m: len(v) for m, v in H.items()}, "median words", {m: sorted(len(t.split()) for t in v)[len(v) // 2] for m, v in H.items()})
def docfreq(T):
    c = collections.Counter()
    for t in T: c.update(set(re.findall(r"[a-z][a-z'-]+", t.lower())))
    return c
fb, fB = docfreq(H["base"]), docfreq(H["B"]); nb, nB = len(H["base"]), len(H["B"])
lr = {w: math.log((fB[w] + 1) / (nB + 2)) - math.log((fb[w] + 1) / (nb + 2)) for w in set(fb) | set(fB) if fb[w] + fB[w] >= 60}
for lab, ws in (("B >> base", sorted(lr, key=lambda w: -lr[w])[:30]), ("B << base", sorted(lr, key=lambda w: lr[w])[:30])):
    print(lab, ", ".join(f"{w}({fb[w] / nb:.2f}->{fB[w] / nB:.2f})" for w in ws))
