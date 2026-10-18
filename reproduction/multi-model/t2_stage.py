"""Free check: does the selected probe separate CONSIDERED hacks and ENACTED hacks equally well? Positives are split
by their strongest judge status (enacted > attempted > considered); each subset is scored against all negatives on
the held-out half, with task-bootstrap CIs, plus the monitor's TPR and a length-only baseline per subset, and a
stage x channel breakdown.

Usage: python3 t2_stage.py OUT.json REPORT.json SCOREDIRS...
"""
import json, os, sys
import numpy as np

OUT = sys.argv[1]
rep = json.load(open(sys.argv[2]))
sys.argv = ["x", "--scores", *sys.argv[3:], "--labels", os.environ.get("LABELS", "data/judge_full/labels.jsonl"), "--out", "/dev/null"]
src = open("t2_analyze.py").read()
exec(src[:src.index("# --- selection ---")])
Lb, key, agg = rep["selected"]["layer"], rep["selected"]["key"], rep["selected"]["agg"]
s = (MX if agg == "max" else MN)[ii, Lb, keys.index(key)]
ln = np.log(np.array([IDX[i]["tok1"] - IDX[i]["tok0"] for i in ii]))
RANK = {"considered": 1, "attempted": 2, "enacted": 3}
stage = np.array(["neg" if not x["gt"] else max(x["judge_statuses"] or ["considered"], key=lambda t: RANK.get(t, 0))
                  for x in rows])
chan = np.array([x["channel"] for x in rows])
rng = np.random.default_rng(0)
X = np.c_[np.ones(len(ln)), ln]
resid = s - X @ np.linalg.lstsq(X[sel], s[sel], rcond=None)[0]

def boot(mask_pos, score):
    t = test & ((stage == "neg") | mask_pos)
    tasks = np.unique(task[t]); vals = []
    for _ in range(500):
        pick = rng.choice(tasks, len(tasks))
        idx = np.concatenate([np.where(t & (task == k))[0] for k in pick])
        v = auroc(stage[idx] != "neg", score[idx])
        if v == v: vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

out = {"selected": rep["selected"], "stages": {}, "stage_x_channel": {}}
negs = test & (stage == "neg")
for st in ("considered", "attempted", "enacted"):
    m = stage == st
    t = test & ((stage == "neg") | m)
    if (test & m).sum() < 5: continue
    thr_mon_fpr = mon[negs].mean()
    out["stages"][st] = dict(n_pos=int((test & m).sum()), auroc=auroc(m[t], s[t]), ci=boot(m, s),
                             len_auroc=auroc(m[t], ln[t]), resid_auroc=auroc(m[t], resid[t]), monitor_tpr=float(mon[test & m].mean()),
                             probe_tpr_at_monitor_fpr=tpr_at_fpr(m[t], s[t], thr_mon_fpr)[0])
for st in ("considered", "enacted"):
    for c in sorted(set(chan)):
        m = (stage == st) & (chan == c); t = test & (((stage == "neg") & (chan == c)) | m)
        if (test & m).sum() >= 5:
            out["stage_x_channel"][f"{st}/{c}"] = dict(n_pos=int((test & m).sum()), auroc=auroc(m[t], s[t]),
                                                       monitor_tpr=float(mon[test & m].mean()))
out["monitor_fpr"] = float(mon[negs].mean())
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps(out, indent=1))
