"""Extra T2 numbers for the report: length baseline, length-residualised probe AUROC, TPR at 5% FPR (Qwen), and the
rollout counts. Uses the probe that t2_analyze.py selected (read from its report JSON).

Usage: python3 t2_extra.py OUT.json data/t2_qwen_final.json data/t2/qwen_part1 data/t2/qwen_part2 data/t2/qwen_part3
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
ln = np.array([IDX[i]["tok1"] - IDX[i]["tok0"] for i in ii])
X = np.c_[np.ones(len(ln)), np.log(ln)]
beta = np.linalg.lstsq(X[sel], s[sel], rcond=None)[0]
t = test
out = {"q_len": auroc(y[t], ln[t]), "q_resid": auroc(y[t], (s - X @ beta)[t]),
       "q_tpr5": tpr_at_fpr(y[t], s[t], 0.05)[0]}
if os.environ.get("LABELS"):
    json.dump(out, open(OUT, "w"), indent=1); print(out); raise SystemExit
uids = {json.loads(l)["uid"] for f in ("qwen_partial_clean", "qwen_new2", "qwen_new3") for l in open(f"data/{f}.jsonl")}
out["q_n"] = len(uids)
out["q_drop"] = (f"Qwen stopped at {len(uids)} of 305 planned rollouts: the rest were cancelled by harness restarts, "
                 "overflowed the 131k context, or were still running in the long tail when the run was cut.")
json.dump(out, open(OUT, "w"), indent=1)
print(out)
