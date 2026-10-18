"""Judge the held-out eval samples (runs/eval/{base,armA_100,armB_100}/{lenient,strict}.jsonl) with GPT-5.6 Sol.

Same judge prompt / schema / quote check as judge_rl.py (and the step-2 calibration judge), with the task shown in the
mode it was sampled under. Loop: rsync runs/eval/ from athena02 -> data/eval_samples/, judge every finished file not yet
judged -> data/judge_eval/{model}_{mode}.jsonl (one row per sample, written atomically per file). Exits once
runs/eval/DONE exists and everything is judged.

Usage: python3 judge_eval.py [--workers 48] [--every 300]
"""
import argparse, concurrent.futures as cf, glob, json, os, subprocess, sys, time

ROOT = "/Users/agastyasridharan/j-lens/rh_rl"
P0 = "/Users/agastyasridharan/j-lens/rh_probe/p0_judge.py"
sys.path.insert(0, "/Users/agastyasridharan/j-lens/rh_probe")
src = open(P0).read()
ns = {"__name__": "p0_judge_defs"}
exec(src[:src.index("ap = argparse.ArgumentParser()")], ns)
INSTR, PROMPT, SCHEMA, call, verify_quote, user_prompt, CATS = (ns[k] for k in
    ["INSTR", "PROMPT", "SCHEMA", "call", "verify_quote", "user_prompt", "JUDGE_CATS"])

ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=48)
ap.add_argument("--every", type=int, default=300)
ap.add_argument("--done", default="DONE", help="marker file in runs/eval/ that ends the loop")
a = ap.parse_args()
HOST = os.environ.get("RL_HOST", "user@gpu-host")  # ssh target holding the run outputs
TASKS = {json.loads(l)["task_id"]: json.loads(l) for l in open(f"{ROOT}/gen/ib_eval_tasks.jsonl")}
SRC, OUT = f"{ROOT}/data/eval_samples", f"{ROOT}/data/judge_eval"
os.makedirs(SRC, exist_ok=True); os.makedirs(OUT, exist_ok=True)


def sync():
    r = subprocess.run(["rsync", "-a", "--include=*/", "--include=*.jsonl", "--include=DONE*", "--exclude=*",
                        "-e", "ssh -o BatchMode=yes -o ConnectTimeout=20",
                        f"{HOST}:/data/agastyas/rh_rl/runs/eval/", f"{SRC}/"],
                       capture_output=True, text=True, stdin=subprocess.DEVNULL)
    return r.returncode == 0


def judge_one(s):
    resp = (s["analysis"] or "") + "</think>" + (s["final"] or "")
    out, _ = call(INSTR, PROMPT.format(cats=", ".join(CATS), task=user_prompt(TASKS[s["task_id"]], s["mode"]), resp=resp),
                  SCHEMA, "p0_verdict", effort="medium")
    if out["hack"] and not verify_quote(out["quote"] or "", resp):
        out["hack"], out["status"], out["why"] = False, None, "withdrawn: quote not found. " + out["why"]
    return out


while True:
    synced = sync()
    files = [f for f in sorted(glob.glob(f"{SRC}/*/*.jsonl"))
             if os.path.basename(f) in ("lenient.jsonl", "strict.jsonl")
             and not os.path.exists(f"{OUT}/{f.split('/')[-2]}_{os.path.basename(f)}")]
    if files:  # all pending files in one pool, written per file
        S = {f: [json.loads(l) for l in open(f)] for f in files}
        with cf.ThreadPoolExecutor(a.workers) as ex:
            futs = {f: [ex.submit(judge_one, s) for s in S[f]] for f in files}
            for f in files:
                V = []
                for fu in futs[f]:
                    try:
                        V.append(fu.result())
                    except Exception as e:
                        V.append({"error": str(e)[:300]})
                dst = f"{OUT}/{f.split('/')[-2]}_{os.path.basename(f)}"
                with open(dst + ".tmp", "w") as fo:
                    for i, (s, v) in enumerate(zip(S[f], V)):
                        fo.write(json.dumps({"i": i, **{k: s[k] for k in ("task_id", "split", "mode", "pass", "probe", "finish", "n_tok")},
                                             "verdict": v}) + "\n")
                os.replace(dst + ".tmp", dst)
                print(time.strftime("%H:%M:%S"), f"judged {dst.split('/')[-1]}: {len(S[f])} samples, errors "
                      f"{sum('error' in v for v in V)}", flush=True)
    if synced and not files and os.path.exists(f"{SRC}/{a.done}"):
        print("all evals judged; exiting", flush=True)
        break
    time.sleep(a.every)
