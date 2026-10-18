"""Judge every RL training sample (arms A and B) with GPT-5.6 Sol, incrementally while training runs.

Loop: rsync runs/{armA,armB}/samples/ from athena02 -> data/rl_samples/{arm}/, judge every finished step file not yet
judged (the newest file is skipped while the arm is still running), append verdicts to data/judge_rl/{arm}.jsonl.
Same judge prompt / schema / quote verification as the step-2 calibration judge (rh_probe/p0_judge.py), so verdicts are
comparable to the calibration labels. A failed sync (VPN down) is retried on the next pass. Exits once both arms have
stopped and every step file is judged.

Usage: python3 judge_rl.py [--workers 48] [--every 300]
"""
import argparse, concurrent.futures as cf, glob, json, os, subprocess, sys, time

ROOT = "/Users/agastyasridharan/j-lens/rh_rl"
P0 = "/Users/agastyasridharan/j-lens/rh_probe/p0_judge.py"
sys.path.insert(0, "/Users/agastyasridharan/j-lens/rh_probe")
src = open(P0).read()
ns = {"__name__": "p0_judge_defs"}
exec(src[:src.index("ap = argparse.ArgumentParser()")], ns)  # INSTR, PROMPT, SCHEMA, call, verify_quote, user_prompt, JUDGE_CATS
INSTR, PROMPT, SCHEMA, call, verify_quote, user_prompt, CATS = (ns[k] for k in
    ["INSTR", "PROMPT", "SCHEMA", "call", "verify_quote", "user_prompt", "JUDGE_CATS"])

ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=48)
ap.add_argument("--every", type=int, default=300)
ap.add_argument("--arms", default="armA,armB")
a = ap.parse_args()
HOST = os.environ.get("RL_HOST", "user@gpu-host")  # ssh target holding the run outputs
ARMS = a.arms.split(",")
TASKS = {json.loads(l)["task_id"]: json.loads(l) for l in open(f"{ROOT}/gen/train_tasks.jsonl")}
os.makedirs(f"{ROOT}/data/judge_rl", exist_ok=True)


def sync():
    ok = True
    for arm in ARMS:
        os.makedirs(f"{ROOT}/data/rl_samples/{arm}", exist_ok=True)
        r = subprocess.run(["rsync", "-a", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4",
                            f"{HOST}:/data/agastyas/rh_rl/runs/{arm}/samples/", f"{ROOT}/data/rl_samples/{arm}/"],
                           capture_output=True, text=True, stdin=subprocess.DEVNULL)
        ok &= r.returncode == 0
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", HOST,
                            f"test -f /data/agastyas/rh_rl/runs/{arm}/STOPPED -o -f /data/agastyas/rh_rl/runs/{arm}/DONE && echo stopped"],
                           capture_output=True, text=True, stdin=subprocess.DEVNULL)
        if "stopped" in r.stdout:
            open(f"{ROOT}/data/rl_samples/{arm}/.stopped", "w").write("1")
    return ok


def judge_one(job):
    s = job
    resp = (s["analysis"] or "") + "</think>" + (s["final"] or "")
    out, _ = call(INSTR, PROMPT.format(cats=", ".join(CATS), task=user_prompt(TASKS[s["task_id"]], "lenient"), resp=resp),
                  SCHEMA, "p0_verdict", effort="medium")
    if out["hack"] and not verify_quote(out["quote"] or "", resp):
        out["hack"], out["status"], out["why"] = False, None, "withdrawn: quote not found. " + out["why"]
    return out


while True:
    synced = sync()
    pending = []
    for arm in ARMS:
        outp = f"{ROOT}/data/judge_rl/{arm}.jsonl"
        done = {json.loads(l)["step"] for l in open(outp)} if os.path.exists(outp) else set()
        files = sorted(glob.glob(f"{ROOT}/data/rl_samples/{arm}/step_*.jsonl"))
        stopped = os.path.exists(f"{ROOT}/data/rl_samples/{arm}/.stopped")
        if not stopped:
            files = files[:-1]  # newest may still be being written
        for f in files:
            step = int(os.path.basename(f)[5:9])
            if step not in done:
                pending.append((arm, step, [json.loads(l) for l in open(f)]))
    if pending:
        jobs = [(arm, step, i, s) for arm, step, S in pending for i, s in enumerate(S)]
        res = {}
        with cf.ThreadPoolExecutor(a.workers) as ex:
            futs = {ex.submit(judge_one, s): (arm, step, i) for arm, step, i, s in jobs}
            for f in cf.as_completed(futs):
                try:
                    res[futs[f]] = f.result()
                except Exception as e:
                    res[futs[f]] = {"error": str(e)[:300]}
        for arm, step, S in sorted(pending, key=lambda x: (x[0], x[1])):
            with open(f"{ROOT}/data/judge_rl/{arm}.jsonl", "a") as fo:
                fo.write(json.dumps({"arm": arm, "step": step, "rows": [
                    {"i": i, "task_id": s["task_id"], "split": s["split"], "pass": s["pass"], "probe": s["probe"],
                     "fires": s["fires"], "n_tok": s["n_tok"], "finish": s["finish"], "verdict": res[(arm, step, i)]}
                    for i, s in enumerate(S)]}) + "\n")
        print(time.strftime("%H:%M:%S"), f"judged {len(jobs)} samples from {len(pending)} step files; "
              f"errors {sum('error' in v for v in res.values())}", flush=True)
    all_stopped = all(os.path.exists(f"{ROOT}/data/rl_samples/{arm}/.stopped") for arm in ARMS)
    if all_stopped and synced and not pending:
        print("all arms stopped and judged; exiting", flush=True)
        break
    time.sleep(a.every)
