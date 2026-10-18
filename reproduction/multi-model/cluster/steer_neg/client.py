"""Sends every cut-point prompt to one steered server (n samples each) and writes continuations. Resumable."""
import concurrent.futures as cf, json, os, sys, time, urllib.request
port, cond, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
KEY = open("/data/agastyas/rh_probe/api_key").read().strip()
STOP = [151329, 151336, 151338]  # <|endoftext|>, <|user|>, <|observation|>
path = f"steer_neg/out_{cond}.jsonl"
done = {json.loads(l)["uid"] for l in open(path)} if os.path.exists(path) else set()
P = [json.loads(l) for l in open("steer_neg/prompts.jsonl")]
P = [p for p in P if p["uid"] not in done]


def go(p):
    body = {"model": "glm-4.5-air", "prompt": p["ids"], "n": n, "max_tokens": 8192, "skip_special_tokens": False,
            "stop_token_ids": STOP, "include_stop_str_in_output": False}
    req = urllib.request.Request(f"http://localhost:{port}/v1/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    for attempt in range(3):
        try:
            res = json.load(urllib.request.urlopen(req, timeout=3600))
            return p, res
        except Exception as e:
            err = e; time.sleep(10)
    raise err


t0 = time.time()
with cf.ThreadPoolExecutor(12) as ex, open(path, "a") as f:
    futs = [ex.submit(go, p) for p in sorted(P, key=lambda p: len(p["ids"]))]
    for i, fu in enumerate(cf.as_completed(futs)):
        p, res = fu.result()
        f.write(json.dumps({"uid": p["uid"], "cond": cond, "step": p["step"],
                            "continuations": [c["text"] for c in res["choices"]],
                            "finish": [c["finish_reason"] for c in res["choices"]],
                            "n_tokens": res["usage"]["completion_tokens"]}) + "\n"); f.flush()
        print(f"{cond}: {i + 1}/{len(P)} {time.time() - t0:.0f}s", flush=True)
print(f"DONE {cond}")
