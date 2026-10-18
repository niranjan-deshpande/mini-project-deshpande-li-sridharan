"""Phase 0 sampler: n samples per task from the vLLM server, single-turn prompt, raw completion text (exact tokens
preserved via /v1/completions with token-id prompts), then the sandboxed test reward.

Usage: python sample.py TASKS.jsonl OUT.jsonl --mode lenient|strict|hint --effort xhigh|medium|low --n 4 [--limit K]
"""
import argparse, concurrent.futures as cf, glob, json, os, random, time, urllib.request
from transformers import AutoTokenizer
from env_lcb import user_prompt, split_response, run_tests

ap = argparse.ArgumentParser()
ap.add_argument("tasks"); ap.add_argument("out")
ap.add_argument("--mode", default="lenient"); ap.add_argument("--effort", default="xhigh")
ap.add_argument("--n", type=int, default=4); ap.add_argument("--max-tokens", type=int, default=16384)
ap.add_argument("--limit", type=int, default=None); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--port", default="8400"); ap.add_argument("--workers", type=int, default=64)
ap.add_argument("--chat", default=None, help="served model name: use /v1/chat/completions (gpt-oss); text = reasoning</think>content")
a = ap.parse_args()
tok = None if "--chat" in os.sys.argv else AutoTokenizer.from_pretrained(glob.glob("/data/hf_cache/models--Qwen--Qwen3.8-27B/snapshots/*")[0])
KEY = open("/data/agastyas/rh_probe/api_key").read().strip()
T = [json.loads(l) for l in open(a.tasks)]
random.Random(a.seed).shuffle(T)
T = T[: a.limit]
done = {json.loads(l)["task_id"] for l in open(a.out)} if os.path.exists(a.out) else set()
T = [t for t in T if t["task_id"] not in done]


def gen(t):
    if a.chat:
        return gen_chat(t)
    ids = tok(tok.apply_chat_template([{"role": "user", "content": user_prompt(t, a.mode)}], tokenize=False,
                                        add_generation_prompt=True, reasoning_effort=a.effort),
              add_special_tokens=False)["input_ids"]
    body = {"model": "qwen3.8-27b", "prompt": ids, "n": a.n, "max_tokens": a.max_tokens, "temperature": 1.0,
            "top_p": 1.0, "skip_special_tokens": False, "stop_token_ids": [248046, 248044]}
    req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    for i in range(4):
        try:
            return t, ids, json.load(urllib.request.urlopen(req, timeout=7200))
        except Exception as e:
            err = e; time.sleep(15)
    raise err


def gen_chat(t):
    body = {"model": a.chat, "messages": [{"role": "user", "content": user_prompt(t, a.mode)}], "n": a.n,
            "max_tokens": a.max_tokens, "temperature": 1.0, "top_p": 1.0, "reasoning_effort": a.effort}
    req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    for i in range(4):
        try:
            res = json.load(urllib.request.urlopen(req, timeout=7200))
            break
        except Exception as e:
            err = e; time.sleep(15)
    else:
        raise err
    ch = [{"text": ((c["message"].get("reasoning_content") or c["message"].get("reasoning") or "") + "</think>" +
                    (c["message"].get("content") or "")), "finish_reason": c["finish_reason"]} for c in res["choices"]]
    return t, range(res["usage"]["prompt_tokens"]), {"choices": ch}


t0 = time.time()
with cf.ThreadPoolExecutor(a.workers) as ex, cf.ProcessPoolExecutor(32) as px, open(a.out, "a") as f:
    for i, fu in enumerate(cf.as_completed([ex.submit(gen, t) for t in T])):
        t, ids, res = fu.result()
        comps = [c["text"] for c in res["choices"]]
        answers = [split_response(c)[1] for c in comps]
        tests = list(px.map(run_tests, [t] * len(comps), answers))
        f.write(json.dumps({"task_id": t["task_id"], "split": t["split"], "mode": a.mode, "effort": a.effort,
                            "prompt_len": len(ids), "completions": comps,
                            "finish": [c["finish_reason"] for c in res["choices"]],
                            "passed": [x[0] for x in tests], "err": [x[1][-200:] for x in tests]}) + "\n"); f.flush()
        if i % 20 == 0:
            print(f"{i + 1}/{len(T)} {time.time() - t0:.0f}s", flush=True)
print("DONE")
