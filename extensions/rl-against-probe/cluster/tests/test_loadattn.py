"""T3b: in-place attention load via the AttnLoader worker extension, verified with real decoding.
Server: one DP1 gpt-oss server on :8400 started with --worker-extension-cls vllm_ext.AttnLoader."""
import json, time, urllib.request
KEY = open("/data/agastyas/rh_probe/api_key").read().strip()
OUT = "/data/agastyas/rh_rl/tests/reload"


def call(path, body=None):
    req = urllib.request.Request(f"http://127.0.0.1:8400{path}", json.dumps(body).encode() if body is not None else b"",
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}, method="POST")
    r = urllib.request.urlopen(req, timeout=900).read()
    return json.loads(r) if r else None


ids = list(range(1000, 1200))
def plp():
    r = call("/v1/completions", {"model": "gpt-oss-120b", "prompt": ids, "max_tokens": 1, "prompt_logprobs": 0, "temperature": 0})
    return [list(x.values())[0]["logprob"] for x in r["choices"][0]["prompt_logprobs"][1:]]


def decode():
    t = time.time()
    r = call("/v1/completions", {"model": "gpt-oss-120b", "prompt": ids[:50], "max_tokens": 512, "n": 16, "temperature": 1.0,
                                 "ignore_eos": True})
    return sum(c["finish_reason"] == "length" for c in r["choices"]), time.time() - t


def load(d):
    t = time.time()
    call("/pause?mode=wait")
    res = call("/collective_rpc", {"method": "load_attn", "args": [f"{OUT}/{d}/model.safetensors"]})
    call("/reset_prefix_cache"); call("/resume")
    return time.time() - t, res


base = plp(); print("decode before:", decode())
dt, res = load("pert"); pert = plp()
print(f"load pert {dt:.2f}s res={res}; mean |dlp| vs base {sum(abs(a - b) for a, b in zip(pert, base)) / len(base):.4f} (>0); decode:", decode())
dt, res = load("orig"); back = plp()
print(f"load orig {dt:.2f}s; mean |dlp| vs base {sum(abs(a - b) for a, b in zip(back, base)) / len(base):.6f} (~0); decode:", decode())
for k in range(3):
    load("pert"); load("orig")
print("after 3 more round trips: mean |dlp|", sum(abs(a - b) for a, b in zip(plp(), base)) / len(base), "decode:", decode())
