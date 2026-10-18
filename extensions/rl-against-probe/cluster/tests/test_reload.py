"""GPU test T3: hot-reload merged attention weights into a no-LoRA vLLM (VLLM_SERVER_DEV_MODE=1, port 8400).
Builds two partial checkpoints of the attention weights (original, and o_proj scaled by 0.9) and checks:
  prompt logprobs change after loading the perturbed one and return exactly to baseline after reloading the original;
  pause -> collective_rpc reload_weights -> reset_prefix_cache -> resume latency."""
import json, os, time, urllib.request, glob, re
import torch
from safetensors import safe_open
from safetensors.torch import save_file
SNAP = glob.glob("/data/agastyas/hf_cache_latekey/models--openai--gpt-oss-120b/snapshots/*")[0]
KEY = open("/data/agastyas/rh_probe/api_key").read().strip()
OUT = "/data/agastyas/rh_rl/tests/reload"
os.makedirs(f"{OUT}/orig", exist_ok=True); os.makedirs(f"{OUT}/pert", exist_ok=True)


def call(path, body=None, method="POST"):
    req = urllib.request.Request(f"http://127.0.0.1:8400{path}", json.dumps(body).encode() if body is not None else b"",
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}, method=method)
    r = urllib.request.urlopen(req, timeout=600).read()
    return json.loads(r) if r else None


wm = json.load(open(f"{SNAP}/model.safetensors.index.json"))["weight_map"]
att = [k for k in wm if re.search(r"self_attn\.(q_proj|k_proj|v_proj|o_proj)\.(weight|bias)$", k)]
t = time.time(); W = {}
for f in sorted({wm[k] for k in att}):
    with safe_open(f"{SNAP}/{f}", "pt") as fh:
        for k in att:
            if wm[k] == f:
                W[k] = fh.get_tensor(k)
print(f"{len(W)} attention tensors, {sum(v.numel() * v.element_size() for v in W.values()) / 2**30:.2f} GiB, read {time.time() - t:.1f}s")
save_file(W, f"{OUT}/orig/model.safetensors")
save_file({k: (v * 0.9 if "o_proj.weight" in k else v) for k, v in W.items()}, f"{OUT}/pert/model.safetensors")
for d in ["orig", "pert"]:
    os.system(f"cp {SNAP}/config.json {OUT}/{d}/")

ids = list(range(1000, 1200))
def plp():
    r = call("/v1/completions", {"model": "gpt-oss-120b", "prompt": ids, "max_tokens": 1, "prompt_logprobs": 0, "temperature": 0})
    return [list(x.values())[0]["logprob"] for x in r["choices"][0]["prompt_logprobs"][1:]]


def reload(d):
    t = time.time()
    call("/pause?mode=wait")
    res = call("/collective_rpc", {"method": "reload_weights", "kwargs": {"weights_path": f"{OUT}/{d}"}})
    call("/reset_prefix_cache")
    call("/resume")
    return time.time() - t, res


base = plp()
dt, res = reload("pert"); pert = plp()
print(f"reload pert {dt:.1f}s res={str(res)[:200]}; mean |dlp| vs base = {sum(abs(a - b) for a, b in zip(pert, base)) / len(base):.4f} (should be > 0)")
dt, res = reload("orig"); back = plp()
print(f"reload orig {dt:.1f}s; mean |dlp| vs base = {sum(abs(a - b) for a, b in zip(back, base)) / len(base):.6f} (should be ~0)")
