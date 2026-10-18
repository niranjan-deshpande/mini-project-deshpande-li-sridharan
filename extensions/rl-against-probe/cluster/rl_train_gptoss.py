"""GRPO with attention-LoRA on gpt-oss-120b in the single-turn Impossible-LCB environment, optional DoM-probe penalty.

Layout (athena02, GPUs 4-7; see run_rl_gptoss.sh):
  * sampler: one vLLM server, DP2 on GPUs 4-5, NO LoRA (keeps the fast MoE kernel), VLLM_SERVER_DEV_MODE=1.
    After every step the trainer merges W0 + (alpha/r) B A into the bf16 attention weights, writes them as a partial
    checkpoint and loads them in place (/pause -> /collective_rpc load_attn [vllm_ext.py] -> /reset_prefix_cache -> /resume).
  * trainer: torchrun, 2 ranks on GPUs 6-7, one full replica each. Expert weights stay MXFP4-resident and are
    dequantised to bf16 per layer per call (tests/dq_experts.py); flex attention (sinks); non-reentrant checkpointing.
  * one-step async: while the trainer trains on batch n (policy n-1 -> n), the sampler generates batch n+1 with
    policy n-1; per-token truncated importance weights min(exp(lp_theta - lp_vllm), 2) correct the lag and any
    vLLM/HF numerical mismatch.

Per step:
  1. rank 0: G samples for each of P prompts (half impossible, half original) as exact token ids (harmony prompt
     rendered with openai_harmony, /v1/completions with token-id prompt; gptoss_rollout.py), test reward per sample.
  2. all ranks, no grad, layers 0..PL only: probe score = max over analysis-message tokens of <v, h_PL> (the
     pre-registered gpt-oss probe span__all @ layer 13; calibrated on the step-1 pilot, AUROC 0.91 at medium effort).
  3. rank 0: reward = pass - lam * [score > tau]; group-normalised advantages; all-equal groups skipped.
  4. all ranks: one forward+backward per active sample, loss = -sum_t w_t A logp_t / N_active_tokens.
  5. rank 0: merged attention weights -> vLLM (after batch n+1 finished); adapter checkpoint every --save-every.
Every sample (texts, pass, probe score, advantage) is logged to {run}/samples/step_XXXX.jsonl for later judging.

Usage: torchrun --nproc_per_node 2 rl_train_gptoss.py --rundir runs/armA --lam 0
"""
import argparse, concurrent.futures as cf, glob, json, os, random, re, sys, threading, time, urllib.request
import numpy as np, torch, torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import save_file
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, "/data/agastyas/rh_rl"); sys.path.insert(0, "/data/agastyas/rh_rl/tests")
from env_lcb import user_prompt, run_tests
from gptoss_rollout import ENC, RETURN, CALL, prompt_ids, split_channels

ap = argparse.ArgumentParser()
ap.add_argument("--rundir", required=True)
ap.add_argument("--tasks", default="/data/agastyas/rh_rl/train_tasks.jsonl")
ap.add_argument("--lam", type=float, default=0.0)
ap.add_argument("--tau", type=float, default=None, help="probe threshold; required if lam > 0")
ap.add_argument("--steps", type=int, default=100)
ap.add_argument("--prompts", type=int, default=32)
ap.add_argument("--group", type=int, default=8)
ap.add_argument("--max-new", type=int, default=16384)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--rank-lora", type=int, default=32)
ap.add_argument("--mode", default="lenient"); ap.add_argument("--effort", default="medium")
ap.add_argument("--ports", default="8400,8401", help="independent DP1 vLLM servers (partial weight reload breaks DP>1 MoE)")
ap.add_argument("--probe", default="span__all:13")
ap.add_argument("--dom", default="/data/agastyas/rh_probe/dom/gptoss_h/dom_vectors.npz")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--save-every", type=int, default=10)
ap.add_argument("--mb-tokens", type=int, default=32768, help="padded tokens per micro-batch (amortises per-layer expert dequant)")
ap.add_argument("--stop-at", type=float, default=None, help="unix time: stop before a step that would not finish by then; save ckpt_final")
ap.add_argument("--resume", default=None, help="checkpoint dir (adapter + trainer_state.pt) to continue from")
ap.add_argument("--no-reload", action="store_true", help="debug: never push weights to vLLM")
ap.add_argument("--eval-out", default=None, help="held-out eval instead of training: write {eval-out}/{mode}.jsonl and exit")
ap.add_argument("--eval-tasks", default="/data/agastyas/rh_rl/ib_eval_tasks.jsonl")
ap.add_argument("--eval-modes", default="lenient:8,strict:4", help="mode:samples_per_task, comma-separated")
ap.add_argument("--penalty", default="fire", choices=["fire", "linear", "filter"],
                help="fire: lam * [score > tau]; linear: lam * (score - tau) / pen_scale (unclipped; NaN score -> 0); filter: reward = pass, samples with score > tau get advantage 0 and are excluded from their group mean/std (groups with < 2 kept or all-equal kept skip)")
ap.add_argument("--pen-scale", type=float, default=None)
ap.add_argument("--score-in", default=None, help="re-score saved texts instead of training: comma list of path:start:end")
ap.add_argument("--score-out", default=None)
ap.add_argument("--score-trace", type=int, default=None, help="with --score-out: also save the per-token span__all projection at this layer + decoded peak windows")
a = ap.parse_args()
assert a.lam == 0 or a.tau is not None
assert a.penalty != "linear" or a.pen_scale

dist.init_process_group("nccl")
RANK, WORLD = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
DEV = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
os.makedirs(f"{a.rundir}/samples", exist_ok=True)
SNAP = glob.glob("/data/agastyas/hf_cache_latekey/models--openai--gpt-oss-120b/snapshots/*")[0]
KEY = open("/data/agastyas/rh_probe/api_key").read().strip()
torch.manual_seed(a.seed); random.seed(a.seed)
T0 = time.time()

# ---------------- model: MXFP4-resident experts, attention LoRA ----------------
import torch._dynamo
torch._dynamo.config.cache_size_limit = 256  # past the limit flex attention silently falls back to an unfused
torch._dynamo.config.recompile_limit = 256   # implementation that materialises the full T x T score matrix (OOM)
from transformers import AutoConfig, GptOssForCausalLM
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding
from peft import LoraConfig, get_peft_model
from dq_experts import make_dq  # MXFP4-resident experts, chunked dequant (no CPU fallback)

cfg = AutoConfig.from_pretrained(SNAP)
if hasattr(cfg, "quantization_config"):
    del cfg.quantization_config
cfg.use_cache = False
with torch.device("meta"):
    model = GptOssForCausalLM._from_config(cfg, attn_implementation="flex_attention", experts_implementation="grouped_mm",
                                           dtype=torch.bfloat16)
for l in model.model.layers:
    del l.mlp.experts.gate_up_proj, l.mlp.experts.down_proj
model.to_empty(device=DEV)
model.model.rotary_emb = GptOssRotaryEmbedding(cfg, device=DEV)
P_ = dict(model.named_parameters()); raw, loaded = {}, set()
WMAP = json.load(open(f"{SNAP}/model.safetensors.index.json"))["weight_map"]
for f in sorted(set(WMAP.values())):
    with safe_open(f"{SNAP}/{f}", "pt", device=str(DEV)) as fh:
        for k in fh.keys():
            if k.endswith(("_blocks", "_scales")):
                raw[k] = fh.get_tensor(k)
            else:
                with torch.no_grad():
                    P_[k].copy_(fh.get_tensor(k))
                loaded.add(k)
assert set(P_) == loaded, set(P_) ^ loaded
for i, l in enumerate(model.model.layers):
    p = f"model.layers.{i}.mlp.experts."
    make_dq(l.mlp.experts, raw[p + "gate_up_proj_blocks"], raw[p + "gate_up_proj_scales"],
            raw[p + "down_proj_blocks"], raw[p + "down_proj_scales"])
del raw, P_
ATT = ["q_proj", "k_proj", "v_proj", "o_proj"]
model = get_peft_model(model, LoraConfig(r=a.rank_lora, lora_alpha=2 * a.rank_lora, lora_dropout=0.0, target_modules=ATT))
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
inner = model.base_model.model
lm, head = inner.model, inner.lm_head
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), weight_decay=0.0)
start_step = 0
if a.resume:  # adapter weights + AdamW state + next step index; the batch RNG is keyed by step, so the task sequence continues
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    set_peft_model_state_dict(model, load_file(f"{a.resume}/adapter_model.safetensors", device=str(DEV)))
    st_ = torch.load(f"{a.resume}/trainer_state.pt", map_location=DEV, weights_only=False)
    opt.load_state_dict(st_["opt"]); start_step = st_["step"]
    if RANK == 0:
        print(f"resumed from {a.resume} at step {start_step}", flush=True)


def save_state(path, next_step):
    """everything needed to resume: adapter, optimizer moments, next step index (rank 0 only)."""
    model.save_pretrained(path)
    torch.save({"opt": opt.state_dict(), "step": next_step, "args": vars(a)}, f"{path}/trainer_state.pt")


if RANK == 0:
    print(f"model ready {time.time() - T0:.0f}s, {torch.cuda.memory_allocated() / 2**30:.1f} GiB resident, "
          f"trainable {sum(p.numel() for p in params) / 1e6:.1f}M", flush=True)

pk, PL = a.probe.split(":"); PL = int(PL)
VEC = torch.tensor(np.load(a.dom)[pk][PL], dtype=torch.float32, device=DEV)
MSG, END = ENC.encode("<|message|>", allowed_special="all")[0], ENC.encode("<|end|>", allowed_special="all")[0]


class _Stop(Exception):
    pass


def analysis_span(c_ids):
    """[m0, m1) completion-token range of the analysis message body."""
    m0 = c_ids.index(MSG) + 1 if MSG in c_ids else 0
    m1 = m0 + c_ids[m0:].index(END) if END in c_ids[m0:] else len(c_ids)
    return m0, m1


def microbatches(idx, lens, budget):
    """longest-first greedy packing of sample indices so that B * max_len <= budget (right padding)."""
    out, cur = [], []
    for i in sorted(idx, key=lambda i: -lens[i]):
        if cur and (len(cur) + 1) * lens[cur[0]] > budget:
            out.append(cur); cur = []
        cur.append(i)
    return out + ([cur] if cur else [])


def padded(seqs):
    L = -(-max(len(x) for x in seqs) // 512) * 512  # bucket lengths: every new shape recompiles flex attention
    x = torch.zeros(len(seqs), L, dtype=torch.long, device=DEV); m = torch.zeros(len(seqs), L, dtype=torch.long, device=DEV)
    for b, q in enumerate(seqs):
        x[b, :len(q)] = torch.tensor(q, device=DEV); m[b, :len(q)] = 1
    return x, m


def probe_scores(samples):
    """max over analysis-body tokens of <v, h_PL> per sample; runs layers 0..PL only."""
    x, m = padded([s["p_ids"] + s["c_ids"] for s in samples])
    cap = {}

    def hk(mod, i, o):
        cap["h"] = o[0] if isinstance(o, tuple) else o
        raise _Stop

    h = lm.layers[PL].register_forward_hook(hk)
    try:
        lm(input_ids=x, attention_mask=m)
    except _Stop:
        pass
    finally:
        h.remove()
    out = []
    for b, s in enumerate(samples):
        m0, m1 = analysis_span(s["c_ids"]); n = len(s["p_ids"])
        out.append(float((cap["h"][b, n + m0:n + m1].float() @ VEC).max()) if m1 > m0 else float("nan"))
    return out


def batch_logprobs(samples):
    """per-sample log p(token_t | <t) over completion tokens, chunked through lm_head."""
    x, m = padded([s["p_ids"] + s["c_ids"] for s in samples])
    # checkpoint early-stop can abort a recompute inside the experts before the dequant post-hook frees the bf16
    # weights, stranding ~6.4 GB per layer during backward; recompute whole layers instead.
    with torch.utils.checkpoint.set_checkpoint_early_stop(False):
        hs = lm(input_ids=x, attention_mask=m).last_hidden_state
    out = []
    for b, s in enumerate(samples):
        n, L = len(s["p_ids"]), len(s["p_ids"]) + len(s["c_ids"])
        hb, tb = hs[b, n - 1:L - 1], x[b, n:L]
        parts = []
        for c in range(0, hb.shape[0], 4096):
            def f(hc, tc):
                lg = head(hc).float()
                return lg.gather(-1, tc[:, None])[:, 0] - torch.logsumexp(lg, -1)
            parts.append(checkpoint(f, hb[c:c + 4096], tb[c:c + 4096], use_reentrant=False))
        out.append(torch.cat(parts))
    return out


# ---------------- sampler side (rank 0) ----------------
TASKS = [json.loads(l) for l in open(a.tasks)]
IMP = [t for t in TASKS if t["split"] != "original"]
ORI = [t for t in TASKS if t["split"] == "original"]


PORTS = a.ports.split(",")


def http(path, body=None, timeout=7200, port=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port or PORTS[0]}{path}", json.dumps(body).encode() if body is not None else b"",
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}, method="POST")
    for i in range(5):
        try:
            r = urllib.request.urlopen(req, timeout=timeout).read()
            return json.loads(r) if r else None
        except Exception as e:
            err = e; time.sleep(10)
    raise err


MAX_LEN = 32768  # sampler --max-model-len


def sample_task(job):
    """G samples for one task; [] (the group is dropped) if the prompt leaves < 2k tokens or the request keeps failing."""
    t, port = job
    ids = prompt_ids(user_prompt(t, a.mode), a.effort)
    budget = min(a.max_new, MAX_LEN - len(ids) - 64)
    if budget < 2048:
        print(f"skip {t['task_id']}: prompt {len(ids)} tokens", flush=True); return []
    try:
        return _sample_task(t, port, ids, budget)
    except Exception as e:
        print(f"dropped {t['task_id']}: {str(e)[:200]}", flush=True); return []


def _sample_task(t, port, ids, budget):
    res = http("/v1/completions", port=port, body={"model": "gpt-oss-120b", "prompt": ids, "n": a.group, "max_tokens": budget,
                                   "temperature": 1.0, "top_p": 1.0, "top_k": -1, "logprobs": 0, "return_token_ids": True,
                                   "skip_special_tokens": False, "stop_token_ids": [RETURN, CALL]})
    out = []
    for c in res["choices"]:
        c_ids, vlp = c["token_ids"], c["logprobs"]["token_logprobs"]
        ana, fin = split_channels(c_ids)
        ok = c["finish_reason"] == "stop" and c_ids and c_ids[-1] == RETURN
        out.append({"task": t, "p_ids": ids, "c_ids": c_ids, "vlp": vlp, "finish": c["finish_reason"],
                    "analysis": ana, "final": fin if ok else None})
    return out


def make_batch(step):
    rng = random.Random(a.seed * 100003 + step)
    ts = rng.sample(IMP, a.prompts // 2) + rng.sample(ORI, a.prompts - a.prompts // 2)
    t = time.time()
    with cf.ThreadPoolExecutor(len(ts)) as ex:
        S = [s for g in ex.map(sample_task, [(t, PORTS[i % len(PORTS)]) for i, t in enumerate(ts)]) for s in g]
    tg = time.time() - t
    with cf.ProcessPoolExecutor(32) as px:
        res = list(px.map(run_tests, [s["task"] for s in S], [s["final"] for s in S]))
    for s, (ok, _) in zip(S, res):
        s["pass"] = ok
    return S, tg


ATT_KEYS = [k for k in WMAP if re.search(r"self_attn\.(q_proj|k_proj|v_proj|o_proj)\.(weight|bias)$", k)]
WDIR = f"/dev/shm/agastyas_rl_{os.path.basename(os.path.abspath(a.rundir))}"


def merged_attention():
    """bf16 W0 + (alpha/r) B A for every attention projection, keyed by HF checkpoint names (+ unchanged biases)."""
    out = {}
    for name, mod in model.named_modules():
        if hasattr(mod, "lora_A") and hasattr(mod, "base_layer") and "default" in mod.lora_A:
            hf = name.replace("base_model.model.", "")
            W = mod.base_layer.weight.float() + mod.scaling["default"] * (
                mod.lora_B["default"].weight.float() @ mod.lora_A["default"].weight.float())
            out[f"{hf}.weight"] = W.to(torch.bfloat16).contiguous().cpu()
            if mod.base_layer.bias is not None:
                out[f"{hf}.bias"] = mod.base_layer.bias.detach().contiguous().cpu()
    assert set(out) == set(ATT_KEYS), set(out) ^ set(ATT_KEYS)
    return out


def push_weights(step):
    t = time.time()
    d = f"{WDIR}/{step % 2}"; os.makedirs(d, exist_ok=True)
    save_file(merged_attention(), f"{d}/model.safetensors")
    tw = time.time() - t
    for port in PORTS:
        http("/pause?mode=wait", port=port)
        # in-place load via the vllm_ext.AttnLoader worker extension (generic reload_weights breaks gpt-oss MXFP4 experts)
        http("/collective_rpc", {"method": "load_attn", "args": [f"{d}/model.safetensors"]}, port=port)
        http("/reset_prefix_cache", port=port); http("/resume", port=port)
    return tw, time.time() - t - tw


# ---------------- re-score saved texts with this model (cross-model / all-layer probe analysis) ----------------
if a.score_out:
    assert WORLD == 1
    os.makedirs(a.score_out, exist_ok=True)
    model.eval()
    DV = np.load(a.dom); KEYS = list(DV.files)
    VALL = torch.tensor(np.stack([DV[k] for k in KEYS], 1), dtype=torch.float32, device=DEV)  # (layers, keys, d)
    NL = len(lm.layers)
    ETASK = {t["task_id"]: t for t in map(json.loads, open(a.eval_tasks))}
    tagm = os.path.basename(os.path.abspath(a.rundir))
    for spec in a.score_in.split(","):
        path, s0, s1 = spec.rsplit(":", 2); s0, s1 = int(s0), int(s1)
        out = f"{a.score_out}/{tagm}__{os.path.basename(os.path.dirname(path))}_{s0}.npz"
        if os.path.exists(out):
            continue
        t = time.time()
        rows = [json.loads(l) for l in open(path)][s0:s1]
        S = [{"p_ids": prompt_ids(user_prompt(ETASK[r["task_id"]], r["mode"]), a.effort),
              "c_ids": ENC.encode("<|channel|>analysis<|message|>" + (r["analysis"] or "") + "<|end|>", allowed_special="all")}
             for r in rows]
        n = len(S)
        MX = np.full((n, NL, len(KEYS)), np.nan, np.float32); MN = MX.copy()
        POOL = np.zeros((n, NL, VALL.shape[-1]), np.float16)
        lens = {i: len(x["p_ids"]) + len(x["c_ids"]) for i, x in enumerate(S)}
        cur = {}; TR = {}; kspan = KEYS.index("span__all")

        def hook(l):
            def f(mod, i, o):
                h = o[0] if isinstance(o, tuple) else o
                for b, (j, off, m0, m1) in enumerate(cur["spans"]):
                    if m1 > m0:
                        hb = h[b, off + m0:off + m1].float()
                        pr = hb @ VALL[l].T
                        if a.score_trace == l:
                            TR[j] = (pr[:, kspan].cpu().numpy(), S[j]["c_ids"][m0:m1])
                        MX[j, l] = pr.max(0).values.cpu().numpy(); MN[j, l] = pr.mean(0).cpu().numpy()
                        POOL[j, l] = hb.mean(0).half().cpu().numpy()
            return f

        hs = [lm.layers[l].register_forward_hook(hook(l)) for l in range(NL)]
        with torch.no_grad():
            for mb in microbatches(list(range(n)), lens, a.mb_tokens):
                cur["spans"] = [(j, len(S[j]["p_ids"]), *analysis_span(S[j]["c_ids"])) for j in mb]
                x, m = padded([S[j]["p_ids"] + S[j]["c_ids"] for j in mb])
                lm(input_ids=x, attention_mask=m)
        for h_ in hs:
            h_.remove()
        tmp = out[:-4] + ".tmp.npz"
        np.savez(tmp, mx=MX, mn=MN, pool=POOL, keys=np.array(KEYS), task_id=np.array([r["task_id"] for r in rows]),
                 idx=np.arange(s0, s0 + n), stored_probe=np.array([r["probe"] for r in rows], dtype=np.float64))
        os.replace(tmp, out)
        if a.score_trace is not None:  # per-token projections + the 3 highest peaks >24 tokens apart, with +-24-token decoded windows
            js = sorted(TR); off_ = np.cumsum([0] + [len(TR[j][0]) for j in js])
            np.savez(out[:-4] + f"_trace{a.score_trace}.npz", tr=np.concatenate([TR[j][0] for j in js]) if js else np.zeros(0),
                     tok=np.array([t_ for j in js for t_ in TR[j][1]], dtype=np.int32), off=off_, idx=np.array(js) + s0)
            with open(out[:-4] + f"_peaks{a.score_trace}.jsonl", "w") as fo:
                for j in js:
                    v, toks = TR[j]; pk = []
                    for p_ in np.argsort(-v):
                        p_ = int(p_)
                        if all(abs(p_ - q[0]) > 24 for q in pk):
                            pk.append((p_, float(v[p_]), ENC.decode(toks[max(0, p_ - 24):p_]) + "⟦" + ENC.decode([toks[p_]]) + "⟧" + ENC.decode(toks[p_ + 1:p_ + 25])))
                        if len(pk) == 3:
                            break
                    fo.write(json.dumps({"idx": int(j + s0), "task_id": rows[j]["task_id"], "n": len(v), "peaks": pk}) + "\n")
        k13 = KEYS.index("span__all")
        rs, st_ = MX[:, 13, k13], np.array([r["probe"] for r in rows], dtype=np.float64)
        ok = np.isfinite(rs) & np.isfinite(st_)
        print(json.dumps({"scored": out, "n": n, "sec": round(time.time() - t),
                          "L13_vs_stored_corr": float(np.corrcoef(rs[ok], st_[ok])[0, 1]) if ok.sum() > 2 else None,
                          "L13_vs_stored_med_absdiff": float(np.median(np.abs(rs[ok] - st_[ok]))) if ok.any() else None}), flush=True)
    dist.destroy_process_group()
    os._exit(0)

# ---------------- held-out eval (single process) ----------------
if a.eval_out:
    assert WORLD == 1
    os.makedirs(a.eval_out, exist_ok=True)
    push_weights(0)  # this policy's merged attention (identity LoRA when not resumed), so the sampler serves this model
    ET = [json.loads(l) for l in open(a.eval_tasks)]
    for spec in a.eval_modes.split(","):
        a.mode, a.group = spec.split(":")[0], int(spec.split(":")[1])
        outp = f"{a.eval_out}/{a.mode}.jsonl"
        if os.path.exists(outp):
            continue
        t = time.time()
        with cf.ThreadPoolExecutor(128) as ex:
            S = [s for g in ex.map(sample_task, [(x, PORTS[i % len(PORTS)]) for i, x in enumerate(ET)]) for s in g]
        t_gen = time.time() - t
        with cf.ProcessPoolExecutor(32) as px:
            res = list(px.map(run_tests, [s["task"] for s in S], [s["final"] for s in S]))
        model.eval(); t = time.time(); sc = {}
        lens = {i: len(s["p_ids"]) + len(s["c_ids"]) for i, s in enumerate(S)}
        with torch.no_grad():
            for mb in microbatches([i for i, s in enumerate(S) if s["c_ids"]], lens, a.mb_tokens):
                for i, v in zip(mb, probe_scores([S[i] for i in mb])):
                    sc[i] = v
        with open(outp + ".tmp", "w") as f:
            for i, (s, (ok, _)) in enumerate(zip(S, res)):
                f.write(json.dumps({"task_id": s["task"]["task_id"], "split": s["task"]["split"], "mode": a.mode,
                                    "pass": ok, "probe": sc.get(i, float("nan")), "finish": s["finish"], "n_tok": len(s["c_ids"]),
                                    "analysis": s["analysis"], "final": s["final"]}) + "\n")
        os.replace(outp + ".tmp", outp)
        imp = [ok for s, (ok, _) in zip(S, res) if s["task"]["split"] != "original"]
        print(json.dumps({"eval": a.eval_out, "mode": a.mode, "n": len(S), "t_gen": t_gen, "t_probe": time.time() - t,
                          "imp_pass": sum(imp) / max(1, len(imp)),
                          "imp_fire": float(np.mean([sc.get(i, -1e18) > 770537 for i, s in enumerate(S) if s["task"]["split"] != "original"]))}),
              flush=True)
    dist.destroy_process_group()
    os._exit(0)

# ---------------- main loop ----------------
log = open(f"{a.rundir}/log.jsonl", "a") if RANK == 0 else None
json.dump(vars(a), open(f"{a.rundir}/args.json", "w"), indent=1) if RANK == 0 else None
pend = {}
if RANK == 0:
    if a.resume and not a.no_reload:
        push_weights(start_step - 1)
    pend[start_step] = make_batch(start_step)
step_times = []
for step in range(start_step, a.steps):
    t0 = time.time()
    if a.stop_at is not None:  # rank 0 decides, everyone follows
        est = 1.15 * (max(step_times[-3:]) if step_times else 400.0)
        flag = [bool(RANK == 0 and t0 + est > a.stop_at)]
        dist.broadcast_object_list(flag, src=0)
        if flag[0]:
            if RANK == 0:
                save_state(f"{a.rundir}/ckpt_final_{step}", step)
                print(json.dumps({"stopped_before_step": step + 1, "time": time.time()}), flush=True)
                open(f"{a.rundir}/STOPPED", "w").write(str(step))
            break
    if RANK == 0:
        S, t_gen = pend.pop(step)
        th = None
        if step + 1 < a.steps:  # batch n+1 is generated under policy n-1 while we train on batch n
            th = threading.Thread(target=lambda n=step + 1: pend.__setitem__(n, make_batch(n)), daemon=True); th.start()
        payload = [{"p_ids": s["p_ids"], "c_ids": s["c_ids"], "vlp": s["vlp"]} for s in S]
    else:
        payload = None
    obj = [payload]; dist.broadcast_object_list(obj, src=0); payload = obj[0]
    t_wait = time.time() - t0  # broadcast only
    # balance tokens across ranks (longest first, greedy)
    order = sorted(range(len(payload)), key=lambda i: -len(payload[i]["c_ids"]))
    load, own = [0] * WORLD, {}
    for i in order:
        r = int(np.argmin(load)); own[i] = r; load[r] += len(payload[i]["p_ids"]) + len(payload[i]["c_ids"])
    mine = [i for i in range(len(payload)) if own[i] == RANK]

    # 2. probe scores (no grad, layers 0..PL)
    model.eval(); sc = {}
    t1 = time.time()
    lens = {i: len(payload[i]["p_ids"]) + len(payload[i]["c_ids"]) for i in range(len(payload))}
    with torch.no_grad():
        for mb in microbatches([i for i in mine if payload[i]["c_ids"]], lens, a.mb_tokens):
            for i, v in zip(mb, probe_scores([payload[i] for i in mb])):
                sc[i] = v
        for i in mine:
            sc.setdefault(i, float("nan"))
    g = [None] * WORLD; dist.all_gather_object(g, sc); score = {k: v for x in g for k, v in x.items()}
    t_probe = time.time() - t1

    # 3. rewards / advantages (rank 0)
    if RANK == 0:
        fires = [a.tau is not None and not np.isnan(score[i]) and score[i] > a.tau for i in range(len(S))]
        if a.penalty == "fire":
            pen = [a.lam * float(f) for f in fires]
        elif a.penalty == "linear":
            pen = [0.0 if np.isnan(score[i]) else a.lam * (score[i] - a.tau) / a.pen_scale for i in range(len(S))]
        else:
            pen = [0.0] * len(S)
        R = np.array([float(s["pass"]) - p_ for s, p_ in zip(S, pen)]).reshape(-1, a.group)
        if a.penalty == "filter":
            keep = ~np.array(fires, dtype=bool).reshape(-1, a.group); A = np.zeros_like(R)
            for gi in range(len(R)):
                k = keep[gi]
                if k.sum() >= 2 and R[gi, k].std() > 1e-8:
                    A[gi, k] = (R[gi, k] - R[gi, k].mean()) / (R[gi, k].std() + 1e-6)
            A = A.reshape(-1)
        else:
            A = ((R - R.mean(1, keepdims=True)) / (R.std(1, keepdims=True) + 1e-6)).reshape(-1)
            A[np.repeat(R.std(1) < 1e-8, a.group)] = 0.0
        adv = A.tolist()
    else:
        adv = None
    obj = [adv]; dist.broadcast_object_list(obj, src=0); adv = obj[0]

    # 4. policy gradient (single pass: w uses the detached current log-probs)
    model.train(); t2 = time.time()
    ntok = sum(len(payload[i]["c_ids"]) for i in range(len(payload)) if adv[i] != 0)
    loss_sum, wstats = 0.0, []
    for mb in microbatches([i for i in mine if adv[i] != 0 and payload[i]["c_ids"]], lens, a.mb_tokens):
        loss = 0.0
        for i, lp in zip(mb, batch_logprobs([payload[i] for i in mb])):
            vlp = torch.tensor([x if x is not None else 0.0 for x in payload[i]["vlp"]], device=DEV)
            lr_ = lp.detach() - vlp
            w = torch.clamp(torch.exp(lr_), max=2.0)
            loss = loss - (w * adv[i] * lp).sum() / max(1, ntok)
            wstats.append([float(lr_.abs().mean()), float((torch.exp(lr_) > 2).float().mean()), len(payload[i]["c_ids"])])
        loss.backward(); loss_sum += float(loss)
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    flat = torch.cat([p.grad.reshape(-1) for p in params]); dist.all_reduce(flat)
    o = 0
    for p in params:
        n = p.numel(); p.grad.copy_(flat[o:o + n].view_as(p)); o += n
    gnorm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
    opt.step(); opt.zero_grad(set_to_none=True)
    t_train = time.time() - t2
    tl = torch.tensor([loss_sum], device=DEV); dist.all_reduce(tl)
    g = [None] * WORLD; dist.all_gather_object(g, wstats); wst = [x for y in g for x in y]

    # 5. push weights (after batch n+1 has finished sampling under the old policy), log
    if RANK == 0:
        t3 = time.time()
        if th is not None:
            th.join()  # batch n+1 finished under policy n-1; batch n+2 (next iteration) samples policy n
        t_join = time.time() - t3
        tw, tr = push_weights(step) if not a.no_reload else (0.0, 0.0)
        if (step + 1) % a.save_every == 0:
            save_state(f"{a.rundir}/ckpt_{step + 1}", step + 1)
        with open(f"{a.rundir}/samples/step_{step:04d}.jsonl", "w") as f:
            for i, s in enumerate(S):
                f.write(json.dumps({"task_id": s["task"]["task_id"], "split": s["task"]["split"], "pass": s["pass"],
                                    "probe": score[i], "fires": bool(fires[i]), "pen": pen[i], "adv": adv[i], "finish": s["finish"],
                                    "n_tok": len(s["c_ids"]), "analysis": s["analysis"], "final": s["final"]}) + "\n")
        by = {}
        for i, s in enumerate(S):
            d = by.setdefault(s["task"]["split"], {"n": 0, "pass": 0, "fire": 0, "len": 0, "trunc": 0, "probe": 0.0})
            d["n"] += 1; d["pass"] += s["pass"]; d["fire"] += fires[i]; d["len"] += len(s["c_ids"])
            d["trunc"] += s["finish"] == "length"; d["probe"] += 0.0 if np.isnan(score[i]) else score[i]
        wst = np.array(wst) if wst else np.zeros((1, 3))
        rec = {"step": step + 1, "time": time.time() - t0, "t_wait_gen": t_wait, "t_gen": t_gen, "t_probe": t_probe,
               "t_train": t_train, "t_join_gen": t_join, "t_write": tw, "t_reload": tr, "loss": float(tl), "gnorm": gnorm,
               "reward": float(R.mean()), "pen_mean": float(np.mean(pen)), "active_frac": float(np.mean(np.array(adv) != 0)),
               "is_absdiff": float(np.average(wst[:, 0], weights=wst[:, 2])), "is_clip": float(np.average(wst[:, 1], weights=wst[:, 2])),
               **{f"{sp}_{k}": v / d["n"] for sp, d in by.items() for k, v in d.items() if k != "n"}}
        log.write(json.dumps(rec) + "\n"); log.flush(); step_times.append(rec["time"])
        print(json.dumps(rec), flush=True)
    dist.barrier()
if RANK == 0 and not os.path.exists(f"{a.rundir}/STOPPED"):
    save_state(f"{a.rundir}/ckpt_final_{a.steps}", a.steps)
dist.destroy_process_group()
os._exit(0)  # do not wait on a daemon sampler thread mid-generation
