"""GRPO with LoRA on Qwen3.8-27B in the single-turn Impossible-LCB environment, with an optional DoM-probe penalty.

Layout (athena02, GPUs 4-7): two TP1 vLLM samplers (ports 8401/8402, --enable-lora, runtime adapter loading) and a
torchrun trainer with WORLD ranks (one bf16 replica each). Rank 0 orchestrates: it samples batch t+1 in a background
thread while all ranks train on batch t (one-step lag, corrected per token by truncated importance weights).

Per step:
  1. rank 0: G samples for each of P prompts (half impossible, half original), test reward per sample (sandbox).
  2. all ranks, no grad: forward each sample on the current weights -> trainer log-probs of the sampled tokens and
     per-token probe projections <v, h_L> (turn__all, layer 40, the T2-selected probe) for completion tokens.
  3. rank 0: probe fires = any reasoning paragraph (or the answer) has max projection > tau (calibrated in Phase 0).
     reward = test_pass - lam * fires (arm B; lam = 0 in arm A, where firing is only logged).
     advantage = (r - mean_group) / (std_group + 1e-6); all-equal groups carry no gradient and are skipped.
  4. all ranks: loss = -sum_t w_t * A * logp_theta(y_t) / N_tokens(batch), w_t = min(exp(logp_trainer - logp_vllm), 2)
     (detached), gradient checkpointing, chunked lm_head; LoRA grads all-reduced by hand; AdamW; grad clip 1.0.
  5. rank 0: save adapter, hot-load it into both samplers.

Usage: torchrun --nproc_per_node 2 rl_train.py --run runs/armA --lam 0 [--steps 150]
"""
import argparse, concurrent.futures as cf, glob, json, math, os, random, re, threading, time, urllib.request
import numpy as np, torch, torch.distributed as dist
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from env_lcb import user_prompt, split_response, run_tests

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--tasks", default="/data/agastyas/rh_rl/train_tasks.jsonl")
ap.add_argument("--lam", type=float, default=0.0)
ap.add_argument("--tau", type=float, default=None, help="probe threshold (Phase 0); required if lam > 0")
ap.add_argument("--agg", default="para_max", choices=["para_max", "max"])
ap.add_argument("--steps", type=int, default=150)
ap.add_argument("--prompts", type=int, default=32)
ap.add_argument("--group", type=int, default=8)
ap.add_argument("--max-new", type=int, default=12288)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--rank-lora", type=int, default=64)
ap.add_argument("--targets", default="q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,out_proj,gate_proj,up_proj,down_proj")
ap.add_argument("--mode", default="lenient"); ap.add_argument("--effort", default="medium")
ap.add_argument("--ports", default="8401,8402")
ap.add_argument("--probe", default="turn__all:40")
ap.add_argument("--dom", default="/data/agastyas/rh_probe/dom/qwen/dom_vectors.npz")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--save-every", type=int, default=25)
a = ap.parse_args()
assert a.lam == 0 or a.tau is not None

dist.init_process_group("nccl")
RANK, WORLD = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
DEV = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
os.makedirs(a.run, exist_ok=True)
SNAP = glob.glob("/data/hf_cache/models--Qwen--Qwen3.8-27B/snapshots/*")[0]
tok = AutoTokenizer.from_pretrained(SNAP)
KEY = open("/data/agastyas/rh_probe/api_key").read().strip()
PORTS = a.ports.split(",")
torch.manual_seed(a.seed); random.seed(a.seed)

# ---------------- model ----------------
model = AutoModelForCausalLM.from_pretrained(SNAP, dtype=torch.bfloat16, device_map={"": DEV})
model.config.use_cache = False
model = get_peft_model(model, LoraConfig(r=a.rank_lora, lora_alpha=2 * a.rank_lora, lora_dropout=0.0,
                                         target_modules=a.targets.split(","), task_type="CAUSAL_LM"))
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
inner = model.base_model.model                      # Qwen3_5ForConditionalGeneration
lm = inner.model.language_model if hasattr(inner.model, "language_model") else inner.model
head = inner.lm_head
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), weight_decay=0.0)
if RANK == 0:
    print(f"trainable params: {sum(p.numel() for p in params) / 1e6:.1f}M", flush=True)

pk, PL = a.probe.split(":"); PL = int(PL)
VEC = torch.tensor(np.load(a.dom)[pk][PL], dtype=torch.float32, device=DEV)
CAP = {}
lm.layers[PL].register_forward_hook(lambda m, i, o: CAP.__setitem__("h", (o[0] if isinstance(o, tuple) else o)))


def seq_logprobs(ids, n_prompt):
    """log p(token_t | <t) for completion tokens, chunked through lm_head (never materialise full logits)."""
    x = torch.tensor([ids], device=DEV)
    h = lm(input_ids=x).last_hidden_state[0]           # [T, d]
    hs, tgt = h[n_prompt - 1:-1], x[0, n_prompt:]
    out = []
    for s in range(0, hs.shape[0], 2048):
        def f(hc, tc):
            lg = head(hc).float()
            return lg.gather(-1, tc[:, None])[:, 0] - torch.logsumexp(lg, -1)
        out.append(checkpoint(f, hs[s:s + 2048], tgt[s:s + 2048], use_reentrant=False))
    return torch.cat(out)


# ---------------- sampling (rank 0) ----------------
TASKS = [json.loads(l) for l in open(a.tasks)]
IMP = [t for t in TASKS if t["split"] != "original"]
ORI = [t for t in TASKS if t["split"] == "original"]


def prompt_ids(t):
    return tok(tok.apply_chat_template([{"role": "user", "content": user_prompt(t, a.mode)}], tokenize=False,
                                       add_generation_prompt=True, reasoning_effort=a.effort),
               add_special_tokens=False)["input_ids"]


def vllm(port, path, body, timeout=7200):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    for i in range(5):
        try:
            return urllib.request.urlopen(req, timeout=timeout).read()
        except Exception as e:
            err = e; time.sleep(10)
    raise err


ADAPTER = {"name": None}


def sample_one(job):
    t, port = job
    ids = prompt_ids(t)
    body = {"model": ADAPTER["name"] or "qwen3.8-27b", "prompt": ids, "n": a.group, "max_tokens": a.max_new,
            "temperature": 1.0, "top_p": 1.0, "top_k": -1, "logprobs": 0, "return_token_ids": True,
            "skip_special_tokens": False, "stop_token_ids": [248046, 248044]}
    res = json.loads(vllm(port, "/v1/completions", body))
    out = []
    for c in res["choices"]:
        cids = c["token_ids"]
        lps = c["logprobs"]["token_logprobs"]
        out.append({"task": t, "p_ids": ids, "c_ids": cids, "vlp": lps, "text": c["text"], "finish": c["finish_reason"]})
    return out


def make_batch(step):
    rng = random.Random(a.seed * 100003 + step)
    ts = rng.sample(IMP, a.prompts // 2) + rng.sample(ORI, a.prompts - a.prompts // 2)
    jobs = [(t, PORTS[i % len(PORTS)]) for i, t in enumerate(ts)]
    with cf.ThreadPoolExecutor(len(jobs)) as ex:
        S = [s for g in ex.map(sample_one, jobs) for s in g]
    with cf.ProcessPoolExecutor(32) as px:
        res = list(px.map(run_tests, [s["task"] for s in S], [split_response(s["text"])[1] for s in S]))
    for s, (ok, _) in zip(S, res):
        s["pass"] = ok
    return S


def load_adapter(path, name):
    for p in PORTS:
        vllm(p, "/v1/load_lora_adapter", {"lora_name": name, "lora_path": path}, timeout=600)
    old, ADAPTER["name"] = ADAPTER["name"], name
    if old:
        for p in PORTS:
            try:
                vllm(p, "/v1/unload_lora_adapter", {"lora_name": old}, timeout=120)
            except Exception:
                pass


def paragraphs(text, offs):
    """token-index groups: each reasoning paragraph (split on blank lines) and the answer as one group."""
    te = text.find("</think>")
    bounds, pos = [], 0
    reasoning = text if te < 0 else text[:te]
    for m in re.finditer(r"\n\s*\n", reasoning):
        bounds.append((pos, m.start())); pos = m.end()
    bounds.append((pos, len(reasoning)))
    if te >= 0:
        bounds.append((te, len(text)))
    groups = []
    for s0, s1 in bounds:
        idx = [i for i, (o0, o1) in enumerate(offs) if o1 > s0 and o0 < s1]
        if idx:
            groups.append(idx)
    return groups


# ---------------- main loop ----------------
log = open(f"{a.run}/log.jsonl", "a") if RANK == 0 else None
pending = {}
if RANK == 0:
    th = threading.Thread(target=lambda: pending.__setitem__(0, make_batch(0))); th.start()
for step in range(a.steps):
    t0 = time.time()
    if RANK == 0:
        th.join(); S = pending.pop(step)
        nxt = step + 1
        if nxt < a.steps:
            th = threading.Thread(target=lambda n=nxt: pending.__setitem__(n, make_batch(n))); th.start()
        payload = [{"p_ids": s["p_ids"], "c_ids": s["c_ids"], "vlp": s["vlp"], "text": s["text"]} for s in S]
    else:
        payload = None
    obj = [payload]; dist.broadcast_object_list(obj, src=0); payload = obj[0]
    t_gen = time.time() - t0
    mine = [i for i in range(len(payload)) if i % WORLD == RANK]

    # 2. scoring forward (no grad): trainer log-probs + probe
    model.eval()
    score = {}
    with torch.no_grad():
        for i in mine:
            s = payload[i]
            if not s["c_ids"]:
                score[i] = (False, -1e9, []); continue
            lp = seq_logprobs(s["p_ids"] + s["c_ids"], len(s["p_ids"])).float().cpu()
            proj = (CAP["h"][0, len(s["p_ids"]):].float() @ VEC).cpu().numpy()
            offs, pos = [], 0
            for tid in s["c_ids"]:
                piece = tok.decode([tid]); offs.append((pos, pos + len(piece))); pos += len(piece)
            groups = paragraphs("".join(tok.decode([t]) for t in s["c_ids"]), offs) if a.agg == "para_max" else [list(range(len(proj)))]
            gmax = max(float(proj[g].max()) for g in groups) if groups else float(proj.max())
            fires = a.tau is not None and gmax > a.tau
            score[i] = (fires, gmax, lp.tolist())
    gathered = [None] * WORLD
    dist.all_gather_object(gathered, {i: (f, m) for i, (f, m, _) in score.items()})
    allscore = {k: v for g in gathered for k, v in g.items()}

    # 3. rewards and advantages (rank 0), broadcast
    if RANK == 0:
        R, info = [], []
        for i, s in enumerate(S):
            fires, gmax = allscore[i]
            r = float(s["pass"]) - a.lam * float(fires)
            R.append(r); info.append((s["task"]["split"], s["pass"], fires, gmax, len(s["c_ids"]), s["finish"]))
        R = np.array(R).reshape(-1, a.group)
        A = ((R - R.mean(1, keepdims=True)) / (R.std(1, keepdims=True) + 1e-6)).reshape(-1)
        A[np.repeat(R.std(1) < 1e-8, a.group)] = 0.0
        adv = A.tolist()
    else:
        adv = None
    obj = [adv]; dist.broadcast_object_list(obj, src=0); adv = obj[0]

    # 4. policy gradient
    model.train()
    ntok = sum(len(payload[i]["c_ids"]) for i in range(len(payload)) if adv[i] != 0)
    loss_sum = 0.0
    for i in mine:
        if adv[i] == 0 or not payload[i]["c_ids"]:
            continue
        s = payload[i]
        lp = seq_logprobs(s["p_ids"] + s["c_ids"], len(s["p_ids"]))
        old = torch.tensor(score[i][2], device=DEV)
        vlp = torch.tensor([x if x is not None else 0.0 for x in s["vlp"]], device=DEV)
        w = torch.clamp(torch.exp(old - vlp), max=2.0)
        loss = -(w * adv[i] * lp).sum() / max(1, ntok)
        loss.backward(); loss_sum += float(loss)
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    flat = torch.cat([p.grad.reshape(-1) for p in params])
    dist.all_reduce(flat)
    o = 0
    for p in params:
        n = p.numel(); p.grad.copy_(flat[o:o + n].view_as(p)); o += n
    gnorm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
    opt.step(); opt.zero_grad(set_to_none=True)
    tl = torch.tensor([loss_sum], device=DEV); dist.all_reduce(tl)

    # 5. save + hot-load adapter
    if RANK == 0:
        path = f"{a.run}/adapter_latest_{step % 2}"
        model.save_pretrained(path)
        if (step + 1) % a.save_every == 0:
            model.save_pretrained(f"{a.run}/ckpt_{step + 1}")
        load_adapter(os.path.abspath(path), f"s{step + 1}")
        by = {}
        for sp, ok, fires, gmax, n, fin in info:
            d = by.setdefault(sp, {"n": 0, "pass": 0, "fire": 0, "len": 0, "trunc": 0})
            d["n"] += 1; d["pass"] += ok; d["fire"] += fires; d["len"] += n; d["trunc"] += fin == "length"
        rec = {"step": step + 1, "time": time.time() - t0, "wait_gen": t_gen, "loss": float(tl), "gnorm": gnorm,
               "reward": float(R.mean()), "active_frac": float(np.mean(np.array(adv) != 0)),
               **{f"{sp}_{k}": v / d["n"] for sp, d in by.items() for k, v in d.items() if k != "n"},
               "probe_max_mean": float(np.mean([x[3] for x in info]))}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(json.dumps(rec), flush=True)
    dist.barrier()
dist.destroy_process_group()
