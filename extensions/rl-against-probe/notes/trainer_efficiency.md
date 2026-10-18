# Trainer efficiency: LoRA GRPO on gpt-oss-120b with at most 4×H200 (2026-10-14)

Scope: the trainer side, and the split between the sampler and the trainer. The sampler and rollout details are in
`notes/sampler_efficiency.md`, written by a parallel agent; this file does not repeat them.

No GPU work was run. The checks were CPU-only Python on athena02, in `/data/agastyas/rh_rl/agent_scratch/`. Other
agents also write to that directory. My files there are:

- `cpu_peft_test.py`
- `dq_experts.py`
- `make_rand_adapter.py`
- `rand_attn_r32/`
- `tiny_adapter_*/`
- `tokstats.py`
- `t1_serve.sh`
- `trainer_smoke.py`

Tags used below:

- **[src]**: read in the installed source. Paths are relative to `venv/lib/python3.12/site-packages/`.
- **[cpu]**: ran on the cluster CPU.
- **[pilot]**: computed from the finished `p1_gptoss_{low,medium}.jsonl`.
- **[inf]**: inference or estimate. Needs a GPU test.

---

## 0. Recommendation (TL;DR)

| Decision | Recommendation |
|---|---|
| GPU layout | **Fixed split 2 + 2, one-step async** (as in rl_train.py). GPUs A,B run vLLM; GPUs C,D run 2 data-parallel trainer ranks. Not colocation (§3). |
| Frozen expert weights | **Keep MXFP4 resident in the trainer (~65 GB/GPU) and dequantise one layer at a time to bf16 inside the forward**, using `dq_experts.py`. Then the backward is plain `grouped_mm` autograd. This gives one full replica per GPU, so pure data parallelism. Do NOT use transformers' MXFP4 path, which has no backward (§1). Fallback: bf16 dequantised and split over 2 GPUs with `device_map`, which is about 2× slower. |
| Attention | `attn_implementation="flex_attention"` (sinks via LSE renorm). Eager is O(T²) memory. FA3 sink backward is reportedly wrong. |
| LoRA targets | **Attention only: `q_proj,k_proj,v_proj,o_proj`**, r=32 to 64 (24 to 48M params). Expert LoRA works end to end in the formats, but costs a lot (§2). |
| Framework | **Adapt rl_train.py.** Do not adopt veRL/TRL/Unsloth/prime-rl (§4). The probe-on-policy-activations reward and the custom MXFP4 trainer make every framework a fork anyway. |
| Free wins | Drop the redundant scoring pass. Compute the probe with a layer 0 to 13 truncated pass. Skip all-equal groups before any trainer compute in arm A. Pack sequences and share the prompt across each group of 8 with flex BlockMasks. Prefer a 12k token cap over 16k (§5). |
| Step time [inf] | **low effort ≈ 60–80 s/step (≈2 h per 100 steps)**. **medium effort ≈ 150–190 s/step (≈4.5–5.5 h per 100 steps)** per arm (§3.3). |

---

## 1. Can the experts stay MXFP4 during training?

### 1.1 transformers' MXFP4 path has no backward, and it would fail silently [src]

- `transformers/integrations/mxfp4.py:372-405` (`Mxfp4GptOssExperts.forward`) calls `triton_kernels_hub.matmul_ogs.matmul_ogs` twice. The kernels come from `get_kernel("kernels-community/gpt-oss-triton-kernels", version=1)` at `mxfp4.py:564-567`.
- In the cached hub build, `matmul_ogs` is a plain Python function that launches Triton kernels. The file is `/data/hf_cache/models--kernels-community--gpt-oss-triton-kernels/snapshots/642d2a…/build/torch-cuda/matmul_ogs.py:375`.
  - The output is `torch.empty(...)` (`matmul_ogs.py:340`).
  - The build has no `torch.autograd.Function` or `backward`; `grep` finds none.
- So the output has no `grad_fn`, and the MoE branch is *detached*. Routing (`triton_kernels_hub.routing.routing`, `mxfp4.py:476-491`) is detached as well.
- A LoRA-on-attention backward would still run, because the residual path carries gradient. The gradients would be silently wrong, since the MoE Jacobian is missing.
- `quantizers/quantizer_mxfp4.py:278-282` `is_trainable` returns False, with the warning "MXFP4 quantization don't support training, please consider dequantizing…".
- In *our* venv the `kernels` package is not installed. In that case `quantizer_mxfp4.py:129-136` silently sets `dequantize=True`. A plain `from_pretrained` therefore materialises about 234 GB of bf16 and OOMs on one GPU.

### 1.2 Fix: MXFP4-resident experts, dequantised just in time [src + cpu]

**What this relies on.**

- With no quantization config, `GptOssExperts` dispatches to `grouped_mm_experts_forward`:
  - The decorator is at `models/gpt_oss/modeling_gpt_oss.py:68`.
  - The default implementation is chosen at `modeling_utils.py:1889-1909` ("grouped_mm" unless unavailable).
  - The function is at `integrations/moe.py:388-497`.
- That function uses `torch.nn.functional.grouped_mm`, which is differentiable.
- `convert_moe_packed_tensors` (`mxfp4.py:316`) dequantises one layer's blocks and scales into exactly the `(E, H, 2I)` / `(E, I, H)` bf16 layout that `GptOssExperts` expects.
- Dequantisation is lossless: e2m1 values times a power of two are exact in bf16. So the trainer and vLLM hold *identical* weights.

**What was checked on CPU.** `agent_scratch/dq_experts.py`:

- A forward pre-hook sets `experts.gate_up_proj/down_proj` to freshly dequantised tensors. A post-hook deletes them.
- The MXFP4 blocks and scales live as buffers, about 1.69 GB per layer, so about 61 GB for 36 layers plus about 4.2 GB of bf16 non-expert weights.
- On a tiny GptOss with PEFT attention LoRA and non-reentrant grad checkpointing, compared with a bf16 reference: **logits identical (max diff 0.0), all 16 LoRA grads identical (rel diff 0.0), all nonzero**.
- Under non-reentrant checkpointing, the bf16 copy (6.4 GB per layer) exists only while one layer runs or back-propagates.
- `make_dq` + meta-device loading (`trainer_smoke.py`) passed a CPU check on a 2-layer meta build: `flex_attention` and `grouped_mm` were accepted, and the expert parameters could be deleted before `to_empty`.

**Memory per trainer GPU [inf].**

| Item | Size |
|---|---|
| MXFP4 experts | 61 GB |
| Non-expert bf16 | 4.2 GB |
| Transient dequant (with the current torch dequant: int32 index temporaries plus a transposed `.contiguous()` copy) | ~13 GB |
| Saved layer inputs at 16k tokens (16k × 2880 × 2 B × 36) | 3.4 GB |
| One layer's recompute at 16k tokens (expert-gathered activations 4T × 5760, fp32 combine, flex outputs) | ~5–6 GB |
| Chunked lm_head (4096 × 201k fp32) | 3.3 GB |
| LoRA + AdamW | <1 GB |
| **Total** | **≈ 90–95 GB at 16k tokens**; about 110 GB if 32k-token packed microbatches are used |

**Cost [inf].**

- `_convert_moe_packed_tensors` (`mxfp4.py:257-313`) does LUT indexing via int32 index tensors, then `ldexp`, then `transpose().contiguous()`. That is about 50 GB of HBM traffic per layer, so roughly 15–25 ms per layer.
- A microbatch needs 36 (fwd) + 36 (recompute) + 14 (probe pass) ≈ 86 dequantisations, so about 1.5–2 s.
- With 32k-token microbatches this adds roughly 15–25% to trainer time.
- A 20-line fused Triton dequant kernel (8 GB of traffic, about 2–3 ms) would make it negligible. That is optional.

### 1.3 Alternatives considered

- **bf16 dequantised, sharded over 2 GPUs** with `device_map` (117 GB per GPU, naive pipeline).
  - It works with stock transformers and PEFT: `from_pretrained(..., quantization_config=Mxfp4Config(dequantize=True), device_map=...)`.
  - Only one GPU computes at a time, so the 2 GPUs give about 1 GPU of throughput.
  - This is the **fallback** if `dq_experts` misbehaves on GPU.
  - FSDP2 over 2 GPUs leaves almost no activation headroom at 117 GB per GPU, and it needs ~0.7 TB of layer all-gathers per 8k-token microbatch. Not worth it.
- **FP8 experts:** `quantizer_finegrained_fp8.py:215` `is_trainable` is False, and it needs `kernels` (not installed). Rejected.
- **Unsloth** (docs: https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune/gpt-oss-reinforcement-learning).
  - Their "4-bit RL" uses bnb-4bit re-quantised experts with on-the-fly dequant. That is the same idea as `dq_experts`, but per expert.
  - Rollouts use their own rewritten HF inference, not vLLM, at a quoted ~21 tok/s (4-bit) and ~30 tok/s (bf16). That is hopeless for 256 × 4k tokens per step.
  - Their stated reason, that vLLM lacks LoRA for gpt-oss, is outdated: vLLM 0.30 has it (§2).
  - Rejected. One useful datapoint from them: FA3 "lacks backward for attention sinks → incorrect training losses", so use flex.
- **Attention backend** [src]:
  - `modeling_gpt_oss.py:386-388`: `_supports_sdpa=False`, flash/flex True.
  - Eager (`modeling_gpt_oss.py:235`) materialises 64 × T × T: about 74 GB at T=17k in fp32. Infeasible.
  - `integrations/flex_attention.py:260-370` applies sinks by renormalising with the returned LSE. This is differentiable on CUDA; it raises on CPU.
  - Flash requires `kernels-community/vllm-flash-attn3` (not installed, and reportedly no sink backward).
  - So use flex. Its backward is verified only by GPU test 2.

---

## 2. LoRA targets

### 2.1 Module names [cpu]

On a GptOss model in transformers 5.18:

- Attention: `self_attn.{q,k,v,o}_proj`, all `nn.Linear` with bias.
- `mlp.router`: `GptOssTopKRouter`. It is an `nn.Module` with its own `weight`, not an `nn.Linear`.
- Experts: `mlp.experts.{gate_up_proj (E,H,2I), down_proj (E,I,H)}`. These are 3-D Parameters (`is_transposed=True`), with an interleaved gate/up layout.

### 2.2 PEFT 0.21.2 supports 3-D expert LoRA via `target_parameters` [src + cpu]

- The implementation is `ParamWrapper` (`peft/tuners/lora/layer.py:2335`).
  - LoRA is per expert: `lora_A = Linear(in, r*E)` and `lora_B = Linear(r*E, out)`.
  - It is applied by parametrising the base weight with `baddbmm(W, lhs, rhs)` (`layer.py:2305-2323, 2572-2600`).
- Cost implications:
  - Every forward materialises a full merged copy of the layer's expert stack (6.4 GB).
  - Backward computes the full dW of the 3-D weight through `grouped_mm` (≈ +1× forward FLOPs for the experts), which then reduces to the LoRA factors.
  - It cannot sit on top of `dq_experts`, because it needs a real Parameter to parametrise.
  - Parameter count is large: r=8 gives about 0.53B params, r=16 about 1.06B.
- Saved key format, from the CPU test (`agent_scratch/cpu_peft_test.py`), with targets `["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]` in this order:
  - `…mlp.experts.base_layer.lora_A/B` = gate_up, with A of shape (r·E, H) and B of shape (2I, r·E).
  - `…mlp.experts.lora_A/B` = down_proj.

### 2.3 vLLM 0.30 accepts both formats [src]

**Model class.** `model_executor/models/gpt_oss.py`:

- `:1408-1412`: `SupportsLoRA`, `is_3d_moe_weight=True`, `packed_modules_mapping={"qkv_proj":[q,k,v]}`.
- `:370-373`: LoRA only disables sequence-parallel MoE.

**Attention keys.** On CPU, `parse_fine_tuned_lora_name` with `GptOssForCausalLM.hf_to_vllm_mapper` maps PEFT keys to `model.layers.N.attn.{q,k,v,o}_proj`; q, k and v are packed into `qkv_proj`. `PEFTHelper` accepts the adapter (r=32, scaling 2.0) [cpu, on `rand_attn_r32`].

**Expert keys.**

- `lora/model_manager.py:874-905` handles exactly "PEFT file format where experts.base_layer is the gate_up_proj and experts is the down_proj". It reshapes (E, r, in) and (out, r, E), matching PEFT's reshapes at `layer.py:2529-2545`.
- It asserts both gate_up and down are present (`:888-891`), so you cannot adapt only one of them.
- The gate/up interleave is handled for GptOss only under TP>1 (`lora/layers/fused_moe.py:551-556`).

**Kernel cost of `--enable-lora`** [src]:

- `fused_moe/modular_kernel.py:591` rejects any experts class without `LoRAExpertsMixin` when `is_lora_enabled` (`fused_moe/layer.py:353`, which is `lora_config is not None`).
- The MXFP4 `TRITON` backend classes (`oracle/mxfp4.py:196-203`: `OAITritonMxfp4ExpertsMonolithic`, `OAITritonExperts`) lack the mixin. `MarlinExperts` (`experts/marlin_moe.py:648`) and `UnfusedOAITritonExperts` have it.
- So on H200 any LoRA-enabled server falls to MARLIN, **even with an attention-only adapter**.
- The pilot log shows `Using 'TRITON' Mxfp4 MoE backend` without LoRA. The throughput delta is unmeasured, and GPU test 1 measures it.
- The sibling note's alternative avoids this: no LoRA in vLLM, and push merged bf16 attention weights (1.9 GB) per step.

### 2.4 Recommendation

- **Attention-only, r=32 to 64.**
  - vLLM serves it, and it is cheap in vLLM.
  - It is trivially cheap in the trainer: no expert dW, and LoRA grads are 24M params, so all-reduce is free.
  - It composes with `dq_experts`.
  - It also enables the per-sample-gradient trick in §5 if ever needed (24M × 4 B = 96 MB per sample).
- Research caveat: "LoRA Without Regret" (Thinking Machines) found attention-only LoRA weaker than MLP LoRA for SFT. For RL, the per-episode information is tiny and low rank suffices, so I expect attention-only to be adequate.
- If capacity looks limiting, phase 2 is expert LoRA at r≤4. It should be implemented as factorised grouped matmuls `x·A_e·B_e`, not through PEFT's merged-weight `ParamWrapper`. Mind the DP slot hazard in sampler note §2.4.

---

## 3. Colocation vs split

### 3.1 Measured rollout cost [pilot]

The pilot ran on 4×H200 with vLLM DP4, no LoRA, the TRITON MoE backend, 120 tasks × n=8, with both efforts running concurrently. Tokens were re-tokenised with the gpt-oss tokenizer (`agent_scratch/tokstats.py`).

| effort | completions | total tok | mean | p50 | p90 | max | truncated@16k | prompt mean |
|---|---|---|---|---|---|---|---|---|
| low | 960 | 1.26M | 1312 | 1045 | 2677 | 5177 | 0 | 946 |
| medium | 960 | 4.07M | 4236 | 3253 | 8804 | 16384 | 13 | 946 |

- Wall time: low finished 187 s after the server came up, medium 337 s. That is 5.33M tokens in 337 s, about **15.8k tok/s on 4 GPUs** (≈ 4k/GPU including the tail). This agrees with the sibling note.
- Learning signal per group (fraction of groups whose 8 test outcomes are all equal, so zero advantage in arm A):

| effort | original | oneoff | conflicting |
|---|---|---|---|
| low | 0.55 | 0.68 | 0.68 |
| medium | 0.57 | 0.28 | 0.15 |

- Pass/hack rates on the impossible splits: low ≈ 0.16 / 0.07; medium ≈ 0.54 / 0.53. Medium gives far more mixed groups on the impossible splits, which is where the signal is.

### 3.2 Trainer cost model [inf]

**Per-token forward FLOPs.**

- Active parameters: attention 26.5M × 36 = 0.95B; experts 4 × 24.9M × 36 = 3.58B; lm_head 0.58B. Total ≈ 5.1B, so about 10.2 GFLOP per token.
- Attention scores: 18 full-attention layers at 16,384 × ctx FLOP per token (ctx ≈ 3–4k) add about 1 GFLOP. The sliding-window-128 layers are negligible.
- Total forward ≈ 11 GFLOP/token.

**Training multiple.** Frozen base weights mean the backward needs only dX: about 1× forward for the matmuls, about 2.5× for attention.

- Forward + recompute + backward ≈ **3.1× forward**.
- The probe pass over layers 0–13 is about **0.39× forward**.
- Total ≈ **3.5× forward ≈ 37–40 GFLOP/token**.
- rl_train.py as written has a full no-grad scoring pass, so it is 4.1×.

**Throughput.** H200 dense bf16 is 989 TFLOP/s. At 20% MFU, one GPU does about 200 TFLOP/s.

- MoE GEMM shapes are fine once packed: a 32k-token microbatch gives about 1k rows per expert.
- rl_train.py's batch=1 per sample at 1k tokens gives about 32 rows per expert, which is poor MoE efficiency.

**Tokens per step.**

- low: 336k completion + 242k prompt (30k with prefix sharing).
- medium: 1.08M + 242k.

### 3.3 Step time estimates [inf]

Assumptions: 256 completions per step; generation is tail-bound by the longest sample at about 100–150 tok/s single-stream; LoRA/Marlin is 1.0–1.5× slower than Triton; trainer at 20% MFU.

| layout | low effort | medium effort (cap 16k; cap 12k ≈ −20%) |
|---|---|---|
| **Split 2 vLLM + 2 trainer DP (dq_experts), async** | gen 45–70 s ‖ train ≈ 55 s naive, ≈ 25–35 s with §5 tricks → **≈ 60–80 s/step** | gen 120–180 s ‖ train ≈ 175 s naive, ≈ 90–120 s with tricks → **≈ 150–190 s/step** |
| Colocated, 4 GPUs time-shared (sleep/wake) | gen 35–50 s (tail-bound, extra GPUs barely help) + train 15–30 s + swap 10–15 s → 75–95 s | gen 110–150 s + train 50–90 s + swap 10–15 s → 190–250 s |
| Split 2 + 2 with bf16 `device_map` fallback (1 GPU effective) | gen ‖ train ≈ 110 s → ~110 s | ≈ 270 s (~7.5 h per 100 steps) |
| 3 vLLM + 1 trainer (dq_experts) | gen ~40–60 s ‖ train ~110 s | train ~350 s: trainer-bound |

**Why split wins.**

- Generation time is dominated by the longest sample in the batch. Giving it 4 GPUs instead of 2 shortens it only modestly.
- Meanwhile async overlap hides all of the trainer time in the split layout.
- Colocation also needs juggling memory on every step. The vLLM weights (65 GB) and the trainer weights (65 GB) cannot both be resident with a useful KV cache, so one side must offload 65 GB each step.

**Colocation facts [src].**

- `--enable-sleep-mode` forces the cumem allocator (`config/model.py:614-621`).
- `/sleep` and `/wake_up` only exist with `VLLM_SERVER_DEV_MODE=1` (`entrypoints/serve/dev/sleep/api_router.py`, router gated in `entrypoints/serve/__init__.py:67`).
- Sleep and wake are broadcast to all DP engines by `call_utility_async` (`v1/engine/core_client.py:1599-1607`).
- Level 1 offloads weights to pinned CPU memory (65 GB per engine; the host has 2 TB).
- Whether LoRA stacked buffers survive level 1 is unverified. Re-loading the adapter after wake is cheap anyway.

**Runtime adapter loading [src].** It needs `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1` and `--api-server-count 1`. The default with DP is api_server_count = DP size (`entrypoints/cli/serve.py:122-128`), and vLLM refuses to start in that combination (`:305-307`). `add_lora` is broadcast to all DP engines (`core_client.py:1599`).

---

## 4. Frameworks (be skeptical)

| framework | gpt-oss-120b + LoRA + vLLM rollouts? | blockers for us |
|---|---|---|
| **Unsloth / ART** | 120B "fits on 120GB" (their docs); 4-bit bnb or bf16 LoRA | Rollouts are HF generate, not vLLM, at ~21–30 tok/s quoted. Far too slow. |
| **TRL GRPOTrainer** | PEFT + vLLM server/colocate exist; no gpt-oss-120b RL example found (only `sft_gpt_oss`) | Search results say TRL supports only a pinned vLLM range, far older than 0.30. From memory, unverified for current TRL: with PEFT it merges the adapter and pushes *all* weights to vLLM, which would push bf16 experts into an MXFP4 engine. The trainer model would be 234 GB of bf16, needing ZeRO-3 or FSDP. Its rewards see only text, not the policy's layer-13 residual. |
| **veRL** | LoRA RL with FSDP/Megatron + vLLM yes; README has no gpt-oss in the model list; vLLM ≥0.8.2 | Ray stack. A torch 2.13/cu130 + vLLM 0.30 combination is untested. The probe reward needs hacking the actor's `compute_log_prob` to return activations. The bf16 actor needs FSDP. |
| **prime-rl** | README lists `gpt_oss` (example `unsloth/gpt-oss-20b-BF16`), MoE, LoRA, FSDP2 + vLLM | Own `uv.lock`, so a new env. Expects a bf16 checkpoint, so the 120b trainer needs ≥2–3 GPUs of FSDP, leaving 1–2 for inference. No hidden-state reward hook. No evidence of 120b on ≤4 GPUs. |
| SkyRL / OpenRLHF | no gpt-oss-120b evidence found | n/a |
| Tinker (hosted) | supports gpt-oss-120b LoRA RL | No activation access, so arm B (probe on the residual) is impossible. |

**Verdict.** Adapting rl_train.py is faster to a correct run. The required changes are well-defined; see §6. Every framework would need:

- the custom MXFP4-resident trainer, or 3+ GPUs of FSDP;
- a fork for the probe reward;
- version reconciliation with torch 2.13/vLLM 0.30.

---

## 5. Other large wins (ranked by payoff/effort)

1. **Delete the scoring pass. Saves about 25% of trainer FLOPs.**
   - rl_train.py computes `old` logprobs in a no-grad pass (`rl_train.py:199-216`), then recomputes logprobs on the *same weights* in the grad pass (`:243`).
   - So `old == lp.detach()` exactly. Use `w = clamp(exp(lp.detach() - vlp), max=2)` inside the grad pass.
   - The probe only needs layers 0–13. Run a truncated no-grad pass with a forward hook on `layers[13]` that raises to stop the forward (`trainer_smoke.py:probe_pass`). Cost is 14/36 of a forward.
   - In arm A the probe is only logged, so read it from a hook in the grad pass, which is free.
2. **Skip all-equal groups before *any* trainer compute in arm A.**
   - rl_train.py already zeroes them (`:228`), but still runs the scoring pass on them.
   - Measured fractions [pilot]: about 64% of groups at low effort and about 33% at medium.
   - Arm B must run the probe pass first, because the probe can break ties. The grad pass then only covers non-zero-advantage groups.
3. **Pack sequences and share the prefix within a group.**
   - The prompt (946 tokens) is about 42% of tokens at low effort and about 18% at medium.
   - Pack one prompt plus 8 completions per flex BlockMask document. Each completion attends to the prompt and to itself, with position_ids restarting at `prompt_len`.
   - `GptOssModel.forward` accepts a precomputed `{"full_attention": BlockMask, "sliding_attention": BlockMask}` dict (`modeling_gpt_oss.py:466-476`).
   - This also gives 32k-token microbatches, which helps MoE GEMM efficiency. Low effort saves about 35%, medium about 15%, plus the efficiency gain.
4. **Effort and length.**
   - Low effort is about 3.2× fewer tokens and a 5k tail versus 16k, so steps are about 2.5× faster. But it has about 2× less arm-A signal on the impossible splits and much lower hack rates.
   - For medium, `max_tokens` 12k cuts the tail at about 1–2% truncation cost.
5. (Only if probe-based arm B wants a single pass.) **Per-sample LoRA gradients.**
   - With attention-only LoRA (24M params ≈ 96 MB fp32 per sample), run one fwd+bwd per sample to get g_i = ∇Σ_t w_t·lp_t, stash it, and combine with the advantages at the end: Σ A_i g_i / N.
   - This removes the probe pass, but costs about 25 GB for 256 samples. It is not needed given item 1.

---

## 6. rl_train.py changes for gpt-oss (not done; listed for whoever writes the trainer)

- **Model and weights.**
  - `SNAP` → gpt-oss-120b.
  - Load via the meta + `make_dq` path in `agent_scratch/trainer_smoke.py`.
  - `attn_implementation="flex_attention"`.
  - `targets=q_proj,k_proj,v_proj,o_proj`.
- **Probe.** `--probe` layer 13 with the gpt-oss DoM file. Truncated probe pass (§5.1). Delete the scoring pass.
- **Stops.** `stop_token_ids` → `[200002, 200012]`.
- **Paragraph splitter.** `paragraphs()` keys on `</think>` (`:163`). For gpt-oss, split analysis and final on harmony tokens. See sampler note §1.3 and `rh_rl/cluster/gptoss_rollout.py`.
- **Batching.** Pack plus prefix-share (§5.3). Pack per rank into roughly equal-token microbatches, not round-robin by index (`mine = i % WORLD` is badly balanced with 1k–16k lengths).
- **Sampler.** Use one LoRA vLLM server with `--api-server-count 1`, or the merged-weight push from the sampler note §3.2. Do not use two TP1 servers on the trainer's GPUs.
- **Norm dtype.** RMSNorm weights load as bf16 in the meta path. `from_pretrained` keeps them fp32 (`_keep_in_fp32_modules`, `modeling_gpt_oss.py:397`). Cast `input_layernorm`, `post_attention_layernorm` and `norm` to fp32 to match.

---

## 7. Top risks

1. **Trainer/sampler logprob mismatch.**
   - vLLM uses Marlin MXFP4 with bf16 activations, plus the LoRA kernels. The trainer uses dequantised bf16 `grouped_mm` with flex attention.
   - Routing flips under small numeric differences can give large per-token gaps. The truncated IS weight (`:245`) only clips at 2.
   - A wrong adapter-key mapping would look like "large mismatch only when the adapter is on". This is silent policy corruption. GPU test 3.
2. **Flex attention on GPU.** Sink-renorm backward correctness, torch.compile recompiles across variable lengths and BlockMasks, and peak memory at 16–32k tokens. GPU test 2.
3. **Throughput assumptions.**
   - The Marlin slowdown under `--enable-lora` is unknown (test 1).
   - The trainer MFU of 20% is a guess (test 2). At 10% the medium step becomes trainer-bound (~260 s).
   - The dequant overhead depends on the torch LUT kernel.
   - Secondary risk: attention-only LoRA capacity (a research risk, §2.4).

---

## 8. GPU tests to run first (each ≤ 15 min; only on GPUs agreed with the user)

### Test 1: vLLM LoRA serving, MoE backend, adapter load, sleep/wake (2 GPUs, ~12 min)

```bash
cd /data/agastyas/rh_rl
GPUS=4,5 DP=2 PORT=8410 nohup bash agent_scratch/t1_serve.sh > agent_scratch/t1_serve.log 2>&1 &
# after "Application startup complete":
grep -E "Mxfp4 MoE backend|Using .*Experts" agent_scratch/t1_serve.log        # expect MARLIN / MarlinExperts
K=$(cat /data/agastyas/rh_probe/api_key)
time venv/bin/python sample.py pilot_tasks.jsonl agent_scratch/t1_base.jsonl --chat gpt-oss-120b --port 8410 --effort low --n 8 --limit 32 --workers 32
time curl -s -X POST -H "Authorization: Bearer $K" -H 'Content-Type: application/json' localhost:8410/v1/load_lora_adapter \
     -d '{"lora_name":"r32","lora_path":"/data/agastyas/rh_rl/agent_scratch/rand_attn_r32"}'
time venv/bin/python sample.py pilot_tasks.jsonl agent_scratch/t1_r32.jsonl --chat r32 --port 8410 --effort low --n 8 --limit 32 --workers 32
time curl -s -X POST -H "Authorization: Bearer $K" 'localhost:8410/sleep?level=1'; nvidia-smi --query-gpu=index,memory.used --format=csv
time curl -s -X POST -H "Authorization: Bearer $K" localhost:8410/wake_up
```

- Compare wall time and tokens (`agent_scratch/tokstats.py`) against the pilot's TRITON numbers.
- Pass condition: Marlin with LoRA is ≥ 0.7× pilot throughput per GPU. Otherwise use the merged-weight push (sampler note §3.2).
- `rand_attn_r32` is a valid PEFT-format random adapter (B ~ N(0, 2e-3)), already created and name-checked on CPU.

### Test 2: trainer smoke (1 GPU, ~10–15 min incl. ~2–4 min load)

```bash
cd /data/agastyas/rh_rl/agent_scratch
CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 /data/agastyas/rh_rl/venv/bin/python -I trainer_smoke.py 2>&1 | tee t2.log
```

`trainer_smoke.py` is untested on GPU; it compiles, and its meta construction was checked on CPU. It reports:

- resident memory (expect ≈ 66 GB);
- dequant ms per layer;
- probe-pass and fwd+bwd time with peak memory at T = 4k/8k/16k/32k, plus an MFU estimate;
- flex-vs-eager LoRA grad agreement at T=1024 (expect rel diff ≲ 1e-2);
- zero-grad LoRA tensors (expect 0/288).

Pass conditions:

- peak memory < 125 GB at 16k;
- MFU ≥ 15%;
- grads agree.

If it fails: use the bf16 `device_map` fallback across 2 GPUs. Same script, but with `from_pretrained(..., quantization_config=Mxfp4Config(dequantize=True), device_map="auto", max_memory={0:"125GiB",1:"125GiB"})`.

### Test 3: logprob consistency, sampler vs trainer (1 vLLM GPU + 1 trainer GPU, ~10 min; reuse test 1 and test 2 setups)

1. Generate 16 samples via `/v1/completions` with `prompt=<harmony ids>, logprobs=0, return_token_ids=true`, once with `model="gpt-oss-120b"` and once with `model="r32"`. See `rh_rl/cluster/gptoss_rollout.py` from the sibling agent.
2. In the trainer, compute `seq_logprobs` with the adapter off, and with `PeftModel.from_pretrained(model, ".../rand_attn_r32")`.
3. Report per-token |Δlp| (mean, p99, max) and the fraction with exp(Δ) > 2, for both cases.

Pass conditions:

- adapter-on mismatch ≈ adapter-off mismatch (otherwise the key mapping or scaling is wrong);
- mean |Δ| ≲ 0.02;
- fewer than 1% of tokens clipped.

---

## 9. Verified vs inferred

**Verified in source or on CPU:**

- The MXFP4 path has no autograd, and dequantises silently because `kernels` is missing.
- `grouped_mm` is the default and is differentiable.
- The `dq_experts` approach is exact on a tiny model, including with grad checkpointing.
- PEFT 3-D LoRA key format.
- vLLM accepts PEFT attention and 3-D expert adapters.
- `--enable-lora` forces a LoRA-capable MoE kernel (Marlin) on H200.
- Runtime LoRA needs api_server_count 1.
- Sleep and LoRA calls broadcast to DP engines.
- Flex attention carries sinks.
- GptOss accepts mask dicts.
- rl_train.py's scoring pass is redundant.
- Pilot token statistics and all-equal fractions.

**Inferred (needs GPU):**

- All throughput and MFU numbers, dequant cost, Marlin slowdown, and step times.
- Flex backward correctness and speed.
- Memory peaks.
- Sampler/trainer logprob agreement.
- LoRA survival across sleep.
- TRL's merge-and-push behaviour, which I recall but did not check in source.
