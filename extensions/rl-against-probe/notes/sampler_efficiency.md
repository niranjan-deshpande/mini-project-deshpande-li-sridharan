# Sampler efficiency for LoRA GRPO on gpt-oss-120b, vLLM 0.30.0 (2026-10-14)

Scope: rollout and sampler side only. No GPU jobs were launched. Two tiny live requests were sent to the running
server (n=2, max_tokens=64 each). All other checks ran on CPU under `/data/agastyas/rh_rl/agent_scratch/`.

Source paths below are relative to `/data/agastyas/rh_rl/venv/lib/python3.12/site-packages/vllm/`.

Tags: **[src]** = read in source, **[live]** = observed on the running server, **[cpu]** = CPU script on the
cluster, **[inf]** = inferred, needs a GPU test.

---

## 0. TL;DR

1. **Rollout path.** Render the prompt ids ourselves with openai_harmony. Call `/v1/completions` with
   `prompt=ids, n=8, logprobs=0, return_token_ids=true, skip_special_tokens=false, stop_token_ids=[200002,200012]`.
   Split analysis and final text from the token ids with the harmony `StreamableParser`. Verified end to end
   [live]. Code: `rh_rl/cluster/gptoss_rollout.py`.
2. **LoRA target set.** Use attention only: PEFT `q_proj,k_proj,v_proj,o_proj`, which vLLM serves as
   `qkv_proj` and `o_proj`. Serve with `--lora-target-modules qkv_proj o_proj`.
   - Expert LoRA is supported, through PEFT `target_parameters` and the 3D path. At r=64 it costs about 8.5 GB per
     adapter slot.
   - With the current DP>1 layout it also has a correctness hazard (§2.4).
3. **Slowdown risk.** Any `--enable-lora` switches the MXFP4 MoE kernel away from the fast fused Triton kernel
   that is used today. The swap happens even if no expert is adapted [src]. This is the largest LoRA throughput
   risk.
4. **Adapter update.** Two options:
   - **Simplest:** `/v1/load_lora_adapter` with a new name each step. It is broadcast to all DP ranks [src]. It
     needs `--api-server-count 1` when DP>1, otherwise vLLM refuses to start [src]. Latency is about 1 s for an
     attention-only adapter [inf].
   - **Fastest:** run with no LoRA in vLLM at all. Merge the attention LoRA in the trainer, then push about 1.9 GB of
     bf16 attention weights each step through `/collective_rpc reload_weights` or the RL weight-transfer endpoints.
     This keeps the fast MoE kernel and has zero LoRA overhead [src for the plumbing, inf for latency].
5. **Throughput.** Measured on the pilot: 15.8k generated tok/s on 4×H200 (DP4), about 4.0k tok/s per GPU averaged
   over the whole run including the tail [live metrics]. That is in line with public H200 numbers (§4).
   - A step of 32×8 medium-effort samples is about 1.1M tokens: roughly 1.5–2.5 min on 4 GPUs, dominated by the
     tail.
   - The same step on the current DP1 server (1 GPU) is roughly 4–6 min.

---

## 1. Exact-token rollouts

### 1.1 Prompt rendering: harmony and the HF template match token for token [cpu]

`agent_scratch/check_render.py` compared two renderings on a real pilot task, at efforts low, medium and high:

- vLLM's chat-path rendering: `build_harmony_preamble(reasoning_effort=…)` + user message + `render_for_completion`
  (`entrypoints/openai/parser/harmony_utils.py:320-358, 457-465`).
- `tok.apply_chat_template(..., add_generation_prompt=True, reasoning_effort=effort)`, then
  `tok(..., add_special_tokens=False)`.

Result: **identical, 2099 tokens each**.

- The prompt ends `…<|end|><|start|>assistant`, ids `200007, 200006, 173781`. The model then generates
  `<|channel|>analysis<|message|>…`.
- Do not pass `model_identity=None` to the HF template; it crashes the jinja template.
- **Pin the date.** Both paths insert `Current date: <today>` into the system message (`harmony_utils.py:139-141`).
  The HF template calls `strftime_now`. Unpinned, prompts change at midnight, which breaks prefix caching and
  reproducibility.
  - `gptoss_rollout.prompt_ids` builds the `SystemContent` directly with a fixed `with_conversation_start_date`.
  - It still reproduces vLLM's chat-path ids exactly [cpu, checked in the script's `__main__`].
- Special ids [cpu], with HF tokenizer and harmony agreeing:

  | token | id |
  |---|---|
  | `<\|start\|>` | 200006 |
  | `<\|end\|>` | 200007 |
  | `<\|message\|>` | 200008 |
  | `<\|channel\|>` | 200005 |
  | `<\|return\|>` | 200002 |
  | `<\|call\|>` | 200012 |
  | `<\|constrain\|>` | 200003 |
  | `<\|endoftext\|>` | 199999 |

- Harmony's `stop_tokens_for_assistant_actions()` is `[200012, 200002]`. **`<|end|>` (200007) must not be a stop
  token**: the model emits `analysis…<|end|><|start|>assistant<|channel|>final…`.

### 1.2 Request and response [src + live]

- `entrypoints/openai/completion/protocol.py`:
  - `return_token_ids` is at :169.
  - Server-default stop ids are merged into the request's ids (:347-356). The defaults come from
    generation_config `eos_token_id = [200002, 199999, 200012]`, so `<|return|>` and `<|call|>` stop even if the
    request omits them.
- `config/model.py:259`: `logprobs_mode` defaults to `raw_logprobs`, i.e. before temperature and top-p. That equals
  the sampling distribution only at T=1 and top_p=1, which we use.

Live result (`agent_scratch/live_rollout.py` and `gptoss_rollout.py`, run against :8400):

- `finish_reason="stop"`. `choice.token_ids` **includes the final `<|return|>` (200002)**, and that token has a
  logprob. So `len(token_ids) == len(token_logprobs)` and there are no `None` entries.
- `choice.prompt_token_ids` is echoed per choice and equals the ids we sent.
- `text` with `skip_special_tokens=false` is
  `'<|channel|>analysis<|message|>Just answer 51.<|end|><|start|>assistant<|channel|>final<|message|>51'`. The
  `<|return|>` is not in the text, so `decode(token_ids) != text`. **Use ids, never text, for the trainer.**
- First completion ids are `[200005, 35644, 200008, …]`, i.e. `<|channel|>analysis<|message|>`.

### 1.3 Analysis vs final split [cpu]

Parse the token ids with `openai_harmony.StreamableParser` (`gptoss_rollout.split_channels`):

- A complete sample gives `(analysis, final)`.
- A sample truncated inside analysis gives `final=None`.
- A sample truncated inside final gives the partial final text.
  - **Recommendation:** use the answer only if `finish_reason=="stop"` and `c_ids[-1]==200002`. Truncated samples
    and `<|call|>` tool calls get no answer, which matches the Qwen rule "no `</think>` means no answer".
- `parse_messages_from_completion_tokens(..., strict=False)` also works for complete samples.
- For the probe's paragraph grouping, work in token space:
  - analysis = the tokens after the first `<|message|>` up to the first `<|end|>`;
  - final = the tokens after `<|channel|>final<|message|>`, i.e. the token pattern `200005, 17196, 200008`.
  - This replaces `paragraphs()`'s `</think>` text search in `rl_train.py`.

### 1.4 Minimal snippet (tested live; full file `rh_rl/cluster/gptoss_rollout.py`)

```python
from openai_harmony import (Conversation, HarmonyEncodingName, Message, ReasoningEffort, Role,
                            StreamableParser, SystemContent, load_harmony_encoding)
ENC = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
RETURN, CALL = 200002, 200012

def prompt_ids(user_text, effort="medium", date="2026-10-14"):
    sys = SystemContent.new().with_reasoning_effort(
        {"low": ReasoningEffort.LOW, "medium": ReasoningEffort.MEDIUM, "high": ReasoningEffort.HIGH}[effort]
    ).with_conversation_start_date(date)
    conv = Conversation.from_messages([Message.from_role_and_content(Role.SYSTEM, sys),
                                       Message.from_role_and_content(Role.USER, user_text)])
    return ENC.render_conversation_for_completion(conv, Role.ASSISTANT)   # ends with <|start|>assistant

body = {"model": ADAPTER_NAME_OR_BASE, "prompt": ids, "n": 8, "max_tokens": 12288,
        "temperature": 1.0, "top_p": 1.0, "top_k": -1, "logprobs": 0,
        "return_token_ids": True, "skip_special_tokens": False, "stop_token_ids": [RETURN, CALL]}
# POST /v1/completions -> per choice: c["token_ids"] (incl. final <|return|>), c["logprobs"]["token_logprobs"]
# trainer sequence = ids + c["token_ids"]; vlp = token_logprobs (same length, no None)
```

Live output of `python gptoss_rollout.py 8400`:

```
matches vLLM chat-path rendering: True
stop 21 200002 -2.525 'Just compute. 17*3=51.' '51'
```

---

## 2. LoRA on gpt-oss in vLLM 0.30

### 2.1 What can carry LoRA [src]

- `GptOssForCausalLM(SupportsLoRA)` has `is_3d_moe_weight=True` and
  `packed_modules_mapping={"qkv_proj": ["q_proj", "k_proj", "v_proj"]}` (`model_executor/models/gpt_oss.py:1408-1412`).
- Supported modules are every `LinearBase` and `MoERunner` (`lora/utils.py:220-237`). For gpt-oss that is:
  - `attn.qkv_proj` (packed from q, k, v);
  - `attn.o_proj`;
  - `mlp.router` (ReplicatedLinear);
  - `mlp.experts` (the FusedMoE3DWithLoRA wrapper);
  - lm_head and embedding.
- HF's router is `GptOssTopKRouter` with raw parameters, not an `nn.Linear`, so PEFT `target_modules` cannot hit
  it. Skip the router.
- `hf_to_vllm_mapper` maps `.self_attn.` to `.attn.` (`gpt_oss.py:1414`). PEFT's `self_attn.q_proj` therefore lands
  in the vLLM `qkv_proj` q-slice.

### 2.2 PEFT adapter format [cpu, PEFT 0.21.2 + transformers 5.18 on a tiny GptOss config]

Attention only, with `target_modules=["q_proj","k_proj","v_proj","o_proj"]`:

```
base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight (r, H)
base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight (q_out, r)   ... same for k, v, o
```

Experts, adding `target_parameters=["mlp.experts.gate_up_proj","mlp.experts.down_proj"]`:

```
...layers.0.mlp.experts.base_layer.lora_A.weight (E*r, H)      # gate_up_proj
...layers.0.mlp.experts.base_layer.lora_B.weight (2I, E*r)
...layers.0.mlp.experts.lora_A.weight            (E*r, I)      # down_proj
...layers.0.mlp.experts.lora_B.weight            (H, E*r)
```

- This is exactly the layout vLLM expects: "experts.base_layer is the gate_up_proj and experts is the down_proj"
  (`lora/model_manager.py:880-935`, `lora/lora_model.py:216-231`).
- The key layout does not depend on the order of `target_parameters` (checked by reversing it).
- HF's expert params are `gate_up_proj (E, H, 2I)` (gate and up interleaved) and `down_proj (E, I, H)`.
  - vLLM handles gpt-oss's interleaving explicitly when TP-sharding expert LoRA (`lora/layers/fused_moe.py:544-570`).
  - With TP1 no slicing is needed.
- Not yet checked: whether vLLM's `lora_b.reshape(out, r, E)` (`model_manager.py:913-918`) matches PEFT's E·r
  column ordering. **Test T1b checks this numerically.**
- `PEFTHelper.from_dict` drops unknown config keys such as `target_parameters` (`lora/peft_helper.py:63-80`). It
  rejects DoRA and `modules_to_save`.

### 2.3 What `--enable-lora` does to the MoE kernel (the `gpt_oss.py:372` question and more) [src]

**The `gpt_oss.py:370-373` condition.** `is_sequence_parallel = parallel_config.use_sequence_parallel_moe and
lora_config is None`.

- `use_sequence_parallel_moe` requires EP, TP>1 and DP>1, plus an all2all backend (`config/parallel.py:715-730`).
- It never applies to our DP-only or TP1 setups. That line costs nothing here.

**The actual cost.**

- `FusedMoEConfig.is_lora_enabled = vllm_config.lora_config is not None`
  (`model_executor/layers/fused_moe/layer.py:353`). This holds regardless of `--lora-target-modules`.
- The kernel gate then rejects any experts class without `supports_lora()` (`fused_moe/modular_kernel.py:591-592`).
- Today the server logs `Using 'TRITON' Mxfp4 MoE backend` and `OAITritonExperts`
  (logs/serve_gptoss.log). This kernel has no LoRA mixin.
- LoRA-capable MXFP4 kernels on Hopper:
  - `UnfusedOAITritonExperts` (`fused_moe/experts/gpt_oss_triton_kernels_moe.py:1077`). It is commented out of the
    auto list ("TRITON_UNFUSED has bug with MTP support"; `fused_moe/oracle/mxfp4.py:332-354`). It is reachable via
    `--moe-backend triton_unfused`.
  - `MarlinExperts` (`fused_moe/experts/marlin_moe.py:648`).
- FlashInfer CUTLASS raises for LoRA (`flashinfer_cutlass_moe.py:396`). The TRTLLM variants need SM100.
- **Net:** with `--enable-lora`, auto selection on H200 will fall from TRITON to MARLIN [inf from the priority list;
  confirm in the startup log].
- The fused-MoE LoRA wrapper also asserts non-monolithic kernels (`lora/layers/fused_moe.py:39`).
- Slowdown magnitude [inf, no measurement available]:
  - Marlin W4A16 vs triton_kernels matmul_ogs at batch ~64–256 on Hopper: expect roughly 0–30% on MoE time.
  - Punica shrink/expand for attention qkv and o at r=64: typically +5–15% decode step time.
  - Expert LoRA (`fused_moe_lora` align, shrink and expand for w13 and w2): plausibly another +20–50%.
  - Treat **10–40% end-to-end** as the planning range until T2 measures it.

**MXFP4 vs bf16.** The LoRA deltas are bf16 side-paths added to the outputs of the quantized base kernel. LoRA on
the MXFP4 experts works through the LoRA-mixin kernels; the base weights stay MXFP4. Attention weights are bf16.

### 2.4 Memory and DP hazards for expert LoRA [src + inf]

- **GPU buffers are allocated at startup** for every supported module, sized at `max_lora_rank × max_loras`, unless
  `--lora-target-modules` restricts them (`lora/model_manager.py:430-438`, `config/lora.py:49`).
- Expert buffers per slot (TP1), 36 layers × 128 experts:
  - r × (H + 2I + I + H) per expert = 64 × 14,400 ≈ **4.25B params ≈ 8.5 GB bf16 per slot**.
  - `--max-loras 2` gives 17 GB per GPU.
  - The current `serve_gptoss_rl.sh` LORA=1 line would allocate this, because it has no target-module
    restriction. **Add `--lora-target-modules qkv_proj o_proj`.**
- Attention-only adapter at r=64: about 48M params, about 96 MB.
- **DP slot hazard.** The current server runs DP4 with MoE TP sharded *across the DP ranks* (no EP):
  - `FusedMoEParallelConfig.make` flattens TP over DP (`fused_moe/config.py:~1209-1229`).
  - The log shows 19.42 GiB of weights per GPU and `MoEPrepareAndFinalizeNaiveDPEPModular`.
  - Tokens from all DP ranks are all-gathered into each rank's MoE. The per-token LoRA *slot index* is gathered with
    them (`fused_moe/prepare_finalize/naive_dp_ep.py:136-177`).
  - Slot indices are assigned per engine by each rank's own LRU and activation order (`lora/worker_manager.py:286-317`).
  - So if two DP ranks ever place the same adapter in different slots, expert LoRA silently applies the wrong
    adapter to the other rank's tokens. That can happen during an adapter transition with `max_loras ≥ 2`.
  - Attention LoRA is unaffected: attention runs rank-locally.
  - If you want expert LoRA, use one of:
    - TP1 replicas as separate servers (external LB);
    - a single DP1 server;
    - `max_loras 1` with a drain between steps.

---

## 3. Adapter update paths

### 3.1 `/v1/load_lora_adapter` (runtime LoRA) [src]

- Needs `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` (`entrypoints/serve/lora/api_router.py:24-27`).
- **With DP>1 the default `api_server_count` equals DP size** (`entrypoints/cli/serve.py:105-127`; the current log
  shows `api_server_count: 4`). **vLLM raises** `VLLM_ALLOW_RUNTIME_LORA_UPDATING cannot be used with
  api_server_count > 1` (`serve.py:304-307`).
  - **Pass `--api-server-count 1`.** One API process is ample for about 256 decode streams.
  - DP1 defaults to 1 already.
- **All DP ranks get it.** `AsyncLLM.add_lora` calls `call_utility_async("add_lora")`, which `DPLBAsyncMPClient`
  fans out to every engine with `asyncio.gather` (`v1/engine/core_client.py:1275-1276, 1599-1608`).
- **Load cost:**
  - The worker reads `adapter_model.safetensors` from disk into CPU in `_load_adapter`, then copies into GPU slots
    on activation (`lora/worker_manager.py:286-317`).
  - For a ~96 MB attention adapter, expect **≈0.2–1 s per step** [inf], dominated by safetensors read and pinning.
    Each DP rank reads the file itself.
  - Expert adapters at r=64 (8.5 GB) would take tens of seconds per rank and use 8.5 GB of pinned CPU per cached
    adapter per rank.
- **Use a new `lora_name` every step** (`rl_train.py` already does this):
  - Prefix-cache block hashes include `lora_request.lora_name` (`v1/core/kv_cache_utils.py:559-571`).
  - `load_inplace=True` with a fixed name would reuse the previous policy's prompt KV.
  - New names also let in-flight requests keep the old adapter. Set `--max-loras 2`.
- `/v1/unload_lora_adapter` only deletes the name in the API process (`entrypoints/openai/models/serving.py:221-235`).
  The engine keeps the adapter until LRU eviction (`--max-cpu-loras`, default `max_loras`). That is harmless.
  `rl_train.load_adapter`'s unload loop can stay but frees nothing.

### 3.2 Faster alternative: merged attention weights, no LoRA in vLLM [src for plumbing; inf for latency]

All of these exist in 0.30. They are dev endpoints, registered only with `VLLM_SERVER_DEV_MODE=1`
(`entrypoints/launchers/api_server/routers.py:34-37`).

- `/pause` (mode abort, wait or keep, optional cache clear), `/resume`, `/reset_prefix_cache`
  (`entrypoints/serve/dev/rlhf/api_router.py:29-91`, `dev/cache/api_router.py:20`).
- `/init_weight_transfer_engine`, `/start_weight_update`, `/update_weights`, `/finish_weight_update` with NCCL, IPC
  or sharded-RDT engines (`distributed/weight_transfer/*`).
  - They load *base* weights and then call `reset_lora_state()`, which drops all adapters
    (`v1/worker/gpu_worker.py:1476-1479`).
  - So they are not a LoRA path. They are the path for merged weights.
- `/collective_rpc {"method":"reload_weights","kwargs":{"weights_path": DIR}}`
  (`dev/rpc/api_router.py:23-54`; `v1/worker/gpu_model_runner.py:5553-5598`).
  - It is broadcast to all DP ranks via the same DPLB fan-out.
  - Partial checkpoints are allowed. Layers that receive no weights get their kernel tensors restored, with a
    "Failed to load weights" warning (`model_executor/model_loader/reload/layerwise.py:228-286`).
  - So a directory holding only `model.layers.*.self_attn.{q,k,v,o}_proj.weight` should reload attention only.
- `/sleep?level=1` and `/wake_up` (`dev/sleep/api_router.py`; `v1/engine/core.py:879-904`). Level 1 offloads
  weights to CPU and drops the KV cache.

**Procedure.** For attention-only LoRA, the trainer computes W' = W0 + (α/r)·B·A for the q, k, v and o weights
(about 0.95B params, **≈1.9 GB bf16**) and pushes them. Then:

- vLLM runs **without `--enable-lora`**. It keeps the TRITON fused MoE kernel, with no punica kernels and no LoRA
  CUDA-graph specialisation. This recovers the whole 10–40% LoRA overhead.
- Latency [inf]: about 1–3 s via safetensors on local disk, well under 1 s via NCCL or IPC over NVLink.
- **Must:** after the reload, clear the prefix cache. Either `/pause?mode=wait&clear_cache=true` → reload →
  `/resume`, or `/reset_prefix_cache` between steps.
- Small cost: bf16 rounding of W0+BA differs slightly from the trainer's unmerged forward. The per-token truncated
  IS weight corrects for it.
- **Not usable for expert LoRA.** Merging into MXFP4 expert weights would requantize. Use path 3.1 for experts.

**Recommendation.** Start with 3.1 (lowest risk, matches `rl_train.py`). Switch to 3.2 if T2 shows more than about
15% LoRA overhead.

---

## 4. Throughput

### 4.1 Measured on the pilot [live metrics + cpu]

Setup: 4×H200, DP4 (MoE TP-sharded across DP), no LoRA, `/v1/chat/completions`, n=8, 240 requests in flight.

| | |
|---|---|
| Samples | 1,920 |
| Generated tokens (`vllm:generation_tokens_total`) | 5.34M |
| Wall time | 337 s (01:38:14 → 01:43:51) |
| Throughput | **15.8k tok/s aggregate ≈ 4.0k tok/s/GPU**, including the tail |

- Prefix-cache hit rate was 55.7%: 253k hits out of 454k queried prompt tokens per engine. n=8 sharing works, but
  only partly, below the ideal 7/8, because children scheduled in the same step can each prefill.
- Completion lengths (harmony tokens):

  | effort | mean | p50 | p90 | p99 | max | truncated at 16,384 |
  |---|---|---|---|---|---|---|
  | low | 1.3k | 1.05k | 2.7k | 4.0k | 5.2k | 0 |
  | medium | 4.24k | 3.26k | 8.8k | 16.4k | (cap) | 13 / 960 |

**Public reference (vLLM is not named on these pages; 1k/1k, FP4).** SemiAnalysis InferenceX reports gpt-oss-120b
on H200 at:

| tok/s/GPU | tok/s/user | concurrency |
|---|---|---|
| ~4.7k | ~58 | ~64 |
| ~3.4–3.6k | ~92–98 | |
| ~1.5k | ~166 | |

Our measured ~4k tok/s/GPU is consistent with this.

### 4.2 Estimate per RL step (32 prompts × 8, medium effort, max_tokens 12,288)

- About 256 × 4.24k ≈ **1.1M tokens**.
- **4 GPUs:** the bulk phase runs at about 15–19k tok/s, so ≈ 60–70 s. Then the tail: the last few sequences run at
  about 120–200 tok/s single-stream [inf], and the longest are about 9–12k tokens, so +50–90 s. Total **≈ 1.5–2.5 min
  per step without LoRA**; add 10–40% with `--enable-lora` [inf].
- **1 GPU** (DP1, as now): about 4–5k tok/s bulk, so **≈ 4–6 min per step**.
- KV is not a constraint:
  - 1.998M KV tokens per GPU with DP4 (log), because sliding-window layers and the MoE sharding leave a lot of HBM.
  - With TP1 full replicas (~65 GB of weights), about 0.9–1.1M tokens per GPU [inf].
  - 256 × 14k is about 3.6M tokens in total.
  - `max_num_seqs` defaults to 1024 on H200 (`engine/arg_utils.py:2740-2748`) and is not binding.

### 4.3 Levers

1. **Keep `n=8` in one request.** The DP load balancer sends one request to one rank
   (`core_client.py:1539-1597`), so the 8 children share the prompt prefix. Eight separate requests would spread
   across ranks and lose the sharing. Already done.
2. **Tail control is the biggest lever.**
   - Use the existing one-step-lag async pipeline.
   - Lower `max_tokens` (medium-effort p90 is 8.8k, so 10–12k is reasonable).
   - Optionally over-sample prompts and abort stragglers. `/abort_requests` exists (dev mode,
     `dev/rlhf/api_router.py:94-136`), as does client-side cancel.
   - At DP4 with MoE TP across ranks, **all 4 GPUs step in lockstep** (dummy batches on idle ranks). One straggler
     keeps all four busy at low batch.
3. **DP4 (MoE TP across DP) vs 4 independent TP1 replicas** [inf]:
   - DP4 shares expert weight reads, which is good for memory-bound decode, but pays a per-layer all-gather and
     reduce-scatter plus lockstep.
   - Independent replicas decouple tails and avoid the DP-LoRA slot hazard (§2.4).
   - Worth a 10-minute A/B (T2b) if 4 GPUs are available for sampling.
4. **No LoRA in vLLM (merged attention path, §3.2):** removes the MoE kernel downgrade.
5. **Sleep-mode colocation.** `--enable-sleep-mode` exists (`config/model.py:342`).
   - `/sleep?level=1` offloads weights and drops KV. Utility calls broadcast to all DP ranks.
   - Not verified with LoRA. The LoRA buffers are model tensors and should go with the weights.
   - Useful only if the trainer must share these GPUs. A bf16 120B trainer needs about 234 GB for weights alone.
6. Pinning the system-prompt date keeps prompts byte-stable, so prefix caching works across steps for repeated
   tasks.

---

## 5. Things in `rl_train.py` and `serve_gptoss_rl.sh` to change for gpt-oss

- `sample_one`:
  - `stop_token_ids` is `[248046, 248044]` (Qwen) → use `[200002, 200012]`.
  - The default model name is `"qwen3.8-27b"` → use `"gpt-oss-120b"`.
  - `prompt_ids` uses the HF template, which is token-exact, but the date is unpinned → use
    `gptoss_rollout.prompt_ids`.
  - `sample_one` already uses `/v1/completions` with `return_token_ids`, which is the right shape.
- `split_response(s["text"])` and `paragraphs()` rely on `</think>`. Replace both with `split_channels(c_ids)` and
  token-index groups (§1.3). `text` no longer contains `</think>`.
- `--targets` should be `q_proj,k_proj,v_proj,o_proj`, optionally with `target_parameters` for experts. The Qwen
  names `gate_proj` and `up_proj` do not exist in GptOss.
- `serve_gptoss_rl.sh` LORA=1 needs:
  - `--api-server-count 1` (if DP>1);
  - `--lora-target-modules qkv_proj o_proj`;
  - `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`;
  - optionally `--max-cpu-loras 2`.
- The logs to check at startup: the `Mxfp4 MoE backend` line and `Maximum concurrency`.

---

## 6. Cheapest GPU tests (each ≤ 15 min, one GPU unless noted)

| # | Test | Pass criterion |
|---|---|---|
| T1 | DP1 server with `--enable-lora --max-lora-rank 64 --max-loras 2 --lora-target-modules qkv_proj o_proj --api-server-count 1`, `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`. Read the MoE backend line. Load a PEFT adapter with **B=0**, time `/v1/load_lora_adapter`. Generate 4×64 tokens at T=0 via base and adapter names. | Startup OK. The backend line shows which LoRA kernel was picked (expect MARLIN). The zero adapter gives identical tokens and logprobs. Load takes < 2 s. |
| T1b | Same server, adapter with random small B. Compare vLLM `prompt_logprobs` (echo a fixed prompt+completion) against an HF+PEFT forward of the same adapter on CPU/GPU for 2 sequences of 512 tokens. Repeat with experts at r=8, which needs the server without target-module restriction. | Mean abs Δlogprob ≲ 0.02 (MXFP4 kernel noise). An expert-ordering bug would show as large Δ. |
| T2 | Throughput A/B on one GPU, 64 prompts × n=4, max_tokens 4096, the same prompts in three configs: (a) no LoRA (TRITON); (b) `--enable-lora` + attention adapter (MARLIN or auto); (c) (b) with `--moe-backend triton_unfused`. | Gives the LoRA overhead. More than about 15% favours the merged path (§3.2). |
| T2b (optional, 4 GPUs) | DP4 (MoE TP across DP) vs 4 independent TP1 servers on the same 256-sequence workload. | Wall time per step, including the tail. |
| T3 | `VLLM_SERVER_DEV_MODE=1`, no LoRA. Write an attention-only safetensors directory (merged W0+BA, HF names). `/pause?mode=wait` → `/collective_rpc reload_weights {weights_path}` → `/reset_prefix_cache` → `/resume`. | Reload < 5 s. Logprobs match T1b's adapter outputs. No errors from the partial reload. |
| T4 | Trainer agreement: for 8 rollouts, HF transformers 5.18 logprobs (MXFP4 dequantized to bf16) vs vLLM `vlp`. | Mean IS weight about 1, under 1% of tokens clipped at 2. Otherwise the IS correction is doing real work and should be investigated. |

---

## 7. Verified vs inferred

- **Verified live:**
  - token-id prompt → `/v1/completions` works;
  - `token_ids` include `<|return|>` with a logprob;
  - logprob count equals token count, with no `None`;
  - `prompt_token_ids` are echoed;
  - harmony split works;
  - pilot throughput and prefix-hit metrics.
- **Verified on CPU:**
  - harmony and HF-template prompt ids are identical (low, medium, high);
  - special ids;
  - the PEFT key layout for attention and 3D experts;
  - truncation parsing.
- **Verified in source:** everything cited with file:line above, including:
  - the api-server-count/runtime-LoRA conflict;
  - DP broadcast of `add_lora` and `collective_rpc`;
  - the MoE kernel switch under `--enable-lora`;
  - the prefix-hash-by-name behaviour;
  - unload being API-only;
  - the dev RL endpoints;
  - the DP LoRA-mapping gather.
- **Inferred, needs the tests above:**
  - which LoRA MoE kernel auto selects on H200, and its speed;
  - LoRA overhead percentages;
  - adapter load latency;
  - partial `reload_weights` behaviour;
  - tail decode rates;
  - DP4 vs TP1-replica comparison;
  - sleep mode with LoRA.

Scratch scripts on the cluster: `/data/agastyas/rh_rl/agent_scratch/{check_render,live_rollout,check_peft,check_parse,pilot_lens,gptoss_rollout}.py`.

Sources: [SemiAnalysis InferenceX gpt-oss-120b H100 vs H200](https://inferencex.semianalysis.com/compare/gptoss-120b-h100-vs-h200),
[InferenceX GB200 vs H200](https://inferencex.semianalysis.com/compare/gptoss-120b-gb200-vs-h200),
[InferenceX H200 vs MI355X](https://inferencex.semianalysis.com/compare/gptoss-120b-h200-vs-mi355x).
