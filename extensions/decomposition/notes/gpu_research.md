# GPU plan for Qwen3.8-27B activation extraction (HF transformers, 1 GPU)

Research date: 2026-10-13. CPU only: no pods, no paid APIs.
Code I read: transformers 5.19.0 from the local venv (`decomp/.venv/lib/python3.13/site-packages/transformers/`, shortened to `tf/` below), and fla v0.5.2 source from GitHub.
Empirical checks: I ran a tiny random-weight `Qwen3_5ForConditionalGeneration` on CPU, using transformers 5.19 with miniconda's torch 2.13 and the reference torch kernels. It has 8 layers in the same 3×linear + 1×full pattern, untied lm_head, and GVA with 2 key heads and 4 value heads.
- Test scripts: `/private/tmp/claude-501/-Users-ndeshpande-Documents-1-classes-fall-2026-cs-2881r-reward-hacking---cs2881r/b73202a8-b36f-4569-961a-00768ab467dc/scratchpad/tests/{cpu_checks,load_checks}.py` (temporary scratchpad; copy into `decomp/tests/` if you want to keep them).
- Results are quoted below as **[CPU-verified]**.
- Anything that needs a GPU (fla kernels, sdpa backend choice, real memory, throughput) is **[unverified]**.

## TL;DR recommendations

1. **Load only layers 0–40.** Set `text_config.num_hidden_layers = 41` and `layer_types = layer_types[:41]`, then call `AutoModelForCausalLM.from_pretrained(..., config=cfg)`.
   - The checkpoint keys for layers 41–63 are dropped as "UNEXPECTED". `model.norm` and the untied `lm_head` still load, so the logit lens works.
   - Weights shrink from 53.8 GB to **36.3 GB** (18.15B parameters).
   - Hook `model.model.layers[40]`. **[CPU-verified: bit-identical to the full model]**
2. **Skip prefix sharing for the synthetic pairs.**
   - It is exact (cache continuation matches a full forward to about 1e-7 in fp32 and 0 in bf16 on CPU).
   - But it saves only about 0.15–0.3 GPU-h on a workload that takes about 0.5 GPU-h.
   - Run each branch as batch=1, or put the 2 branches of a pair in one right-padded batch.
3. **Padding:** use **right padding with `attention_mask`**. It is exact. Left padding is only approximately equal in bf16. **Do not pack sequences:** the linear-attention state leaks across sequence boundaries.
4. **Long rollouts:** a single forward pass with `use_cache=False` is enough. Use sdpa, which is the default, and forbid the math backend.
   - With the truncated model, a 70k-token pass should peak around **50–55 GB** and a 140k-token pass around **60–75 GB**. **H100 80GB SXM is enough; H200 is not needed.**
   - Chunked prefill through `DynamicCache` is exact and is the fallback if a pass runs out of memory.
5. **Packages:**
   - `transformers==5.19.0`. Qwen3.5 support first shipped in 5.2.0.
   - **`flash-linear-attention==0.5.2` is required.** The torch fallback is more than 10× slower and memory-hungry.
   - `causal-conv1d` is optional; its fallback is a plain depthwise `F.conv1d`.
   - Image: `runpod/pytorch:1.4.0-cu1281-torch2121-ubuntu2404`, or the `torch2130` variant.
6. **Test model:**
   - `Qwen/Qwen3.5-0.8B` (1.75 GB) for plumbing. It runs even on a laptop CPU.
   - `Qwen/Qwen3.5-9B` (19.3 GB) for an exact-feature check, because it has an untied head and GVA.
   - There is no small Qwen3.8 model.
7. **Budget:** about **1.5–2.5 H100-hours** in total, including setup; plan for 3–4 to cover debugging.
   - Layer-40 fp16 cache: **10,240 bytes per token**, so 1M assistant tokens is 10.2 GB.

---

## 1. Early exit at layer 40

**How the decoder loop works.** In `tf/models/qwen3_5/modeling_qwen3_5.py`:
- Line 1281 is `for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers])`.
- Line 1285 picks the mask with `causal_mask_mapping[self.config.layer_types[i]]`.
- Then `self.norm` is applied.

The loop takes the first N layers in order, so either kind of truncation is safe. Each layer keeps its own `layer_idx` (used by the cache), and those indices stay correct because we keep the prefix 0..40. Layer 40 is a `linear_attention` layer.

**Option A (recommended): a truncated config at load time.**
```python
cfg = AutoConfig.from_pretrained(MODEL)
cfg.text_config.num_hidden_layers = 41
cfg.text_config.layer_types = cfg.text_config.layer_types[:41]
model = AutoModelForCausalLM.from_pretrained(MODEL, config=cfg, dtype=torch.bfloat16, device_map="cuda:0").eval()
lm = model.model  # Qwen3_5TextModel; hook lm.layers[40]
```
- `AutoModelForCausalLM` maps `qwen3_5` to `Qwen3_5ForCausalLM`, commented as "VLM compatibility" (`tf/models/auto/modeling_auto.py:856`). It strips the `language_model.` key prefix (`tf/conversion_mapping.py:1054`) and ignores `^model.visual.*` and `^mtp.*` (`modeling_qwen3_5.py:1670`).
- The vision tower (0.46B) is therefore never loaded. That is what `t2_acts.py` already does, which is why its `lm = model.model` fallback works.
- **[CPU-verified]**
  - The model has 6 of 8 layers after loading.
  - The load report lists only the dropped layers as UNEXPECTED, and nothing is missing.
  - The layer-5 output is bit-identical to the full model's (max|diff| = 0.0).
  - `lm_head` and `norm` are bit-identical.
- Parameter counts, from meta-device instantiation with the real config:
  - Full CausalLM: 26.896B (53.79 GB bf16).
  - 41 layers: 18.147B (36.29 GB).
  - The ForConditionalGeneration wrapper with vision is 27.357B.
  - Truncation saves 17.5 GB (33%) and a matching share of load time.
- The forward still applies final norm + lm_head to the layer-40 output, but `logits_to_keep=1` makes that negligible. For the logit lens, call `model.lm_head(model.model.norm(v))` on any layer ≤40.

**Option B: `lm.layers = lm.layers[:41]` after a full load.** Also bit-identical **[CPU-verified]**, but you load 54 GB first. It is only useful if one script needs all 64 layers sometimes.

**Option C: raise an exception in a hook.** It works, but it saves nothing over A. Skip it.

**Skipping the download of later layers.** The 27B checkpoint has 18 shards totalling 55.56 GB (hub index). Layers 0–40 plus embed, norm and lm_head need shards 1–12, 16 and 18, which come to **45.3 GB**:
- Shard 12 holds layers 40–45.
- Shard 16 holds `model.language_model.norm` and layers 56–61.
- Shard 18 holds `lm_head` and `mtp.*`.

The loader raises FileNotFoundError on missing shards. It works if you also rewrite `model.safetensors.index.json` to list only the present shards **[CPU-verified]**. That saves about 10 GB of download, which is optional and minor.

## 2. Prefix sharing: exact, but not worth it here

**Mechanics** (`modeling_qwen3_5.py:540-653`, `tf/cache_utils.py:942-1155`):
- When a linear-attention layer has a previous state, a multi-token forward goes through `cache_params.update_conv_state` (line 576).
  - That call prepends the cached last-3 pre-conv inputs, stored in bf16, which are exactly what a full forward would see.
  - Then `chunk_gated_delta_rule(..., initial_state=recurrent_state)` runs (line 635).
- The recurrent state is fp32. fla allocates `final_state` as fp32 (`fla/ops/common/chunk_delta_h.py` L704/707), and the torch reference also produces fp32.
- fla's `chunk_gated_delta_rule` natively supports `initial_state` and `output_final_state`, with shape `[N, HV, K, V]` (`fla/ops/gated_delta_rule/chunk.py` L397-435, v0.5.2).
- Full-attention layers use the ordinary DynamicLayer KV cache.
- The only mathematical difference is that the suffix's 64-token chunk boundaries are shifted, so results differ only at the fp-roundoff level.

**[CPU-verified]** Cached prefix + suffix vs a full forward on prefix+suffix, as relative L2 error on the suffix tokens:
- fp32, prefix 200 (not chunk-aligned): 2.6e-7 at both layer 5 and layer 7 (after a full-attention layer).
- fp32, prefix 192 (aligned): ≤8e-8.
- bf16, prefixes 200 and 1000: 0.0.

On GPU, expect bf16-noise-level differences, about 1e-3, from different GEMM and attention kernels **[unverified]**. Check against the batch=1 vs batched noise floor on 0.8B.

**Gotchas [CPU-verified]:**
- **The cache is mutated in place**: `copy_` happens in `update_recurrent_state` and `update_conv_state`. If you reuse the prefix cache for branch B without copying, B's activations are wrong, with relative error 5–10%. Fork with `copy.deepcopy(cache)` per branch.
  - Alternatively, run both branches as one batch with `cache.reorder_cache(torch.tensor([0, 0]))`, which also verified exact.
- `cache.batch_repeat_interleave(2)` **crashes**: `AttributeError: 'LinearAttentionLayer' object has no attribute 'batch_repeat_interleave'`.
- Share the **token-level** longest common prefix of the two tokenized branches, not the character divergence point. BPE can merge across the boundary.
- Position ids are handled: text-only `compute_3d_position_ids` returns None (`modeling_qwen3_5.py:1575`), and the text model uses `arange + past_seen_tokens` (line 1252).
- With a cache, sdpa gets a materialized `[1,1,S,P+S]` mask. That is fine at a few thousand tokens.

**Cost comparison.** Assumptions:
- H100 SXM bf16 dense peak is 989 TFLOPS (NVIDIA H100 datasheet, nvidia.com/en-us/data-center/h100; the PCIe version is 756).
- HF eager forward at about 35% MFU, so **~350 TFLOPS effective**; the pessimistic case is 200 TFLOPS. This is an assumption, not measured. GEMMs with M ≥ 2k tokens should be efficient, while the GDN, norm and elementwise fp32 work adds overhead.

FLOPs per token for 41 layers: 2 × 15.6B layer params gives **31.2 GFLOP**. Attention adds 2·L·6144 per full-attention layer: about 0.3 GFLOP at L ≈ 2k, but about 4 GFLOP at L = 33k.

| Synthetic, 4,000 pairs × 2 × ~2.25k tok = 18M tok | Tokens | GPU-h @350 TF | GPU-h @200 TF |
|---|---|---|---|
| Plain (each branch full) | 18M | **0.45** | 0.8 |
| Prefix-shared, prefix = 60 / 70 / 80% | 12.6 / 11.7 / 10.8M | 0.32 / 0.30 / 0.27 | 0.55 / 0.5 / 0.47 |

The speedup is 1/(1 − p/2): 1.43× / 1.54× / 1.67×. The absolute saving is about 10–20 GPU-minutes, under $1. **Not worth a second code path.** Use plain forwards and keep prefix sharing in reserve.

## 3. Padding and packing

The linear-attention input is zeroed at padding positions: `apply_mask_to_padding_states` (`modeling_qwen3_5.py:227, 547`), with the 2D mask from `create_recurrent_attention_mask` (`tf/masking_utils.py:1471-1501`). That mask is None if it is all ones, and is trimmed to the current segment when continuing from a cache.
- A zero input gives k = v = 0 after the bias-free projections, and l2norm(0) = 0. So the delta update is zero and a zero state stays zero.
- Left padding is therefore mathematically exact for the recurrence.

| Setup ([CPU-verified], rel. err vs unbatched) | fp32 | bf16 |
|---|---|---|
| Right pad + mask | 3e-8 | **0.0** |
| Right pad, no mask | 3e-8 | 0.0 (causal: later pads can't affect earlier tokens) |
| Left pad + mask | 3e-7 | 5e-3 |
| Left pad + mask + explicit `position_ids` | 3e-7 | 6e-4 |
| Packing via reset `position_ids` | 5.7e-3 | 7e-3 (**wrong**) |

Why left padding is only approximate in bf16: the text-only path ignores the mask when building positions. It uses `arange` at line 1252, so pads shift RoPE positions. RoPE is shift-invariant in exact arithmetic but not in bf16.

Why packing is wrong:
- The full-attention layers do detect packing from `position_ids` (`masking_utils.py:731-760`).
- But the GDN layers only get boundaries via a `cu_seq_lens_q` kwarg (lines 625/638), which the torch fallback ignores.
- The conv never receives `seq_idx`: the kernel wrapper filters kwargs to the implementation's signature (`tf/integrations/hub_kernels.py:1039`). So even with fla, 3 conv taps leak across each boundary.

**Recommendation:**
- Use right padding with `attention_mask`, then slice off the pads before averaging.
- The sdpa mask gets materialized for padded batches, `[B,1,L,L]` bool, which is about 70 MB at B = 8 and L = 3k. That is fine.
- Never pack.

## 4. Long rollouts: memory

**Full attention.** sdpa is the default (`attn impl: sdpa` in the CPU test).
- With no mask and q_len = kv_len, the mask is skipped and `is_causal=True` is used (`masking_utils.py:273`).
- GQA goes in natively through `enable_gqa` when there is no mask and head_dim ≤ 256. Ours is 256 (`tf/integrations/sdpa_attention.py:30-37, 109-114`).
- This should dispatch to the flash or mem-efficient backend, which is linear in memory **[unverified on GPU]**.
- **Guard:** wrap the forward in `torch.nn.attention.sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])`. A math-backend fallback would then raise an error instead of materializing 24 × L² scores.
- Do not pass a 2D attention_mask that has padding on long sequences. That would materialize an L × L mask: 19.6 GB at 140k.

**Activations** (no_grad, bf16, estimated per token, transient within one layer):
- MLP: gate, act, up and product at 17408 × 2 B each, about 104 KB, plus the residual and normed copies. **~125 KB** in total.
- GDN layer: proj/conv about 60 KB, GVA-repeated q and k about 25 KB, fla internals about 75 KB, and the fp32 `RMSNormGated` temporaries about 75 KB.
  - The fla internals are `h` `[NT,HV,K,V]` bf16 (24.6 KB/token), plus `w`, `u`, `A` (fp32), `v_new` and `o`.
- Peak: **~150–250 KB/token**, which is 10–18 GB at 70k and 21–35 GB at 140k. This is an estimate.

| Peak memory estimate | 70k tok | 140k tok |
|---|---|---|
| 41-layer weights (36.3 GB) + activations | ~47–55 GB | ~58–72 GB |
| 64-layer weights (53.8 GB) + activations | ~64–72 GB | ~75–89 GB (OOM risk on 80 GB) |

**The fla kernels need fla installed.** Without fla, the torch reference `torch_chunk_gated_delta_rule` (`modeling_qwen3_5.py:290-424`) is slow and memory-hungry:
- It casts q, k, v and β to fp32 and keeps about 11 tensors of `[48, T, 128]` fp32 plus 3 of `[48, T/64, 64, 64]` fp32. That is about 300 KB/token: **42 GB at 140k**, and about 21 GB at 70k.
- It then loops over chunks in Python.
- HF's own warning says it is more than 10× slower on H100 (`hub_kernels.py:1028-1031`).

With fla, the largest intermediate is `h`, at 1.7 GB for 70k tokens.

**Hooks.** Do not upcast the whole layer output. Copy the assistant-token slice as `h[0, idx].to(torch.float16).cpu()`, and compute the 30 projections as `h[0].float() @ V40` (V40 has shape `[5120, 30]`) under a per-chunk loop if needed. This is the pattern in `t2_acts.py`, restricted to one layer.

**Chunked prefill.** It is not needed with the truncated model for ≤140k tokens.
- To use it: `cache = DynamicCache(config=lm.config)`, then feed 16–32k-token chunks with `use_cache=True` and `past_key_values=cache`.
- It is exact, per §2. KV memory is 10 full-attention layers × 4 KB/token = 40 KB/token, which is 5.6 GB at 140k. The GDN state is about 3 MB per layer.

**Verdict:** **H100 80GB SXM** is enough with the 41-layer load. The rollouts are ~70k+ tokens, so there is about 25 GB of headroom.
- Use H200 only if you insist on the full 64 layers at more than 120k tokens without chunking.
- First job on the pod: run the longest rollout and print `torch.cuda.max_memory_allocated()`.
- I could not confirm which GPU the athena02 cluster used for the 140k-token runs; `serve_qwen.sh` and `hfrun.sh` do not say.

## 5. Packages, versions and image

- **transformers:**
  - Qwen3.5 support was added in commit fc91372 "Adding Support for Qwen3.5 (#43830)" on 2026-02-09. It is in v5.2.0 (released 2026-02-16) and not in v5.1.0 (checked with GitHub compare).
  - Qwen3.8-27B uses the same `model_type: qwen3_5` and `Qwen3_5ForConditionalGeneration`. Its config was saved by 5.8.0.dev0.
  - Its config diff vs Qwen3.5-27B is only extra or ignored keys: `language_model_only`, `output_gate_type: swish`, `bos/pad`, `partial_rotary_factor`, `tie_word_embeddings: false`.
  - `output_gate_type` is only read by `qwen4_exp`. Qwen3_5's `RMSNormGated` hard-codes silu (= swish), so the behaviour is consistent.
  - **Pin 5.19.0** (latest, 2026-10-06; identical to the code read here). It requires torch ≥ 2.5 and huggingface-hub ≥ 1.31.
- **flash-linear-attention 0.5.2** (fla-core 0.5.2, 2026-07-27): **required**.
  - transformers picks it up at import time through `use_kernel_func_from_hub_with_fallback("chunk_gated_delta_rule", "fla")`, which imports `fla.ops.gated_delta_rule` (`hub_kernels.py:67-80, 984-1044`).
  - It needs torch ≥ 2.7 and triton ≥ 3.3 (its pyproject `[cuda]` extra).
  - Since v0.5, a bare `pip install flash-linear-attention` does not pull in torch.
  - **Verify on the pod** that no "`chunk_gated_delta_rule` is falling back" warning appears.
- **causal-conv1d 1.7.0:** optional. The fallback (`modeling_qwen3_5.py:259-281`) is one depthwise `F.conv1d` with kernel size 4, which is cheap. Building it needs nvcc and is not worth the risk.
- **flash-attn:** not needed; sdpa is enough.
- **Image:** `runpod/pytorch:1.4.0-cu1281-torch2121-ubuntu2404`, or `...-torch2130-...` to match the local CPU test's torch 2.13.
  - These are stable tags from 2026-09-30 on Docker Hub. CUDA 12.8 needs host driver ≥ 570, so filter for CUDA ≥ 12.8 when choosing the pod.
  - The fla + triton combination on these torch versions is **[unverified]**. Fallback image: `1.4.0-cu1281-torch291-ubuntu2404`.
- Install: `pip install transformers==5.19.0 flash-linear-attention==0.5.2 fla-core==0.5.2 einops accelerate hf_xet`

## 6. Small compatible test models (hub configs fetched with curl)

| Model | Class | Layers | Hidden | GDN k/v heads | Tied head | Size |
|---|---|---|---|---|---|---|
| **Qwen/Qwen3.5-0.8B** | Qwen3_5ForConditionalGeneration | 24 (3:1) | 1024 | 16/16 | yes | 1.75 GB |
| Qwen/Qwen3.5-2B | same | 24 (3:1) | 2048 | 16/16 | yes | 4.5 GB |
| Qwen/Qwen3.5-4B | same | 32 (3:1) | 2560 | 16/32 | yes | 9.3 GB |
| **Qwen/Qwen3.5-9B** | same | 32 (3:1) | 4096 | 16/32 | **no** | 19.3 GB |
| Qwen/Qwen3.8-27B (target) | same | 64 (3:1) | 5120 | 16/48 | no | 55.6 GB |

- No small Qwen3.8 models exist. Qwen's Qwen3.8 repos are only 27B, Flash-Next and 2.4T-A95B.
- **Use 0.8B for plumbing.** It shares the vocab (248,320), head_dim 256, and the GDN 128-dim heads.
  - It can even run on the laptop CPU with the torch fallback. The base miniconda env has torch 2.13 and works with the venv's transformers via `PYTHONPATH`.
  - It does **not** exercise the untied lm_head or the GVA q/k repeat, and it has only 24 layers. Truncate at layer 15, which is roughly 40/64 of the depth.
- **Use 9B** for a final check: untied head, GVA, about 20 GB on any 40 GB+ GPU.

## 7. Estimates

**Synthetic workload:** about 0.45 GPU-h at the assumed throughput (0.8 GPU-h pessimistic). Batch=1 is fine: about 0.2 s of compute per 2.25k-token forward, against roughly 20 ms of launch overhead.

**Real rollouts:**
- About 271 × ~33k ≈ **9M tokens**, assuming mean ≈ 1.1× median. Measure this.
- About 35.5 GFLOP/token including attention at L ≈ 33k.
- Total 3.2e17 FLOP, which is **~0.25 GPU-h** (0.45 GPU-h pessimistic). Writing to disk adds a little.

**Overheads:**
- Image pull plus pip: about 10 min.
- Weight download: 45–56 GB, about 5–15 min. Use `hf_xet` and store on a network volume so later runs skip it.
- Load: about 2 min.
- 0.8B smoke tests: about 0.25–0.5 h.

**Total: about 1.5–2.5 H100-hours. Budget 3–4.**

**Layer-40 fp16 cache:** 5120 × 2 B = **10,240 B per token** (10.24 GB per 1M tokens).
- 1M assistant tokens: 10.2 GB.
- 2.5M (about 25% of 9M): 25.6 GB.
- All 9M tokens: 92 GB.

Write one `.npy` per rollout and memory-map it; do not accumulate the cache in RAM. The 30 projections cost only 120 B/token in fp32.

## Not verified / open risks
- GPU behaviour: fla kernel numerics vs the torch reference, the sdpa backend chosen at head_dim 256 with `enable_gqa`, real peak memory, and throughput. MFU of 35% is an assumption.
- Whether the triton bundled with torch 2.12 or 2.13 works with fla 0.5.2 on H100. fla's CI badge says it runs on H100, but I did not check versions.
- What GPU athena02 had.
- The rollout token distribution: the mean and the assistant-token fraction need measuring.
- Exact shard-download savings assume the hub index layout as of today.
