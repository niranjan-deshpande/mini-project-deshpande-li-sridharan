# J-lens and J++ lens: what they are, cost for us, recommendation

Research date: 2026-10-13. CPU-only literature pass. Sources are listed at the bottom.

## TL;DR

- **J-lens** (Jacobian lens) comes from Gurnee, Sofroniew, Pearce, ... Batson, Lindsey (Anthropic), *Verbalizable Representations Form a Global Workspace in Language Models*. It was posted on arXiv as 2607.15495 on 16 July 2026, with a transformer-circuits.pub version, and its code is at `anthropics/jacobian-lens`. It decodes a residual vector `h` at layer `l` as `unembed(J_l h)`, with `J_l = E[∂h_final/∂h_l]`. The expectation is averaged over a text corpus, over source positions, and over current and future target positions. Setting `J = I` gives the logit lens.
- **R-lens** comes from Blank, Bhatia and Nanda, *R-lens: Making J-lens More Faithful on Early Layers* (Alignment Forum, 5 Aug 2026). It is the J-lens with its Jacobians computed through an LRP-modified backward pass, which adds stop-gradients.
- **J++ lens** comes from Kola Ayonrinde (Anthropic Fellows) and Jack Lindsey (Anthropic), *J++ Lens: Jacobian Filtering Enables More Faithful Workspace Lenses*. It was posted on LessWrong on 8 Oct 2026; no arXiv version had appeared as of 13 Oct. Code: https://github.com/safety-research/jpp_lens (Apache-2.0). Lenses: https://huggingface.co/koayon/jpp-lenses. I am confident this is the method the brief refers to.
- **A native J++ refit on Qwen3.8-27B is not cheap.** At layer 40 alone I estimate about 2–5 H100-hours, so **skip the refit.**
- **A pretrained J++ lens exists for Qwen3.6-27B, and layer 40 is one of its 7 J++-fitted readout layers.** Reports say Qwen3.8-27B has the same architecture as Qwen3.6-27B (an empty config diff, the same 248,320 vocabulary, a working 0.9/0.1 weight-merge of the two). So we can **borrow J_40 from the Qwen3.6 lens** and decode with Qwen3.8's own final norm and lm_head. That costs about 0 GPU-minutes. Validate it with a cheap native finite-difference J-lens check (about 5–15 GPU-min) and label the result "J++ (Qwen3.6-fit, transferred)".

## 1. Method definitions

### J-lens (Gurnee et al., 2026)
- `lens_l(h) = unembed(J_l @ h)`, where `unembed` means the final RMSNorm followed by lm_head.
- `J_l = E_{prompts, source pos s, target pos t ≥ s}[∂h_t^{final} / ∂h_s^{(l)}]`. The reference estimator sums cotangents over target positions and then averages over source positions. The "final" target is the output of the last block (`target_offset = -1`, block 63 for a 64-block model).
- Fitting data: 1,000 sequences × 128 tokens of web text in the paper. The README says about 100 prompts already works, because quality saturates quickly.
- Each J_l is a dense d×d matrix (5120×5120 for us). It is built row by row with VJPs (`jacobian-rows-per-pass` = 16 in the J++ code). That needs about d/16 = 320 batched backward passes per prompt.

### R-lens (Blank, Bhatia, Nanda, 2026)
This is the same estimator, but the backward pass uses LRP rules. All of them are stop-gradients, and the forward output is unchanged.
- **LN-rule:** the RMSNorm denominator on the residual-stream norms is detached, which makes the norm linear.
- **Identity rule:** the nonlinear factor of SiLU is detached, so `silu(z) = z * stopgrad(sigmoid(z))` and the gradient is sigmoid(z).
- **Half rule:** relevance through the gated-MLP product `act(gate) * up` is split evenly between the two branches, so it is not counted twice.
- Attention, q/k norms, Gated DeltaNet and linear layers use the plain gradient. MoE routing weights are frozen.

### J++ lens (Ayonrinde & Lindsey, 2026)
J++ is still **one fixed linear map per layer**, so it adds no inference cost. It changes the J-lens in three ways:
1. **Jacobian Filtering** (the biggest gain):
   - Fit a per-layer router: PCA to 64 dimensions on centred, unit-normalised layer-l activations, then k-means with K = 8. The router is fit on 1,000 WikiText-103 sequences.
   - Fit one "expert" averaged Jacobian per cluster, using only the source positions routed to that cluster.
   - Set `J_l = Σ_k w_k J_l^{(k)}`. The signed weights `w_k` are learned with Adam (lr 0.05) on a small dev split of the latent-variable-extraction eval items (`fit-weights`, default fit-fraction 20%).
   - Experts with fewer than 50 kept positions fall back to the pooled Jacobian.
2. **LRP backward pass**, as in the R-lens above.
3. **Readout filtering:** before ranking, drop vocabulary tokens that contain no letter or digit. That is 6,116 tokens for the Qwen3.6 tokenizer. This is a post-hoc mask.

Fit recipe for Qwen3.6-27B, from Table 5:
- 64 WikiText-103 sequences × 128 tokens.
- Skip the first 16 positions (attention-sink outlier gradients) and the final position, leaving 111 fit positions per sequence (7,104 total).
- 7 source layers (8, 16, ..., 56) and target block 63.

Results on Qwen3.6-27B (recall@10 on 5 latent-variable tasks):
- J++ 55.2%, J-lens 35.7%, R-lens 37.7%.
- Ablations: without readout filter 52.3, without LRP 48.6, without Jacobian filtering 43.5.
- Causal swap success: J++ 59.7% vs J-lens 49.3%.

Reported cost: about 9 H200-GPU-hours in total (≈7.3 h for Jacobians, ≈1 h for clustering, ≈0.5 h to merge and fit weights). Wall clock was about 1 h on 16 GPUs.

## 2. Cost of a native J++ refit for Qwen3.8-27B, layer 40 only, on one H100

- Building the full J_40 takes 5120 VJP rows × 64 prompts ≈ 3.3e5 VJPs over 128 tokens, back-propagated through blocks 40–63 (≈10B params). That is roughly 4 × 10e9 × 128 × 3.3e5 ≈ 1.7e18 FLOP, or about 1.3 h at an optimistic 350 TFLOP/s sustained. Scaling the paper's 7.3 H200-h by depth (24 of about 56 layers) gives about 3 H200-h, which is roughly 4–5 H100-h. Expect **2–5 H100-hours**.
- On top of that:
  - Router fit: 1,000 forward passes to layer 40, a few minutes.
  - Expert-weight fitting: needs the jacobian-lens eval items plus forward passes.
  - Engineering: porting the LRP hooks to Qwen3.8 is probably easy, since the J++ repo already supports the same-architecture Qwen3.6.
- **Verdict: far over the 30 GPU-minute budget, so skip.**

## 3. Cheap options (recommended)

### A. Borrowed J++ lens (≈0 GPU-min)
1. Download `koayon/jpp-lenses/qwen3.6-27b/lens.pt`. This is a 6.6 GB single file: fp32 5120×5120 matrices for layers 0–62. Load it with `torch.load(..., weights_only=True)` and keep `parameters["jacobians"][40]` (≈105 MB). Layer 40 is one of the 7 layers that were J++-fitted.
2. Compute `logits = lm_head(final_norm(J_40 @ v))` with **Qwen3.8's** final norm and lm_head. Mask tokens that contain no `[A-Za-z0-9]` (or the Unicode letter/digit equivalent), recomputed from the Qwen3.8 tokenizer. Rank top and bottom tokens.
3. This is a CPU or GPU matvec for about 40 directions.

**Caveat:** this rests on the assumption that Qwen3.8-27B keeps Qwen3.6-27B's residual basis. The evidence is circumstantial: an identical architecture and config, and a 0.9/0.1 weight-merge that apparently works. That is consistent with shared pretraining lineage, but it is not confirmed. Two checks:
- **Weight check (CPU, needs a download):** compare cosine similarity of `embed_tokens`, `lm_head`, and the layer 40–63 weight tensors between the two checkpoints. Values near 1 support shared lineage; near 0 means the bases are unrelated, so do not use A.
- **Functional check (B below):** compare the borrowed `J_40 v` against Qwen3.8's own Jacobian-vector product.

### B. Native finite-difference J-lens JVP for our ~40 directions (≈5–15 GPU-min)
We only need `J_40 v` for about 40 vectors, not the full matrix, so we can use forward-mode JVPs. Because attention and DeltaNet are causal, perturbing **all** source positions s ∈ [16, T−2] at once by `ε v` and summing `Δh_63` over target positions gives `Σ_s Σ_{t≥s} J_{ts} v`. That is exactly the J-lens estimator up to a scalar, and the final RMSNorm in `unembed` makes the scalar irrelevant to rankings.

Recipe:
- Data: about 64 WikiText-103 sequences × 128 tokens.
- Cache `h_40` once.
- For each direction, run blocks 41–63 on `h_40 ± ε v·mask` (central difference).
- Compute `Jv ∝ Σ_t (h63⁺ − h63⁻)`, then sum over sequences.

Cost:
- About 40 × 64 × 2 ≈ 5k sequence-passes through 24 blocks, roughly 1e16 FLOP. That is about 1 min of ideal compute, or 5–15 min in practice.
- bf16 noise: pick ε so that ‖ε v‖ is a few percent of the typical ‖h_40‖. Alternatively, drop blocks 0–39 after caching and upcast blocks 40–63 to fp32 (≈42 GB, which fits on an H100).

What B gives us:
- (i) a native plain-gradient J-lens readout for each direction.
- (ii) the cosine and top-k overlap between it and the borrowed J++ `J_40 v` (and the Qwen3.6 J-lens from `camilablank/workspace-lenses`). That validates or invalidates A.

B does not include Jacobian filtering or LRP, so call it "J-lens (FD, native)", not J++.

**Optional, about 1–2 h of engineering, not GPU:** an R-lens-style native readout. In forward mode, stop-gradient means "freeze that factor at its clean-pass value". Cache the clean RMSNorm denominators and sigmoid(gate) values, and apply the half rule to `act(gate)*up` in the perturbed pass. Probably not worth it.

## 4. Gotchas for Qwen3.8-27B (hybrid Gated DeltaNet / Gated Attention, 3:1, 64 blocks)
- **Layer convention:** in jpp_lens, "layer l" is the **output of decoder block l**, which is HF `hidden_states[l+1]`. Our `agastya-repo/rh_probe/cluster/build_dom.py` uses `hs[1:]`, so L there is also the output of `model.layers[L]`, which matches. But `agastya-repo/probe/extract_and_score.py` (and the olmo copy) uses `hs[L]`, the output of block L−1. Check which script produced each direction, because they differ by one.
- **Forward-mode AD:** `torch.func.jvp` and double-backward will likely fail through the fla Triton `chunk_gated_delta_rule` and causal-conv1d kernels, which are custom autograd Functions without jvp. Use finite differences (B). The pure-torch HF fallback path may support forward-mode AD, but it is slow. Reverse-mode, which the J++ refit uses, works with fla installed.
- J++ uses the plain gradient through Gated DeltaNet and through attention output gates. LRP applies only to RMSNorms and the gated MLP.
- **Final norm:** decode with the model's own `model.model.language_model.norm` module, not a hand-rolled RMSNorm. Qwen3-Next/3.5-family norms may use a `(1 + weight)` parameterisation; I have not verified this for 3.8. The VLM wrapper is `Qwen3_5ForConditionalGeneration`. Ignore the MTP head.
- **Positions:** skip the first 16 positions as sources (attention sinks). With chat-formatted data the sink and template tokens matter more. WikiText is the paper's fit corpus, so use it for comparability.
- **Readout filter:** recompute the filter from the Qwen3.8 tokenizer. The two tokenizers are reportedly identical, but don't assume it.
- **Inputs:** the lens was validated on activations, not on difference-of-means directions. Because the lens is linear, `J(h_a − h_b) = J h_a − J h_b` is well defined, and bottom tokens are meaningful in the same way as with the logit lens.

## 5. Recommendation
- **Include:** (A) the borrowed Qwen3.6 J++ J_40 plus readout filter, about 0 GPU-min plus a 6.6 GB download, and (B) the native FD J-lens JVP check, about 5–15 GPU-min. Together they stay well under 30 GPU-min.
- **Decision rule for A:** report A as "J++ (transferred)" only if its readouts agree reasonably well with B's native readouts (for example, the median cosine of `J_40 v` is clearly above the cosine for shuffled directions), or if the weight-similarity check shows shared lineage. Otherwise report only the native FD J-lens and the logit lens, and record that J++ was skipped.
- **Skip:** a native J++ refit. It needs about 2–5 H100-hours of VJPs, plus clustering and dev-set weight fitting.

## Sources
- Gurnee et al. 2026, Verbalizable Representations Form a Global Workspace in Language Models: https://arxiv.org/abs/2607.15495 ; code https://github.com/anthropics/jacobian-lens
- Blank, Bhatia, Nanda 2026, R-lens: https://www.alignmentforum.org/posts/nv8oedrnLXKRzNEL9 ; lenses https://huggingface.co/camilablank/workspace-lenses (J- and R-lenses for Qwen3.6-27B, Qwen3.5-27B, and others; no Qwen3.8)
- Ayonrinde & Lindsey 2026, J++ Lens: https://www.lesswrong.com/posts/nc9dHfcB22JdMzGbr/paper-j-lens-jacobian-filtering-enables-more-faithful ; code https://github.com/safety-research/jpp_lens ; lenses https://huggingface.co/koayon/jpp-lenses (Qwen3.6-27B, Qwen3.5-9B, gemma-4-31B, Olmo-3-32B, Qwen3.5-122B-A10B, DeepSeek-V4-Flash; no Qwen3.8)
- A related but distinct paper: Yan et al. 2026, Short Horizons and Sparse Concepts (top-energy position-pair filtering of J-lens Jacobians), https://arxiv.org/abs/2608.25347
- Qwen3.8 vs 3.6 architecture (secondary sources): https://codersera.com/blog/qwen-3-8-vs-qwen-3-6-2026/ ; merge https://featherless.ai/models/vvsotnikov/Qwen3.8-27B-0.9-Qwen3.6-27B-0.1
