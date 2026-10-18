# Implementation plan (living document)

The brief (`../claude_code_prompt.md`) is unchanged. This file tracks where the plan deviates from it or fills in details, as agreed with Niranjan. Reasons for each choice are in `DECISIONS.md`; spending is in `COSTS.md`.

## Changes to the brief

| Area | Brief | Now |
|---|---|---|
| Generator | GPT-5.6 Terra for everything | One generator for everything, chosen by a 2-arm pilot: GPT-6 Luna (high effort) vs GPT-5.6 Terra. Same prompts and task seeds, blind-graded by DeepSeek V4 Pro (a third family, so neither arm grades itself). Pick Luna if it's within ~5 points of Terra on auto-check pass rate and contamination rate and isn't noticeably more templated. (Claude subagents dropped: nonstandard and hard to document.) |
| Pair counts | ~same n as `pairs_full`, ~200 per candidate | Fixed: 300 hack pairs; 200 per candidate, baseline and general-cheating format. No adaptive sizing. Reliability is reported after the GPU pass. |
| Contamination judge | GPT-6 Luna | A model from a different family than the chosen generator: DeepSeek V4 Pro if the generator is Luna; Luna is fine if the generator is Terra or Claude. Every boundary pair is judged once; flagged pairs are re-judged twice. |
| Phase 2 second judge | Flash-tier Gemini or DeepSeek | DeepSeek V4 Pro. Pilot on ~20 rollouts and check κ first, then the rest. The original split is optional; it doesn't feed any Phase 2 analysis. |
| Label-shuffled null | unspecified | Balanced sign flips within pairs (exactly half swapped). |
| Lenses | Logit lens + J++ lens | Logit lens; native J-lens and native R-lens readouts for every direction (per-direction finite differences at L40, ~5–15 GPU-min); J++ borrowed from the published Qwen3.6-27B lens, reported only if it agrees with the native readouts. No native J++ refit (stretch goal after CHECKPOINT D at most). |
| Spending | caps $60 / $80 | Caps, not targets. Pilot first, then stop once an answer is clear. |

## Order of work
1. **CHECKPOINT 0** (CPU, in progress): AUROC reproduction ✅, split-half reliability ✅, token counts ✅, layer indexing ✅, pricing ✅, analysis library ✅, GPU efficiency research (running), generation prompts and rubrics (running).
2. **Generation pilot**: ~20 pairs × 5 hardest directions × 2 generators (Luna, Terra), plus a Phase 2 rubric pilot on ~20 rollouts with both judges. → **CHECKPOINT A**
3. **Bulk generation** (Batch API if an API generator wins) + contamination judging + Phase 2 labels. → **CHECKPOINT B**
4. **GPU session 1** (one unattended script, tested end to end on a small same-architecture model first): pipeline check vs Agastya's vector and scores; pair activations (L40 turn/span mean plus a few lens layers); rollout pass with per-token projections onto all directions and the L40 fp16 cache (~96 GB of assistant tokens) on the network volume; native J-/R-lens readouts; manifest; self-terminate.
5. **CHECKPOINT C** (selection half): the new `v_hack` vs Claude `v_hack`.
6. **Phase 4 analyses** (CPU, `rhdecomp`). → **CHECKPOINT D**
7. Optional Phase 5 items, each with separate approval.

## Current estimates
- **API:** ~$55 if the generator is Terra. Much less with Luna (~$5–10 for generation). Judging is ~$8 in total.
- **GPU:** ~1.5–2.5 H100-hours, budget 3–4 (≈ $10–15 incl. a network volume of ~150 GB). Measured inputs: rollouts total 12.0M tokens (max 99k, 9.36M assistant tokens).
- **GPU recipe** (from `notes/gpu_research.md`): load only blocks 0–40 (`num_hidden_layers=41`, keep `norm` + untied `lm_head` for lenses; 36 GB of weights); plain forwards, right padding with masks, never packing; no prefix sharing (exact but saves only ~15 min); `transformers==5.19.0` + `flash-linear-attention==0.5.2` (fail if the torch fallback activates); sdpa limited to flash/efficient backends; H100 80GB is enough. Test end to end on `Qwen/Qwen3.5-0.8B` (plumbing) and `Qwen/Qwen3.5-9B` (untied head, grouped heads) first.
