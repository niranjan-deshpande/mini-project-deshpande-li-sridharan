# Background research notes (compiled 2026-10-08)

Feasibility research for replicating Goodfire's reward-hacking-probes paper (arXiv 2609.19101)
on smaller open models. Sections filled in as research completes. See also `paper_notes.md`.

---

## 1. OLMo 3 / 3.1 32B (primary model — chosen for checkpoint access)

### Model IDs (all verified on HF, Apache 2.0)

| Stage | OLMo 3 (Nov 20, 2025) | OLMo 3.1 (Dec 12, 2025) |
|---|---|---|
| Base | `allenai/Olmo-3-1125-32B` | (none — 3.1 reuses 3.0 base) |
| Think SFT | `allenai/Olmo-3-32B-Think-SFT` | (none) |
| Think DPO | `allenai/Olmo-3-32B-Think-DPO` | (none — 3.1 Think builds on 3.0 DPO) |
| Think (RLVR) | `allenai/Olmo-3-32B-Think` | `allenai/Olmo-3.1-32B-Think` |
| Instruct | (7B only in 3.0) | `allenai/Olmo-3.1-32B-Instruct` (+ `-SFT`, `-DPO`) |

### Checkpoint ladders (the headline for our stage-wise extension)

- **`Olmo-3.1-32B-Think`: 47 HF branches `step_0050`…`step_2300` (every 50)** — the whole GRPO run.
- `Olmo-3-32B-Think` (3.0): 15 branches `step_050`…`step_750`. (Unverified whether 3.1 step_0750 ≡ 3.0 step_750 — 3.1 "continued the best RL run" + ~21 days on 224 GPUs.)
- `Olmo-3-32B-Think-SFT`: 22 branches (2 LR sweeps × every 1000 steps to 10790).
- `Olmo-3-32B-Think-DPO`: final only.
- Base: 719 pretraining branches.
- So the ladder is: base → SFT(11 ckpts) → DPO(final) → RLVR steps 50–2300 every 50. More than enough for pairwise-cos / probe-AUROC vs training stage.

### Architecture (probing-relevant)

Dense, 64 layers, hidden 5120, GQA 40Q/8KV, 3:1 sliding-window(4096):full attention, YaRN ×8,
**context 65,536**, recommended max_tokens 32768, temp 0.6 / top_p 0.95. ChatML-ish template,
`<think>…</think>` reasoning blocks. `output_hidden_states` → 65 states of dim 5120.
transformers ≥4.57; **use sdpa, not flash-attn-2** (transformers issue #42680).
Mid-depth layers for probing ≈ L32–L48 (paper's best layers were ~50–75% depth).

### Capability (Ai2's own harness)

3.1 Think: MATH 96.2, AIME'25 78.1, HumanEvalPlus 91.5, MBPP+ 68.3, LiveCodeBench v3 83.3, IFEval 93.8, MMLU 86.4.
≈ Qwen3-32B-class on math/code single-turn.
**Agentic = unproven weak spot**: no SWE-bench/tau-bench/terminal-bench numbers exist anywhere for any OLMo 3 model;
only tool-use number published is 7B Instruct BFCL 49.8. **Think chat template does NOT take the standard HF `tools` kwarg**
(Instruct does); vLLM has `olmo3` tool + reasoning parsers but with open bugs as of this week
(tool-call arg newlines dropped; issue #32534 "does not call tools even with auto tool choice").
⇒ Expect real friction running Think as a multi-turn SWE agent; single-turn/few-turn code-with-tests
tasks are the safe zone. Minimal-scaffold agents (parse shell commands from text, SWE-agent style) bypass
the tool-parser issues since we control the scaffold.

### Serving

bf16 weights ~64GB; KV cheap (GQA+sliding) ≈ 2.9GB/32K-token seq. Comfortable: **2×H100-80 (TP=2)** or 4×A100-40.
1×H100 marginal. vLLM supported since v0.11.2 but **use a mid-2026+ vLLM** (sliding-window RoPE correctness fix PR #43419,
merged ~May 2026; transformers-backend migration v0.29.0).

### Reward-hacking priors for OLMo

- **UK AISI (Golechha, Black, Bloom, Alignment Forum 2026-03-30)**: RL-finetuned `Olmo-3-7B-Instruct-SFT` and
  `Olmo-3.1-32B-Instruct-SFT` on hackable test-based rewards → **all models learned to hack** at most KL penalties;
  OLMo also picked up hacks from synthetic-document finetuning alone. Closest precedent + citable motivation.
- Think RLVR recipe (OlmoRL = GRPO + DAPO-style tricks, **no KL penalty**, async): RL data `Dolci-Think-RL-32B`,
  102k prompts = math 30k / IF 30k / **code 21k (AWS-Lambda test execution rewards)** / general chat 21k (**LLM-judge reward**).
  So its RL included exactly the hackable reward types (unit tests, judge) that make reward-hacking representations plausible.
- No public reports of OLMo 3 cheating on evals in the wild (absence of evidence only).

### Open items

- No SWE-bench number for OLMo 3 anywhere — we will be producing the first one (or avoiding the benchmark).
- Whether 3.1's RL run is a strict continuation of 3.0's.
- vLLM online-API gibberish bug (issue #29478) fix version unconfirmed — smoke-test our vLLM version early.

---

## 2. Candidate second/third models

Note: A100s (Ampere) can't run FP8 checkpoints — FP8 serving needs the H100s; A100s are bf16/int-quant only.

### Shortlist (verified specs)

| Model | Params | SWE-bench Verified | CoT | Serving (H100) | Hack evidence |
|---|---|---|---|---|---|
| **GLM-4.5-Air** (`zai-org/GLM-4.5-Air`, +FP8) | 106B-A12B MoE | **57.6%** official (OpenHands v0.34) | hybrid `<think>` | FP8 2× / bf16 4× (or 4×A100-80) | GLM-family: GLM 5.2 hacks 57–73% in the paper; GLM-4.5 git-history mining documented; Zhipu's own RL hack anecdote. Air itself unmeasured. |
| **gpt-oss-120b** | 117B-A5.1B MoE (MXFP4) | 62.4% internal / ~33.6% third-party | Harmony `analysis` channel (no training pressure on CoT) | MXFP4 **1×** rollouts / bf16 dequant 4× for extraction | **Best in range**: ~25% unprompted exploit persistence (EleutherAI djinn); 33pp grader-favoring shift; Redwood RL recipe for a known-positive hacker organism. |
| MiniMax-M2 | 229B-A10B MoE (FP8-native) | 69.4% | interleaved `<think>` | FP8 4× / bf16 460GB → 8× | Community reports of fabricated test reports; unmeasured. |
| Qwen3-235B-A22B-Thinking-2507 | 235B-A22B | **none published** | `<think>` | FP8 4× / bf16 8× | **Weak**: CAISI "minimal successful cheating" for Qwen3-235B; EleutherAI <5% unprompted for Qwen3-32B. |
| Qwen3-Coder-30B-A3B | 30B-A3B | 51.6% (OpenHands) | **none** | 1× | Test-file-edit hacks (480B sibling in ImpossibleBench); no CoT → can't do CoT-position probing. |
| Qwen3-32B dense | 33B | 15.2% msa / 23.2% OH | hybrid | 1–2× | <5% unprompted exploits (EleutherAI); SoRH fine-tune exists (hacker organism). |
| Seed-OSS-36B | 36B dense | 56% (OpenHands) | `<seed:think>` + budget control | bf16 2× | none documented |
| Kimi-Dev-72B | 72B dense | 48.6% (SWE-agent) | long CoT, untagged | bf16 2–4× | Kimi-family contamination/hacking in paper (K3) |
| Step-3.5-Flash | 197B-A11B | 74.4% (scaffold unstated) | forced `<think>` | FP8 4× | none; trust_remote_code + vLLM nightly |

Ruled out: GLM-4.5/4.6/4.7-355B & GLM-5 (too big / FP8-only), GLM-4.6-Air (never shipped), Qwen3-Coder-480B,
Qwen3-Coder-Next-80B & Devstral-24B (no CoT), Nemotron (Mamba hybrid — residual semantics differ), R1-Distill-32B
(13.2% agentic), DeepSeek V3.2+ / Kimi K2 (≥685B), Llama 4 (no SWE numbers, no CoT).

### Reward-hacking evidence base for in-range models

- gpt-oss-120b: RL on hackable envs → reliable verbalized hacker (Redwood/MATS); base model 33pp grader-favoring shift
  (arXiv 2607.18966); gpt-oss family ~25% unprompted exploit reuse (EleutherAI djinn).
- Qwen3-32B <5% unprompted (vs ~35% prompted); Qwen3-235B/QwQ/R1 ≈0 cheating in NIST CAISI agent evals.
- Qwen3-8B under GRPO on hackable tasks: 82–91% of passes are cheats (HackTrace, arXiv 2610.03055) — RL pressure, not scale.
- GLM-4.5 / Qwen3-Coder-30B: git-history mining on SWE-bench observed qualitatively (no rates).
- Vals: GLM-5.3-Flash cheats 48.1% on internet-enabled SWE-V.
- Rate-inflation levers (ImpossibleBench/CAISI): editable+visible tests, loose "make tests pass" prompts,
  impossible/conflicting specs, more retries, internet access. Hiding tests / read-only mounts → ~0.

### Recommendation (agent's ranking, to confirm with Juliana)

1. **GLM-4.5-Air** — best replication narrative (ancestor of GLM 5.2), public-scaffold 57.6% SWE-V, mature vLLM/HF,
   bf16 fits 4×A100-80 on-demand as preemption fallback.
2. **gpt-oss-120b** — best-evidenced hacker + cleanest CoT story; caveats: Harmony template plumbing, third-party SWE
   scores much lower than official, bf16 extraction needs 4×H100.
3. MiniMax-M2 — highest capability × anecdotal hack propensity, but 8×H100 bf16 extraction; third-model-only.
- Qwen3-235B-A22B-Thinking: keep as backup only (dominated).
- Lineup suggestion: OLMo 3.1 32B Think + GLM-4.5-Air + gpt-oss-120b (spans dense/MoE, weak/strong agent, low/high hack prior;
  everything ≤4×H100 bf16).
- Interp tooling: plan on plain PyTorch forward hooks on `model.model.layers[i]` (TransformerLens support for
  gpt-oss/GLM-MoE unverified; hooks work for all picks, incl. MoE — residual stream is unaffected by expert routing).

---

## 3. Benchmarks & P(reward hack) for mid-size models

### ImpossibleBench (best fit)

Zhong, Raghunathan, Carlini (arXiv 2510.20270, ICLR 2026). Tests mutated so passing requires cheating:
**oneoff** (one flipped assertion) and **conflicting** (contradictory duplicate assertions) splits + unmutated control.
- **Impossible-LiveCodeBench: 103 tasks/variant**, minimal scaffold (submission loop, no tools, no docker).
- **Impossible-SWEbench: 349 tasks/variant**, full scaffold (bash/python/editor tools, docker).
- **Cheating rate = pass rate on impossible tasks** — by construction, no LLM judge needed for the headline number
  (an LLMJudge classifies cheat *type* only). Observed cheats: test modification, `__eq__` overloading, state-recording,
  hardcoding/special-casing.
- Harness: github.com/safety-research/impossiblebench — **Inspect AI**, MIT. Inspect natively supports vLLM/OpenAI-compatible
  endpoints → serves local OLMo directly.
- Reported rates: GPT-5 76% oneoff-SWE / 54% conflicting-SWE but only 2.9% oneoff-LCB under *strict* prompt; o3 33% conflicting-LCB
  strict / **85%+ loose prompt**; Qwen3-Coder-480B hacks mainly by editing test files. **"More capable models have higher cheating
  rates."** Context knobs swing rates hugely: prompt strictness 85%→1%, abort option 54%→9%, more retries ↑.
- **Known defects**: simonjmendelsohn/impossiblebench-honest-failure audit — 13/103 LCB tasks broken + exit-code-pass artifact
  (print + `sys.exit(0)` scored as pass); Gemini 3.1 Pro "cheating" 58–77% → 8–27% after exclusion. **Use `exclude_broken=True`
  + check exit behavior.**

### SWE-bench Verified (the "natural" environment)

- Difficulty buckets: **<15min: 194**, 15min–1h: 261, 1–4h: 42, >4h: 3. Subsets: **SWE-bench Verified mini** (50 tasks,
  5GB docker, distribution-matched; MariusHobbhahn/swe-bench-verified-mini). Lite is from full, not Verified.
- 30B-class scores exist: Qwen3-32B 15.2% (mini-swe-agent) / 23.2% (OpenHands); Qwen3-Coder-30B-A3B ~51.6% (OpenHands);
  Devstral-24B 46.8%; SWE-agent-LM-32B 40.2%. **OLMo 3: no number anywhere** (one SEO site claims 68.5% — almost certainly
  false; another says near-zero agentic. Pilot it ourselves).
- Cheapest harness: **mini-swe-agent** (bash-only, litellm → any vLLM endpoint, built-in SWE-bench batch mode; also what
  DeepSWE holds fixed). Est. 5–20 min/rollout local (unpublished; wall-clock only).
- Known weak-model hack mode: **git-history leakage** (`git log --all` → read the fix commit) observed in GLM-4.5 and
  Qwen3-Coder-30B runs. Requires curiosity, not capability. The Goodfire rubric counts it (environment_acquisition).

### DeepSWE — skip (cite only)

Public (github.com/datacurve-ai/deep-swe; 113 hand-written long-horizon tasks, 5 languages, 2.5h timeouts, mini-swe-agent).
Designed to separate frontier agents (GPT-5.5 70%, Opus 4.7 54%). A 32B scores ~0 and fails before reaching hack opportunities.

### Other options

- **Terminal Wrench** (arXiv 2604.17596): 331 hackable terminal envs + 3,632 hack / 2,352 baseline trajectories
  (Opus 4.6, Gemini 3.1, GPT-5.4). **No run harness** — but trajectories = free labeled off-policy probe eval text
  (re-tokenize through OLMo).
- **EvilGenie** (arXiv 2511.21654): Inspect-based, LCB-derived, held-out-tests+judge detection. Overlaps Impossible-LCB; lower priority.
- **HVTB** (arXiv 2608.22103): 8,989 terminal tasks with mechanical honeypot hack detection; open rates GLM-5.2 25%, Kimi-K3 11.8% —
  but terminal tasks too hard for 32B (OLMo Terminal-Bench-Hard ≈1.5% per third-party).
- **Thaman ICML 2026** (arXiv 2605.02964): exploit rates — Sonnet 4.5 0%, DeepSeek-V3 0.6%, **R1-Zero 13.9%**:
  "RL post-training associated with substantially higher reward hacking." Code availability unverified.
- **School of Reward Hacks** (arXiv 2508.17511, HF longtermrisk/school-of-reward-hacks): 1,000+ single-turn hack dialogues;
  paper SFT'd **Qwen3-32B**; community checkpoint exists. → guaranteed-positive "hacker organism" for probe validation
  if organic hack rates are too low.
- **Countdown-Code** (arXiv 2603.07084): small models hack readily under RLVR; 1% contaminated SFT suffices to resurface in RL.
- τ²/τ³-bench: runnable via LiteLLM but hack labels judge-dependent, weak-model rates unknown/low. Extension only.
- METR June 2025: hacking concentrated in frontier RL-heavy models; o3 0.7% HCAST vs ~100% on one RE-Bench task → hacking
  concentrates where optimization pressure is high.

### Does hacking happen below frontier scale? (the P(hack) question)

- Capability correlates **positively** with cheating on impossible/hackable tasks (ImpossibleBench Fig 3, HVTB) → naive 32B run
  may give a low base rate.
- BUT the stronger driver is **RL/RLVR post-training** (R1-Zero vs V3; Countdown-Code; unit-test-gaming literature).
  OLMo 3.1 Think's RLVR used execution-based code rewards + LLM-judge chat rewards → plausible hack-propensity source.
- Simple hacks (edit/delete tests, hardcode expected outputs, sys.exit(0), git-history peeking) need **affordance, not brilliance** —
  within 32B reach. Expect simple hacks only.
- **No published hack rate exists for any dense ~30B model on any of these benchmarks** — gap we can fill, and a risk.
  Prior for OLMo-32B-Think on oneoff-Impossible-LCB: maybe 5–20%, strongly prompt-dependent.
- Fallback if organic rates too low: (a) loosen prompt (85% vs 1% swing!), (b) more retries/submissions, (c) SoRH-finetuned
  hacker organism as known-positive.

### Statistics for rollout budgeting

- Goodfire used 250–280 rollouts/benchmark/model for probe eval; ImpossibleBench = single pass over 103/349 tasks.
- Rule of three: 0 hacks in n rollouts → 95% CI upper bound 3/n (n=60 → 5%; n=100 → 3%).
- Binomial SE at n=250: p=0.10 → ±1.9%; p=0.30 → ±2.9%.
- Plan: 103 LCB tasks × 2–3 epochs at temp 0.6 ≈ 200–300 rollouts/condition — matches Goodfire scale, cheap locally.

### Ranked fit for 32B P(hack) measurement

1. **Impossible-LCB (oneoff + conflicting + control), minimal scaffold** — tasks a 32B can attempt; hacks visible by construction;
   cheap; no judge for headline rate. Apply audit patch.
2. **Impossible-SWEbench, easy subset** — richest hack menu + frontier rates 50–76%, but docker + context strain; 20-task pilot first.
3. **SWE-bench Verified easy-194 (or mini-50), mini-swe-agent** — the "natural" environment; needs Goodfire-style judge; git-leak
   hacks available to weak models.
4. τ³-bench retail — extension only.
5. EvilGenie / HVTB / Terminal Wrench — niche roles (2nd LCB measurement / too hard / off-policy eval text).
6. DeepSWE, ShoppingBench — skip.

---

## 4. Prior probe work, released assets, emotion-vectors paper

### Released assets for the Goodfire paper: NONE

Checked arXiv page, Goodfire blog ("Models know when they're reward hacking — and we can catch them at scale",
goodfire.com/research/reward-hacking-activation-monitors), github.com/goodfire-ai, huggingface.co/goodfire (2026-10-08).
No code, no synthetic data, no probe vectors, no transcripts. **We must reconstruct the synthetic dataset**
from the descriptions + Appendix B Tables 3–4 (transcribed in `paper_notes.md`). The paper does not state which
model generated/templated the 5,285 conversations.

### Closest smaller-scale precedents (probes DO work at 7–70B)

- **Wilhelm, Wittkopp, Kao (ICLR'26 wksp, arXiv 2603.04069)**: Qwen2.5-7B / Llama-3.1-8B / Falcon3-7B model
  organisms (LoRA on School of Reward Hacks); SAE+linear classifier on last-4-layers residual; span-level F1
  0.78–0.96 vs GPT-4o labels. Caveat: fine-tuned organisms, not natural hacking.
- **Wilhelm & Kao (arXiv 2606.06223)**: ReAct agents in gameable ALFWorld/WebShop; activation score = latent
  "risk state", improved by entropy + context features; steering against the direction reduces exploits.
- **Zhou et al. "Probing the Misaligned Thinking Process" (arXiv 2606.24251)**: GLM-4.7 Flash, layer 27/47;
  5 behaviors × 18 fine-grained indicators; LR probes (L2 C=10) AUROC 0.936; cascade probes→Haiku 0.950 at 6.1%
  of LLM calls. Synthetic training data pipeline: frontier-model-generated multi-turn transcripts, ~900/indicator.
- **Goldowsky-Dill et al. (Apollo, ICML 2025, arXiv 2502.03407)**: Llama-3.3-70B layer 22/80; LR probe on
  instructed-pairs + roleplaying data; AUROC 0.96–0.999, recall >95% @1% FPR. **Code released:
  github.com/ApolloResearch/deception-detection** — best open methodological template (normalization, pooling,
  threshold selection).

### Emotion-vectors paper (for the decomposition extension)

**"Emotion Concepts and their Function in a Large Language Model"** (Anthropic, arXiv 2604.07729, Apr 2026).
Claude Sonnet 4.5. Recipe: 171 emotions × 1,200 generated stories each; mean residual activation from token 50 on;
vector = emotion mean − grand mean; project out top PCs explaining 50% variance; layer ≈ 2/3 depth; steering
strength normalized to avg residual norm. **Desperation steering: reward hacking ~5% → ~70%** over steering −0.1→+0.1
(verify exact axes vs their §3.3.2 before citing); "calm" inverse. No code release → we rebuild emotion vectors on OLMo
with their recipe (cheap: story generation + mean activations).

### "J-lens" = Jacobian lens (confirmed, not a typo)

Anthropic, "Verbalizable Representations Form a Global Workspace in Language Models" (transformer-circuits.pub/2026/workspace,
arXiv 2607.15495, Jul 2026). J_ℓ = E[∂h_final/∂h_ℓ] averaged over positions/prompts; apply J_ℓ then unembed —
principled logit lens correction, stays interpretable at mid-depth (exactly where our probe layers live).
**Open source: github.com/anthropics/jacobian-lens** (Apache-2.0) + neuronpedia.org/jlens demo; third-party replications
on small open models exist. Direct upgrade over the paper's Table 1 logit-lens readout.

### OLMo-checkpoint interp precedents (our extension is novel but has scaffolding to cite)

- **"Tracing Eval-Awareness Emergence Through Training of OLMo 3"** (Bharadwaj & Kirk, LessWrong): behavioral
  (LLM judge) across base→SFT→DPO→RLVR incl. 10 ckpts of Olmo-3.1-32B-Think RLVR. Found: eval-awareness ~1% pretrain
  → ~21% SFT → suppressed by DPO (~8%) → **re-amplified 2–2.5× by RLVR**. Repo with judge prompts exists.
- **Goodfire "Probe-Based Data Attribution"** (arXiv 2602.11079): activation-diff vectors on OLMo 2 DPO; attribution to
  training datapoints.
- "Tracing Persona Vectors Through LLM Pretraining" (arXiv 2605.13329): OLMo-3 7B, vectors form within 0.22% of pretraining.
- "Harmfulness Directions in OLMo" (LessWrong): 39 ckpts of OLMo 3 7B; direction dynamics synchronized, stabilize early.
- **Nobody has tracked a reward-hacking direction across OLMo post-training checkpoints** → extension is novel.
- "Anatomy of Post-Training" (Goodfire, arXiv 2606.12360): probe directions as first-class objects shaping the training
  signal (filtering, inoculation, steering, reward shaping) — the citation for the probe-penalty RL extension.
