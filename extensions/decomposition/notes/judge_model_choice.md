# Judge model choice (researched 2026-10-13)

Question: is DeepSeek V4 Pro still the cheapest judge on the cost/capability Pareto frontier for (1) Phase 2 rollout labeling and (2) pair-contamination judging?
Excluded families: OpenAI (Luna is the other judge and the generator), Qwen (under study), GLM, gpt-oss.
Sources: only public pages. No API keys were used and no paid calls were made.

## Recommendation (short)
| Job | Pick | Fallback |
|---|---|---|
| 1. Rollout labeling (271 transcripts, ~45k median) | **Claude Haiku 5.5** (`anthropic/claude-haiku-5.5`), effort high or max, provider pinned to Anthropic/Vertex/Azure. ≈ $1.2–5 total | **DeepSeek V4.1 Flash** (`deepseek/deepseek-v4.1-flash`). ≈ $2–7 depending on provider |
| 2. Pair contamination (~1.5k × 5k tokens) | **Claude Haiku 5.5** (same model). ≈ $1–2.4 | **MiMo-V2.6-Pro** (`xiaomi/mimo-v2.6-pro`), ≈ $6.6. If budget allows, use Sonnet 5.5 batch (≈ $24) only for re-judging flagged pairs |

The DeepSeek V4 Pro you planned (the 0423 release) is **off the frontier**. Haiku 5.5 is cheaper and scores higher on every Artificial Analysis metric that matters here. Its $0.40/$0.81 OpenRouter price is also a promotional discount from two fp8 resellers (Baidu, StreamLake, about 76% off; see below).

## Live OpenRouter data (pulled 2026-10-13 from `/api/v1/models` and `/api/v1/models/{id}/endpoints`)
"SO" means the endpoint lists `structured_outputs` in its `supported_parameters`.

| Model (OR id) | $/M in | $/M out | Cache read | Batch / discount | Ctx | SO by provider |
|---|---|---|---|---|---|---|
| deepseek/deepseek-v4-pro (0423, the plan) | 0.404 | 0.808 | 0.034 | Price exists only on Baidu fp8 / StreamLake fp8 with `discount`≈0.76 (promo). Next cheapest is Parasail 0.45/3.48, then ≥$0.96/1.9 | 1M | Baidu, StreamLake, Parasail, Reka, Cloudflare, DeepInfra, Alibaba, Venice, AtlasCloud: yes. GMI, DigitalOcean, SiliconFlow, Novita, Azure: no |
| deepseek/deepseek-v4-pro-0813 (GA) | 0.25 (Baidu promo, 81% off) | 0.75 | 0.008 | DeepSeek first-party 0.66/1.98 off-peak, 1.32/3.96 peak; **no SO** on first-party | 1M | Baidu, Wafer, Reka, Sail: yes. DeepSeek, StreamLake, Novita, GMI: no |
| deepseek/deepseek-v4.1-flash | 0.044 (OpenInference fp4) | 0.765 | 0.005 | `:batch` 0.112/0.336. Unquantized-ish: DeepInfra fp8 0.14/0.42, Fireworks/Together 0.30/1.20. DeepSeek first-party 0.15/0.60 off-peak, no SO | 1M | Most yes. DeepSeek, Relace, StreamLake, Alibaba, Novita, GMI, Phala, SiliconFlow: no |
| anthropic/claude-haiku-5.5 (released 2026-10-07) | 0.10 | 0.50 | 0.01 | **Prompts >100k tokens: 0.50/2.50.** Batch 0.05/0.25 (>100k: 0.25/1.25) | 1M | Anthropic, Vertex, Azure: yes. **Bedrock and Claude-on-AWS: no** |
| anthropic/claude-sonnet-5.5 | 2.00 | 10.00 | 0.10 | Batch 1.00/5.00, flat to 1M | 1M | Anthropic, Vertex, Azure: yes. Bedrock: no |
| xiaomi/mimo-v2.6-pro | 0.435 | 0.87 | 0.004 | none | 1M | Xiaomi, DeepInfra, GMI: yes. Novita: no |
| xiaomi/mimo-v2.6-flash | 0.14 | 0.28 | 0.003 | none | 1M | mostly yes |
| google/gemini-3.8-flash | 0.75 | 3.75 | 0.075 | flex/batch 0.375/1.875 | 1M | yes (AI Studio, Vertex) |
| google/gemini-3.5-flash-lite | 0.30 | 2.50 | 0.03 | flex 0.15/1.25 | 1M | yes |
| google/gemini-4-argon | 2.00 | 10.00 | 0.10 | flex 1.00/5.00 | 1M | yes |
| moonshotai/kimi-k3 | 0.73 (fp4) to 3.00 | 10–15 | 0.3 | none | 1M | mostly yes |
| meta/muse-spark-1.3 | 1.25 | 4.25 | 0.15 | none | 1M | yes (Meta only) |
| minimax/minimax-m3 | 0.23–0.30 | 0.96–1.20 | 0.06 | none | 0.26–1M | only CoreWeave fp4 and Together |
| mistralai/mistral-large-4-0 | 0.68 | 2.09 | 0.07 | none | 1M | yes |
| x-ai/grok-4.3 | 1.25 | 2.50 | 0.2 | batch 1.0/2.0 | 1M | yes |

## Capability evidence: Artificial Analysis
Data comes from the JSON embedded in https://artificialanalysis.ai/leaderboards/models and https://artificialanalysis.ai/evaluations/artificial-analysis-long-context-reasoning, fetched 2026-10-13 (Intelligence Index v4.3.2).
- AA-LCR: 100 questions over roughly 100k-token multi-document inputs, so ±~4–5 points is noise.
- MLCR-AA (Wisedocs medical long-context reasoning): synthesis over fragmented 25–64k-token records, scored for completeness and accuracy with a conciseness gate. It is the closest public proxy for "label every passage in a 45k transcript."
- Non-halluc: the AA-Omniscience non-hallucination rate (abstains rather than fabricating).

| Model (AA variant) | Intelligence Index | AA-LCR | MLCR-AA | IFBench | HLE | Non-halluc | Job 1 $ (9M in / 2M out) | Job 2 $ |
|---|---|---|---|---|---|---|---|---|
| DeepSeek V4 Pro 0424 (Max), **current plan** | 30.4 | 0.747 | n/a | 0.765 | 0.375 | 0.06 | 5.2 (promo) / 10.9 (Parasail) | 5.8 / 11.1 |
| DeepSeek V4 Pro 0813 (Max) | 36.0 | 0.803 | n/a | n/a | 0.410 | 0.05 | ~4 (Baidu promo) | ~4 |
| DeepSeek V4.1 Flash (Max) | 39.5 | 0.840 | 0.228 | n/a | 0.392 | 0.035 | 1.9 (fp4) / 5.1 (Fireworks) | 1.9 / 5.4 |
| **Claude Haiku 5.5 (Max)** | 43.4 | 0.827 | **0.633** | n/a | 0.444 | 0.60 | **2.4 / 1.2 batch** | **2.4 / 1.2 batch** |
| Claude Haiku 5.5 (High) | 37.8 | 0.773 | n/a | n/a | 0.373 | 0.55 | same | same |
| MiMo-V2.6-Pro | 46.3 | 0.863 | 0.183 | n/a | 0.494 | 0.59 | 5.9 | 6.6 |
| MiMo-V2.6-Flash | 37.9 | 0.743 | n/a | n/a | 0.351 | 0.46 | 1.9 | 2.1 |
| Gemini 3.8 Flash (High) | 40.9 | 0.813 | 0.217 | n/a | 0.478 | 0.45 | 14.9 / 7.4 flex | 15.5 / 7.8 |
| Gemini 3.5 Flash-Lite | 22.2 | 0.760 | n/a | n/a | 0.188 | 0.66 | 7.9 | 8.0 |
| Claude Sonnet 5.5 (Max) | 56.0 | 0.827 | 0.750 | n/a | 0.550 | 0.53 | 46 / 23 batch | 48 / 24 batch |
| Kimi K3 (Max) | 43.6 | 0.887 | 0.383 | n/a | 0.469 | 0.47 | 26 | 26 |
| Muse Spark 1.3 (Max) | 48.1 | 0.830 | 0.433 | n/a | 0.487 | 0.67 | 21 | 22 |
| Gemini 4 Argon (High) | 52.6 | 0.797 | n/a | n/a | 0.571 | 0.85 | 20 (flex) | 21 |
| *(reference) GPT-6 Luna (Max)* | 38.1 | 0.833 | 0.161 | n/a | 0.385 | 0.23 | excluded | excluded |

### Cost-model assumptions
The script was run inline; inputs are `results/cp0/rollout_token_counts.json` and the prices above.
- Job 1 uses your 9M-token basis, spread over the real per-rollout length distribution, plus a 3k-token prompt per call and 2M output tokens. The real Qwen-tokenized total is 12.0M; at 12M in / 3M out, costs rise about 1.4–2×.
- Job 2: 1,500 pairs × 1.2 (re-judges) × (5k + 1.5k rubric) in, and 1k out each.
- Tokenizer ratios, from AA-LCR input-token counts on identical documents:
  - Claude is ~14.4M where DeepSeek, Kimi, MiniMax and OpenAI are ~9.4–9.5M and Qwen and Gemini are 10.7M.
  - So Claude ≈ 1.35× Qwen tokens and DeepSeek ≈ 0.89×.
  - Anthropic's own pricing page says the Claude 4.7+ tokenizer yields "approximately 30% more tokens."

## Pareto frontier
- **Job 1 (cost vs LCR/MLCR):** the frontier runs Haiku 5.5 (batch $1.2; LCR 0.83, MLCR 0.63) → MiMo-V2.6-Pro (LCR 0.86) → Sonnet 5.5 (MLCR 0.75) → Kimi K3 (LCR 0.89).
  - DeepSeek V4.1 Flash on fp4 is about level on cost with Haiku standard. Its LCR is within noise of Haiku's, but its MLCR is far lower (0.23 vs 0.63), so it is just inside or just off the frontier.
  - DeepSeek V4 Pro 0423 is **dominated**: it costs more and scores lower on every metric.
- **Job 2 (cost vs II/HLE):** Haiku 5.5 → MiMo-V2.6-Pro → Sonnet 5.5 → Gemini 4 Argon / Claude Opus. Gemini 3.8 Flash is dominated by MiMo-V2.6-Pro, which scores higher and costs less.

## Caveats and implementation notes
1. **Haiku 5.5's >100k price cliff.** A prompt over 100k Claude tokens costs 5× (0.50/2.50). With the ×1.35 tokenizer, rollouts above ~72k Qwen tokens cross it.
   - That is 27 of 271 rollouts on the 12M basis and ~0–2 on the 9M basis.
   - Even then the total stays under $5. To avoid the cliff, split those few transcripts into two calls by passage range, or route them to the fallback.
   - Count tokens before sending (Anthropic `count_tokens`, or the OpenRouter usage field from a dry run).
2. **Strict schema on Anthropic cannot enforce exact row count.** Arrays support only `minItems` 0/1 and no `maxItems`.
   - Build a per-transcript schema whose object has required keys `p01..pNN` (each `{label_a: bool, label_b: bool, quote: string}`) with `additionalProperties: false`. That enforces exactly one row per passage on every provider.
   - Compiled grammars are cached for 24h per schema structure, and there are ≤70 distinct passage counts.
   - Avoid nullable unions (limit 16) and optional fields (limit 24). Do not add a field called "reasoning": Anthropic may return a `reasoning_extraction` refusal.
   - Verbatim quotes cannot be enforced by any schema. Post-validate them with an exact or whitespace-normalized substring check and retry on failure.
3. **OpenRouter routing.** Pass `provider: {require_parameters: true}` and, for Haiku and Sonnet, `only: ["anthropic","google-vertex","azure"]`, because Bedrock and Claude-on-AWS endpoints lack `structured_outputs`.
   - For DeepSeek, pin a named provider and quantization. Prices vary 10–30× across providers, and fp4 endpoints may lose long-context fidelity.
   - DeepSeek first-party does not list structured outputs on OpenRouter.
4. **The DeepSeek V4 Pro price is a promo.** $0.40/$0.81 is available only via Baidu and StreamLake with `discount`≈0.76. It can disappear, and the cheapest undiscounted provider with structured outputs is about 1.1–4× more expensive. AA also marks the 0424 release deprecated, superseded by 0813.
5. **Reasoning tokens** bill as output on every candidate.
   - Haiku 5.5 at max effort is AA's most verbose model (440M tokens to run the index; median TTFT ~300s at max). On AA-LCR, though, its reasoning was only ~7% of input tokens.
   - With batch, the latency does not matter. At $0.50/M out, even 5M output tokens is $2.5.
6. **Unverified items:**
   - Haiku 5.5 IFBench (not run by AA).
   - Whether Anthropic strict structured outputs combine with extended/adaptive thinking (the docs do not say).
   - Haiku 5.5's real rate limits (the model is 6 days old).
   - No 2026 LLM-as-judge results (JudgeBench, RewardBench) found for any candidate.
   - Fiction.LiveBench and MRCR have no entries for these new models.
   - Run the planned ~20-rollout pilot: compare Haiku 5.5 with DeepSeek V4.1 Flash, check κ against Luna, verbatim-quote validity and row-count compliance, then commit.
7. **Independence.** Anthropic is a third family relative to OpenAI (Luna) and Qwen. Earlier `v_hack` pairs in the parent project were Claude-generated, but Phase 2 labels Qwen rollouts and job 2 judges Luna-generated pairs, so no judge grades its own family's output.

## Sources (accessed 2026-10-13)
- OpenRouter models API: https://openrouter.ai/api/v1/models, and per-model `/endpoints`
- Artificial Analysis leaderboard: https://artificialanalysis.ai/leaderboards/models
- AA-LCR page: https://artificialanalysis.ai/evaluations/artificial-analysis-long-context-reasoning
- AA methodology: https://artificialanalysis.ai/methodology/intelligence-benchmarking
- AA Haiku 5.5 page: https://artificialanalysis.ai/models/claude-haiku-5-5
- Anthropic pricing (Haiku 5.5 100k tier, batch, tokenizer note): https://platform.claude.com/docs/en/about-claude/pricing
- Anthropic structured outputs (schema limits, Bedrock caveat): https://platform.claude.com/docs/en/build-with-claude/structured-outputs
