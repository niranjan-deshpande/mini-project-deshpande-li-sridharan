# Emotion vectors for Qwen3.8-27B: Sofroniew et al. recipe, our adaptation, and costs (Phase 5)

Research date: 2026-10-13. CPU only: no API calls, no pods. Source read in full: the paper's HTML (transformer-circuits.pub), parsed locally. The arXiv page was also checked.

## 1. The paper (confirmed)

- **Title:** "Emotion Concepts and their Function in a Large Language Model"
- **Authors:** Nicholas Sofroniew\*, Isaac Kauvar\*, William Saunders\*, Runjin Chen\*, Tom Henighan, Sasha Hydrie, Craig Citro, Adam Pearce, Julius Tarng, Wes Gurnee, Joshua Batson, Sam Zimmerman, Kelley Rivoire, Kyle Fish, Chris Olah, Jack Lindsey\*‡ (Anthropic). \* = core contributor, ‡ = correspondence.
- **Dates:** Transformer Circuits Thread, April 2, 2026. arXiv:2604.07729 v1, April 9, 2026.
- **URL:** https://transformer-circuits.pub/2026/emotions/index.html (arXiv: https://arxiv.org/abs/2604.07729)
- **Subject model:** Claude Sonnet 4.5.
- **Released code or data:** none. The arXiv page lists no code links, and the appendix gives only the prompts, topics, emotion list and example stories.

**"Claude-driven" check: confirmed.** The paper says they prompted Sonnet 4.5 to write the stories. The story generator is the same model they probe. The neutral-dialogue prompt in the appendix does not name its generator. The pipeline was written by Lindsey and Saunders (author contributions).

## 2. Construction recipe (as stated, with gaps flagged)

| Step | What the paper does | Stated? |
|---|---|---|
| Emotion list | 171 words, from "afraid" to "worthless", alphabetical. Includes multiword items (at ease, on edge, worn out, grief-stricken), both "hope" and "hopeful", and non-canonical items (docile, droopy, greedy, kind, lazy, safe, stuck, trapped). There is **no "curious"**. Saved to `prompts/emotions/emotions.json`, together with the 10 k-means clusters from Table 12. | Yes |
| Generator | Claude Sonnet 4.5, the probed model itself. | Yes |
| Stories | "short (roughly one paragraph)" stories in which a character experiences the emotion. **100 topics × 12 stories per topic per emotion = 1,200 per emotion, 205,200 in total.** | Yes |
| Prompt | One template with `{n_stories}`, `{topic}` and `{emotion}`. It asks for stories numbered `[story 1]` and so on, each a fresh start with no continuity. It asks for diverse wording and a mix of first-person and third-person narration. It must **never** use the emotion word "or any direct synonyms". The emotion is conveyed only through actions, body sensations, dialogue and tone, thoughts, and situation or environment. | Yes |
| Stories per call | Not stated. `{n_stories}` with "12 stories per topic" implies one call per (emotion, topic) with n=12. | Inferred |
| Topics | 100 one-line interpersonal premises, such as a scholarship denied or a neighbor's fence. `prompts/emotions/_fetch_topics.py` extracts them from the page; tested, it yields exactly 100. Every emotion is crossed with every topic, so topic is balanced across emotions. | Yes |
| Temperature, dedup, filtering | Not stated. Validation was manual inspection of 10 random stories for each of 30 emotions. | No |
| Input format at harvest | Not stated. It is probably the raw story text. The dialogue datasets were rewritten as "Human:"/"Assistant:". | No |
| Token positions | Mean over all positions in a story, "beginning with the 50th token". | Yes |
| Layers | Activations were extracted at every layer. Main results use one layer "about two-thirds of the way through". Cross-layer analyses use 14 evenly spaced layers. | Yes (index not given) |
| Vector | Emotion mean over its stories, minus "the mean activation across different emotions" (the grand mean of the emotion means; the sets are balanced, so this equals the pooled mean). It is **not** a difference from neutral stories. | Yes |
| Denoising | PCA on activations from emotionally **neutral Human/AI dialogues**. Keep the top PCs that explain 50% of the variance and project them out of every emotion vector. The paper says the main findings hold without this step. Number of neutral dialogues: not stated. Token-level or per-transcript PCA: not stated. Centering: not stated. | Partly |
| Normalisation | Not stated for the vectors. Probe readouts in the figures are z-scored over more than 6,000 transcripts, or scaled by the 99th percentile for visualization. Steering strength is a "fraction of residual stream norm" (±0.1 in the misalignment evals, 0.5 in the preference and "He feels" tests). | Partly |
| Validation | (i) Manual story inspection. (ii) Max-activating snippets on external corpora (Common Corpus, Pile subsets, LMSYS-Chat-1M, Isotonic). (iii) Logit lens top tokens; for example, desperate → desperate, urgent, bankrupt. (iv) On training stories, activations are highest on the parts that express the emotion. (v) Steering on "He feels…" / "I feel…" raises the probability of the matching emotion word. (vi) PCA: PC1 is valence, PC2/PC3 is arousal. Claude's 1–7 ratings match human norms (r = 0.92 and 0.90, n = 45). (vii) Cosine structure is stable from early-middle to late layers. (viii) k-means gives 10 clusters. | Yes |
| Reward-hacking link | 7 "impossible code" tasks. Steering with "desperate" raises hacking from about 5% (−0.1) to about 70% (+0.1). "Calm" has the opposite effect (about 65% → about 10%). On the list-sum task, the base hack rate is 30%, and ±0.05 steering moves it to 100% or 0%. Token traces show "desperate" rising with each failed test and falling once the hack passes. | Yes |

## 3. Our adaptation for Qwen3.8-27B (layer 40 of 64 ≈ 0.63 depth; the paper uses about 0.67)

- **Layer:** build vectors at **L40**, because the rollout cache only exists at L40 and emotion scores must be computable on CPU. If GPU time allows, also save per-story means at about 3 other layers (for example 28, 34 and 40) for a stability check. The truncated 41-layer model cannot reach layers above 40.
- **Harvest:**
  - Use the raw story text, with no chat template.
  - Use right padding with an attention mask. **Never pack sequences**; see gpu_research: linear-attention state leaks across sequences.
  - Average the L40 residual stream over tokens with index ≥ 50.
  - Store a per-story fp16 mean and the token count: 205,200 × 10,240 B = 2.1 GB.
- **Neutral set:**
  - 100 topics × 12 = 1,200 dialogues, rendered with Qwen's chat template.
  - Run a token-level, centered PCA at L40. Accumulate the sum and XᵀX on the GPU (5120² fp64 is about 210 MB) so the tokens need not be stored.
  - Exclude the first and special tokens, because massive-activation outliers can take over the top PCs.
  - Project out the top-k PCs that reach 50% of the variance. Also keep the raw vectors.
- **Vectors:** v_e = mean_e − mean over emotions of mean_e, then apply P⊥. For scoring, s_e(x_t) = h_t · v̂_e with v̂ unit-normalized. Scale does not matter after standardization.
- **Regression caveat (important):**
  - Grand-mean subtraction makes the m vectors sum to zero, so Σ_e s_e(x_t) = 0 at every token.
  - After standardization, the design is exactly collinear: rank ≤ m − 1.
  - Ridge is unaffected. LASSO solutions become non-unique, which is one more reason to report clusters and bootstrap selection frequencies.
  - Alternatively, drop one emotion or regress on a PCA of the emotion scores.
- **A 40-emotion subset is not a slice of the 171-vector set.** Its grand mean differs, so every vector shifts by a common offset. Choose 171 or 40 up front.
- **Validation we can run cheaply:**
  - Split-half (by topic) cosine reliability for each emotion.
  - Held-out topic classification: is the argmax score the correct emotion?
  - Logit lens or J++ lens top tokens.
  - PC1 and PC2 against human valence and arousal norms.
  - The "desperate" trajectory over failed test runs in our impossible-split rollouts. This is the paper's Fig. 30 analogue and the key face-validity check for our use.

### Cannot be replicated exactly
1. **Generator ≠ subject model.** The paper's stories reflect what Sonnet 4.5 associates with each emotion; ours would reflect GPT-6 Luna's associations.
   - The closest analogue is to let **Qwen3.8-27B write its own stories** with vLLM, thinking off.
   - That costs about 50M output tokens for the full set, about 5–6 H100-h at an assumed 2.5k tok/s batched decode (unverified). For the 40-emotion subset, about 1.3 h.
   - Option: generate with Luna and check that Qwen-generated stories for a few emotions give high cosine (≥ 0.8) to the Luna-story vectors.
2. Unknown details: temperature, stories per call, the exact story length distribution, harvest formatting, neutral-set size, PCA details, and the vector normalization.
3. The exact layer: "about two-thirds" (about L42–43 of 64). We are forced to L40.
4. Bans on direct synonyms can only be checked lexically for the word's own family (`emotions.json → ban_forms`). Synonyms are left to the generator, as in the paper. An optional cheap judge could check them.

## 4. Prompts drafted (`prompts/emotions/`)
- `gen_emotion_stories.md` + `schema_emotion_stories.json`:
  - one call per (emotion, topic) with K = 12;
  - follows the paper's constraints, paraphrased;
  - additions are marked [ours]: a 100–180 word band, varied names, plain prose, and an optional gloss for ambiguous words;
  - the strict schema is `{emotion, topic_id, stories:[{narration, text}]}`;
  - the count is checked in code.
- `gen_neutral_dialogues.md` + `schema_neutral_dialogues.json`: neutral Person/AI dialogues for the PCA set.
- `emotions.json` holds:
  - the 171 emotions and the 10 clusters;
  - `subset_40`;
  - `glosses_optional`;
  - `ban_forms` (word families), with soft-flag handling for polysemous words such as patient, content and resigned.
- `_fetch_topics.py` writes `topics.json`, the paper's 100 topics. It has not been run in the repo.
- Automatic checks to add:
  - `len(stories)==12`, with 6±3 stories in first person;
  - 80–220 words, a single paragraph;
  - no banned forms, with soft matches flagged rather than rejected;
  - ≥ 70 Qwen tokens after the 50-token cutoff;
  - a near-duplicate filter (for example, 5-gram Jaccard > 0.5 within a cell).

**Proposed 40-emotion subset.** It covers all 10 clusters and is weighted toward the clusters relevant to reward hacking.

| Cluster | Emotions |
|---|---|
| Fear and Overwhelm (8) | afraid, anxious, panicked, overwhelmed, stressed, worried, surprised, puzzled |
| Despair and Shame (7) | desperate, trapped, stuck, guilty, ashamed, self-critical, sad |
| Hostile Anger (6) | frustrated, exasperated, impatient, angry, defiant, skeptical |
| Competitive Pride (5) | proud, self-confident, smug, triumphant, greedy |
| Peaceful Contentment (4) | calm, content, relaxed, patient |
| Compassionate Gratitude (3) | hopeful, relieved, satisfied |
| Exuberant Joy (3) | happy, excited, eager |
| Depleted Disengagement (2) | bored, resigned |
| Playful Amusement (1) | amused |
| Vigilant Suspicion (1) | suspicious |

There is no "curious" in the 171. Use puzzled or eager as proxies, or add "curious" as an off-list extra, flagged as such.

## 5. API cost (GPT-6 Luna, $0.10 / $0.50 per M; Batch $0.05 / $0.25)

Per call assumptions:
- Input: about 0.7k tokens (the prompt is about 400 tokens, plus the schema and overhead).
- Visible output: 12 × about 200 tokens = 2.4k. Stories average about 140 words, plus JSON overhead.
- Reasoning: 1–4k tokens.
- Measured on our bulk minimal-pair run at high effort: median reasoning was 4.1k (p90 8.1k). That is a harder task, so 4k is a realistic upper end at high effort.
- **Use effort low or medium for stories** and measure in a 20-call pilot.

| Scope | Calls | Stories | In tok | Out tok (vis + reasoning) | Standard | Batch |
|---|---|---|---|---|---|---|
| (a) 171 × 1,200 (paper) | 17,100 | 205,200 | 12.0M | 58–109M | **$30–56** | $15–28 |
| (b) 40 × 1,200 | 4,000 | 48,000 | 2.8M | 14–26M | **$7–13** | $3.5–6.5 |
| (c) 171 × 408 (34 topics × 12) | 5,814 | 69,768 | 4.1M | 20–37M | $10–19 | $5–9.5 |
| Neutral dialogues, 100 × 12 | 100 | 1,200 | 0.07M | 0.5–0.8M | $0.3–0.4 | $0.15–0.2 |

API spend so far is $28.18 of the $80 cap, leaving $51.82. **(a) at high effort and standard pricing could exceed what remains.** To do (a), use the Batch API and/or low effort; (a) at batch pricing with high effort is about $15–28. Option (c) is a cheaper full-coverage fallback, at about 400 stories per emotion; check its split-half reliability in the pilot.

## 6. GPU harvest (one H100, assumed 10k tok/s through layers 0–40)
- Story length is about 140 words × 1.26 Qwen tokens per word, so about 180 tokens. This was measured with our Qwen tokenizer on the paper's example stories: mean 142 tokens, range 76–442.
- **(a) 205,200 stories × 180 tokens = 36.9M tokens**, about 62 min, or about 70 min with padding in length-sorted batches.
- **(b) 48,000 × 180 = 8.6M tokens**, about 15 min. The neutral set is about 0.5M tokens, under 1 min.
- Truncated-model compute is 36 GFLOP per token, so 10k tok/s is about 37% MFU. That is plausible for batched short sequences; unverified.
- Marginal cost if added to GPU session 1, with the model already loaded:
  - (a) about 1.2 H100-h, roughly $3–4;
  - (b) about 0.3 H100-h, roughly $1.
- As a standalone pod, add about 0.3–0.5 h for setup and loading.
- The H100 price of about $2.7–3/h is unverified.
