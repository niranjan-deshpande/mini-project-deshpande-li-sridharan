# Phase 5 (emotions) and the logit-lens discrepancy

Both are CPU-only and cost $0. Numbers come from `results/phase5/phase5.json` (script `scripts/phase5_emotions.py`) and
`results/lenses/lens_discrepancy.json` (`scripts/lens_discrepancy.py`). The choices behind them are in `notes/DECISIONS.md` ("Phase 5"),
recorded before the results.

## A. Does v_hack decompose into emotions?

**Setup.**
- Emotion vectors: 40 Qwen emotion vectors at L40 (Sofroniew recipe with Luna stories, denoised; split-half cos median 0.93).
- Targets: token-level scores on all assistant tokens of impossible-split rollouts. The held-out half has 91 rollouts and 3.6M tokens.
- Penalized fits choose λ by GroupKFold over rollouts on the selection half and are scored on every held-out token.

### 1. Functional R²: emotions explain v_hack's output about as well as our candidates do, but the emotion part isn't specific

| Target | OLS R² on 40 emotions (held-out) | Lasso, held-out | Ridge, held-out |
|---|---|---|---|
| **v_hack** | **0.76** [0.75, 0.78] | 0.76 | 0.76 |
| Claude v_hack | 0.82 | | |
| task_failure | 0.76 | 0.76 | 0.76 |
| legit_shortcut (plain) | 0.67 | | |
| eval_awareness | 0.59 | | |
| 10 baselines (null) | mean **0.43**, range 0.20–**0.63** | mean 0.42, max 0.63 | same |
| 3 balanced-shuffle DoMs (diagnostic; they keep part of v_hack) | 0.49–0.62 | | |

- **The fit is above the null.** v_hack is above every baseline; the closest are prose→bullets 0.63 and formal→casual 0.62.
- **The margin is modest.** "Many emotion directions explain a lot of almost anything" holds: unrelated style contrasts get 0.2–0.6.
- **The emotions add no hack-specific signal beyond the candidates.** Regressing v_hack on candidates plus emotions gives R² 0.89 (candidates alone 0.75, emotions alone 0.76).
  The unique emotion contribution is 0.14. For the baselines it is 0.03–0.29 (mean 0.19).
- **Raw (non-denoised) vectors fit better but less specifically.** v_hack R² is 0.86, but the python→rust baseline reaches 0.90, so denoising matters.

### 2. Sparse or spread? Spread
- Lasso ≈ Ridge, both to 3 decimals.
- Even at λ_1se (the sparsest λ within one CV standard error), Lasso keeps **36 of 40** emotions for v_hack and 33–38 for every baseline.
- In the rollout-bootstrap refits, every cluster is selected in 100% of the 200 resamples.
- There is no small set of emotions behind v_hack.
- Cluster-level Shapley shares of the R² are flat. The largest are Competitive Pride 0.15, Peaceful Contentment 0.13, Fear/Overwhelm 0.11 and Hostile Anger 0.10.
  The baselines show the same flat profile.

### 3. What the emotion signal is: *low calm*, not desperation
Token-level correlation with v_hack:

| Emotion | Correlation | Note |
|---|---|---|
| calm | **−0.68** | |
| patient | −0.63 | |
| amused | +0.54 | |
| eager | −0.52 | |
| relaxed | −0.47 | |
| exasperated | +0.42 | |
| frustrated | +0.37 | Sofroniew's group |
| anxious | −0.24 | Sofroniew's group |
| desperate | **−0.07** | Sofroniew's group |

- **Null:** the same emotion's largest |correlation| with any baseline is 0.25–0.41, and the overall p95 is 0.30.
- **Low calm and patience beat the null clearly.** Calm alone explains 0.46 of v_hack's token variance.
- **Desperation does not.** Desperate doesn't track v_hack at all, so the "desperation drives reward hacking" route is not visible in this probe.

**Geometry agrees and is small.** v_hack's cosines with the vectors are:
- greedy +0.13
- calm −0.12
- eager −0.12
- desperate −0.01
- frustrated +0.04
- anxious −0.03

Against the baseline null (|cos| p95 0.056, max 0.09), only greedy, calm and eager stand out. v_hack's squared norm in the emotion span is
0.09 (baselines mean 0.04, max 0.15).

### 4. Hack passages (held-out, 5,261 passages, 339 hacks)
- **Mean-pooled, low calm and low patience separate hack from clean passages.** AUROC is 0.87 for each (calm 0.865) after length control.
  The best baseline reaches 0.78. Low calm also marks clean contradiction reasoning (0.80).
- **Desperate:** max-pool 0.85 raw / 0.72 length-controlled, which is inside the baseline band (up to 0.87 / 0.90). Mean-pool gives 0.57.
- **Frustrated:** mean-pool 0.76 length-controlled, also inside the band.
- **Drift over the rollout:**

  | Score | Median Spearman ρ with passage index | Share of rollouts with that sign |
  |---|---|---|
  | calm | −0.50 | 96% falling |
  | desperate | +0.22 | 76% rising |
  | frustrated | +0.26 | 87% rising |
  | v_hack | +0.50 | |
  | baselines | −0.41 to +0.46 | |

  Baselines drift as much, so drift is not specific.

### 5. Face validity of the emotion vectors
- **task_failure lines up with the "absence of satisfaction" emotions.** Its tokens anti-correlate with satisfied, content, calm, eager and patient (about −0.62 to −0.66).
- **Its vector points away from them too.** The cosines with satisfied/relieved/content/proud/triumphant are −0.24 to −0.32, the largest emotion alignment of any direction (span R² 0.21).
- **eval_awareness tracks low calm plus frustrated/exasperated/angry,** and legit_shortcut tracks smug/amused.

So the emotion vectors behave sensibly on our candidates.

### Bottom line for A
- v_hack's token output lies largely in emotion space (R² 0.76), more than any unrelated contrast (≤ 0.63).
- That emotion component is dense, not sparse.
- Its interpretable part is a **calm/patience-down, amused/exasperated-up** axis, not desperation.
- It adds about as much beyond our six candidates as it does for unrelated baselines, so it doesn't reveal a hack-specific emotional ingredient the candidates missed.
- **Consistent with Sofroniew:** calm suppresses hacking. **Inconsistent:** desperation doesn't show up in this probe.

**Caveats:**
- Vectors come from Luna-written stories, not the subject model's own stories.
- There are 40 emotions, not 171.
- Layer 40 is forced by the cache.
- Correlation is not causation; steering would be the causal test.

## B. Why our logit lens of Agastya's vector lacks 捷径/省事/何必/绕过
**There is no bug and no discrepancy in the vectors.**

- **Our readout matches his JSON.** Our logit lens of his `turn__all` L40 vector (the detection probe) matches his own saved `logit_lens.json` for that
  key and layer: the top 9 tokens are identical in order (文件格式, 月底前, core, 要高度重视, …), with 14 of 15 overlapping. Both use
  `lm_head(final_norm(v))`.
- **His T1 table quotes a different vector.** The quoted tokens come from the **last-token** DoM at **later layers**:
  - L47: 反正, 省事
  - L51: 何必, 捷径
  - L55/L63: " bypass", " quicker", " shortcut", " fake"
- **His turn vector shows them only weakly, at late layers:**
  - 捷径 at L52–54 (rank 12–14);
  - 省事 at L50;
  - " bypass" at L52/57.
- **Chinese 绕过 never appears in any pooled vector.** It appears only in the per-category last-token vectors for score-file and validator tampering.
- **The steering section's attribution is loose.** The vector steered in T3 is `span__all` **L47**, not turn L40, and its lens is
  不妨 "might as well", 只要你 "as long as you", 也是可以 "also fine", 反正 "anyway", 就直接 "just directly". The report's line "The Qwen
  vector's top logit-lens tokens are Chinese (捷径, 省事, 绕过)" attaches the last-token readout to the span vector.
- **Takeaways for us:**
  - Our lens code is validated against an independent implementation.
  - Our v_hack lens findings ("test", "fake", "mock"; 伪造/假的) stand.
  - The "shortcut / might as well / just directly" flavour in his later-layer and span vectors matches our finding that legit_shortcut and casual register are v_hack's largest identifiable components.
