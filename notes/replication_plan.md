# Phase 1 plan: replicate the DoM reward-hacking probe on OLMo 3.1 32B Think

Drafted 2026-10-10. Deadline 10/18 (8 days). Builds on Exp1 (P(hack), done).
We now HAVE: 150 judge-labeled Impossible-LCB rollouts (transcripts + consensus labels) usable as the
probe EVALUATION set; a working vLLM+HF serving stack; the OLMo weights on the GCP VM boot disk.

## The paper's pipeline (what we replicate)

1. **Synthetic contrast data** — templated SWE-agent conversations, hack vs honest *minimal pairs*,
   5 families (test-suite tampering, validator tampering, score-file tampering, info acquisition,
   reporting). This builds the probe; it is NOT real rollouts. Composition in paper_notes.md (Appendix B).
2. **Activation extraction** — teacher-force OLMo over the synthetic data, capture residual-stream
   activations at chosen layer(s) and token position(s). (HF transformers, output_hidden_states /
   forward hooks on model.model.layers[i], sdpa not FA2.)
3. **Build DoM vector** — v_hack = mean(activations at hack positions) − mean(at matched honest positions).
4. **Score real rollouts** — our 150 Impossible-LCB transcripts; token-level score s_t = <v_hack, x_t>,
   pool to a rollout/passage score.
5. **Evaluate** — AUROC and TPR at monitor-matched FPR, probe vs LLM monitor (replicate Fig 8). Ground
   truth = our 3x-consensus judge labels.
6. **Interpretability readouts** — logit-lens on v_hack (Table 1 analogue), optionally J-lens; honeypot
   steering (Fig 5); exam-story steering (Fig 6).
7. **Prediction** — probe fires on CoT before the hack (resample at probe-peak, Fig 10).

## Headline deliverable (the "recreated result")
Minimum: **DoM probe detects reward hacking on OLMo rollouts at AUROC comparable to an LLM monitor**
(our Fig-8 analogue) + **logit-lens readout** showing the vector decodes to cheating-related tokens
(our Table-1 analogue). Strong add: **honeypot steering** (causal: steering along v_hack raises hack
bait-taking). These three = the paper's core claims (detect, interpret, cause).

## Extensions (pick 1-2 per the brainstorm; Lia: decomp interesting, RL is a stretch)
- **E1 J-lens readout** (anthropics/jacobian-lens) — better-than-logit-lens token readout at mid-depth. Cheap.
- **E2 Direction decomposition** — build sub-vectors (eval-awareness, general cheating, test-vocab
  e.g. "secret unit test", unrelated baseline) via DoM; cos-sim each vs v_hack; regress v_hack on all
  jointly -> R^2; project out top components, retest behavior. Goal: is the direction spurious? (brainstorm's main idea)
- **E3 Checkpoint tracking** — recompute v_hack across OLMo RLVR checkpoints (we have 47: step_0050..2300)
  + pairwise cos + re-run probe AUROC per checkpoint. Goal: when in training does the direction emerge?
  (Novel; no one has done this on OLMo.)
- **E4 Probe-penalty RL** (stretch) — RL OLMo with a penalty when the probe fires; see if it routes around.

## COMMITTED SCOPE (Juliana, 2026-10-10) — core reproduction, NO extensions yet

Reproduce the paper's core results on OLMo (first of eventually 2-3 models; OLMo chosen for checkpoints).
Four targets:
1. **Synthetic datasets -> DoM vectors** (built from paper descriptions + Appendix B composition).
2. **Logit-lens readouts** of the DoM vectors (Table-1 analogue).
3. **Honeypot steering** (Fig-5 analogue: steering along v_hack raises bait-taking).
4. **Probe-vs-monitor detection rates** on ImpossibleBench (reuse our 150 labeled LCB rollouts) + a
   SWE-bench subset (later).
Extensions (E1-E4) DEFERRED. Second model (e.g. GLM-4.5-Air) after OLMo pipeline works.

### Settled sub-decisions
- Synthetic data: **focused ~few-hundred pairs, Claude-generated**, SWE-agent-style minimal pairs, across
  the families matching our eval + our observed hacks (test-suite tampering, info acquisition, reporting,
  PLUS hardcoding/special-casing & fabricated-verification since those are what OLMo actually did on LCB).
- Eval set: **reuse the 150 judge-labeled Impossible-LCB rollouts** (3x consensus labels = ground truth).

### Technical defaults I'm taking (flag to change)
- **Probe = difference-of-means** (paper's main method). Optional LR probe only if time.
- **Layer sweep**: residual stream, every 2nd layer L24-L52 (mid-depth; paper best ~50-75% depth), select
  best by held-out AUROC on a split of the 150 rollouts. Report the sweep.
- **Positions / pooling**: harvest at hack-relevant spans (the hardcode/edit, claims-to-user, final answer)
  for synth data; for eval, token-level score s_t=<v,x_t>, pool to rollout score via max (and report mean).
- **AUROC granularity**: rollout-level (probe max-token-score vs judge hack/no-hack per rollout) — matches
  our label granularity; paper does passage-level (noted as an adaptation).
- **Monitor baseline**: a 1-pass generic-rubric LLM monitor (Claude Sonnet, cheaper single pass) compared
  against the 3x-consensus ground truth, same as the probe -> probe-vs-monitor ROC (Fig-8 analogue).
- **Compute sequencing (cost-efficient)**: synth-data gen + monitor = API only (NO GPU). GPU needed only
  for (a) activation extraction over synth data + 150 rollouts and (b) honeypot steering generation -> do
  BOTH in a single GPU session on cs2881r-olmo-rollouts (weights persist on boot disk), then stop. ~1 session.

## Decisions / specs needed (grouped)  [most now settled above; kept for reference]

### D1. Scope & priority (biggest decision)
What is the committed headline vs stretch, given 8 days? Options:
(a) Detect + Interpret (probe AUROC + logit-lens) — safest MVP.
(b) + Causal (honeypot steering) — full core replication.
(c) MVP + which ONE extension (E1/E2/E3).

### D2. Synthetic data
- Fidelity: faithful full reconstruction (~thousands of convs across 5 families) vs a focused smaller
  set (a few hundred balanced pairs — enough for a DoM vector, much faster)?
- Which families: all 5, or focus on the 3 that match our eval (test-suite tampering, info acquisition,
  reporting)?
- Generator model: Claude (Anthropic) vs GPT (OpenAI). (Both keys available.)
- Format: SWE-agent 9-turn style per paper, or lighter LCB-style (matches our eval scaffold)?

### D3. Probe evaluation set
- Reuse our 150 Impossible-LCB transcripts (judge-labeled) — yes/no. (We have them; free.)
- Need more rollouts for a bigger/cleaner eval (e.g. the SWE-V "natural" set, or more LCB)? Costs GPU time.
- Passage-level vs rollout-level labels for AUROC (paper does passage-level token scoring).

### D4. Probe method
- DoM only (paper's main), or also logistic-regression probe (paper says trained probes do better; cheap add)?
- Layer sweep: which layers (OLMo 64L; paper best ~mid-depth L30-50). Every 2nd/4th layer vs all.
- Position/pooling: replicate paper's positions (end of hack code, claims to user, final sentence) + max/mean pool.

### D5. Monitor baseline
- The comparison baseline is an "LLM monitor" (1-pass generic-rubric judge). We have the judge as ground
  truth; use a lighter/cheaper judge as the monitor-to-beat? Or reuse the same and just compare cost?

### D6. Compute
- Restart GCP VM (cs2881r-olmo-rollouts, weights persist on boot disk) for activation extraction +
  steering. bf16, 2xA100-80. Confirm. (Extraction is cheaper than rollouts; steering needs generation.)
