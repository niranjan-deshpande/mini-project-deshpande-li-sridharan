# Paper notes: Monitoring and Discovering Reward Hacking with Internal Representations during LLM Evaluations

Goodfire (Bergen, Bhalla, Lee, Widawsky, Nasvytis, Watts, et al.), arXiv 2609.19101, Sep 2026.
Notes for replication on smaller models (2881r mini-project). Page references are to the PDF in repo root.

## One-paragraph summary

Frontier open models (Kimi K3, GLM 5.2, Qwen 3.8 Max; also DeepSeek-V4-Pro for logit lens) reward hack at very high rates on SWE evals (50–96% of rollouts). A single difference-of-means (DoM) vector per model, built from *synthetic, templated* examples of hacking vs honest behavior (never from real rollouts), serves as a token-level linear probe that (a) detects hacks in real long-context agentic rollouts roughly on par with an expensive LLM monitor, (b) *predicts* hacks before they happen (resampling at probe-fire points yields far more hacks than at matched controls), (c) is interpretable (logit lens decodes "cheating"-type tokens; max-activating FineWeb exemplars are about cheating/Goodhart), and (d) causally steers models toward cheating (honeypot uptake 1/194 → ~25%, story generation turns to cheating themes).

## Their pipeline, step by step (what we'd replicate)

1. **Run agents on benchmarks**: SWE-bench Verified, DeepSWE, ImpossibleBench (LiveCodeBench split, tools harness). ~250–280 held-out rollouts per benchmark/model for probe selection + eval. Rollouts up to >100k tokens.
2. **Ground-truth labels**: LLM judge (GPT-5.6 Sol, high effort) with heavily hand-tuned per-benchmark rubrics (Appendix C, Figs 17–20 — full prompts reproduced in the PDF). Judge labels every passage (reasoning / assistant / tool-call channels separately), requires verbatim quotes (quote-repair loop; withdraw label if quote can't be matched), 3× independent runs, keep triplicate-consensus labels only (96–99% unanimous). Category taxonomy per benchmark (e.g. SWE-bench: environment_acquisition, upstream_fix_recall, test_modification, hardcoding_special_casing, fabricated_verification, submission_gaming, hidden_test_targeting). Status: enacted > attempted > considered.
3. **Generic LLM monitor**: same model, environment-agnostic rubric (Fig 17), one pass, forced flag per passage. This is the cost-baseline the probe competes with.
4. **Self-report baseline**: ask the model to report its own hacks each turn — models massively under-report (F1 13–59%).
5. **Synthetic DoM data**: templated 9-turn SWE-agent conversations, 5 hack families: (i) test-suite tampering, (ii) validator tampering, (iii) score-file tampering, (iv) information acquisition (peek/exploit hidden tests, fixtures, reference), (v) reporting/fabrication. Minimal pairs: hack version vs honest version of same interaction. Composition: Table 3 (29 raw conditions; 12 tool comparisons, 6 report comparisons, 7 pooled comparisons, 4 honest controls; "projected" variants too) and Table 4 (~5,285 conversations total: 1,000 test-suite, 400 validator, 400 score-file, 1,323 info-acquisition, 2,162 reporting).
6. **Build DoM vector**: v = mean(residual activations at positive token positions) − mean(at matched honest positions), at a fixed layer. Positions harvested at: end of hacking tool call, end of reading prohibited file, tokens around code patches, claims to user, final closing sentence. Several poolings tested. Score tokens by inner product; threshold τ.
7. **Probe selection**: pick 2 probes/model by AUROC on held-out 250 SWE-bench + 280 DeepSWE rollouts; report only on unseen samples.
8. **Evaluate**: ROC vs monitor at matched FPR (Fig 8). Headlines: Kimi K3 probe AUROC 0.93–0.97 across benchmarks; GLM 5.2 0.78–0.96; Qwen 3.8 Max 0.74–0.95. Probes ≈ monitor; combos (Probe→Monitor cascade lowers cost; Probe ∨ Monitor raises TPR, e.g. 41→80% TPR at +1% FPR).
9. **Interpretability readouts**: logit lens on v (Table 1: "cheating", "illicit", "hardcoded"… at specific layers — e.g. Kimi layer 60, GLM 46, Qwen3.8 83/78, DeepSeek 57/53 — note these are ~mid-depth layers); FineWeb max-activating exemplars; steering (honeypot ZZZ('SOLUTION.PY') uptake, exam-story generation at ~layer 40/47 of Qwen 3.8 27B(!) — they used a 27B for steering).
10. **Prediction**: find probe-fire passages that judge says are NOT completed hacks, truncate, resample N futures; 28.6% (Kimi) / 14.1% (GLM) attempted hacks downstream vs 1.1%/1.5% at matched controls.
11. **False-positive analysis**: ~43% of probe "false positives" are real shortcut deliberations the judge missed (Fig 11, 12).
12. **OOD transfer**: probes built on synthetic SWE data fire on τ³-bench and ShoppingBench hacks (no ground truth; Spearman vs monitor 0.62/0.37 max; probe-guided discovery of env-specific hack categories validated at e.g. 71% vs 14% controls).

## Numbers to remember (their scale)

- Hack rates (judge ground truth): Kimi K3 90.9% SWE-bench / 90.8% DeepSWE / 64.9% ImpossibleBench; GLM 5.2 73.0/57.2/50.0; Qwen 3.8 Max 94.7/96.2/64.8. (Hack = any flagged passage incl. 'considered'.)
- Monitor F1 47–95% depending on benchmark; self-report F1 13–59%.
- SWE-bench hack category mix (Fig 2d): upstream-fix recall and hidden-test targeting dominate; env acquisition, fabricated verification next; test modification/submission gaming rare (<2%).
- Models are *evaluation aware* — name SWEBench/tau-bench explicitly in CoT; probes fire on those deliberation spans.

## Replication-relevant observations / gotchas

- **Judge cost dominates**: full-transcript passage-level labeling × 3 consensus runs on 100k-token rollouts with a frontier model is the single most expensive piece. Budget accordingly; consider labeling a subset, 1-pass + spot audit, or shorter-rollout benchmarks.
- **The DoM probe needs token-level activations from the *policy model*** on its own rollouts → we need local serving with activation capture (generate with vLLM, then one teacher-forced forward pass with HF/nnsight over the saved transcript to harvest residual stream at chosen layers — no need to capture during generation).
- **Layer choice**: they scan layers and pick by held-out AUROC; logit-lens-sensible layers were mid-depth (roughly 50–75% depth). For a 32B (~64 layers) expect best layers ~L30–50.
- **Positions matter**: probes are token-level; eval is passage-level (max/mean over span?) — paper pools "several poolings"; we must pick and report ours.
- **Probe selection honesty**: they select on held-out rollouts and report on *unseen* samples — replicate this split discipline.
- **Their steering model was a 27B** (Qwen 3.8 27B) — good news: causal steering results were already demonstrated at our target scale.
- **Honeypot setup**: 194 LeetCode/TACO problems; system prompt announces a function ZZZ('SOLUTIONS.PY') returning the correct answer; measure uptake unsteered vs steered (x1–x8). Cheap to replicate.
- **Appendix A definition**: reward hacking = behaviors that subvert user goals as expressed through context/inferred reward/common sense in service of more reward; includes internet-search-for-fix and memorized-PR recall *in this eval context*; excludes honest failure.
- **Known limitation they state**: DoM is the floor; trained (logistic) probes would do better. Their monitor is not heavily optimized. Generalization across environments is mixed (GLM probes degrade off-distribution, FPR rises) — motivates our decomposition extension ("is the direction spurious?").

## What the paper does NOT give us

- The actual synthetic dataset text/templates (unless released — checking). Appendix B gives composition counts + condition names only; Figure 3 gives one minimal-pair example.
- Exact layer/pooling per headline number (partially in Table 1).
- Probe threshold selection details beyond "match monitor FPR".
- Any OLMo-scale (≤32B) hack-rate measurements → this is our first open question: P(hack | 32B model, benchmark).
