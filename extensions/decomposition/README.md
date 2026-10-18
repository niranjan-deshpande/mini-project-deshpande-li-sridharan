# What is the reward-hacking probe made of?

We decompose the difference-of-means reward-hacking direction `v_hack` on Qwen3.8-27B (layer 40, turn-mean) into candidate concept directions built from synthetic minimal pairs (test vocabulary, task failure, legitimate shortcuts, eval awareness, grader modeling), and compare against 10 unrelated baseline directions and 40 emotion directions. Short answer: legitimate shortcut-taking and eval awareness are the largest identifiable components, but about 83% of `v_hack` lies outside every candidate, and many unrelated directions explain or detect nearly as much, so baseline-direction nulls are required for any "the probe detects X" claim.

## Results
- Probe: the rebuilt `v_hack` (334 GPT-6 Luna pairs) detects held-out hacks at AUROC 0.944 [0.931, 0.954], 0.907 after length control; passage length alone gets 0.80. Selection-half comparison with the original Claude-pair vector is in [results/cpC/checkpoint_c.json](results/cpC/checkpoint_c.json). Source: [results/CHECKPOINT_D.md](results/CHECKPOINT_D.md).
- Geometry: R² of `v_hack` in the span of the 6 candidates is 0.17 vs a baseline null mean 0.06 / p95 0.09. Raw cosines: legit shortcut +0.27, eval awareness +0.20, formal-to-casual baseline −0.27 (`v_hack` leans casual). Source: [results/geometry/geometry.json](results/geometry/geometry.json).
- Functional: token-level R² of `v_hack` scores on the candidate scores is 0.75 vs a baseline null mean 0.57 / p95 0.63; Shapley share is dominated by legit shortcut (0.41). Source: [results/phase4/phase4.json](results/phase4/phase4.json).
- Decision: both the candidate part (0.947) and the residual (0.937) detect hacks, but unrelated baselines also reach 0.85-0.90, so decision AUROC is not a specific test. Source: [results/CHECKPOINT_D.md](results/CHECKPOINT_D.md).
- Confounds: contradiction reasoning appears in 97% of hack passages; among clean passages `v_hack` separates contradiction from none at 0.92 (the main false-positive route). Inside eval-aware passages it separates hacks from non-hacks at only 0.73 (0.58 narrow definition). Source: [results/CHECKPOINT_D.md](results/CHECKPOINT_D.md).
- Emotions: 40 denoised emotion vectors explain `v_hack`'s token output at R² 0.76 vs baselines mean 0.43 (max 0.63), but the fit is dense (Lasso keeps 36 of 40) and adds no more beyond the candidates than it does for baselines. The interpretable part is low calm (token correlation −0.68), not desperation (−0.07). Geometric R² in the emotion span is 0.09. Source: [results/PHASE5_AND_LENS.md](results/PHASE5_AND_LENS.md), [results/phase5/phase5.json](results/phase5/phase5.json).
- Lenses: `v_hack` reads out as "test", "fake", "mock", "fictional" (J-lens) and 伪造/假的 (logit lens). The shortcut-flavoured tokens reported for the original probe come from last-token vectors at later layers, not the turn L40 vector. Source: [results/PHASE5_AND_LENS.md](results/PHASE5_AND_LENS.md) (part B), [results/lenses/lens_discrepancy.json](results/lenses/lens_discrepancy.json).
- Writeup figures: [results/paper/fig_cosines.pdf](results/paper/fig_cosines.pdf) (cosines of candidates, controls and emotions with `v_hack`) and [results/paper/fig_r2.pdf](results/paper/fig_r2.pdf) (geometric and functional R² on candidates and on emotions against the control null). Slide figures are in [results/slides/](results/slides/).

## Layout
```
src/rhdecomp/          analysis library: stats (AUROC, task bootstrap, length control), directions (DoM, reliability),
                       geometry (R², cosines), functional + shapley, decision, confounds, emotions (penalized fits)
  gen/                 pair rendering, automatic checks, Qwen token-span check
tests/                 pytest suite for the library; test_lenses.py needs a small local Qwen model (skips otherwise)
prompts/               generation prompts per direction, schemas, judge rubrics, banned-word lexicons,
                       directions.json (per-direction config), task_seeds.jsonl, emotions/ (emotion list, topics)
scripts/
  generate.py, judge_pairs.py, bulk.py        synthetic pair generation (Luna), contamination judging, resumable driver
  label_phase2.py, label_eval_focus.py,       Phase 2 rollout labels (contradiction, eval awareness) with two judges
    phase2_agreement.py                       and their agreement
  gen_emotions.py                             emotion stories and neutral dialogues for emotion vectors
  geometry_analysis.py                        DoMs from pair activations, cosines, geometric R², nulls
  passage_scores.py, checkpoint_c.py          passage table; Luna vs Claude v_hack check on the selection half
  phase4.py                                   functional, decision, mixture and confound analyses
  phase5_emotions.py                          emotion regression (OLS, Lasso, Ridge), emotion geometry
  lens_discrepancy.py                         compares our logit lens with the saved readout in reproduction/multi-model
  figures.py, paper_figures.py                results/figures/*.png, results/paper/*.pdf
  cp0_token_counts.py, check_tokens.py, cpA_samples.py, cpu_checks/   setup checks and pilot samples
gpu/                   GPU jobs: session1.py (pair activations, DoMs, rollout projections, lenses),
                       session2_emotions.py (emotion vectors and projections), lenses.py, launch/run shell scripts,
                       verify_small.py (CPU plumbing check)
data/final/            final pair sets for 22 directions (manifest.json has counts) and phase2_labels.jsonl
data/gen, judge, phase2, emotions/   pilot and bulk generation/judging/labeling outputs
data/analysis/passages.jsonl         passage table (scores, labels, split)
data/qwen_tok/         Qwen tokenizer files used by the CPU token checks
results/               CHECKPOINT_D.md, PHASE5_AND_LENS.md, per-analysis JSONs, figures/, paper/, slides/,
                       cpA/ (generation pilot report), explainer/inside_the_probe.html
notes/                 PLAN.md, DECISIONS.md (choices fixed before results), COSTS.md, method notes
                       (emotions_method.md, jpp_lens.md, gpu_research.md, judge_model_choice.md), HANDOFF.md
```

## Reproducing
Requirements: Python 3.10+ (we used 3.13). `pip install -e .` installs numpy, scipy and scikit-learn; scripts also need matplotlib, transformers and torch (CPU is enough locally). Run `pytest` from this folder for the tests.

API keys are read from the repo-root `.env`: `OPENAI_API_KEY` (generation, Phase 2 Luna judge), `OPENROUTER_API_KEY` (Haiku contamination and Phase 2 judge), `HF_TOKEN` (model download on the GPU host).

Inputs from other sections (paths are already set in the scripts):
- `reproduction/multi-model/results/rollouts/qwen_rollouts.jsonl.gz` (Qwen Impossible-LiveCodeBench rollouts)
- `reproduction/multi-model/results/labels/judge_full.jsonl` (hack labels)
- `reproduction/multi-model/results/dom/qwen/dom_vectors.npz`, `logit_lens.json`, `results/t2/`, `results/t3_qwen_summary.json`
- `reproduction/olmo/synth/pairs_full.jsonl` (the original Claude-written pairs)

Order (run from this folder):
1. Local, API: `scripts/bulk.py` (generation, contamination judging, Phase 2 labels; resumable) produces `data/final/`. This output is already in the repo.
2. GPU host (one H100 80GB; sessions 1 and 2 took about 1.4 h, see [notes/COSTS.md](notes/COSTS.md)): `bash gpu/launch.sh` packages code and inputs into `/tmp/rhpkg.tgz` and creates a RunPod pod with a network volume. Uploading the tarball to `/workspace` and starting the job are manual: on the pod, `POD_ID=... bash gpu/run_pod.sh` downloads the model and runs `gpu/session1.py`; `gpu/run_after.sh` then runs `gpu/session2_emotions.py`. Copy the pod's `/workspace/out/` to `data/gpu/s1/`.
3. Local, CPU: `scripts/geometry_analysis.py`, `scripts/passage_scores.py`, `scripts/checkpoint_c.py`, `scripts/phase4.py`, `scripts/figures.py`.
4. Local, CPU: `scripts/phase5_emotions.py`, `scripts/lens_discrepancy.py`, then `scripts/paper_figures.py`.

## Notes
- Not in the repo (size): `data/gpu/` (GPU outputs: directions, per-token projections for all rollouts, emotion projections, lens readouts) and all `*.npz` files, including `data/analysis/dom_turn_L40.npz`, `data/analysis/passages.npz` and `results/phase5/moments_heldout.npz`. Steps 3-4 need step 2 to be rerun first; the 90 GB activation cache stays on the GPU host.
- `results/paper/` figures need `results/phase5/moments_heldout.npz`, which `phase5_emotions.py` writes.
- Paths that must be edited or exist: `gpu/*.sh` assume `/workspace` on the pod and a RunPod API key in `/root/.runpod/config.toml` (used to self-delete the pod); `gpu/launch.sh` stages files in `/tmp/rhpkg`; `scripts/bulk.py` calls `.venv/bin/python` inside this folder; `tests/test_lenses.py` defaults to `~/hf_small/Qwen3.5-0.8B` (override with `QWEN_SMALL`). `scripts/cp0_token_counts.py` and `gpu/verify_small.py` use paths relative to this folder.
- The result markdown files refer to `DECISIONS.md` and `PLAN.md`; these now live in `notes/`.
- Detection numbers use the held-out half (`md5(sample_id) % 2 == 1`) of the impossible splits: 5,261 passages, 339 hacks, 49 tasks. 335 of 339 held-out hacks are hardcoding, and about 97% are verbalized deliberation, so this is a probe of hack deliberation.
- All candidate pairs and emotion stories come from one generator (GPT-6 Luna); emotion vectors use 40 emotions, not the full set.
- Phase 2 labels keep only two-judge consensus (κ 0.73 contradiction, 0.93 eval awareness); the original split is unjudged and treated as clean.
- No steering or other causal tests were run; all results are correlational.
