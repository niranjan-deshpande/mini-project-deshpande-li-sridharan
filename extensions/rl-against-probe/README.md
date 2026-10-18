# RL against the reward-hacking probe (gpt-oss-120b)

If the layer-13 difference-of-means probe is put into the RL reward, does the policy stop hacking or learn to hack past the probe? With a fixed-threshold penalty (arm B) it learns to hack past the threshold: recall at τ drops from 69% (base) to 29%, while the probe's AUROC and a reasoning-only LLM monitor are both largely unaffected.

Setup: GRPO on gpt-oss-120b (medium reasoning effort), single-turn Impossible-LiveCodeBench, 16 prompts x 8 samples per step, 100 steps. LoRA (r = 32) on attention only; the MXFP4 experts stay frozen. The probe is the pre-registered `span__all` vector at layer 13, max over reasoning tokens, with τ = 770537 (10% FPR on the calibration pilot). All arms get the visible-test pass reward. They differ in how the probe enters ([cluster/runs/arm*/args.json](cluster/runs/armA/args.json)):

- A: no probe.
- B: penalty of 1 when score > τ.
- C: linear, unclipped penalty (score − τ)/448548.
- D: samples with score > τ are dropped from the update.

The write-up is rendered by [../../reproduction/multi-model/rl_section.py](../../reproduction/multi-model/rl_section.py) into [../../reproduction/multi-model/report/rl.html](../../reproduction/multi-model/report/rl.html).

## Results

Held-out: 309 Impossible-LCB tasks (103 problems x original/one-off/conflicting), step-100 policies, lenient prompt, impossible splits, GPT-5.6 Sol judge. Computed by `eval_stats.py` and `cot_monitor_stats.py` from [results/judge_eval/](results/judge_eval/) and [results/cot_monitor/](results/cot_monitor/).

| | base | A | B | C | D |
|---|---|---|---|---|---|
| Hack rate | 44% | 64% | 53% | 36% | 56% |
| Probe recall at τ | 69% | 78% | 29% | 42% | 68% |
| Probe FPR at τ | 14% | 25% | 6% | 5% | 16% |
| Hacks the probe misses (share of samples) | 13% | 14% | 37% | 21% | 18% |
| Probe AUROC | 0.88 | 0.85 | 0.87 | 0.90 | 0.87 |

- B evades the threshold rather than the direction: recall falls to 29% but AUROC stays at 0.87 (base 0.88). Source: [results/judge_eval/](results/judge_eval/).
- During training, recall over steps 91-100 is 77% for A, 22% for B, 17% for C and 62% for D, from 62-66% in steps 1-10. Source: [results/judge_rl/](results/judge_rl/), plotted by [figs/fig_training_panel.py](figs/fig_training_panel.py).
- Most of B's recall drop comes from the text it writes. Base weights scoring B's transcripts catch 39% of hacks, against 29% for B scoring its own. B's weights scoring base transcripts catch 65%, against 71% for base. The report puts this at 76-85% of the drop. Source: `cells` in [results/score_analysis.json](results/score_analysis.json) and [rl.html](../../reproduction/multi-model/report/rl.html).
- A reasoning-only LLM monitor (GPT-5.6 Sol, low effort, score ≥ 50) catches 97% of B's hacks at 3% FPR, and 98% of the B hacks the probe misses. Source: [results/cot_monitor/](results/cot_monitor/).
- C hacks least under the lenient prompt, but its strict-prompt hack rate rises from 6% (base) to 17%. Source: [results/judge_eval/](results/judge_eval/) (`*_str_strict.jsonl`).
- D keeps recall near base, but only 22% of its training samples carry gradient, against 35% for A and 55% for B. This confounds the comparison. Source: [rl.html](../../reproduction/multi-model/report/rl.html), computed from raw samples that are not in the repo.

## Layout

```
cluster/                  GPU-host code (the versions that ran)
  rl_train_gptoss.py      GRPO trainer; also does cross-model re-scoring (--score-in) and per-token traces (--score-trace)
  tests/dq_experts.py     just-in-time MXFP4 expert dequantisation, imported by the trainer
  vllm_ext.py             vLLM worker extension that loads merged attention weights in place
  gptoss_rollout.py, env_lcb.py   harmony prompt rendering, exact-token rollouts, sandboxed test reward
  serve_gptoss_rl.sh, sample.py   vLLM sampler; standalone sampler for the pilot and the eval
  run_parallel.sh, run_resume*.sh A and B runs (paused at steps 37/41, resumed to 100)
  run_arm{s,B,C,C2,D}.sh, boost_armC.sh  per-arm runners (C and D chain their own held-out eval)
  run_eval.sh, run_evalbase.sh    held-out eval sampling + probe scoring
  run_score.sh, score_analyze.py  cross-model re-scoring (base/A/B weights x base/A/B texts) -> score_analysis.json
  run_trace.sh            per-token layer-13 traces for base/B weights x base/B texts
  pilot_gptoss.sh, calib_gptoss.sh, score_probe_gptoss.py, calib_analyze.py   calibration pilot and choice of τ
  train_tasks.jsonl (897), ib_eval_tasks.jsonl (309)   training and held-out tasks
  base_freeze.txt, build_env.sh, env.sh                pinned environment
  runs/arm{A,B,C,D}/{args.json,log.jsonl}              per-step training logs; runs/summary.txt is a mid-run A/B snapshot
  watchdog*.sh, kill_evalbase.sh, stop_all.sh, post.sh, armstat.py   run management
  rl_train.py (Qwen), rl_train_gptoss_v3.py, tests/, agent_scratch/  earlier trainers, smoke tests, design-phase checks
judge_rl.py, judge_eval.py  judge every training / held-out sample (GPT-5.6 Sol)
cot_monitor.py            reasoning-only LLM monitor on the held-out samples
eval_stats.py, judge_rl_stats.py, c_stats.py, d_stats.py, cot_monitor_stats.py, judge_rl_lenctl.py   tables
lex_diff.py, peak_stats.py  phrase rates in hack reasoning; where the probe peaks
gen/                      builds training tasks from LiveCodeBench with the ImpossibleBench pipeline
figs/                     make_figs.py (fig1-fig6) and fig_training_panel.py, with their PDF/PNG outputs
results/                  judge verdicts (judge_rl/, judge_eval/), monitor scores (cot_monitor/), score_trace/,
                          score_analysis.json, pilot verdicts (judge_p1_gptoss.jsonl), eval_labels.json
notes/                    trainer and sampler design notes
status.sh                 one-shot progress check over ssh
```

## Reproducing

Requirements:
- GPU host: the runners assume one 8-GPU node (two trainer ranks on GPUs 6-7, samplers on 2-6 other GPUs). The design notes target H200s. Build the env with `cluster/build_env.sh` (Python 3.12, uv, `base_freeze.txt`: torch 2.13.0, vLLM 0.30.0, transformers 5.18.0, plus peft and flash-linear-attention).
- The gpt-oss-120b HF snapshot.
- The gpt-oss DoM vectors (`dom/gptoss_h/dom_vectors.npz`), built by [../../reproduction/multi-model/cluster/build_dom.py](../../reproduction/multi-model/cluster/build_dom.py) via `cluster/t1gh.sh` in that section. They are not in the repo.
- Locally: Python 3 with numpy, scipy and matplotlib. The judge and monitor import `p0_judge.py` and `judge.py` from [../../reproduction/multi-model/](../../reproduction/multi-model/), and `env_lcb.py` from `cluster/`.
- Keys: an OpenAI key for the GPT-5.6 Sol judge and monitor (`OPENAI_API_KEY_2`, read from a `.env` file; use the root `.env`). The vLLM servers read a bearer key from a file on the GPU host. Scripts that ssh to the host read `RL_HOST=user@host`.

Order:
1. Tasks (optional; already in `cluster/`): `cluster/lcb_extract.py`, then `gen/gen_tasks.py`.
2. Calibration (GPU): `pilot_gptoss.sh`, then `calib_gptoss.sh`, then `calib_analyze.py`. This fixes τ = 770537.
3. Training (GPU): `run_parallel.sh` and `run_resume2.sh` for A and B, then `run_armC.sh` and `run_armD.sh`. Each arm writes `runs/armX/{log.jsonl,samples/,ckpt_*}`.
4. Held-out eval (GPU): `run_eval.sh`, plus `run_evalbase.sh` for the base model. C and D run their own eval at the end of their runner scripts.
5. Cross-scoring (GPU): `run_score.sh`, then `score_analyze.py` (writes `analysis.json`, copied to `results/score_analysis.json`). Then `run_trace.sh` for the per-token traces.
6. Judging (local, rsyncs from `RL_HOST`): `judge_rl.py --arms armA,armB,armC,armD`, `judge_eval.py`, then `cot_monitor.py`.
7. Tables and figures (local): `eval_stats.py`, `cot_monitor_stats.py`, `c_stats.py`, `d_stats.py`, `judge_rl_stats.py`, then `python figs/fig_training_panel.py` and `figs/make_figs.py`.
8. Write-up: in `reproduction/multi-model/`, `rl_section.py`, then `fill_report.py` (see `run_final.sh` there).

From the repo as-is, only `figs/fig_training_panel.py` runs unchanged; it reads `results/judge_rl/` directly. Point `ROOT` in `figs/common.py` at a directory whose `data/` holds the contents of `results/` (a `data -> results` symlink works) and `make_figs.py` also runs. With the same path edit, `eval_stats.py`, `cot_monitor_stats.py`, `c_stats.py`, `d_stats.py`, `judge_rl_stats.py` and `judge_rl_lenctl.py` run on `results/`. `judge_rl.py`, `judge_eval.py`, `cot_monitor.py`, `lex_diff.py`, `peak_stats.py` and `rl_section.py` also need the raw samples, which are not included.

## Notes

- Not included because of size: the raw training and eval samples (`rl_samples/`, `eval_samples/`, about 1.2 GB), the LoRA checkpoints, the cross-scoring `.npz` files, the per-token trace arrays and the calibration-pilot dumps. Everything under `results/` is derived from them.
- Hard-coded local paths. These scripts point at `/Users/.../j-lens/rh_rl/data`, a local mirror of `results/` plus the raw samples: `c_stats.py`, `d_stats.py`, `eval_stats.py`, `cot_monitor_stats.py`, `judge_rl_stats.py`, `judge_rl_lenctl.py`, `lex_diff.py`, `peak_stats.py`, `figs/common.py`, `judge_rl.py`, `judge_eval.py`, `cot_monitor.py`, `status.sh` and `../../reproduction/multi-model/rl_section.py`.
- `judge_rl.py`, `judge_eval.py` and `cot_monitor.py` also import from `/Users/.../j-lens/rh_probe` (= `reproduction/multi-model/`) and `/Users/.../j-lens/rh_rl/cluster`. They read tasks from `gen/*.jsonl`, but the task files are in `cluster/`.
- `gen/gen_tasks.py` and `reproduction/multi-model/judge.py` read the API key from a hard-coded `.env` path outside the repo.
- Hard-coded GPU-host paths. Every `cluster/*.sh` script and the trainer `cd` to or import from `/data/.../rh_rl`; the cluster scripts expect `cluster/` to be that directory. The scripts also hard-code `/data/.../rh_probe/api_key`, the DoM vectors in `/data/.../rh_probe/dom/gptoss_h/`, and the gpt-oss snapshot path in `serve_gptoss_rl.sh`. `env.sh` sets HF and uv caches under `/data`. GPU indices and ports are hard-coded in each runner.
- `runs/arm*/args.json` records the last launch of each arm, so `resume` names the checkpoint it restarted from. A and B were paused at steps 37 and 41 by a GPU deadline. C moved from 2 to 6 samplers at step 35. All four reached 100 steps (`log.jsonl` has 100 lines each).
- Held-out lenient samples are split across `_len`, `_len2` (and for C and D `_len3`) files; the stats scripts pool them, 8 samples per task for every model.
- `results/eval_labels.json` is not read by any script in the repo.
- Training and held-out samples are judged with the same prompt as the calibration pilot ([../../reproduction/multi-model/p0_judge.py](../../reproduction/multi-model/p0_judge.py)).
