# Reward-hacking DoM probes on Qwen3.8-27B, gpt-oss-120b and GLM-4.5-Air

Does the paper's difference-of-means probe pipeline transfer across model families on Impossible-LiveCodeBench
(strict prompt, tools agent, oneoff / conflicting / original splits, one epoch)? Partly. On Qwen3.8-27B the probe
works (held-out AUROC 0.93, beats a generic text monitor), but Qwen almost never carries its hacks out. On gpt-oss-120b
and GLM-4.5-Air, which do carry hacks out, the probe is weak (AUROC about 0.75) and the monitor wins. Split by stage,
the probe detects hacks that are considered in reasoning much better than hacks that are enacted.

The full write-up, with T0-T6 verdicts, is [report/rh_probe_report_filled.html](report/rh_probe_report_filled.html).

## Results

- T0 hack rate on the impossible splits: Qwen 67-70% (2.2-2.3% enacted), gpt-oss 41-71% (19-42% enacted)
  ([results/t0_final.txt](results/t0_final.txt)); GLM 47-56% (37-46% enacted), from `t0_summary.py` on
  [results/labels/judge_glm_full.jsonl](results/labels/judge_glm_full.jsonl).
- Capability control, original split pass rate: Qwen 82/91, gpt-oss 80/103, GLM 65/103; a judge audit found 17 of
  GLM's 65 passes were enacted hacks ([results/t0_capability.json](results/t0_capability.json)).
- T2 held-out passage AUROC: Qwen 0.927 [0.912, 0.942] (turn vector, L40, max pooling); at the monitor's FPR (24%) the
  probe's TPR is 95% vs the monitor's 48% ([results/t2_qwen_final.txt](results/t2_qwen_final.txt)). gpt-oss 0.756
  (L13; monitor TPR 80% at 3% FPR, probe 13%) ([results/t2_gptoss_v1.txt](results/t2_gptoss_v1.txt); 0.728 with
  harmony-rendered vectors, [results/t2_gptoss_h.txt](results/t2_gptoss_h.txt)). GLM 0.746 (L45; monitor 89% vs probe
  50% TPR at 18% FPR) ([results/t2_glm.txt](results/t2_glm.txt)).
- T2b by stage, AUROC considered / enacted: Qwen 0.93 / 0.79 (only 9 enacted positives), gpt-oss 0.92 / 0.71,
  GLM 0.80 / 0.74 ([results/t2_stage_qwen.json](results/t2_stage_qwen.json), `t2_stage_gptoss.json`, `t2_stage_glm.json`).
- T3 honeypot steering toward v: Qwen mentions of the bait rise from 5% to 69% at 1.5x (L47); gpt-oss 12% to 23% at 2x
  (L26); GLM with reasoning 0/103 at 0x and 2x ([results/t3_qwen_summary.json](results/t3_qwen_summary.json),
  [results/t3_gptoss_summary.json](results/t3_gptoss_summary.json),
  [results/t3_raw/glm_think/summary.json](results/t3_raw/glm_think/summary.json)).
- T3b steering GLM against v at 84 enacted-hack cut points, 2 samples each: re-hack rate 32% unsteered, 32% at -1x,
  35% at -1.5x, 21% at -2x but with 43% incoherent outputs (35% among coherent ones); a random direction at 3x gives
  53% among coherent ([results/steer_neg/summary_84.json](results/steer_neg/summary_84.json)). -3x is 95% incoherent
  on a 40-cut subset ([results/steer_neg/summary_40.json](results/steer_neg/summary_40.json)).
- T4 resampling at Qwen probe peaks: 0/136 hacking continuations vs 1/136 at random cut points
  ([results/t4_qwen_judged.jsonl](results/t4_qwen_judged.jsonl)). T5: at 5% FPR the Qwen probe catches 55% of hack
  passages ([results/t2_extra.json](results/t2_extra.json)); the false-positive audit is in the report.
- T1 logit-lens tokens per model are in the report; raw readouts are in
  [results/dom/qwen/logit_lens.json](results/dom/qwen/logit_lens.json) and [results/glm_logit_lens.json](results/glm_logit_lens.json).

## Layout

```
run_rollouts.py           Inspect + ImpossibleBench LCB rollouts against a vLLM server (GPU host)
export_rollouts.py        Inspect .eval logs -> JSONL with passages and judge/monitor renderings
judge.py                  3-run GPT-5.6 Sol judge (Fig 20 rubric) + generic monitor (Fig 17 rubric)
rubric_judge_lcb.txt, rubric_monitor_generic.txt
t0_summary.py, t0_capability.py               T0 hack rates and capability control
t2_analyze.py, t2_stage.py, t2_extra.py       T2 probe vs monitor, by stage, length baselines
t3_detail.py                                  T3 generation-health stats
t4_judge.py, t4_judge_prompt.py               T4 judging of resamples
steer_judge.py, steer_analyze.py              T3b judging and bootstrap summary
p0_judge.py               single-turn judge used for RL Phase 0 / RL eval
*_section.py, fill_report.py, run_final.sh    report fragments and the filled report
rl_section.py             T6 fragment (report/rl.html), arms A-D; reads ../../extensions/rl-against-probe
rl_section_v8.py, rl_section_v10.py           earlier versions of it (2 and 3 arms)
cluster/                  GPU-host scripts
  serve_{qwen,gptoss,glm}.sh, dl_*.py         vLLM serving and weight download
  build_dom.py                                T1: DoM vectors, logit lens, per-pair activations
  t2_acts.py, t2_acts_gptoss.py, t2_acts_glm.py, t2_check.py   T2 per-passage probe scores
  t3_steer.py, t4_resample.py                 T3 honeypot steering, T4 resampling
  t1g*.sh, t2g*.sh, t3g.sh, glm_*.sh, hfrun.sh, env.sh         launch wrappers
  steer_neg/                                  T3b: cut-point builder, vLLM steering plugin, servers, client
report/                   HTML template, filled report, generated fragments
results/
  rollouts/qwen_rollouts.jsonl.gz             271 Qwen rollouts (chat, passages, tools, score)
  labels/                                     judge + monitor labels per passage (Qwen/gpt-oss; GLM; GLM original audit)
  dom/qwen/                                   dom_vectors.npz [64 layers, 5120] per vector key, logit_lens.json
  t2/qwen_part{1,2,3}/                        per-passage probe scores (scores_0.npz) and index (index_0.jsonl)
  t0_*, t2_*, t3_*, t4_*                       summaries behind the report
  t3_raw/                                     raw steering outputs and summaries per model and calibration run
  steer_neg/                                  T3b judged resamples and summaries
```

## Reproducing

Requirements: Python 3.10+; `numpy` for the analysis scripts; `torch`, `transformers`, `vllm` and `huggingface_hub` on
the GPU host; Inspect and ImpossibleBench patched with
[../olmo/patches/impossiblebench-lcb-fixes.patch](../olmo/patches/impossiblebench-lcb-fixes.patch) for rollouts.
The judge scripts use only the standard library and read `OPENAI_API_KEY_2` from a `.env` (use the repo-root `.env`).

On the GPU host:
1. Download and serve the model: `cluster/dl_*.py`, `cluster/serve_{qwen,gptoss,glm}.sh`.
2. Rollouts: `run_rollouts.py --model openai-api/local/<name> --port <port> --tag <tag>` (GLM needs
   `--no-strict-tools`), then `export_rollouts.py <log_dir> <out.jsonl>`.
3. T1: `cluster/build_dom.py --model <snapshot> --family {qwen,gptoss,glm} --pairs ../olmo/synth/pairs_full.jsonl --out dom/<model>`.
   The synthetic pairs come from [../olmo/synth/pairs_full.jsonl](../olmo/synth/pairs_full.jsonl).
4. T2 scores: `cluster/t2_acts.py` (Qwen), `t2_acts_gptoss.py`, `t2_acts_glm.py` (wrappers `t2g*.sh`, `glm_t2.sh`).
5. T3: `cluster/t3_steer.py` (wrappers `t3g.sh`, `glm_t3*.sh`). T4: `cluster/t4_resample.py` against a served Qwen.
6. T3b: `cluster/steer_neg/build.py`, then `run.sh` and `run2.sh` (start steered vLLM servers via `serve.sh` with the
   `steerplug` plugin and query them with `client.py`).

Locally:
7. Judge: `judge.py <rollouts.jsonl> <outdir> --runs 3`; `t4_judge.py`, `steer_judge.py` for resamples.
8. Analysis: `t0_summary.py`, `t0_capability.py`, `t2_analyze.py`, `t2_stage.py`, `t2_extra.py`, `steer_analyze.py`.
9. Report: `run_final.sh` chains the analysis, `*_section.py` and `fill_report.py`.

The Qwen T2 number reproduces on CPU from files in this folder (checked: held-out AUROC 0.927):
```
python3 t2_analyze.py --scores results/t2/qwen_part1 results/t2/qwen_part2 results/t2/qwen_part3 \
    --labels results/labels/judge_full.jsonl --out t2_qwen.json
```

## Notes

- Used by [../../extensions/decomposition](../../extensions/decomposition): `results/rollouts/qwen_rollouts.jsonl.gz`,
  `results/labels/judge_full.jsonl`, `results/dom/qwen/` and `results/t2/`.
- Not in the repo (size): the per-pair activation caches `results/dom/qwen/pair_acts_{span,turn,last}_{hack,honest}.npz`
  (6 files, 65-80 MB each; `acts` [135 pairs, 64 layers, 5120] float16 plus `families`). `cluster/build_dom.py`
  regenerates them as one `pair_acts.npz` with keys `{span,turn,last}_{hack,honest}`; split one key per file.
  Also not included: GLM and gpt-oss rollouts, vectors and per-passage scores, raw Inspect logs, T3b prompts and raw
  continuations, and the exported rollout files `t0_capability.py` and `t2_extra.py` read.
- Path mismatch: the local scripts read a `data/` directory with an older layout (e.g. `data/judge_full/labels.jsonl`,
  `data/t2_qwen_final.json`, `data/t3_raw/`), while this folder stores them under `results/` (e.g.
  `results/labels/judge_full.jsonl`). Symlink or edit before running `fill_report.py`, the `*_section.py` scripts,
  `t0_*.py`, `t3_detail.py` or `run_final.sh`. `t2_extra.py` and `t2_stage.py` take the label path from `LABELS`.
- Hard-coded paths (not edited):
  - `cluster/*.sh`, `cluster/steer_neg/*`, `build_dom.py` (`--pairs` default), `t2_check.py`, `t3_steer.py`,
    `t4_resample.py`, `run_rollouts.py`: a cluster working dir `/data/<user>/rh_probe` (API key file, logs, tmp),
    HF caches under `/data/hf_cache` and `/data/<user>/hf_cache_*`, and a vLLM venv under `/data/<user>/`.
  - `judge.py`: `.env` at a `/Users/<user>/...` path. `p0_judge.py`: `sys.path` to the RL extension's `cluster/`.
  - `rl_section*.py`: `/Users/<user>/.../rh_rl/data`, which corresponds to
    [../../extensions/rl-against-probe/results](../../extensions/rl-against-probe/results); its `eval_samples/` and
    `rl_samples/` inputs are not in the repo. `run_final.sh`: `cd` to a local working dir.
- GLM rollouts and T3b use the FP8 checkpoint on vLLM; GLM activations and T3 use bf16 in HF.
- Only unanimous judge labels are scored. Qwen has 271 of 305 planned rollouts. Layer/vector/pooling are selected on
  one hash-split half of tasks and reported on the other, over a wide grid.
- T3 strengths are multiples of the raw vector, not comparable to the paper's x1-x8.
