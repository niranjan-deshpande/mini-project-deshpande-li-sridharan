# OLMo 3.1 32B reproduction

Do the paper's main claims hold on a mid-sized open model? On OLMo 3.1 32B (Think for rollouts and probing, Instruct for steering), three of five do: the model reward-hacks Impossible-LCB at a substantial rate, the difference-of-means (DoM) direction reads out as "cheating" under the logit lens, and the probe detects hacking well above chance. Two do not: the probe is clearly weaker than an LLM monitor at low false-positive rate, and steering along the direction does not cause honeypot use.

Summary table and discussion: [results/EXPERIMENT_SUMMARY.md](results/EXPERIMENT_SUMMARY.md).

## Results

- Hack rate (Think, 150 rollouts, 3x Claude Sonnet 4.5 judge consensus, defective tasks excluded): hack propensity 46% with the loose prompt and 36% with the strict prompt on impossible splits, enacted 38% / 21%. 0% on solvable controls, which it solves 74-88% of the time. All of its hacks are hardcoding of tested inputs. [results/exp1/FINAL_RESULTS.md](results/exp1/FINAL_RESULTS.md)
- Pass rate understates hacking: hacks pass 29% on loose/oneoff but only 15% on loose/conflicting, and the strict prompt drops oneoff success to 0%. The CoT judge is needed to see the behavior. [results/exp1/FINAL_RESULTS.md](results/exp1/FINAL_RESULTS.md)
- Logit lens: the DoM direction at layers 28-48 decodes to bypass, fake, pretend, cheating, dummy, artificially, none of which are used to build it. The Instruct direction decodes to a similar set. [results/probe/RESULTS.md](results/probe/RESULTS.md), [results/probe/logit_lens.json](results/probe/logit_lens.json), [results/probe/instruct/logit_lens.json](results/probe/instruct/logit_lens.json)
- J-lens follow-up (Think): finite-difference J-lens and a borrowed J++ lens read the same cheating cluster, surface `hardcoded` at blocks 40/48, and agree with each other at cosine 0.82-0.86. Instruct not run. [../../extensions/jlens/results/JLENS.md](../../extensions/jlens/results/JLENS.md)
- Detection: rollout-level AUROC 0.78 on a held-out half, 0.84 for the best config (span harvest, L48, max pooling); logistic-regression probe 0.83. Eval set n=112 (31 hacked). Paper range is 0.74-0.97. [results/probe/RESULTS.md](results/probe/RESULTS.md)
- Probe vs monitor: at about 10% FPR the probe catches 52% of hacks; a single-pass Claude monitor catches 81% at 9% FPR. [results/probe/RESULTS.md](results/probe/RESULTS.md)
- Honeypot steering (Instruct, announce-only framing, L40, n=48 per strength): bait use 0-4.2% at every strength from -8 to +8, baseline 2.1%, no dose-response (paper: about 0.5% to 25%). A diagnostic shows the hook works: output degrades to gibberish by strength 24 before any cheating appears. [results/probe/INSTRUCT_STEERING.md](results/probe/INSTRUCT_STEERING.md)
- Think steering is not measurable: within 448-1024 new tokens the model never finishes reasoning and writes code. [results/probe/RESULTS.md](results/probe/RESULTS.md) section 3

## Layout

```
exp1/                     Experiment 1: P(hack) on Impossible-LCB
  vm_serve_and_harness.sh   GPU host: install patched ImpossibleBench, start vLLM (TP=2, 65k ctx) in tmux
  smoke_test.py             check the vLLM endpoint (think block, multi-turn template, code extraction)
  run_pilot.py              25 tasks x {original, oneoff, conflicting} x {strict, loose} prompts via Inspect
  analyze.py                pass rates with Wilson CIs from Inspect .eval logs
  judge_cot.py              single-pass Claude CoT judge (used as the monitor baseline)
  judge_consensus.py        3x judge, majority vote (ground-truth labels)
synth/
  generate_pairs.py         Claude-generated hack/honest minimal pairs, 6 families
  pairs_full.jsonl          231 generated pairs; 211 survive the loader filter and build the probe
probe/
  extract_and_score.py      GPU: span/turn activations -> DoM + LR directions, logit lens, rollout scores
  smoke_template.py         tokenizer-only check of chat template and span finding
  analyze_probe.py          local: held-out AUROC, config sweep, probe vs monitor TPR/FPR
  steer_honeypot.py         GPU: honeypot steering sweep (announce-only framing)
  diag_steer.py             GPU: one problem at rising strength, to confirm the hook changes output
  analyze_steering.py       local: dose-response table with Wilson CIs
patches/impossiblebench-lcb-fixes.patch   fixes to ImpossibleBench LCB scoring (see Notes)
results/
  EXPERIMENT_SUMMARY.md     scorecard vs the paper
  exp1/                     FINAL_RESULTS.md, transcripts_full.jsonl, labels_consensus.jsonl, monitor_labels.jsonl
  probe/                    RESULTS.md, INSTRUCT_STEERING.md, dom_vectors.npz, logit_lens.json,
                            rollout_scores.csv, honeypot.json (Think), instruct/ (Instruct DoM, lens, honeypot)
```

[../../extensions/jlens](../../extensions/jlens) reads `results/probe/dom_vectors.npz` and `results/probe/logit_lens.json` from here. Background and the Impossible-LCB protocol are in [../../notes/](../../notes/), especially [exp1_impossible_lcb_protocol.md](../../notes/exp1_impossible_lcb_protocol.md).

## Reproducing

Requirements: Python 3.10+, `inspect_ai`, ImpossibleBench (patched), Docker for the Inspect sandboxes, `vllm`, `openai`, `anthropic`, `torch`, `transformers`, `numpy`, `scikit-learn` (optional, for the LR probe). GPU host: two GPUs for vLLM serving the 32B at 65k context; the HF scripts load the model in bf16 with `device_map="auto"`.

Environment: `ANTHROPIC_API_KEY` for the judge and pair generator (keep it in the repo-root `.env` and export it; the scripts read the environment directly). `LOCAL_BASE_URL` (default `http://localhost:8000/v1`) and `LOCAL_API_KEY` for the vLLM endpoint.

1. GPU: put the model at `~/models/olmo-3.1-32b-think` and the patch at `~/impossiblebench-lcb-fixes.patch`, then `bash exp1/vm_serve_and_harness.sh`. Check with `python exp1/smoke_test.py`.
2. GPU: `python exp1/run_pilot.py --log-dir ./logs/exp1_pilot`, then `python exp1/analyze.py ./logs/exp1_pilot` for pass rates.
3. Export the Inspect logs to `transcripts_full.jsonl` (fields `task_id, split, prompt, score, errored, messages, ...`). No export script is included; the committed file is in `results/exp1/`.
4. Local: `python exp1/judge_cot.py transcripts_full.jsonl --out monitor_labels.jsonl` (monitor) and `python exp1/judge_consensus.py transcripts_full.jsonl --runs 3 --out labels_consensus.jsonl` (labels).
5. Local: `python synth/generate_pairs.py --n-per-family <N> --out synth/pairs_full.jsonl`.
6. GPU: `python probe/smoke_template.py <model>` (run from a directory containing `pairs_full.jsonl` and `transcripts_full.jsonl`), then `python probe/extract_and_score.py --model <model> --synth synth/pairs_full.jsonl --rollouts transcripts_full.jsonl --outdir <out>`. Writes `dom_vectors.npz`, `synth_vectors.npz`, `logit_lens.json`, `rollout_scores.csv`.
7. Local: `python probe/analyze_probe.py --scores rollout_scores.csv --labels labels_consensus.jsonl --monitor monitor_labels.jsonl`.
8. GPU, Instruct: `extract_and_score.py --model <instruct> --no-score` for the Instruct direction, then `python probe/steer_honeypot.py --model <instruct> --dom <out>/dom_vectors.npz --layer span_L40 --rollouts transcripts_full.jsonl --n 48 --strengths -8 -4 0 2 4 6 8 --out honeypot_dom.json` (add `--random` for a random-direction control). `python probe/diag_steer.py <instruct> <dom.npz>` checks the hook.
9. Local: `python probe/analyze_steering.py --dom honeypot_dom.json [--random honeypot_random.json]`.

To skip steps 1-4, use the committed transcripts and labels in `results/exp1/`.

## Notes

- Hard-coded paths: `exp1/vm_serve_and_harness.sh` assumes `~/venvs/vllm`, `~/models/olmo-3.1-32b-think`, `~/impossiblebench`, `~/impossiblebench-lcb-fixes.patch` and writes `~/vllm.log`. `probe/smoke_template.py` defaults to `~/models/olmo-3.1-32b-think`. `probe/smoke_template.py` and `probe/diag_steer.py` open `pairs_full.jsonl` / `transcripts_full.jsonl` from the working directory.
- `steer_honeypot.py` defaults to `--layer L40`, but the `.npz` keys are `span_L40` etc., so pass `--layer span_L40`.
- The patch fixes ImpossibleBench LCB measurement: last-code-block extraction, `sys.exit(0)` and similar no longer count as a pass, `allow_test_modifications` forwarding, and an `exclude_broken` option for 4 unrunnable mutated tests. 13 further defective tasks are excluded in the analysis scripts (`DEFECT` set).
- Not committed: Inspect `.eval` logs, model weights, and `synth_vectors.npz` (per-pair activations; regenerate with step 6).
- The committed `results/probe/instruct/honeypot_dom.json` and `honeypot_random.json` are from the earlier run that told the model not to use the helper (all near 0%). The announce-only numbers above are only recorded in [results/probe/INSTRUCT_STEERING.md](results/probe/INSTRUCT_STEERING.md). `results/probe/honeypot.json` (Think) and `probe/diag_steer.py` also use the forbid wording.
- [results/probe/RESULTS.md](results/probe/RESULTS.md) section 3 describes the superseded Think steering run; section 1b and the summary link `JLENS.md` in `results/probe/`, but that file is in [../../extensions/jlens/results/](../../extensions/jlens/results/).
- `extract_and_score.py` drops any synthetic pair whose JSON line contains the substring `error`; that is how 231 pairs become 211.
- Scale of evidence: one benchmark (no SWE-bench Verified), minimal scaffold (no test-file editing, so only simple hacks are available), 211 pairs vs about 5,285 in the paper, n=112 for detection and n=48 per steering cell, rollouts truncated to the last 16k tokens for scoring, steering at a single layer.
