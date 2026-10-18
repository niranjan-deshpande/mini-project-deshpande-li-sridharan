# Handoff: v_hack decomposition project (state as of 2026-10-13, ~22:45 local)

Read this first, then `../claude_code_prompt.md` (the brief), `PLAN.md`, `DECISIONS.md`, `COSTS.md`, and
`results/CHECKPOINT_D.md` (the current results). Everything below is in the `decomp/` git repo unless noted.

## Who / how to work
- The user is Niranjan (CS 2881R, due **Oct 18**). Project: decompose Goodfire-style reward-hacking DoM probe `v_hack` on Qwen3.8-27B.
- **Stop at checkpoints and wait for replies.** The user gives explicit go-aheads and likes short progress updates while jobs run.
- **Never start GPU or other billable RunPod resources without an explicit go.** State hourly prices first. The budget is a cap, not a target ($60 RunPod, $80 API).
- They prefer simple, standard, documentable methods: API generation over Claude subagents, no over-engineered adaptive schemes, baseline directions as the null. They pushed back on overkill (no GPT-6 Sol).
- They use Wispr Flow, so messages may arrive as pasted blocks. They confirmed that's them.
- They asked: once a checkpoint is approved, run things; when a cheap fix saves wall-clock (e.g. SSH-debugging a pod), just do it.

## Where we are
Checkpoints 0, A, B, C and D are done; CHECKPOINT D was reported. **The user approved next steps A and B**, then asked for this handoff:
- **A. Phase 5 emotion regression (CPU only, data already downloaded).**
- **B. The logit-lens discrepancy:** our logit lens of Agastya's saved `turn__all` L40 vector does NOT show his reported 捷径/省事/何必/绕过.
- **Then D:** the writeup, and pushing the code to a new GitHub repo (the user must authenticate `gh`/git; the GitHub MCP was failing earlier).
- **C (steering) is optional** and needs separate approval plus GPU.

## Update 2026-10-14
A (emotion regression) and B (lens discrepancy) are DONE: `results/PHASE5_AND_LENS.md`, `results/phase5/phase5.json`, `results/lenses/lens_discrepancy.json`. Lasso uses `precompute=True` (Gram) — 40x faster. Code pushed (public) to fork branch https://github.com/niranjan-deshpande/2881r-mini-project/tree/rh-probe-decomposition (subtree at `rh_decomp/`; update with `git subtree pull --prefix=rh_decomp <path-to-decomp> main` in a clone of the fork). Collaborator explainer artifact: https://claude.ai/artifact/6C4tN4WzD4oVN5gwSZVkQ9 (source `results/explainer/inside_the_probe.html`). Next: formal writeup; steering (C) optional.

## Key results so far (details in results/CHECKPOINT_D.md)
- **Probe:** Luna v_hack held-out AUROC 0.944 (0.907 length-controlled). Claude/Agastya 0.927. cos(Luna, Claude) = 0.81. CP C on the selection half: Luna 0.951 vs Claude 0.937, so go.
- **Geometry:**
  - R²_geo = 0.17 vs baseline null p95 0.09.
  - Cosines: legit_shortcut_plain +0.28, eval_awareness +0.20, testvocab_swe +0.05, task_failure −0.02, grader −0.14, testvocab_noncode −0.18.
  - The formal_casual baseline is at −0.27 (v_hack leans casual).
- **Functional:** token R² 0.75 vs baseline null mean 0.57 / p95 0.63. Shapley: shortcut 0.41, task_failure 0.13, eval 0.12.
- **Decision:**
  - Cβ̂ 0.947 and v_res 0.937 both detect.
  - Baselines also detect: up to 0.87 raw, and 0.90 for loop_comprehension negated after length control. So decision AUROC is non-specific.
- **Mixture:** max detector 0.87 < best single 0.91 < Cβ̂ 0.95. Shortcut dominates 328 of 339 peaks; residual share 0.67.
- **Confounds:**
  - P(contradiction | hack) = 0.97 vs 0.30 for clean. v_hack separates clean contradiction from no contradiction at 0.92.
  - Within eval-aware passages, v_hack AUROC is only 0.73 (broad) / 0.58 (narrow).
  - Candidate validity: task_failure detects contradiction at 0.95; eval_awareness detects eval awareness at 0.98.
- **General cheating:** narrative formats +0.08 to +0.13; AI off-domain +0.22.
- **Lenses:** v_hack J-lens → "test", "fake", "mock", "fictional"; logit lens → 伪造/假的. The borrowed J++ agrees with the native lenses.

## Files and data map (decomp/)
- **Code:**
  - `src/rhdecomp/` is the analysis library (23 tests: `.venv/bin/pytest`); its `gen/` subpackage holds the render/checks/tokcheck modules.
  - `scripts/`:
    - `generate.py`: Luna pair generation with a repair round.
    - `judge_pairs.py`: contamination judging with Haiku via OpenRouter.
    - `label_phase2.py` + `label_eval_focus.py`: Phase 2 labels (eval awareness uses snippets).
    - `bulk.py`: resumable driver.
    - `gen_emotions.py`, `geometry_analysis.py`, `passage_scores.py`, `checkpoint_c.py`, `phase4.py`, `figures.py`.
  - `gpu/`: `session1.py`, `session2_emotions.py`, `lenses.py`, `run_pod.sh`, `run_after.sh`, `run_final.sh`, `verify_small.py`.
- **Venv:** `decomp/.venv` (Python 3.13, torch CPU, transformers 5.19, sklearn, matplotlib). Keys are in `../.env` (OPENAI_API_KEY, OPENROUTER_API_KEY, HF_TOKEN). Never print them.
- **Prompts/config:** `prompts/` (directions.json, gen_*.md, schemas, banned_words.json, task_seeds.jsonl, rubrics). `prompts/emotions/` holds emotions.json with `subset_40`, plus topics.json (our own 50 topics).
- **Final synthetic data:** `data/final/pairs_<direction>.jsonl` (22 directions) and `data/final/phase2_labels.jsonl` (consensus labels: contradiction_failure, eval_awareness [broad, primary], eval_awareness_narrow; None means the judges disagreed).
- **GPU outputs, downloaded (gitignored):** `data/gpu/s1/`
  - `directions.npz`: V [K, 5120] and names, 67 directions at L40, including `hack__turn`, `claude_rebuilt__turn`, `agastya__turn`, `*__span`, 20 `null_hack_shuffle*`.
  - `pairs/<dir>.npz`: per-pair acts at layers 24/32/40, keys turn/span/last _hack/_honest, plus ids.
  - `claude_pair_acts.npz`.
  - `rollouts/<uid>.npz`: `pos` (absolute token positions of assistant tokens), `proj` [T_a, 67] (⟨v, x_t⟩, unnormalised), `names`, `n_tokens`; plus `<uid>.passages.json` (passage token ranges tok0/tok1 in absolute positions).
  - `extra/<uid>.npy`: [T_a, 2] projections on legit_shortcut_plain / colloquial (`extra/columns.json`).
  - `emotions/`:
    - `emotion_vectors.npz`: names (40), raw, denoised, split_half_cos (median 0.93).
    - `story_acts.npz`, `neutral_pca.npz` (46 PCs = 50% variance).
    - `rollouts/<uid>.npy`: [T_a, 80], **unit-normalised** emotion vectors; columns in `rollouts/columns.json`: first 40 `<emo>__raw`, then 40 `<emo>__denoised`. Rows are aligned with `rollouts/<uid>.npz` `pos`.
  - `lenses.json` (logit / j_lens / r_lens / jpp_borrowed + agreement), `manifest.json`, `session1.log`, `check.json`.
- **Analysis intermediates:** `data/analysis/dom_turn_L40.npz` (all turn DoMs, including legit_shortcut_plain), `extra_directions.npz`, `passages.npz/.jsonl`.
- **Results:** `results/geometry/geometry.json`, `results/phase4/phase4.json`, `results/cpC/checkpoint_c.json`, `results/cpA/`, `results/figures/fig1–4`, `results/CHECKPOINT_D.md`.
- **Inputs from the other repos (read-only):**
  - `../agastya-repo/rh_probe/results/` holds the rollouts gz, `labels/judge_full.jsonl` (`gt` = unanimous label), the `dom/qwen/` vectors and `logit_lens.json`, and `t2/` scores.
  - `../olmo-repo/synth/pairs_full.jsonl` holds the Claude pairs.

## Task A: emotion regression (Phase 5; brief's spec)
- **Target:** token-level v_hack scores (`hack__turn` column of `rollouts/<uid>.npz` proj) on held-out impossible-split rollouts (md5(sample_id) % 2 == 1; splits conflicting/oneoff).
- **Predictors:** the 40 standardised **denoised** emotion scores (primary; raw secondary) from `emotions/rollouts/<uid>.npy`. Same row order as `pos`.
- **Collinearity:** emotion vectors sum to zero (grand-mean subtraction), so the scores are exactly collinear. Ridge is fine; Lasso is non-unique, so make claims at cluster level (clusters in `prompts/emotions/emotions.json` → `subset_40_by_cluster`).
- **Method:**
  - Lasso and Ridge with λ chosen by GroupKFold CV grouped by rollout, then held-out R². Use `src/rhdecomp/emotions.py`: `fit_penalized`, `selection_frequencies`; sklearn alpha ↔ λ as noted there.
  - Subsample tokens (e.g. ≤300k, stratified by rollout) for speed.
  - Report Lasso selection frequencies over rollout-bootstrap refits.
- **Null:** run the same procedure with each of the 10 baseline DoM scores (and a couple of shuffles) as the target. "A subspace of many emotion directions can explain a lot of almost anything."
- **Also worth reporting:**
  - cosines of v_hack with each emotion vector, against the baseline null;
  - desperate/frustrated/anxious/calm in particular, because Sofroniew found desperation drives reward hacking and calm suppresses it;
  - optionally, regressing the other candidates on the emotion subspace.
- **Generator caveat:** the stories came from GPT-6 Luna (low effort), 600 per emotion across our own 50 topics, not Sofroniew's 1,200 from Claude. Averaged from token 50 onward, layer 40.

## Task B: the logit-lens discrepancy
- `data/gpu/s1/lenses.json['logit']['agastya__turn']` gives nonsense such as 文件格式 and 月底前. Agastya's report (`../agastya-repo/rh_probe/report/rh_probe_report_filled.html` and `results/dom/qwen/logit_lens.json`) says 捷径/省事/何必/绕过.
- Check which key and layer his `logit_lens.json` used: `lens[key][L]['top']` for `turn__all` at L40, and other layers/harvests such as span.
- Check whether his decode uses the final norm on the vector (his `build_dom.py`: `head(norm(v))` with the model's norm). Ours uses `gpu/lenses.py::logit_lens`; check norm handling and the readout filter.
- This needs only CPU: compare our L40 readout vs his JSON, and read his report's text for which vector and layer he quoted.

## Infrastructure state (RunPod)
**2026-10-14: volume i2qaejmqzv DELETED at Niranjan's request; no pods or volumes remain. The notes below are historical. A new direction now needs a fresh pod: re-download the model and rerun the rollout pass (~1 H100-h, ~$4).**

- **No pods running.** Network volume `rh-decomp`, id **i2qaejmqzv**, 200 GB, AP-JP-1, bills about $0.46/day. It holds:
  - `/workspace/models/Qwen3.8-27B` (the 41-layer use loads the full model),
  - `/workspace/models/jpp/qwen3.6-27b/lens.pt`,
  - `/workspace/venv`,
  - `/workspace/code`, `/workspace/data`,
  - `/workspace/out` (including the **90 GB fp16 L40 activation cache** in `out/cache/<uid>.npy`, rows = assistant tokens).

  Delete it (`runpodctl network-volume delete i2qaejmqzv`) only when the user says so.
- **Getting a pod in AP-JP-1:**
  - AP-JP-1 has no CPU pods, only H100/H200: `runpodctl pod create --image runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 --gpu-id "NVIDIA H100 80GB HBM3" --data-center-ids AP-JP-1 --network-volume-id i2qaejmqzv --container-disk-in-gb 40 --ports "22/tcp" --wait`. The actual price is **$3.99/h** (not the catalog $2.69).
  - SSH key: `~/.runpod/ssh/runpodctl-ssh-key`. The pod has no rsync, so use scp or tar over ssh.
- **Self-termination:**
  - Copy `~/.runpod/config.toml` to the pod's `/root/.runpod/config.toml` (container disk only) and delete via the REST API: `curl -X DELETE https://rest.runpod.io/v1/pods/$POD_ID` with the parsed apikey. See `gpu/run_final.sh`.
  - Never `set -x` around the key. A key leak into a volume log happened once and was redacted; the user was advised to rotate the RunPod key after the project.
- RunPod balance is about $25.87.

## Gotchas learned
- **Laptop sleep kills in-flight HTTP requests.** Scripts now use client timeouts (OpenAI 480 s, OpenRouter 300–900 s) and are resumable; just rerun them. Run long jobs with `nohup caffeinate -i`.
- **`rhdecomp/__init__` imports sklearn,** so any pod needs scikit-learn and scipy.
- **`length_residualize` takes a dict.** Scripts wrap it: `length_residualize = lambda s, n: _lr({"x": s}, n)["x"]`.
- **Baseline-sign issue:** a baseline's sign is arbitrary, so use max(AUROC, 1−AUROC) for baselines as a decision null.
- **The label-shuffled null is invalid here,** because v survives balanced flips via per-pair magnitude variation. Only baselines are the null (DECISIONS.md).
- **Phase 2 judge transcripts** get [P#] ids written into the block headers (`label_phase2.tag_passages`). Without them Luna mis-addresses quotes.

## Costs so far
API $30.55 of $80 (OpenAI $23.22, OpenRouter $7.33; the OpenRouter baseline usage before the project was $25.05). RunPod $6.48 of $60 plus the volume. Keep `COSTS.md` updated.
