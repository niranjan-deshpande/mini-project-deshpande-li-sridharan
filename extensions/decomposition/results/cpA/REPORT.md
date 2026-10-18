# CHECKPOINT A report

See also: `prompts/EXAMPLES_rendered.md` (every generation prompt, rendered), `results/cpA/samples.md` (5 pairs per direction).

## Generation pilot (GPT-6 Luna, 20 pairs per boundary direction)

| direction | v1 pass (no repair) | v2/v3 first pass | after 1 repair round | contamination drops (≥2/3 Haiku runs) |
|---|---|---|---|---|
| hack | 13/20 | 17/20 | 20/20 | not judged (A is a hack) |
| testvocab_swe | 15/20 | 16/20 (v3) | 20/20 | 3/20 (v3) — judged; re-judge pending in bulk |
| task_failure | 6/20 | 16/20 | 19/20 | 2/19 |
| legit_shortcut | 16/20 | 16/20 | 20/20 | 0/20 |
| eval_awareness | 20/20 | 20/20 (v3) | 20/20 | 1/20 (v3) |
| grader_modeling | 19/20 | 19/20 (v3) | 20/20 | 0/20 (v3) |

Other directions (5 samples each): all pass after repair. Token check: every passing pair renders with the Qwen template and its span maps to tokens inside the measured turn (0 failures, ~800 tokens/branch).

Cost per pair ≈ $0.003–0.007 (Luna, high effort, incl. repair).

## Phase 2 rubric pilot (20 selection-half rollouts: 8 conflicting, 8 oneoff, 4 original; 1,144 passages)

| label | method | Cohen's κ (Luna vs Haiku) |
|---|---|---|
| contradiction_failure | full transcript, original | 0.65 |
| contradiction_failure | full transcript + inline passage ids | **0.79** |
| eval_awareness | full transcript (+ ids) | 0.48–0.51 |
| eval_awareness | cue-word sentence snippets (±1 sentence), broad definition | **0.88** (passage level; 37 both-positive) |
| eval_awareness | same, narrow (explicit + task-design only, no scoring speculation) | 0.87 (20 both-positive) |
