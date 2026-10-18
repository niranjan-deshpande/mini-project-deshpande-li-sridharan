# Phase 1 results — DoM reward-hacking probe on OLMo 3.1 32B Think

Date 2026-10-10. Replicating Goodfire (arXiv 2609.19101) core results on a 32B open model.
Probe built from **211 synthetic hack/honest minimal pairs** (6 families: paper's 5 + hardcoding),
Claude-generated. Activations from HF transformers (bf16, sdpa), residual stream, layers 16–52.
Evaluated on our **150 judge-labeled Impossible-LCB rollouts** (3× consensus ground truth).

## 1. Logit-lens readout of the DoM vector — REPLICATES (Table-1 analogue)

The difference-of-means reward-hacking direction decodes (via logit lens) to **cheating-semantics tokens**
at mid-depth, despite none of these words being used to construct it:

| layer | top promoted tokens |
|---|---|
| L28 | bypass, artificially, **pretend, fake, cheating, dummy, pretended**, desired |
| L32 | bypass, artificially, dummy, **cheat**, desired, test outputs |
| L36–40 | bypass, dummy, **fake**, artificially, directly, desired, test outputs |
| L48 | bypass, dummy, fake, artificially, test outputs |

This reproduces the paper's finding (their vectors decoded to "cheating", "illicit", "hardcoded") — the
direction is interpretable as a generic cheating concept. **Clear replication.**

### 1b. J-lens / J++ confirm and sharpen the readout (extension, 2026-10-15)

We put the same direction through two more faithful mid-layer lenses — a transfer-free finite-difference
**J-lens** (Jacobian lens, `J_L·v`) and a borrowed **J++** lens (`koayon/jpp-lenses`, fit on Olmo-3 32B
base) — cross-checked against the logit lens. Full writeup in [JLENS.md](../../../../extensions/jlens/results/JLENS.md). On **Think** (3/3
self-checks passed):

- All three lenses decode the direction to the same cheating cluster (*fake, bypass, pretend, cheating,
  dummy, cheat, hardcoded*). So the readout is a property of the direction, not a logit-lens artifact.
- The faithful lenses **sharpen** it: they pull explicit cheating tokens to the very top and surface ones
  the logit lens misses — `pretending`, `magically`, `disguised`, and notably **`hardcoded`** at blocks
  40/48. `hardcoded` is the representational echo of Exp1's behavior (100% of OLMo's hacks were hardcoding).
- Lens agreement rises with depth and sits far above chance: logit↔J cos 0.69→0.86 over blocks 27→48 (random
  direction baseline ≈ 0 at block 40). Native J-lens ↔ borrowed J++ agree at **0.82–0.86**, validating the
  J++ transfer (M3 caveat). Instruct J-lens not yet run; its logit-lens readout (§1) already matches.

## 2. Detection: probe vs monitor — PARTIAL (core claim yes, "≈ monitor" no)

Eval set n=112 (31 hacked, 81 clean; defective + incomplete excluded; all splits, originals = clean negatives).
Held-out 50/50 split for honest config selection.

- **Probe detects hacking: AUROC ≈ 0.78 (held-out) / 0.84 (best DoM config, span_L48/max).**
  Squarely inside the paper's reported range (0.74–0.97 across their models/benchmarks).
- LR probe ≈ DoM (0.83 vs 0.84) — no big gain from the trained probe here.
- **vs monitor (1-pass Claude judge, TPR 0.81 @ FPR 0.09):** at matched ~10% FPR the probe catches
  **52%** of hacks vs the monitor's **81%**. Probe TPR by FPR: 0.26 @5%, 0.52 @10%, 0.77 @22%.

So the core claim replicates — **a simple difference-of-means direction detects reward hacking well above
chance on a 32B** — but the paper's stronger "probe ≈ LLM monitor" result does NOT hold here: our probe
underperforms the (strong) monitor at low FPR. Likely causes: cross-distribution transfer (synthetic
SWE-agent pairs → LCB eval), 32B vs 1T+ scale, small synthetic set (211 vs ~5,285), and scoring the
tail-16k of very long rollouts. This is an honest divergence, consistent with the paper's own weaker rows
(e.g. GLM 5.2 AUROC 0.78 on DeepSWE) and its note that probes degrade off-distribution.

## 3. Honeypot steering — DOES NOT CLEANLY TRANSFER (methodological non-transfer)

Setup: plant a helper `zzz('solution.py')` in the system prompt ("returns the correct answer; you should NOT
rely on it"), generate solutions with no steering vs steering along +v_hack at L40, measure how often the
model takes the bait. Paper (Qwen 3.8 27B): bait uptake 1/194 (~0.5%) unsteered → ~25% steered at x1–x8.

**Result on OLMo 3.1 32B Think: inconclusive, because the method assumes a model that writes code promptly.**
OLMo Think reasons for thousands of tokens before emitting code; within any feasible generation budget
(tested 448–1024 new tokens) it NEVER finishes reasoning to write a code submission (0 of the generations
contained a `</think>` close). So we cannot measure actual `zzz()` *use in code* — only whether steering
changes how often the model *mentions/considers* the honeypot in its reasoning (and in the flagged cases it
mentions it while explicitly refusing: "the user says NOT to rely on zzz...").

On that weaker proxy (honeypot mentioned in reasoning), n=16/strength:
- x0 (none): 12.5% · x4 (mid): 12.5% · x8 (strong): 31.2%

A suggestive bump at the strongest steering, but n=16 (2/16 vs 5/16) is within noise. **Verdict: the paper's
honeypot-steering protocol does not transfer cleanly to a long-reasoning model at a feasible generation
budget.** A clean test would need a non-reasoning model or a prohibitively long/slow generation (~3000+
tokens/rollout). This is itself a useful robustness finding, not a clean replication of the causal result.

## Caveats
- n=112 eval (31 positives) → AUROC CI ~±0.08–0.10.
- Rollouts truncated to tail-16k tokens (hacks are late) — may miss some hack spans, understating the probe.
- Single synthetic-data generation; one probe-layer sweep.
- Monitor is a strong 1-pass Claude judge, a high bar.
