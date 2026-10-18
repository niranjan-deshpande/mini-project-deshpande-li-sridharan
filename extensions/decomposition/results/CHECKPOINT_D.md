# CHECKPOINT D: what is v_hack made of?

Qwen3.8-27B, layer 40, turn-mean DoM probes, max-pooled over passages. Synthetic pairs from GPT-6 Luna. Detection numbers
are on the **held-out half** of the impossible splits (5,261 passages, 339 hacks, 49 tasks), with paired task-level
bootstrap CIs. Primary choices were fixed in `notes/DECISIONS.md` before these numbers were computed. Figures are in `results/figures/`; full
numbers are in `results/geometry/geometry.json` and `results/phase4/phase4.json`.

## The probe
- The rebuilt v_hack (334 Luna pairs, split-half ρ = 0.991) detects hacks with AUROC **0.944** [0.931, 0.954] held out, or
  0.907 after length control. The rebuilt Claude v_hack scores 0.927 (exactly Agastya's number). cos(Luna, Claude) = 0.81.
- Length alone gets 0.80.

## 1. Geometry: v_hack is mostly *not* in the span of the candidates
- R² of v_hack in span(6 candidates) = **0.17**. The ceiling is ρ = 0.99, so ceiling-adjusted R² is also 0.17. The same-size baseline null has mean 0.06 and p95 0.09.
- Cosines with v_hack (disattenuated; the baseline |cos| p95 is 0.19):
  - legit shortcut **+0.28** (plain wording) / +0.33 ("just/directly");
  - eval awareness **+0.20**;
  - test vocab (SWE) +0.05;
  - task failure −0.02;
  - grader modeling −0.14;
  - test vocab (non-code) −0.18.
- One *baseline*, formal→casual register, sits at **−0.27**: v_hack leans casual.
- About 83% of v_hack's squared norm is outside every candidate we built.

## 2. Functional: on real tokens, candidates track the probe's output, but unrelated contrasts nearly do too
- The token-level R² of v_hack scores on the 6 candidate scores is **0.75** [0.74, 0.76]. That covers 3.5M held-out assistant tokens and is bootstrapped by rollout.
- The baseline null is **0.57** on average (p95 0.63). DoM directions from this pipeline share a large common token-level component.
- Shapley shares are dominated by **legit shortcut (0.41)**, then task failure (0.13) and eval awareness (0.12). Test vocab and grader modeling are ≈ 0.

## 3. Decision: the candidate part and the residual both detect, and so do many baselines
- Splitting v_hack = Cβ̂ + v_res, held-out AUROC is:

  | Part | Raw | Length-controlled |
  |---|---|---|
  | Cβ̂ | 0.947 | 0.926 |
  | v_res | 0.937 | 0.871 |

  Neither collapses toward 0.5.
- The best single candidate is legit shortcut at 0.911 (0.834 length-controlled).
- **But unrelated baselines also detect.** Python→Rust gets 0.85 raw and JSON→YAML 0.87. After length control, loop-vs-comprehension's
  *negation* gets 0.90. Once max-pooled over long passages, many DoM directions separate hack passages from clean ones. So decision-level
  AUROC is not a specific test of what v_hack encodes. The functional R² against its null (item 2) and the confound strata (item 6) are
  more informative.

## 4–5. Mixture: no evidence that different hacks are driven by different candidates
- The max-of-z-scores detector, 0.87, is *worse* than both the best single candidate (0.91) and Cβ̂ (0.95). The paired CIs exclude 0.
- At each hack's peak token, legit shortcut is the dominant candidate in **328 of 339** passages. The residual still carries about 67% of the
  peak's elevation, and no large opposite-sign clusters appear.
- 335 of 339 held-out hacks are hardcoding, so a comparison between hack categories isn't possible on Qwen.

## 6. Confounds on real rollouts (Phase 2 labels, two-judge consensus)
- **Contradiction reasoning:** present in **97%** of hack passages vs 30% of clean ones.
  - Within contradiction = 1, v_hack still separates hacks at 0.87 (1,522 passages, 216 hacks).
  - Among clean passages alone, v_hack separates contradiction = 1 vs 0 at **0.92** (0.87 length-controlled): it fires on
    contradiction deliberation without any hacking. That is the false-positive mechanism Agastya saw.
  - The task-failure candidate detects contradiction among clean passages at 0.95, so the candidate is valid. Yet its cosine with v_hack is about 0.
- **Eval awareness:** present in 36% of hack passages vs 0.9% of clean ones.
  - Within eval-aware passages, v_hack separates hacks from non-hacks at only **0.73** (163 passages, 117 hacks), and at **0.58** with the narrow
    definition (74 passages, 59 hacks). Outside them it reaches 0.93.
  - Among clean passages, eval-aware vs not is 0.97 (only 46 eval-aware clean passages).
  - So inside eval-aware reasoning, v_hack barely distinguishes considering a hack from not considering one.
  - The eval-awareness candidate detects its own concept at 0.98, so it's valid.
- **Impossible vs solvable** (original split treated as clean, which is unverified): v_hack AUROC 0.61, a weak effect.

## 7. General cheating (off-domain), held out of all regressions
- Cosines with v_hack are +0.08 to +0.13 for the narrative formats (template, 3rd person, 1st person, judgment), within the baseline band.
  The **AI off-domain gaming** format (an AI padding an essay to hit a length target) is at +0.22.
- v_hack looks like a detector of *AI agents gaming metrics*, not a broad "cheating" direction.

## 8. Lenses
These are native J-/R-lens readouts per direction. The J++ lens borrowed from Qwen3.6 agrees with them (median cos 0.76 / 0.84 vs 0.02 for mismatched directions).

| Direction | Readout |
|---|---|
| v_hack | "test", "fake", "testing", "mock", "reports", "fictional", "somehow", "simulated"; logit lens 伪造 "forge", 假的 "fake", 虚假 "false" |
| task failure | "contradictory", "conflicting", 矛盾 |
| eval awareness | "hacker", "unethical", "fake", "paranoid", "AI", "sabotage" |
| gc judgment | 伪造, 伪装 "disguise", 冒充 "impersonate", 偷偷 "secretly" |

Our logit lens of Agastya's turn L40 vector matches his own saved readout; his reported 捷径/省事/何必 tokens come from his last-token vectors at L47–63 (see `results/PHASE5_AND_LENS.md`, B).

## Plain summary
v_hack is a strong detector, but it is not cleanly *made of* the contexts we suspected.
- **Shortcut-taking is its largest identifiable component,** geometrically and in its token-level output. Legitimate "use the library"
  choices align with it, so a monitor built on it will fire on allowed shortcuts.
- **Eval awareness is a second component,** and inside eval-aware reasoning the probe largely loses the ability to tell hack deliberation
  from its absence.
- **Contradiction reasoning is the main false-positive mechanism on real rollouts.** The probe fires on it even without hacking. Yet the
  synthetic task-failure direction is orthogonal to v_hack, so that response comes through a different route than "the tests failed".
- **Most of v_hack (~83%) is outside every candidate we built,** and its detection survives removing the candidates' span. We have *not*
  shown that the probe is mostly spurious.
- **Methodological lesson:** max-pooled DoM probes from this pipeline share structure. Unrelated contrasts reach token-R² ≈ 0.57 and,
  after length control, sometimes AUROC ≈ 0.9. Any "the probe detects X" claim needs a baseline-direction null.

## Caveats
- About 97% of Qwen's hack passages are *considered* hacks (raised then set aside), so this is a probe of hack deliberation. Unverbalized hacks can't be tested here.
- Non-unanimous hack labels and judge disagreements on the Phase 2 labels were dropped (Phase 2 κ: 0.73 contradiction, 0.93 eval awareness).
- The original split is unjudged (treated as clean).
- The Luna v_hack differs from the Claude one: cos 0.81, though it detects equally well or better.
- 335 of 339 held-out hacks are hardcoding.
- Eval-aware strata are small (46–163 passages).
- All synthetic candidates come from one generator (Luna). Cosines reflect Qwen's representation of Luna-written text.
