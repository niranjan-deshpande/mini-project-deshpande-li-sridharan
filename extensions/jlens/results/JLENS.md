# J-lens / J++ readout of the reward-hacking direction — OLMo 3.1 32B Think

Date 2026-10-15. Extension to Phase 1 (see [RESULTS.md](../../../reproduction/olmo/results/probe/RESULTS.md)). We put the difference-of-means
reward-hacking direction through **more faithful mid-layer lenses** than the plain logit lens, to test
whether the "cheating-concept" readout is a real property of the direction or an artifact of the logit
lens. Three lenses, same direction, same layers:

- **Logit lens** (filtered baseline): `unembed(norm(v))`. What Phase 1 reported.
- **Native J-lens** — finite-difference Jacobian lens. Computes `J_L·v` directly by perturbing the
  layer-L residual stream along `v` and reading the change in final logits; `J_L = E[∂h_final/∂h_L]`.
  **Transfer-free and architecture-agnostic** (no pre-fit matrix, nothing OLMo-specific). This is the
  primary lens. Tool: Niranjan's `fd_lens` (fork branch `rh-probe-decomposition`).
- **J++ lens** (borrowed cross-check) — J-lens + Jacobian filtering + LRP backward + readout-token
  filtering; the most faithful of the family, but needs a pre-fit per-layer map. We use
  `koayon/jpp-lenses → olmo-3-1125-32b/lens.pt`, fit on **Olmo-3 32B *base*** (fitted layers
  8/16/24/32/40/48/56). See the transfer caveat below.

R-lens was skipped on purpose: J++ already subsumes its LRP fix, there is no pre-fit OLMo R-lens, and our
signal lives at mid-depth (L28–48) where plain J is already faithful.

**Scope note:** this writeup covers **Think**. Instruct was in the locked scope but its run did not complete
(the GPU box hit its failsafe before the second model finished) and the box is gone; re-running Instruct is
~16 min of compute (~$3–5 on a fresh box) and is pending. Instruct's *logit*-lens readout is already in
RESULTS.md and replicates the same cheating vocabulary.

## Layer indexing

Our extractor harvests at `hidden_states[L]` (hs[0] = embeddings), so a vector named `span_L28` is the
**output of transformer block 27**. The J-lens/J++ tooling indexes by block output, so block **27 ↔
span_L28**, **32 ↔ span_L33**, **40 ↔ span_L41**, **48 ↔ span_L49**. The DoM was re-harvested at the
hs-indices that line up with J++'s fitted layers (32/40/48) for an apples-to-apples J-vs-J++ comparison
(re-harvest cosine vs the saved vectors > 0.999, self-check C).

## Self-checks (all passed — the run is trustworthy)

- **A — suffix plumbing:** `unembed(suffix(prefix(ids, 40), 40))[-1]` vs a plain `model(ids).logits[-1]`:
  **cos 0.99999998, top-1 token matches.** The split-forward used by the finite-difference lens reproduces
  the model exactly.
- **B — logit-lens parity:** recomputed logit-lens top-15 for `span_L28` matches the Phase-1 saved JSON
  **15/15.**
- **C — re-harvest fidelity:** DoM re-harvested at `span_L40` vs the saved vector, **cos 0.99999763.**

## Result — all three lenses read "cheating", and the faithful lenses sharpen it

Top promoted tokens (readout-filtered), reward-hacking direction, Think, by block:

| Block (vec) | Logit lens | Native J-lens (finite-diff) | J++ (borrowed base) |
|---|---|---|---|
| 27 (`span_L28`) | bypass, directly, artificially, pretend, fake, cheating, dummy, pretended | **fake, bypass, pretending, pretend, cheating, pretended, magically, disguised** | — (not a J++-fitted layer) |
| 32 (`span_L33`) | bypass, directly, outputs, output, artificially, dummy, **cheat** | bypass, fake, pretend, irrelevant, pretending, cheating, dummy, cheat | fake, irrelevant, cheating, cheat, pretending, **Fake, useless** |
| 40 (`span_L41`) | directly, bypass, output, outputs, dummy, fake | fake, bypass, output, dummy, cheat, cheating, pretend | fake, output, bypass, pretending, cheat, pretend, **hardcoded** |
| 48 (`span_L49`) | output, outputs, directly, bypass, dummy, fake | output, fake, bypass, dummy, cheat, pretend, **hardcoded** | output, outputs, bypass, fake, cheat, dummy, **hardcoded** |

**Lens agreement** (median over the span/turn pair at each block; "baseline" = the same metric on a random
direction, so it's the chance floor):

| Block | logit ↔ J-lens cos (baseline) | J-lens ↔ J++ cos |
|---|---|---|
| 27 | 0.69 (0.56) | — |
| 32 | 0.75 (0.59) | 0.63 |
| 40 | **0.86 (−0.00)** | **0.86** |
| 48 | 0.84 (0.65) | 0.82 |

### What this shows

1. **The cheating readout is real, not a logit-lens artifact.** All three independent lenses decode the same
   direction to the same semantic cluster — *fake / bypass / pretend / cheating / dummy / cheat / hardcoded*.
   None of these words are in the synthetic pairs the direction was built from.
2. **The more faithful lenses *sharpen* the result.** Where the logit lens mixes in neutral plumbing tokens
   (`output`, `test`, `directly`, `files`), the native J-lens and J++ push the explicit cheating tokens to
   the very top and surface sharper ones the logit lens misses — `pretending`, `magically`, `disguised`
   (block 27) and especially **`hardcoded`** at blocks 40/48.
3. **`hardcoded` ties the direction straight to behavior.** In Exp1, **100% of OLMo's enacted hacks were
   hardcoding** (`if input==X: return expected`). The deep J-lens/J++ readout naming `hardcoded` is the
   representational echo of exactly that behavioral signature.
4. **Agreement rises with depth and sits far above chance.** logit↔J climbs 0.69 → 0.86 over blocks 27→48;
   at block 40 the random-direction baseline is ≈ 0 while the hack direction is 0.86 — i.e. the lenses agree
   on *this* direction specifically, not on directions in general.
5. **The J++ transfer validates.** The borrowed base-model J++ and the transfer-free native J-lens agree at
   cos 0.82–0.86 at the fitted layers 40/48. The two lenses are built differently and one is borrowed from a
   different (base) checkpoint, yet they give the same answer — so the borrowed lens is reading real OLMo
   structure, not its own fitting artifacts.

## Logit-lens depth band (continuity, for reference)

The saved-DoM logit sweep stays in the cheating cluster across the whole mid-depth band, strongest on the
`span` harvest at L28 and on the `turn` harvest at L40–48:

- `span` L28→L48: bypass/artificially/pretend/fake/cheating/dummy → bypass/output/dummy/fake/artificially
- `turn` L40→L48: test/bypass/**artificially/cheating/fake**/superficial → test/**fake/bypass/artificially/cheating**

## Caveats

- **M3 — J++ transfer.** The J++ map is fit on Olmo-3 32B **base**; we apply it to the post-trained
  Think weights (Think/Instruct are post-trained *from* that base). This is a borrowed/transfer lens, not a
  native fit — hence we lead with the transfer-free finite-difference J-lens and use J++ only as a
  cross-check. The 0.82–0.86 native↔J++ agreement is the evidence the transfer is sound here.
- Single direction (the DoM hack vector), one harvest config (`n_wiki=32`, 128-token baselines).
- Instruct J-lens not yet run (see scope note). Native-FD compute was ~16 min/model (reharvest 395 s + four
  finite-diff blocks ~10–23 s each).

## Files
- `jlens_think.json` — full per-layer readouts (logit / jlens_fd / jpp), agreement metrics,
  self-checks, timings.
- Code: `lenses.py` (`logit_lens` / `fd_lens` / `load_jpp_layer` / `lens_agreement`, from the
  `rh-probe-decomposition` fork, with a multi-GPU device fix) + `jlens_run.py` (the driver + 3 self-checks),
  in the J-lens payload.
