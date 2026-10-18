# J-lens / J++ readout of the OLMo reward-hacking direction

The OLMo reproduction decodes the difference-of-means reward-hacking direction with the logit lens and gets
cheating vocabulary. This section asks whether that readout is real or a logit-lens artifact, by decoding the same
direction with a native finite-difference Jacobian lens (J-lens) and a borrowed J++ lens. On OLMo 3.1 32B Think,
all three lenses read out the same cheating cluster, and the Jacobian lenses rank `hardcoded` near the top at
blocks 40 and 48.

## Results

All numbers are for `allenai/Olmo-3.1-32B-Think`. The full writeup is [results/JLENS.md](results/JLENS.md); the raw
readouts, agreement metrics, self-checks and timings are in [results/jlens_think.json](results/jlens_think.json).

- Self-checks passed: split-forward reconstruction cos 0.99999998 with matching top-1 token (A); recomputed
  logit-lens top-15 for `span_L28` matches the saved Phase-1 readout 15/15 (B); re-harvested `span_L40` vs saved
  vector cos 0.99999763 (C). Source: `self_checks` in [results/jlens_think.json](results/jlens_think.json).
- All three lenses decode the direction to the same cluster: fake, bypass, pretend(ing), cheating, cheat, dummy.
  The native J-lens at block 27 adds `magically` and `disguised`; J-lens (block 48) and J++ (blocks 40, 48) surface
  `hardcoded`. Table in [results/JLENS.md](results/JLENS.md).
- Logit-lens vs J-lens agreement (median cosine of transported vectors): 0.69 / 0.75 / 0.86 / 0.84 at blocks
  27 / 32 / 40 / 48 ([results/JLENS.md](results/JLENS.md), `layers.*.agreement` in the JSON).
- Native J-lens vs borrowed J++: 0.63 / 0.86 / 0.82 at blocks 32 / 40 / 48. The J++ map was fit on the Olmo-3 32B
  base model, so this agreement is the evidence that the transfer to Think is usable.
- `hardcoded` matches the behavioural result that all of OLMo's enacted hacks in Exp1 were hardcoding
  ([../../reproduction/olmo/results/EXPERIMENT_SUMMARY.md](../../reproduction/olmo/results/EXPERIMENT_SUMMARY.md)).
- Compute: about 16 min per model on the GPU host (`total_s` 979; DoM re-harvest 395 s; each finite-difference
  block 10-23 s).
- OLMo 3.1 32B Instruct has not been run through the J-lens. Its logit-lens readout is in
  [../../reproduction/olmo/results/probe/RESULTS.md](../../reproduction/olmo/results/probe/RESULTS.md).

## Layout

```
extensions/jlens/
  lenses.py         logit_lens, fd_lens (native finite-difference J-lens; R-lens mode unused here),
                    matrix_lens + load_jpp_layer (borrowed J++), wikitext_texts, lens_agreement
  jlens_run.py      driver: loads the model, runs self-checks A/B/C, re-harvests the DoM at hs 33/40/41/49,
                    runs logit / J-lens / J++ at blocks 27/32/40/48, plus a logit-lens band sweep; writes JSON
  bootstrap.sh      GPU-host runner: installs deps, downloads the J++ lens, runs jlens_run.py, shuts the host down
  results/
    JLENS.md          writeup: readout tables, agreement table, self-checks, caveats
    jlens_think.json  full output of the Think run
```

Method definitions (J-lens, R-lens, J++) are in
[../decomposition/notes/jpp_lens.md](../decomposition/notes/jpp_lens.md), which the `lenses.py` docstring refers to.

Layer indexing: the extractor saves `hidden_states[L]` with `hs[0]` = embeddings, so a vector named `span_L28` is the
output of block 27. The lens code indexes by block output: block 27 = `span_L28`, 32 = `span_L33`, 40 = `span_L41`,
48 = `span_L49`. J++ is fitted at blocks 8/16/24/32/40/48/56, so block 27 gets logit lens and J-lens only.

## Reproducing

Requirements: one GPU host with enough memory for OLMo 3.1 32B in bf16 (loaded with `device_map="auto"`, so
multi-GPU works), Python 3, `torch`, `transformers==5.19.0`, `accelerate`, `huggingface_hub`, `hf_transfer`,
`numpy`. `datasets` is optional; without it WikiText-103 prompts are fetched from the Hugging Face datasets-server
API. If Hugging Face access needs a token, put it in the root `.env` and export it on the host.

Inputs from other sections (copy to the host working directory):

- `../../reproduction/olmo/results/probe/dom_vectors.npz` (Think DoM vectors)
- `../../reproduction/olmo/results/probe/logit_lens.json` (Think logit-lens readout, used by self-check B)
- `../../reproduction/olmo/synth/pairs_full.jsonl` (synthetic hack/honest pairs, used for the re-harvest)
- For Instruct: `../../reproduction/olmo/results/probe/instruct/dom_vectors.npz` and `.../instruct/logit_lens.json`

Steps:

1. Copy `lenses.py`, `jlens_run.py`, `bootstrap.sh` and the inputs into `~/jlens` on the GPU host.
2. Think run (what produced the committed result), from `~/jlens`:
   ```
   python3 jlens_run.py --model allenai/Olmo-3.1-32B-Think --dom dom_vectors.npz \
     --existing-logit logit_lens.json --pairs pairs_full.jsonl --jpp koayon/jpp-lenses \
     --out jlens_think.json --tag think
   ```
3. Instruct run: rename the Instruct inputs to `dom_instruct.npz` and `logit_instruct_existing.json`, then
   `bash bootstrap.sh`. It writes `jlens_instruct.json` and a `DONE` or `FAILED` marker.
4. Copy the output JSON back into `results/`. The run aborts before the lens loop if any self-check fails.

The J++ file (`koayon/jpp-lenses`, `olmo-3-1125-32b/lens.pt`) and the model weights are downloaded from the
Hugging Face Hub on first use. Nothing in this section runs locally except reading the results.

## Notes

- `bootstrap.sh` is set up for the Instruct-only re-run: it hard-codes `cd "$HOME/jlens"`, `HF_HOME=$HOME/hf`, the
  Instruct model and input filenames. It runs `sudo shutdown -h +70` at start as a failsafe, and `shutdown -h +3`
  on success; it assumes a host launched with shutdown-behaviour set to terminate. Edit these before reusing it.
  It also deletes cached `allenai` model weights under `$HF_HOME` after each run.
- `lenses.py` caches WikiText prompts in `~/.cache/decomp_lenses/`. `jlens_run.py` asserts `d_model == 5120`.
- A copy of `lenses.py` also lives in `../decomposition/gpu/`; this copy differs only in a device-placement fix
  for multi-GPU sharding in `fd_lens`.
- J++ transfer: the J++ map is fit on the base model and applied to the post-trained Think weights, so J++ is used
  as a cross-check, with the transfer-free J-lens as the primary lens.
- Agreement baselines: `lens_agreement` reports a mismatched-pair baseline (vector i under one lens vs vector
  j != i under the other), not a random-direction score. At block 40 the medians are over five vectors (two hack
  directions plus three random ones); at the other blocks only the span/turn pair is compared, so the baseline
  is span-vs-turn and is high.
- Scope: one direction (the DoM hack vector), one harvest config (`n_wiki=32`, 128-token prompts), Think only.
- Model weights, the J++ lens file and per-direction transported vectors are not stored here; the JSON keeps only
  top/bottom token rows and summary metrics.
