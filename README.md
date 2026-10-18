# Reward-hacking directions in open models

This repo contains our code for CS 2881R HW2. We reproduced Goodfire's *Monitoring and Discovering Reward Hacking with Internal Representations
during LLM Evaluations* ([arXiv:2609.19101](https://arxiv.org/abs/2609.19101)) on open-weight models, and then extend it
three ways. You can see our report at: **FILL THIS IN**. 

In brief, the paper builds a difference-of-means "reward-hacking direction" from synthetic hack/honest pairs of SWE-agent rollouts. They then use this direction as a probe and steering vector. We replicate the paper (this is non-trivial, because they do not publish source code and exclude many engineering choices) on smaller open-weight models. Then, we extend their results in three ways:
1. We build a J-lens and use it to decode their reward-hacking vector.
2. We RL gpt-oss-120b with various ways of penalizing activations along the reward-hacking vector. We test whether this reduces reward hacking, and whether it teaches the model to evade the probe.
3. We decompose the reward hacking vector by constructing plausible confounders, a set of emotion vectors, and controls. 

Note that the git history for our repo might be unusual: we each worked in separate repos and combined them into this repo before submitting, so you cannot observe intermediate results.

## Sections

| Folder | Question | Model(s) |
|---|---|---|
| [reproduction/olmo](reproduction/olmo) | Do the paper's main claims replicate on a mid-sized open model? (hack rates, logit-lens readout, probe AUROC, probe vs. monitor, steering) | OLMo 3.1 32B |
| [reproduction/multi-model](reproduction/multi-model) | Does the probe pipeline transfer across model families? Does steering change behaviour? | Qwen3.8-27B, gpt-oss-120b, GLM-4.5-Air |
| [extensions/jlens](extensions/jlens) | The authors read out their `v_hack` as "cheating" with logit lens. Does J-lens confirm this? | OLMo 3.1 32B |
| [extensions/decomposition](extensions/decomposition) | What is the direction made of, in terms of candidate concepts and emotion vectors? | Qwen3.8-27B |
| [extensions/rl-against-probe](extensions/rl-against-probe) | If the probe is used as an RL penalty, does the model stop hacking or learn to evade the probe? | gpt-oss-120b |
| [notes](notes) | Paper notes, background reading, replication plan, Impossible-LCB protocol | — |

In each folder, there is a README with results, layout and reproduction steps. The report itself outlines our research decisions, results, and interpretations. The other READMEs are AI generated and lightly human-edited; they're most useful for reproduction and documentation of engineering decisions.

## Dependencies between sections

- `extensions/jlens` reads the OLMo direction and logit-lens output from `reproduction/olmo/results/probe/`.
- `extensions/decomposition` reads Qwen rollouts, judge labels and DoM vectors from `reproduction/multi-model/results/`.
- `extensions/rl-against-probe` uses the gpt-oss probe built in `reproduction/multi-model`. Its write-up section is
  rendered by `reproduction/multi-model/rl_section.py`.

## Setup

- Python 3.10+. Each section lists its own packages. GPU work used vLLM and Hugging Face `transformers` on rented
  GPU hosts.
- Rollouts use [ImpossibleBench](https://github.com/safety-research/impossiblebench), patched with
  `reproduction/olmo/patches/impossiblebench-lcb-fixes.patch`.
- API keys (Anthropic, OpenAI, Hugging Face) go in a `.env` at the repo root, which is git-ignored.
- Large regenerable caches (activation dumps, raw RL samples, checkpoints) are not committed. Each README says what is
  missing and how to rebuild it.
