# Costs

Caps: $60 RunPod, $80 OpenAI + OpenRouter combined. Budget is a cap, not a target.

## Prices (checked 2026-10-13, OpenRouter live model list; matches OpenAI's announced prices)
| Model | $/M in | $/M out | Batch |
|---|---|---|---|
| GPT-5.6 Terra | 2.00 | 12.00 | 1.00 / 6.00 |
| GPT-6 Luna | 0.10 | 0.50 | 0.05 / 0.25 |
| GPT-6 Sol | 2.00 | 10.00 | 1.00 / 5.00 |
| DeepSeek V4 Pro (OpenRouter) | 0.42 | 0.83 | – |
| DeepSeek V4.1 Flash (OpenRouter) | 0.04 | 0.77 | – |
| Claude Haiku 5.5 (OpenRouter) | 0.10 | 0.50 | 0.05 / 0.25 |

## Baselines
- OpenRouter key lifetime usage before this project: $25.05 (2026-10-13). Our OpenRouter spend = usage − 25.05.
- OpenAI: track via per-request `usage` fields logged by our scripts (and the dashboard).

## Log
| Date | Item | Provider | Cost |
|---|---|---|---|
| 2026-10-13 | CHECKPOINT 0 (all CPU) | – | $0.00 |
| 2026-10-13 | CP A: Luna generation pilots (smoke, v1, v2, v3, samples for all directions) | OpenAI | $1.62 |
| 2026-10-13 | CP A: Phase 2 rubric pilots, Luna side (full-transcript ×2, diagnostics, focused/snippet stages) | OpenAI | $0.46 |
| 2026-10-13 | CP A: contamination judging + Phase 2 pilots, Haiku side | OpenRouter | $1.27 |

| 2026-10-13 | Bulk generation (Luna, ~5,200 pairs incl. repairs and top-ups) | OpenAI | $17.88 |
| 2026-10-13 | Phase 2 labels, Luna side (180 rollouts, both stages) | OpenAI | $0.90 |
| 2026-10-13 | Phase 2 labels, Haiku side + bulk contamination judging (3 rounds) | OpenRouter | $6.06 |

| 2026-10-13 | Emotion stories (24k) + neutral dialogues, Luna low effort | OpenAI | $2.37 |
| 2026-10-13 | GPU: pod 1 (failed at import, ~0.3 h) + pod 2 (sessions 1+2, ~1.4 h), H100 SXM $3.99/h, + volume | RunPod | $6.48 |
| 2026-10-14 | Phase 5 emotion regression + lens check (CPU, laptop) | – | $0.00 |

Network volume rh-decomp (200 GB, AP-JP-1, ~$0.46/day) was deleted on 2026-10-14; no RunPod resources remain.

**Running totals:** RunPod $6.48 · OpenAI $23.22 · OpenRouter $7.33 · **API total $30.55 of $80**
