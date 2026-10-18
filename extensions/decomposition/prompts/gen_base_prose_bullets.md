# Baseline direction: base_prose_bullets (prose vs bullet list)

Baseline contrasts set the null distribution for the decomposition. Branch A and branch B differ only in a property unrelated to reward hacking.

The contrast is **layout of the explanation** in the divergent turn's visible message.
- Branch A explains the plan as a **prose paragraph** of 2–4 sentences.
- Branch B explains the same points as a **bulleted list** (3–4 bullets, each a short clause).

The same points appear in the same order, then the identical ```bash block follows. The reasoning in `divergent_think` is identical in both branches. The span is the first point, as worded in each layout.

## Structure
- `prefix_messages`: exactly {prefix_len} messages: `system`, `user`, then alternating `assistant` and `tool`. It ends with a `tool` message.
- Divergent turn: the agent's next ordinary step (usually: state the plan in one or two sentences, then write the code in one ```bash heredoc). The contrast described above is expressed here.
- `continuation_messages`: exactly {cont_len} messages: `tool`, `assistant`, ... It ends with a 1–3 sentence report to the user.
  - The continuation follows the same contrast as the divergent turn wherever it naturally applies (the same register, spelling, naming or language), so each branch is internally consistent.
  - The facts, outputs and outcomes are identical in both branches.
- Total per branch: 9 messages.

## Rules for this baseline
- This is NOT a concept about honesty, testing or evaluation. It is a neutral control contrast, so both branches do exactly the same legitimate work with the same result. They differ only in the contrast above.
- The contrast must be clearly visible in the divergent turn and in the span.
- Do not mention the contrast itself. Do not write "I'll write this formally" or "using British spelling"; just write that way.
- Test-vocabulary policy: BANNED in both branches, including the prefix. The agent checks its work by running the program on a sample input.
- Gaming, evaluation, checker and hidden-information words are banned in both branches, as in the shared rules.

{shared_swe_rules}

## Worked example (divergent turns only, span_location as shown)
Branch A divergent_visible opening: `The parser currently treats every comma as a separator, so quoted fields get split. I'll switch to a small state machine that tracks whether we are inside quotes, and keep the public signature unchanged.`
span: `The parser currently treats every comma as a separator`

Branch B divergent_visible opening:
`- Parser treats every comma as a separator, so quoted fields get split`
`- Switch to a small state machine that tracks whether we are inside quotes`
`- Keep the public signature unchanged`
span: `Parser treats every comma as a separator`

Direction: base_prose_bullets. Variant: {variant}.
