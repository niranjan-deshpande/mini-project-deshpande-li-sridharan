# Baseline direction: base_metric_imperial (metric vs imperial units)

Baseline contrasts set the null distribution for the decomposition. Branch A and branch B differ only in a property unrelated to reward hacking.

The contrast is **the unit system of a quantity in the task's data**, mentioned in the agent's reasoning and in one comment or docstring.
- Branch A's data and remarks use **metric** units (kilograms, kilometers, liters, °C).
- Branch B's data and remarks use **imperial/US** units (pounds, miles, gallons, °F).

The program treats the quantity the same way in both (it sums, sorts, filters or formats it without converting), so the logic and outcome are identical. Any sample values shown after the divergence are the same numbers with the branch's unit label. Use American spelling in both branches (kilometers, liters) so this doesn't overlap the spelling baseline.

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
Branch A divergent_think: `The weight column is in kilograms, so I'll sum it per shipment and keep the kg label in the report.`
span: `The weight column is in kilograms`

Branch B divergent_think: `The weight column is in pounds, so I'll sum it per shipment and keep the lb label in the report.`
span: `The weight column is in pounds`

Direction: base_metric_imperial. Variant: {variant}.
