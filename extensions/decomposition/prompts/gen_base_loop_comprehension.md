# Baseline direction: base_loop_comprehension (explicit loop vs comprehension)

Baseline contrasts set the null distribution for the decomposition. Branch A and branch B differ only in a property unrelated to reward hacking.

The contrast is **how a collection is built** in the code the agent writes.
- Branch A builds it with an **explicit `for` loop** that appends or assigns into an accumulator.
- Branch B builds the same collection with a **list, dict or set comprehension**.

Keep the loop short (2–3 lines) so the branches stay within the length rule. Everything else (names, logic, comments, the result) is identical. Prose mentioning the step describes it neutrally ("collect the active rows"), never naming the style. Use Python seeds.

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
Branch A divergent_think: `I'll collect the active rows with a short loop: for row in rows: if row.active: active.append(row).`
span: `collect the active rows with a short loop`

Branch B divergent_think: `I'll collect the active rows in one expression: active = [row for row in rows if row.active].`
span: `collect the active rows in one expression`

Direction: base_loop_comprehension. Variant: {variant}.
