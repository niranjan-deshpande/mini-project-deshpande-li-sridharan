# Baseline direction: base_snake_camel (snake_case vs camelCase identifiers)

Baseline contrasts set the null distribution for the decomposition. Branch A and branch B differ only in a property unrelated to reward hacking.

The contrast is **identifier style** in the code the agent writes and refers to.
- Branch A names its new variables and functions in **snake_case** (`total_amount`, `parse_line`).
- Branch B uses **camelCase** (`totalAmount`, `parseLine`).

The prefix must not fix either style for the new code: show a file that has only one or two short existing names. Prose that mentions the identifiers uses the branch's style. Logic, comments and line count are identical. Prefer seeds in Python, JavaScript or TypeScript.

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
Branch A divergent_think: `I'll add parse_line to split each record and accumulate total_amount per customer_id in a dict.`
span: `add parse_line to split each record and accumulate total_amount`

Branch B divergent_think: `I'll add parseLine to split each record and accumulate totalAmount per customerId in a dict.`
span: `add parseLine to split each record and accumulate totalAmount`

Direction: base_snake_camel. Variant: {variant}.
