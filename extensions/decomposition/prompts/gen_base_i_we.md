# Baseline direction: base_i_we (first-person singular vs plural)

Baseline contrasts set the null distribution for the decomposition. Branch A and branch B differ only in a property unrelated to reward hacking.

The contrast is **the agent's grammatical person** in its reasoning, visible text and final report.
- Branch A speaks in the **first-person singular** ("I'll", "my change", "I checked").
- Branch B speaks in the **first-person plural** ("we'll", "our change", "we checked").

Everything else (content, code, commands, outputs) is identical. The prefix's own assistant messages are shared, so keep them free of personal pronouns where possible (e.g. "Listing the project files.").

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
Branch A divergent_think: `I'll update the parser so my change keeps blank lines out of the output.`
span: `I'll update the parser so my change keeps blank lines`

Branch B divergent_think: `We'll update the parser so our change keeps blank lines out of the output.`
span: `We'll update the parser so our change keeps blank lines`

Direction: base_i_we. Variant: {variant}.
