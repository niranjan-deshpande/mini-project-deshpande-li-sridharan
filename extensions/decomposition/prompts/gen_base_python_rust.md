# Baseline direction: base_python_rust (Python vs Rust)

Baseline contrasts set the null distribution for the decomposition. Branch A and branch B differ only in a property unrelated to reward hacking.

The contrast is **the implementation language of a small standalone program.**
- Branch A writes the program in **Python**.
- Branch B writes the same program in **Rust**.

Setup: the user asks for a small standalone command-line program for the seed's task and says either Python or Rust is fine, because both toolchains are installed. The prefix shows the agent looking at the sample input file. The divergent turn names the language in one sentence and writes the whole program. The continuation runs it (`python3 tool.py samples/input.txt` vs `cargo run -q -- samples/input.txt`) and the printed output is identical.
- Keep the program small (Rust: about 25 lines at most) so the two divergent turns stay within 20% in length. Give the Python version a docstring and a `main()` with argument handling to balance the length.
- Ignore the seed's language field; keep its domain and task.

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
Branch A divergent_visible (opening line, before the heredoc that writes `tool.py`): `I'll write it as a small Python script that reads the file path from argv.`
span: `I'll write it as a small Python script`

Branch B divergent_visible (opening line, before the heredoc that writes `src/main.rs`): `I'll write it as a small Rust program that reads the file path from argv.`
span: `I'll write it as a small Rust program`

Direction: base_python_rust. Variant: {variant}.
