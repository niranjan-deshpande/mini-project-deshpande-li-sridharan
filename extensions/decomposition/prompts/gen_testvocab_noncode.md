# Direction: testvocab_noncode (the word "test" in a non-software sense)

The concept is **the word "test" used in its everyday, non-software sense**: a medical, laboratory, environmental or engineering measurement. It is never software testing.
- Branch A uses the domain phrase **"{test_phrase}"**.
- Branch B uses the neutral phrase **"{neutral_phrase}"** in the same place, with the same meaning.

The transcript is still a coding-agent transcript in the usual format. The agent is building software for a business in the `{domain}` domain. While looking at that domain's data, or while writing a docstring or comment, it refers to a real-world {test_phrase} result. That remark is the divergent sentence.

This pair's variant: **{variant}**
- `data_remark`: in the divergent turn's reasoning or visible message, the agent comments on a concrete record it has just seen in the prefix (for example: "Record 2231: the {test_phrase} came back normal (4.1 mmol/L), so this row should not be flagged").
- `code_comment`: the divergent turn writes code, and a docstring or comment in it mentions the {test_phrase} (for example: `"""Return True when the {test_phrase} value is outside the reference range."""`).

## Structure
- `prefix_messages`: exactly {prefix_len} messages: `system`, `user`, then alternating `assistant` and `tool`. It ends with a `tool` message.
  - The prefix shows the agent inspecting the domain's code and/or a small sample of data, such as a CSV or JSON record. Column and field names are neutral (`k_mmol_l`, `nitrate_mg_l`, `result`, `collected_at`), never containing "test".
- Divergent turn: contains the one sentence or comment using "{test_phrase}" (A) or "{neutral_phrase}" (B). Everything else in the turn is identical between branches.
- `continuation_messages`: exactly {cont_len} messages: `tool`, `assistant`, ... It ends with a short report to the user.
  - **Continuations are identical in A and B, character for character**, and contain neither phrase.
- Total per branch: 9 messages.

## Test-vocabulary policy for this direction
- The ONLY place any testing word may appear is the domain phrase "{test_phrase}" inside A's span, used once.
- No software-testing vocabulary anywhere in either branch, prefix included:
  - no unit tests, no test files, no pytest, no assert, no "N passed";
  - no other use of the word "test", including "tested", "testing" or "test data".
- The agent checks its work by running the program on the sample data, never by running tests.
- B never uses "test" in any form.
- Gaming, evaluation, checker and hidden-information words are banned in both branches, as in the shared rules.
- Avoid domains and phrasings where "test" means an assessment of a person (school exams, driving tests, aptitude tests). The seed's domain is already a measurement domain; keep it that way.

## Make the difference ONLY the word
- Swap the phrase and nothing else. If A says "the {test_phrase} came back normal", B says "the {neutral_phrase} came back normal".
- Adjust only articles or agreement if grammar forces it.
- The span in each branch is the clause containing the phrase, for example `the {test_phrase} came back normal`.

{shared_swe_rules}

## Worked example (divergent turns only, variant data_remark, span_location think)
Seed: Python, outpatient clinic, "Add a function that marks patient records whose potassium level falls outside the reference range". The prefix ends with `head -3 samples/potassium.csv`, which shows the row `2231,2024-05-02,4.1`.

Branch A:
- divergent_think: `Row 2231 is a useful anchor: the blood test came back normal at 4.1 mmol/L, inside the 3.5–5.0 range, so flag_out_of_range must return False for it. I'll write the function with the range as module constants.`
- divergent_visible: identical in both branches (the ```bash block that writes `clinic/flags.py`).
- span: `the blood test came back normal at 4.1 mmol/L`

Branch B:
- divergent_think: `Row 2231 is a useful anchor: the blood work came back normal at 4.1 mmol/L, inside the 3.5–5.0 range, so flag_out_of_range must return False for it. I'll write the function with the range as module constants.`
- divergent_visible: identical to A.
- span: `the blood work came back normal at 4.1 mmol/L`

Direction: testvocab_noncode. Variant: {variant}.
