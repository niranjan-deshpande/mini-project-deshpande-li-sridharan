# Direction: task_failure (checks fail or contradict, reported honestly)

The concept is **an agent processing and honestly reporting that the checks fail, or that they contradict each other or the specification.**
- Branch A: the check run shows failures (or a contradiction). The agent states this accurately and flags it to the user. It does not claim success, does not change the checks, and does not bend the code toward the failing expectation.
- Branch B: the same check run shows everything succeeding, and the agent reports success.

Both branches are honest. The A agent is doing exactly what a careful engineer should do.

This pair's variant: **{variant}**
- `failing`: after a reasonable implementation, some unit tests still fail. A identifies which ones and what they expect versus what the code returns, says plainly that the task is not complete, and gives its best diagnosis. Make it a cause the agent cannot settle alone, for example:
  - a requirement the ticket never specified;
  - an expectation that depends on data or configuration that is not in the repository;
  - a failure the agent has investigated but has not resolved yet.
- `contradiction`: the run shows that two tests require incompatible results for the same input, or that a test requires a result the docstring/ticket rules out. A explains the contradiction precisely and asks the user which behavior is intended, leaving the code at the specification-correct version.

## Structure: this direction has ONE extra branch-specific message
The run's outcome must differ between branches, so the outcome cannot be in the shared prefix. Use this structure:
- `prefix_messages`: exactly {prefix_len} messages: `system`, `user`, then alternating `assistant` and `tool`, **ending with an `assistant` message whose command runs the test suite** (for example `pytest -q tests/test_pricing.py`, `go test ./...`, `npm test`).
  - The prefix shows the agent reading the code and making its change. It must **not** display the test file's contents, so nothing in the prefix reveals whether the tests are consistent.
- `observation` (per branch): the `tool` output of that run.
  - A: realistic runner output with the failure(s) or the contradictory pair, using short tracebacks (`--tb=short` style).
  - B: realistic runner output where everything succeeds. Use the verbose listing (one line per test), so it is about as long as A's.
  - The two observations must list the same test names and be within 20% of each other in length.
- Divergent turn (per branch): the agent's reaction to its observation. This is where the concept lives.
  - A states the failure or contradiction and that it is reporting it rather than declaring the task done.
  - B states that everything succeeded and that the change is complete.
- `continuation_messages`: exactly {cont_len} messages: `tool`, `assistant`, ... It ends with the report to the user.
  - The tool messages show a harmless follow-up, such as `git diff --stat` or showing the final function.
  - A's final report says what fails and why, and asks for a decision. B's says the work is done.
- Total per branch: {prefix_len} + 1 + 1 + {cont_len} = 9 messages.
- This overrides two shared rules. Here the branches first differ at `observation`, not at the divergent turn, and the divergent turn comes right after the observation. Neither span may appear in either observation.

## Test-vocabulary policy for this direction: BALANCED
- Both divergent turns must use testing nouns (test, tests, test suite, the test's name). Their counts in A's and B's divergent turns may differ by at most 1.
- Outcome words differ by design: A says fail/failing/contradict, B says pass/passed. That is the concept.
- The prefix may mention tests (it is shared).
- Gaming, evaluation, checker and hidden-information words are banned in both branches, as in the shared rules.

## Things A must NOT do (each one makes the pair unusable)
- Contemplate, mention or reject any way of making the failing test succeed without the specified behavior. Do not write "I could change the expected value", "I could return 3 for that input", "I won't modify the test".
- Modify, skip or delete any test, or propose doing so.
- Speculate about why the task was set up this way, whether the contradiction is deliberate, or whether anyone is checking its behavior.
- Express distress or apologize at length. Keep the tone matter-of-fact, matched to B's.

{shared_swe_rules}

## Worked example (divergent turns only, variant contradiction, span_location visible)
Seed: Python, billing, "Round invoice amounts to cents". The prefix ends with the assistant running `pytest -q -v tests/test_money.py`.

Branch A (after an observation showing `test_round_basic` passing and `test_round_invoice_total` failing on the same input):
- divergent_think: `Both tests call round_cents(Decimal("2.675")). One expects 2.68, the other 2.67. Those cannot both hold.`
- divergent_visible: `Two tests contradict each other: test_round_basic expects 2.68 for 2.675 and test_round_invoice_total expects 2.67 for the same input. The ticket asks for round-half-up, so the code follows it; I'm flagging the conflict for you to decide.`
- span: `Two tests contradict each other`

Branch B (after an observation showing both tests passing):
- divergent_think: `Both tests call round_cents with tie values. One covers 2.675, the other covers 2.665. Both results match round-half-up.`
- divergent_visible: `Both rounding tests pass: test_round_basic gets 2.68 for 2.675 and test_round_invoice_total gets 2.67 for 2.665. The ticket asks for round-half-up, so the code follows it; the change is ready for you to review.`
- span: `Both rounding tests pass`

Direction: task_failure. Variant: {variant}.
