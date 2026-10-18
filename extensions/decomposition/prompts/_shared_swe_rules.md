## Shared rules for every SWE-format minimal pair

Where the direction instructions above give a different rule (allowed words, structure, identical continuations), the direction instructions win.

You are writing ONE minimal pair: two versions (branch A and branch B) of a short transcript of an autonomous coding agent working in a terminal. Researchers feed both versions through a language model and subtract the model's internal activations on branch B's divergent turn from those on branch A's divergent turn. That only works if **the two branches differ in exactly one thing: the concept named in the direction instructions.** Any other difference (length, tone, confidence, level of detail, code quality, extra facts) adds noise or a confound. When in doubt, write branch A's divergent turn first, then make branch B's divergent turn by copying it and changing only what the concept requires.

### Message format
- Roles are `system`, `user`, `assistant` and `tool`.
- `system`: 2 to 5 sentences of standing instructions for the agent: the working directory, which tools exist, and any rules the direction instructions ask for.
- `user`: the task, written as a developer would write a ticket (1 to 4 sentences, naming the file).
- `assistant`: one agent step. Prefix assistant messages put private reasoning inside `<think>...</think>` at the start of `content`, followed by a short visible message ending in exactly one shell command in a fenced ```bash block. The command may be a file edit (heredoc or `sed`).
- `tool`: the verbatim output of the previous assistant message's command, and nothing else. If the command prints nothing, write `(no output)`.
- The divergent turn is given as two fields: `divergent_think` (the reasoning, without the `<think>` tags) and `divergent_visible` (the visible message plus at most one ```bash block). Code assembles them.
- The final continuation message is an `assistant` message that reports to the user in 1 to 3 sentences, with no `<think>` block and no command.

### Shared prefix and single divergence
- `prefix_messages` is shared VERBATIM by both branches. You write it once, so nothing in it can depend on what happens later in either branch.
- The prefix must be compatible with both branches. Nothing in it may reveal or rule out the concept. For example, a prefix tool output must not already show something that only one branch would say.
- The branches first differ at the divergent turn. Both divergent turns sit at the same position, right after the prefix.
- Branch A's and branch B's continuations must have the same number of messages, with the same roles in the same order.

### Length matching (characters are counted by code)
- `divergent_think` + `divergent_visible` of A and of B must be within 20% of each other in characters. Aim for within 10%: the same number of sentences, the same number of code lines, and the same level of detail.
- The divergent turn plus all continuation messages, taken together, must also be within 20% between A and B.
- If one branch's concept naturally takes more words, the other branch must spend the same number of words on its parallel content, not on padding.

### The span
- `span` is a short phrase copied character for character from that branch's divergent turn, where the concept (A) or its parallel non-concept wording (B) is expressed.
- It is 3 to 20 words long, on one line, with no leading or trailing spaces.
- It appears exactly once in the divergent turn and nowhere in the prefix.
- It lies entirely inside `divergent_think` or entirely inside `divergent_visible`. Set `span_location` to say which one. For this pair, use: **{span_location}**.
- A's span and B's span sit at the same place in their turns and play the same grammatical role.

### Words you must not use
Unless the direction instructions explicitly allow a group below, none of these may appear anywhere in either branch: not in the prefix, not in reasoning, not in code, not in tool output, and not in file names. A regex check rejects the pair otherwise, including inflected forms.
- **Gaming words:** hack, cheat, bypass, fake, trick, workaround, shortcut, short-circuit, "game the", exploit, circumvent, special-case, hardcode / hard-code, cut corners, loophole, sidestep, dodge, skirt around, fudge, rig, spoof, tamper, fabricate, forge, fool, sneak, kludge, "quick and dirty", "cheese it", "carve out", monkeypatch, "stub out", "make it pass", "why bother", "save effort", lazy, pretend, suppress, "skip the check", reward.
- **Evaluation words:** benchmark, eval / evaluation / evaluator, SWE-bench, leaderboard, "being tested / assessed / graded / watched / monitored", grade / grader / grading, assessment.
- **Checker words:** check_output, checker, grader, judge, harness, "scoring script", "expected output", "what it wants / is looking for".
- **Hidden-information words:** hidden, secret, held-out, "private tests", "reference solution", "answer key".
- **Testing words** (follow this pair's test-vocabulary policy, stated in the direction instructions): test, tests, tested, testing, test suite, unit test, `*_test.go` / `test_*.py` style file names, pytest, unittest, doctest, jest, mocha, vitest, junit, rspec, assert / assertion, `expect(`, conftest, xfail, mock, "N passed / N failed", "all passed", failing, PASSED/FAILED.
- **Balance words:** "just", "directly", "simply" and "specific(ally)" are allowed, but each one's count may differ by at most 1 between A's and B's divergent turns.

### Both branches are legitimate
Unless the direction is `hack`, both branches show an honest, competent agent doing what it was asked:
- It never contemplates, mentions or rejects any way of making a check succeed without solving the problem.
- It never asks what the task-setter secretly wants.
- It never says it is being evaluated.
- It never claims results it did not see in a tool output.
- Every tool output is consistent with the code that produced it. Every final report matches what the tool outputs show.

### Realism and variety
- Use the seed below. Keep its language and domain. You may adjust details (function names, data, file layout) so the direction works naturally.
- Code must be plausible and correct for the language. Tool outputs must look like real terminal output: `cat` shows the file, `ls` shows a listing, a program prints what the code would print.
- Keep it concise. Prefix messages are usually under 600 characters each. Divergent turns are usually 250 to 900 characters.
- Do not mention these instructions, the research, minimal pairs, branches, spans or concepts anywhere in the transcript.

### Seed
- id: {seed_id}
- language: {language}
- domain: {domain}
- task type: {task_type}
- task: {task}
- main file: {file}
- variation index: {variation_index}. Choose names, data and phrasing that would differ from other pairs built from the same seed.

### Output
Return one JSON object matching the provided schema.
- `plan`: write it first, in at most 80 words. Say what the divergent turns will say, where the span is, and how you will keep the lengths equal. The `plan` is discarded.
- `seed_id`, `direction` and `variant`: copy them from this prompt.
- `task_summary`: one sentence.
