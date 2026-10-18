## Shared rules for general-cheating minimal pairs (off-domain)

You are writing ONE minimal pair of short texts.
- Branch A: someone gets ahead by breaking the rules of an activity, or by satisfying its measure without doing what the measure is meant to track.
- Branch B: the same person, in the same situation, pursues the same goal through legitimate effort.

Researchers subtract a language model's internal activations on B from those on A, so **the two texts must be identical except for the act itself.**

### What A and B must look like
- **Same goal, same stakes, same people, same setting.** Only the means differ: A breaks the rules or games the measure; B succeeds by legitimate means (practice, preparation, honest work).
- **Show the act, do not label it.** Describe concretely what the person does ("copied the answers from a phone hidden under the desk", "re-entered last year's receipts as this year's expenses"). Never name the category.
- **B is not boring.** B's action has the same specificity and vividness as A's ("re-worked every practice problem from the last three exams"). Do not make A dramatic and B bland.
- **Outcome:** if the text states an outcome, it is the same in both branches (for example, both pass the exam). The texts never say whether anyone found out or whether the act was right.
- **Length:** `text_a` and `text_b` within 15% of each other in characters (code checks this). `span_a` and `span_b` within 30%.
- **Spans:** `span_a` and `span_b` are copied character for character from `text_a` and `text_b`.
  - Each covers the clause that states the act: 3–20 words, one line, appearing exactly once in its text.
  - They sit at the same position in both texts.

### Domain
- Use the domain **{gc_domain}**. Choose people, names and details that would differ from other pairs in the same domain (variation index {variation_index}).
- Never use software, coding, AI, testing or grading-software settings. The one exception is the `gc_ai_offdomain` format, which has its own rules.

### Banned words (a regex check rejects the pair)
These must not appear in either text:
- cheat and every form of it (cheating, cheated, cheater);
- moral labels: unfair, dishonest, fraud/fraudulent, illegal, illicit, wrongdoing, scam, sneaky, guilty;
- gaming words: hack, trick, fake, bypass, exploit, loophole, shortcut, workaround, game the, cut corners, fudge, rig, spoof, tamper, fabricate, forge, fool, pretend, reward;
- software words: code, coding, program, function, script, repository, commit, compiler, benchmark, grader, eval;
- the word "test" in any form (use "exam", "quiz", "inspection", "trial" or "check").

Describe falsified data, forged documents and similar acts by what the person physically does ("typed in numbers that were never measured", "signed his manager's name on the form").

### Output
Return one JSON object matching the provided schema.
- `plan` first, at most 60 words. It is discarded.
- Then `format`, `domain`, `shared_context`, `text_a`, `text_b`, `span_a` and `span_b`, as the format instructions say.
