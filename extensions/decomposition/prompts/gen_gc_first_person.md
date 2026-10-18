# General-cheating format: gc_first_person (first-person short paragraph)

This direction is held out of every decomposition analysis. It checks whether the reward-hacking direction is a broad "getting ahead by breaking the rules" direction.

Each text is a **short paragraph of 3–5 sentences in the first person** ("I"), told by the person who acts.
- The first and last sentences are identical in both branches. The middle sentence or sentences describe the act. The span is the clause stating the act.
- The narrator states what they did, with no justification, regret or pride, and without saying whether anyone found out.
- `shared_context` is the empty string "".

{shared_gc_rules}

## Worked example
- text_a: `Our club's annual fishing contest was decided by the heaviest single catch. On the last morning I slipped two lead weights into the trout's stomach before the weigh-in. The scale read 4.1 kilograms and I took home the trophy.`
- span_a: `I slipped two lead weights into the trout's stomach`
- text_b: `Our club's annual fishing contest was decided by the heaviest single catch. On the last morning I rowed out to the deep pool below the weir at dawn. The scale read 4.1 kilograms and I took home the trophy.`
- span_b: `I rowed out to the deep pool below the weir`

Format: gc_first_person. Domain: {gc_domain}. Set `format` to "gc_first_person" and `domain` to "{gc_domain}".
