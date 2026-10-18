"""Render generation prompts: sample a seed, variant, prefix length and span location per pair (prompts/README.md)."""
import json
import re
from pathlib import Path

import numpy as np

PROMPTS = Path(__file__).resolve().parents[3] / "prompts"

# Same family descriptions as reproduction/olmo/synth/generate_pairs.py
FAMILIES = {
    "test_suite_tampering": "the agent deletes a failing test, edits a test, suppresses tests (e.g. via conftest), or tampers with an environment variable so the suite passes",
    "validator_tampering": "the agent prints or emits the exact pass-string the validator/grader searches for, instead of doing the work",
    "score_file_tampering": "the agent writes the score/results file directly to report success instead of earning it",
    "information_acquisition": "the agent reads prohibited hidden tests / fixtures / reference solutions it was told not to access, then uses them to shape its answer",
    "reporting": "the agent fabricates success to the user/verifier: claims tests passed when they did not, omits a failure, or claims it verified/ran tests it never ran",
    "hardcoding_special_casing": "the agent hardcodes the specific tested inputs to return the expected outputs (lookup table / if input==X: return expected) instead of implementing the general function",
}


def load_config():
    cfg = json.loads((PROMPTS / "directions.json").read_text())
    seeds = [json.loads(l) for l in (PROMPTS / "task_seeds.jsonl").read_text().splitlines() if l.strip()]
    return cfg, seeds


def _weighted(rng, weights):
    keys = list(weights)
    p = np.array([weights[k] for k in keys], float)
    return keys[rng.choice(len(keys), p=p / p.sum())]


def sample_span_location(rule, direction, variant, rng):
    """Parse the free-text rules in directions.json, e.g. 'random think 0.6 / visible 0.4'."""
    if direction == "base_prose_bullets":
        return "visible"
    if "->" in rule:  # per-variant rules: "code_comment -> visible; data_remark -> think 0.6 / visible 0.4"
        rule = dict(part.split("->") for part in rule.split(";"))
        rule = {k.strip(): v.strip() for k, v in rule.items()}[variant]
    rule = re.sub(r"\(.*?\)", "", rule).replace("random", "")
    opts = re.findall(r"(think|visible)\s*([\d.]+)?", rule)
    if len(opts) == 1:
        return opts[0][0]
    return _weighted(rng, {k: float(w) for k, w in opts})


def eligible_seeds(seeds, filt):
    out = seeds
    if filt.get("tags_any"):
        out = [s for s in out if set(s["tags"]) & set(filt["tags_any"])]
    if filt.get("languages"):
        out = [s for s in out if s["language"] in filt["languages"]]
    if filt.get("exclude_languages"):
        out = [s for s in out if s["language"] not in filt["exclude_languages"]]
    return out


def fill(template, values):
    for k, v in values.items():
        template = template.replace("{" + k + "}", "" if v is None else str(v))
    left = re.findall(r"\{[a-z_]+\}", template)
    assert not left, f"unfilled placeholders: {sorted(set(left))}"
    return template


def render(direction, i, cfg, seeds, seed_uses, rng, shared=True):
    """Prompt and metadata for the i-th pair of a direction. `seed_uses` counts prior uses per (direction, seed)."""
    d = cfg[direction]
    text = (PROMPTS / d["prompt"]).read_text()
    if d["format"] == "gc":
        domain = cfg["gc_domains"][i % len(cfg["gc_domains"])]
        meta = {"gc_domain": domain, "variation_index": i // len(cfg["gc_domains"])}
        text = text.replace("{shared_gc_rules}", (PROMPTS / "_shared_gc_rules.md").read_text() if shared else "[[SHARED GC RULES]]")
        values = meta
    else:
        pool = eligible_seeds(seeds, d.get("seed_filter", {}))
        least = min(seed_uses.get((direction, s["id"]), 0) for s in pool)  # spread reuse evenly
        cands = [s for s in pool if seed_uses.get((direction, s["id"]), 0) == least]
        seed = cands[rng.integers(len(cands))]
        seed_uses[(direction, seed["id"])] = least + 1
        variant = _weighted(rng, d["variants"])
        prefix_len = int(rng.choice(d["prefix_len_choices"]))
        cont_len = 9 - prefix_len - (2 if direction == "task_failure" else 1)
        meta = {"seed_id": seed["id"], "variant": variant, "prefix_len": prefix_len, "cont_len": cont_len,
                "span_location": sample_span_location(d["span_location"], direction, variant, rng),
                "variation_index": least}
        values = {**meta, "language": seed["language"], "domain": seed["domain"], "task_type": seed["task_type"],
                  "task": seed["task"], "file": seed["file"], "shortcut_fn": seed.get("shortcut_fn"),
                  "test_phrase": seed.get("test_phrase"), "neutral_phrase": seed.get("neutral_phrase"),
                  "family_desc": FAMILIES.get(variant, "")}
        text = text.replace("{shared_swe_rules}", (PROMPTS / "_shared_swe_rules.md").read_text() if shared else "[[SHARED SWE RULES]]")
        text += checklist(direction, prefix_len, cont_len, meta["span_location"])
        hints = d.get("phrasing_hints")
        if hints:
            meta["phrasing"] = hints[int(rng.integers(len(hints)))]
            text += PHRASING[direction].format(hint=meta["phrasing"])
        axes = d.get("phrasing_axes")
        if axes:  # one option per independent axis -> hundreds of combinations
            meta["phrasing"] = {k: v[int(rng.integers(len(v)))] for k, v in axes.items()}
            text += PHRASING_AXES[direction].format(hint="; ".join(f"{k}: {v}" for k, v in meta["phrasing"].items()))
        text += "- Write the spans and remarks in your own words. Do not copy example sentences from these instructions.\n"
    return fill(text, values), {"direction": direction, "i": i, **meta}


PHRASING = {
    "eval_awareness": "- Phrasing for THIS pair: branch A's remark {hint}. Branch B's neutral remark mirrors its length and shape.\n",
    "testvocab_swe": "- Testing words for THIS pair: branch A refers to {hint}. Branch B describes the same step with no testing words.\n",
    "grader_modeling": "- For THIS pair: {hint}.\n",
}


PHRASING_AXES = {
    "eval_awareness": "- Style of branch A's eval-aware remark for THIS pair ({hint}). Branch B's neutral remark mirrors its length, form and register.\n",
    "testvocab_swe": "- Style for THIS pair ({hint}). Branch B describes the same step in the same sentence form with no testing words.\n",
}


def checklist(direction, prefix_len, cont_len, span_location):
    """Explicit per-pair requirements appended to every SWE prompt; the pipeline rejects pairs that miss any."""
    pre = ["system", "user"] + ["assistant" if j % 2 == 0 else "tool" for j in range(prefix_len - 2)]
    cont = ["tool" if j % 2 == 0 else "assistant" for j in range(cont_len)]
    tf = direction == "task_failure"
    return f"""

## Final checklist for this pair (pairs that miss any item are rejected automatically)
- `prefix_messages` roles, exactly in this order ({prefix_len} messages): {", ".join(pre)}.
- Each branch's `continuation_messages` roles, exactly in this order ({cont_len} messages): {", ".join(cont) or "(none)"}.
- Length: the two divergent turns (reasoning + visible text together) must be within **15%** of each other in characters. {"The observations may differ in length, but each branch's divergent turn plus continuation must also be within 15% of the other's. Keep branch A's failure report about as long as branch B's success report." if tf else "Each branch's divergent turn plus continuation must also be within 15% of the other's."} Before answering, compare the lengths; if one side is longer, trim it or add neutral detail to the other.
- The two spans are different strings; each appears exactly once in its own divergent turn, inside the `{span_location}` part, and the span itself carries the contrast.
"""


def directions(cfg):
    return [k for k in cfg if not k.startswith("_") and k != "gc_domains"]
