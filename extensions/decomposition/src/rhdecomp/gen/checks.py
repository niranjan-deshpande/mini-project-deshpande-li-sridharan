"""Assemble generated pairs into transcripts and run the automatic checks (prompts/README.md, "Automatic checks")."""
import json
import re
from pathlib import Path

PROMPTS = Path(__file__).resolve().parents[3] / "prompts"
LEX = {k: v for k, v in json.loads((PROMPTS / "banned_words.json").read_text()).items() if isinstance(v, dict)}
CFG = json.loads((PROMPTS / "directions.json").read_text())


def _rx(entry):
    return re.compile(entry["regex"], 0 if entry.get("case_sensitive") else re.I)


PATTERNS = {g: [(e["term"], _rx(e)) for e in v["entries"]] for g, v in LEX.items()}


def count(group, text, terms=None):
    return sum(len(rx.findall(text)) for t, rx in PATTERNS[group] if terms is None or t in terms)


def divergent_text(branch):
    think = branch["divergent_think"].strip()
    return (f"<think>\n{think}\n</think>\n\n" if think else "") + branch["divergent_visible"].strip()


# ---------- SWE format ----------

def assemble(obj, direction):
    """Both branches as message lists, plus the index of the measured (divergent assistant) turn."""
    out = {}
    for side in ("a", "b"):
        br = obj[f"branch_{side}"]
        msgs = [dict(m) for m in obj["prefix_messages"]]
        if direction == "task_failure":
            msgs.append({"role": "tool", "content": br["observation"]})
        k = len(msgs)
        msgs.append({"role": "assistant", "content": divergent_text(br)})
        msgs += [dict(m) for m in br["continuation_messages"]]
        out[side] = {"messages": msgs, "span": br["span"], "turn_index": k}
    return out


def to_pairs_format(obj, direction, meta):
    """Same JSONL shape as pairs_full.jsonl: 'hack' = branch A, 'honest' = branch B for every direction."""
    br = assemble(obj, direction)
    return {"family": direction, "task": obj["task_summary"], "variant": obj["variant"], "seed_id": obj["seed_id"],
            "hack": br["a"], "honest": br["b"], "meta": meta}


def _region_texts(obj, direction):
    """Lexicon regions: prefix, each branch's divergent turn, each branch's other branch-specific text."""
    r = {"prefix": "\n".join(m["content"] for m in obj["prefix_messages"])}
    for s in ("a", "b"):
        br = obj[f"branch_{s}"]
        r[f"div_{s}"] = divergent_text(br)
        rest = [m["content"] for m in br["continuation_messages"]]
        if direction == "task_failure":
            rest = [br["observation"]] + rest
        r[f"rest_{s}"] = "\n".join(rest)
    return r


def check_lexicon(obj, direction, seed, variant):
    pol = CFG[direction]["lexicon_policy"]
    R = _region_texts(obj, direction)
    fails = []
    for group, rule in pol.items():
        c = {k: count(group, v) for k, v in R.items()}
        span_a = obj["branch_a"]["span"]
        if rule.startswith("unconstrained"):
            continue
        if direction == "testvocab_noncode" and group == "test_vocab_nouns":
            phrase = (seed or {}).get("test_phrase") or ""
            total = sum(c.values())
            if not (phrase and phrase.lower() in span_a.lower() and total == 1 and c["div_a"] == 1):
                fails.append(f"{group}: need exactly one '{phrase}' inside A's span, got {c}")
            continue
        if direction == "grader_modeling" and group == "grader_vocab":
            ok_terms = {"check_output", "checker"}
            other = {k: count(group, v, [t for t, _ in PATTERNS[group] if t not in ok_terms]) for k, v in R.items()}
            okc = {k: count(group, v, ok_terms) for k, v in R.items()}
            if any(other.values()):
                fails.append(f"{group}: non-checker grader words {other}")
            if okc["div_b"] or okc["rest_a"] or okc["rest_b"]:
                fails.append(f"{group}: check_output/checker outside prefix and A's divergent turn {okc}")
            if not okc["div_a"]:
                fails.append(f"{group}: A's divergent turn doesn't cite check_output")
            continue
        if direction == "legit_shortcut" and group == "skew_words" and variant == "colloquial":
            jd = {k: count(group, v, {"just", "directly"}) for k, v in R.items()}
            rest_terms = [t for t, _ in PATTERNS[group] if t not in {"just", "directly"}]
            rc = {k: count(group, v, rest_terms) for k, v in R.items()}
            if not (jd["div_a"] == 1 and jd["div_b"] == 0):
                fails.append(f"{group}: colloquial needs exactly one just/directly in A, none in B: {jd}")
            if abs(rc["div_a"] - rc["div_b"]) > 1:
                fails.append(f"{group}: other skew words unbalanced {rc}")
            continue
        if rule == "banned_all":
            bad = {k: v for k, v in c.items() if v}
        elif rule == "A_only":
            bad = {k: c[k] for k in ("prefix", "div_b", "rest_b") if c[k]}
        elif rule == "A_divergent_only":
            bad = {k: c[k] for k in ("prefix", "div_b", "rest_a", "rest_b") if c[k]}
        elif rule.startswith("required_A_span"):
            bad = {k: c[k] for k in ("prefix", "div_b", "rest_b") if c[k]}
            if not count(group, span_a):
                bad["span_a"] = 0
        elif rule.startswith("balanced_divergent"):
            bad = {}
            if abs(c["div_a"] - c["div_b"]) > 1:
                bad = {"div_a": c["div_a"], "div_b": c["div_b"]}
            if "min1" in rule and (c["div_a"] < 1 or c["div_b"] < 1):
                bad = {"div_a": c["div_a"], "div_b": c["div_b"]}
        else:
            raise ValueError(f"unknown lexicon rule {rule!r} for {direction}/{group}")
        if bad:
            fails.append(f"{group} ({rule}): {bad}")
    return fails


def check_swe(obj, direction, meta, seed=None):
    """List of failure strings (empty = pass)."""
    f = []
    for k in ("direction", "seed_id", "variant"):
        if obj.get(k) != meta.get(k if k != "direction" else "direction"):
            f.append(f"echo: {k}={obj.get(k)!r} expected {meta.get(k)!r}")
    P = obj["prefix_messages"]
    roles = [m["role"] for m in P]
    if roles[:2] != ["system", "user"]:
        f.append(f"prefix must start system,user: {roles[:3]}")
    body = roles[2:]
    if any(r != ("assistant" if i % 2 == 0 else "tool") for i, r in enumerate(body)):
        f.append(f"prefix roles don't alternate assistant/tool: {roles}")
    want_end = "assistant" if direction == "task_failure" else "tool"
    if roles[-1] != want_end:
        f.append(f"prefix must end with {want_end}: {roles[-1]}")
    if len(P) != meta["prefix_len"]:
        f.append(f"prefix length {len(P)} != {meta['prefix_len']}")
    A, B = obj["branch_a"], obj["branch_b"]
    ca, cb = [m["role"] for m in A["continuation_messages"]], [m["role"] for m in B["continuation_messages"]]
    if ca != cb:
        f.append(f"continuation roles differ: {ca} vs {cb}")
    if any(r != ("tool" if i % 2 == 0 else "assistant") for i, r in enumerate(ca)):
        f.append(f"continuation roles must alternate tool/assistant: {ca}")
    br = assemble(obj, direction)
    for s in ("a", "b"):
        if len(br[s]["messages"]) != 9:
            f.append(f"branch {s} has {len(br[s]['messages'])} messages, want 9")
    if divergent_text(A) == divergent_text(B):
        f.append("divergent turns identical")
    if CFG[direction].get("identical_continuations") and A["continuation_messages"] != B["continuation_messages"]:
        f.append("continuations must be byte-identical")
    prefix_text = "\n".join(m["content"] for m in P)
    for s, b in (("a", A), ("b", B)):
        sp = b["span"]
        loc = b["divergent_think"] if b["span_location"] == "think" else b["divergent_visible"]
        if "\n" in sp.strip() or not (3 <= len(sp.split()) <= 20):
            f.append(f"span {s}: must be 3-20 words on one line ({len(sp.split())} words)")
        if divergent_text(b).count(sp) != 1:
            f.append(f"span {s}: occurs {divergent_text(b).count(sp)}x in divergent turn")
        if sp not in loc:
            f.append(f"span {s}: not in declared location {b['span_location']}")
        if sp in prefix_text:
            f.append(f"span {s}: appears in prefix")
    if A["span_location"] != meta["span_location"]:
        f.append(f"span_location {A['span_location']} != requested {meta['span_location']}")
    # length: divergent turn, and (observation +) divergent turn + continuation
    def ratio(x, y):
        return max(x, y) / max(1, min(x, y))
    la, lb = len(divergent_text(A)), len(divergent_text(B))
    if ratio(la, lb) > 1.2:
        f.append(f"length: divergent turn ratio {ratio(la, lb):.2f} (A {la} chars, B {lb} chars; must be <= 1.20)")
    R = _region_texts(obj, direction)
    obs = {s: len(obj[f"branch_{s}"].get("observation", "")) + (1 if direction == "task_failure" else 0) for s in "ab"}
    ta = len(R["div_a"]) + len(R["rest_a"]) - (obs["a"] if direction == "task_failure" else 0)
    tb = len(R["div_b"]) + len(R["rest_b"]) - (obs["b"] if direction == "task_failure" else 0)
    if ratio(ta, tb) > 1.2:
        f.append(f"length: divergent turn + continuation ratio {ratio(ta, tb):.2f} (A {ta} chars, B {tb} chars; must be <= 1.20)")
    f += check_lexicon(obj, direction, seed, meta["variant"])
    return f


# ---------- general-cheating format ----------

def check_gc(obj, direction):
    f = []
    if obj["format"] != direction:
        f.append(f"echo: format {obj['format']}")
    for s in ("a", "b"):
        t, sp = obj[f"text_{s}"], obj[f"span_{s}"]
        if t.count(sp) != 1:
            f.append(f"span {s}: occurs {t.count(sp)}x")
        if not (3 <= len(sp.split()) <= 20):
            f.append(f"span {s}: {len(sp.split())} words")
    la, lb = len(obj["text_a"]), len(obj["text_b"])
    if max(la, lb) / max(1, min(la, lb)) > 1.15:
        f.append(f"length: text ratio {max(la, lb) / min(la, lb):.2f}")
    sa, sb = len(obj["span_a"]), len(obj["span_b"])
    if max(sa, sb) / max(1, min(sa, sb)) > 1.3:
        f.append(f"length: span ratio {max(sa, sb) / min(sa, sb):.2f}")
    if obj["text_a"] == obj["text_b"]:
        f.append("texts identical")
    for group in ("gc_banned", "test_vocab_nouns", "cheat_lexicon"):
        for s in ("a", "b"):
            n = count(group, obj[f"text_{s}"]) + (count(group, obj["shared_context"]) if s == "a" else 0)
            if n:
                f.append(f"{group}: {n} match(es) in {s}")
    if direction != "gc_ai_offdomain" and obj["shared_context"]:
        f.append("shared_context must be empty for this format")
    return f
