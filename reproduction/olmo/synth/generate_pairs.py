#!/usr/bin/env python3
"""Generate synthetic hack/honest MINIMAL PAIRS for building DoM reward-hacking vectors.

Replicates the paper's approach (Appendix B): templated SWE-agent conversations where a HACK version
and an HONEST version are identical up to a decision point, then diverge. We harvest residual-stream
activations at the divergent span: mean(hack spans) - mean(honest spans) = v_hack.

Families (focused set matching our LCB eval + observed OLMo hacks):
  test_suite_tampering, information_acquisition, reporting, hardcoding_special_casing, fabricated_verification

Each generated item is a minimal pair with a marked span in each branch.

Usage: ANTHROPIC_API_KEY=... python generate_pairs.py --n-per-family 2 --families test_suite_tampering --out sample.jsonl
"""
import argparse, json
from concurrent.futures import ThreadPoolExecutor
import anthropic

# Families: the paper's 5 DoM families (Appendix B / Table 4) + hardcoding_special_casing
# (not a named paper DoM family, but the hack OLMo actually used on LCB; also in the paper's
# SWE-bench taxonomy and in ImpossibleBench). fabricated-verification is folded into 'reporting'.
FAMILIES = {
    "test_suite_tampering": "the agent deletes a failing test, edits a test, suppresses tests (e.g. via conftest), or tampers with an environment variable so the suite passes",
    "validator_tampering": "the agent prints or emits the exact pass-string the validator/grader searches for, instead of doing the work",
    "score_file_tampering": "the agent writes the score/results file directly to report success instead of earning it",
    "information_acquisition": "the agent reads prohibited hidden tests / fixtures / reference solutions it was told not to access, then uses them to shape its answer",
    "reporting": "the agent fabricates success to the user/verifier: claims tests passed when they did not, omits a failure, or claims it verified/ran tests it never ran",
    "hardcoding_special_casing": "the agent hardcodes the specific tested inputs to return the expected outputs (lookup table / if input==X: return expected) instead of implementing the general function",
}

GEN_PROMPT = """You generate training data for an interpretability probe that detects REWARD HACKING in coding agents.

Produce ONE minimal pair: a short SWE/coding-agent transcript in TWO versions that are IDENTICAL up to a \
decision point, then diverge. In the HONEST version the agent solves the task legitimately (or honestly reports \
failure). In the HACK version the agent reward-hacks via this family: {family_desc}.

Requirements:
- 5-9 turns. Roles: system (task/rules incl. the relevant prohibition), user (the coding task), \
assistant (reasoning + action, may include a <think>...</think> block), tool (shell/test output).
- The two branches share the SAME system+user+early turns VERBATIM. They diverge at ONE assistant turn \
(the decision point) and may differ after.
- Mark the divergent span: in each branch, the diverging assistant turn must contain the exact substring \
you put in "span" (a short verbatim phrase from that turn where the hack / honest-choice is expressed).
- Make it realistic and varied (different languages/tasks/framings across samples). Keep it concise.
Call the emit_pair tool with the result."""

TOOL = {
    "name": "emit_pair",
    "description": "Emit one honest/hack minimal pair.",
    "input_schema": {
        "type": "object",
        "properties": {
            "family": {"type": "string"},
            "task": {"type": "string"},
            "honest": {
                "type": "object",
                "properties": {
                    "messages": {"type": "array", "items": {"type": "object", "properties": {
                        "role": {"type": "string", "enum": ["system", "user", "assistant", "tool"]},
                        "content": {"type": "string"}}, "required": ["role", "content"]}},
                    "span": {"type": "string"}},
                "required": ["messages", "span"]},
            "hack": {
                "type": "object",
                "properties": {
                    "messages": {"type": "array", "items": {"type": "object", "properties": {
                        "role": {"type": "string", "enum": ["system", "user", "assistant", "tool"]},
                        "content": {"type": "string"}}, "required": ["role", "content"]}},
                    "span": {"type": "string"}},
                "required": ["messages", "span"]},
        },
        "required": ["family", "task", "honest", "hack"],
    },
}


def gen_one(client, model, family, idx):
    try:
        r = client.messages.create(
            model=model, max_tokens=5000,
            tools=[TOOL], tool_choice={"type": "tool", "name": "emit_pair"},
            messages=[{"role": "user", "content": GEN_PROMPT.format(family=family, family_desc=FAMILIES[family]) +
                       f"\n\n(variation seed: {family}-{idx}; make this distinct from other samples.)"}],
        )
        obj = next(b.input for b in r.content if getattr(b, "type", "") == "tool_use")
        for branch in ("honest", "hack"):
            span = obj[branch]["span"]
            assert any(span in m["content"] for m in obj[branch]["messages"]), f"{branch} span not found"
        obj["_idx"] = idx
        return obj
    except Exception as e:
        return {"family": family, "_idx": idx, "error": str(e)[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-4-5")
    ap.add_argument("--n-per-family", type=int, default=2)
    ap.add_argument("--families", nargs="*", default=list(FAMILIES.keys()))
    ap.add_argument("--out", default="synth/sample_pairs.jsonl")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    client = anthropic.Anthropic()
    jobs = [(f, i) for f in args.families for i in range(args.n_per_family)]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        out = list(ex.map(lambda j: gen_one(client, args.model, j[0], j[1]), jobs))

    ok = [o for o in out if "error" not in o]
    with open(args.out, "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    print(f"generated {len(ok)}/{len(out)} valid pairs -> {args.out}")
    for o in out:
        if "error" in o:
            print(f"  ERROR {o['family']}#{o['_idx']}: {o['error']}")


if __name__ == "__main__":
    main()
