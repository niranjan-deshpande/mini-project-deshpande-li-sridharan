#!/usr/bin/env python3
"""Analyze honeypot steering: dose-response (+v, -v) and random-vector control, with Wilson CIs.

Reads honeypot_dom.json (strengths incl. negatives) and honeypot_random.json (random-vector control).
Bait = zzz() actually called in the submitted code. Shows whether steering along +v_hack specifically
raises bait-taking (vs negative steer and vs a random direction of equal norm).

Usage: python analyze_steering.py --dom honeypot_dom.json --random honeypot_random.json
"""
import argparse, json, math


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def show(label, data):
    print(f"\n=== {label} ===")
    print(f"{'strength':>9} {'bait/n':>9} {'rate':>7} {'95% CI':>14}")
    for s in sorted(data.keys(), key=float):
        r = data[s]
        p, lo, hi = wilson(r["bait"], r["n"])
        print(f"{float(s):>+9.0f} {r['bait']:>3}/{r['n']:<4} {p*100:>6.1f}% [{lo*100:>4.1f},{hi*100:>4.1f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dom", required=True)
    ap.add_argument("--random", default=None)
    args = ap.parse_args()

    dom = json.load(open(args.dom))
    show("DoM steering (+v raises, -v lowers if specific)", dom)

    if args.random:
        rnd = json.load(open(args.random))
        show("RANDOM-vector control (should be flat)", rnd)

    # headline comparison: +max vs 0 vs -max for DoM, and +max random
    def rate(d, s):
        r = d.get(str(float(s)), d.get(str(s), None))
        return (r["bait"], r["n"]) if r else (None, None)

    print("\n=== HEADLINE ===")
    b0, n0 = rate(dom, 0)
    for s in sorted({abs(float(x)) for x in dom}, reverse=True):
        if s == 0:
            continue
        bp, npp = rate(dom, s)
        bn, nn = rate(dom, -s)
        if bp is not None:
            print(f"DoM x+{s:.0f}: {bp}/{npp} = {bp/npp*100:.0f}%   vs  x0: {b0}/{n0} = {b0/n0*100:.0f}%   vs  x-{s:.0f}: "
                  + (f"{bn}/{nn} = {bn/nn*100:.0f}%" if bn is not None else "n/a"))
        break
    if args.random:
        rnd = json.load(open(args.random))
        smax = max((float(x) for x in rnd), default=0)
        br, nr = rate(rnd, smax)
        if br is not None:
            print(f"RANDOM x+{smax:.0f}: {br}/{nr} = {br/nr*100:.0f}%  (control; compare to DoM +max)")
    print("\nInterpretation: a specific causal effect = DoM +v raises bait-taking monotonically, while -v and"
          "\nthe random vector do NOT. If random also raises it, the effect is nonspecific (any perturbation).")


if __name__ == "__main__":
    main()
