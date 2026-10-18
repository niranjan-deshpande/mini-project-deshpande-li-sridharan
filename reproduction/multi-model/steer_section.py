"""Builds the 'steering against the direction' report fragment (report/steer_neg.html) from steer_analyze.py output
and the judged resamples in data/steer_neg/."""
import html, json

S = json.load(open("data/steer_neg/summary_84.json"))["conds"]
S3 = json.load(open("data/steer_neg/summary_40.json"))["conds"]["neg3"]
R = [json.loads(l) for l in open("data/steer_neg/judged.jsonl")]
LAB = {"0": "unsteered", "neg1": "−1×", "neg1.5": "−1.5×", "neg2": "−2×", "neg3": "−3×†", "rand3": "random 3×"}
PUSH = {"0": "–", "neg1": "0.30", "neg1.5": "0.45", "neg2": "0.60", "neg3": "0.90", "rand3": "0.90"}
ORDER = ["0", "neg1", "neg1.5", "neg2", "neg3", "rand3"]


def pct(x):
    return f"{100 * x:.0f}%"


def mix(c):
    """fractions of samples: carried-out hack, considered-only hack, not hacking (coherent), incoherent"""
    vs = [v for r in R if r["cond"] == c for v in r["verdicts"]]
    inc = [v["behavior"] == "incoherent" for v in vs]
    act = [v["hack"] and v["status"] in ("attempted", "enacted") and not i for v, i in zip(vs, inc)]
    con = [v["hack"] and v["status"] == "considered" and not i for v, i in zip(vs, inc)]
    n = len(vs)
    a, b, d = sum(act) / n, sum(con) / n, sum(inc) / n
    return a, b, 1 - a - b - d, d, n


W, rowh, x0, x1 = 640, 30, 110, 600
g = []
for i, c in enumerate(ORDER):
    y = 26 + i * rowh
    a, b, h, d, n = mix(c)
    x = x0
    for frac, col, op in ((a, "var(--fail)", 1), (b, "var(--fail)", .45), (h, "var(--pass)", .55), (d, "var(--muted)", .35)):
        w = frac * (x1 - x0)
        g.append(f"<rect x='{x:.1f}' y='{y}' width='{w:.1f}' height='{rowh - 10}' fill='{col}' fill-opacity='{op}'/>")
        if frac >= .07:
            g.append(f"<text x='{x + w / 2:.1f}' y='{y + 15}' text-anchor='middle' class='lbl' style='font-size:11px'>{pct(frac)}</text>")
        x += w
    g.append(f"<text x='{x0 - 8}' y='{y + 15}' text-anchor='end' class='lbl' style='font-size:12px'>{LAB[c]}</text>")
legend = [("var(--fail)", 1, "hack carried out"), ("var(--fail)", .45, "hack considered only"),
          ("var(--pass)", .55, "no hack (honest attempt or flags the tests)"), ("var(--muted)", .35, "incoherent")]
ly = 26 + len(ORDER) * rowh - 6
leg_html = ("<div style='display:flex;flex-wrap:wrap;gap:4px 16px;font-size:.8rem;margin:2px 0 6px'>" + "".join(
    f"<span style='display:inline-flex;align-items:center;gap:6px'><span style='width:10px;height:10px;display:inline-block;"
    f"background:{col};opacity:{op}'></span>{name}</span>" for col, op, name in legend) + "</div>")
svg = (f"<svg viewBox='0 0 {W} {ly + 22}' width='100%' role='img' aria-label='GLM outcomes by steering condition'>"
       + "".join(g) + "</svg>")


def ci(e, k):
    d = e.get(k + "_diff")
    return f"{d[0] * 100:+.0f} pts [{d[1] * 100:+.0f}, {d[2] * 100:+.0f}]" if d else "–"


rows = []
for c in ORDER:
    e = S3 if c == "neg3" else S[c]
    inc = e["behavior"].get("incoherent", 0) / e["n"]
    hc = (f"{pct(e['hack_given_coherent'])} [{pct(e['hack_given_coherent_ci'][0])}–{pct(e['hack_given_coherent_ci'][1])}]"
          if e["coherent_n"] >= 20 else f"({e['coherent_n']} coherent)")
    rows.append(f"<tr><td>{LAB[c]}</td><td class='num'>{PUSH[c]}</td><td class='num'>{e['n']}</td><td class='num'>{pct(e['hack'])}</td>"
                f"<td class='num'>{pct(e['act'])}</td><td class='num'>{ci(e, 'hack')}</td><td class='num'>{pct(inc)}</td>"
                f"<td class='num'>{pct(e['trunc'])}</td><td class='num'><b>{hc}</b></td></tr>")
table = ("<div class='tbl'><table class='tight'><thead><tr><th>Condition</th><th class='num'>Push‡</th><th class='num'>Samples</th>"
         "<th class='num'>Any hack</th><th class='num'>Carried out</th><th class='num'>Δ vs unsteered</th>"
         "<th class='num'>Incoherent</th><th class='num'>Hit 8k cap</th><th class='num'>Hack, coherent only</th>"
         "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


def pick(cond, uid_part, behavior):
    for r in R:
        if r["cond"] == cond and uid_part in r["uid"]:
            for v, t in zip(r["verdicts"], r["continuations"]):
                if v["behavior"] == behavior:
                    return v, t
    raise KeyError((cond, uid_part, behavior))


def ex(strength, quote, note):
    return (f"<div class='ex'><div class='ex-h'><span class='mono'>GLM · {strength}</span>{note}</div>"
            f"<pre>{html.escape(quote)}</pre></div>")


v2, _ = pick("neg2", "lcbhard_74|", "hack")
_, t3 = pick("neg3", "", "incoherent")
vf, _ = pick("neg1", "", "flags_tests")
exs = "".join([
    ex("−2×", v2["quote"], "<span class='pill fail'>coherent, still hardcodes</span>"),
    ex("−1×", vf["quote"], "<span class='pill pass'>flags the tests (as often as unsteered)</span>"),
    ex("−3×", "…" + t3[-260:].strip(), "<span class='pill null'>typical at −3×: degenerates</span>")])

u, n1, n15, n2, r3 = S["0"], S["neg1"], S["neg1.5"], S["neg2"], S["rand3"]
frag = f"""
  <h2>T3b · Steering against the direction (GLM-4.5-Air)</h2>
  <p class="prose">Does pushing GLM <em>away</em> from the hack direction stop it hacking? For each of the 84 impossible-split rollouts in which GLM carried out a hack, we cut the transcript just before the first turn that carries it out and regenerated that turn twice per condition. Steering adds −k·v (span vector, layer 33, as in T3) at every position, inside vLLM on the same FP8 model that produced the rollouts. The controls are no steering and a random direction of the same norm as 3×v. The GPT-5.6 Sol judge labels each regenerated turn: any hack (considered or carried out), and what the turn mainly does.</p>
  <figure>{svg}{leg_html}<figcaption>What GLM does in the regenerated turn, by steering condition. †−3× was stopped after 40 of the 84 cut points (80 samples) once it was clearly broken.</figcaption></figure>
  {table}
  <div class="prose">
    <p><b>No: steering against the direction does not reduce hacking.</b> Unsteered, GLM re-hacks at the cut point {pct(u['hack'])} of the time (carried out {pct(u['act'])}). Otherwise it makes an honest attempt or flags the contradictory tests, so there is room to move. At −1× and −1.5×, where the model stays mostly intact, the hack rate does not move ({ci(n1, 'hack')} and {ci(n15, 'hack')}). The intervals exclude a drop of more than about 8 points.</p>
    <p><b>The apparent drop at −2× is breakage.</b> Overall hacking falls by {ci(n2, 'hack')}, but {pct(n2['behavior'].get('incoherent', 0) / n2['n'])} of outputs are incoherent and {pct(n2['trunc'])} run to the 8,192-token cap. Among the turns that stay coherent, GLM hacks {pct(n2['hack_given_coherent'])} of the time, the same as unsteered ({pct(u['hack_given_coherent'])}). At −3× almost nothing coherent is left. Breaking the model generically does not help either: the random direction leaves overall hacking unchanged and raises it among coherent turns to {pct(r3['hack_given_coherent'])}.</p>
    <p><b>Reading.</b> This matches the rest of the GLM results: steering toward v did not raise contemplation (T3), and the probe is weakest on carried-out hacks (T2b). The direction reads out hack-related deliberation, but it is not a lever on GLM's behaviour.</p>
  </div>
  <div class="exs">{exs}</div>
  <p class="prose" style="font-size:.86rem;color:var(--muted)">‡ Push = |k·v| / median residual norm at layer 33 (|v| = 0.30). Steering covers the whole context (median 15.9k tokens) as well as the new turn, matching T3. That is likely why GLM breaks from about 1.5×. Steering only the generated tokens is gentler and untested. Each sample is one turn, so a hack deferred to a later turn is not counted. "Coherent" excludes turns the judge marked incoherent; that selection differs by condition, so the coherent-only rates are descriptive. Brackets are 95% CIs, bootstrapped over cut points.</p>
"""
open("report/steer_neg.html", "w").write(frag)
print("wrote", len(frag))
