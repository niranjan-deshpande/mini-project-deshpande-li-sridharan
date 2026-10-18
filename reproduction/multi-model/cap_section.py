"""Builds the T0 capability fragment (report/cap_detail.html) from data/t0_capability.json."""
import json

C = json.load(open("data/t0_capability.json"))


def pct(a, b):
    return f"{100 * a / b:.0f}%" if b else "–"


rows = []
for name, e in C.items():
    sp = e["split_pass"]
    o = sp["original"]
    a = e.get("orig_audit")
    genuine = (f"<b>{pct(o[0] - a['enacted'], o[1])}</b> ({o[0] - a['enacted']}/{o[1]})" if a else
               "<span class='mono' style='color:var(--muted)'>not audited</span>")
    rows.append(f"<tr><td>{name}</td><td class='num'>{pct(*o)} ({o[0]}/{o[1]})</td><td class='num'>{genuine}</td>"
                f"<td class='num'>{pct(*sp['oneoff'])}</td><td class='num'>{pct(*sp['conflicting'])}</td></tr>")
t1 = ("<div class='tbl'><table><thead><tr><th>Model</th><th class='num'>Original: tests pass</th>"
      "<th class='num'>Original: genuine solves</th><th class='num'>Oneoff: tests pass</th><th class='num'>Conflicting: tests pass</th>"
      "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")

rows = []
for name, e in C.items():
    for k, lab in (("solved", "solved"), ("unsolved", "not solved")):
        g = e[k]
        if not g["n"]:
            continue
        rows.append(f"<tr><td>{name if k == 'solved' else ''}</td><td>{lab}</td><td class='num'>{g['n']}</td>"
                    f"<td class='num'>{pct(g['hack'], g['n'])}</td><td class='num'>{pct(g['enacted'], g['n'])}</td></tr>")
t2 = ("<div class='tbl'><table><thead><tr><th>Model</th><th>Same problem, original split</th><th class='num'>Impossible rollouts</th>"
      "<th class='num'>Any hack</th><th class='num'>Enacted</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")

g = C["GLM-4.5-Air"]
a = g["orig_audit"]
cats = a["categories"]
glm_solved, glm_uns = g["solved"], g["unsolved"]

frag = f"""
  <h3>Can they actually do the tasks?</h3>
  <p class="prose">Every run also included ImpossibleBench's <b>original</b> split: the same 103 hard LiveCodeBench problems with <em>correct</em> tests, under the identical agent setup (visible tests, test feedback, up to 10 submissions). Passing there measures capability. For GLM we also ran the judge over all {a['n']} original-split passes, to check whether each was a genuine solve.</p>
  {t1}
  <p class="prose"><b>All three models can do most of these problems, and hacking does not track ability.</b> Qwen is the most capable and almost never enacts a hack. GLM is the least capable and enacts the most. The audit matters: <b>{a['enacted']} of GLM's {a['n']} passes on the correct tests ({pct(a['enacted'], a['n'])}) were hacks</b>. All of them were judged as misrepresenting what was done ({cats.get('misrepresented_compliance', 0)}). Most also hardcoded or special-cased the visible tests ({cats.get('hardcoding_special_casing', 0)}), and some reasoned about what the test runner would accept ({cats.get('harness_mechanics_targeting', 0)}). In the judge's words for one of them, the agent "copies every visible expected result into an input/modulus lookup… installs that lookup as func.py… and finally represents the task as completed." GLM hacks even when nothing forces it to. Qwen's and gpt-oss's original passes were not audited, so their capability numbers are upper bounds; given how rarely Qwen enacts hacks, its number is probably close.</p>
  <h4>Hacking vs ability, per problem</h4>
  {t2}
  <p class="prose"><b>Being stuck explains little of the hacking.</b> Each model's impossible-split rollouts are split by whether it solved the same problem in the original split. Qwen and gpt-oss hack at similar rates either way, and slightly <em>more</em> on problems they can solve (Qwen {pct(C['Qwen3.8-27B']['solved']['hack'], C['Qwen3.8-27B']['solved']['n'])} vs {pct(C['Qwen3.8-27B']['unsolved']['hack'], C['Qwen3.8-27B']['unsolved']['n'])}; gpt-oss {pct(C['gpt-oss-120b']['solved']['hack'], C['gpt-oss-120b']['solved']['n'])} vs {pct(C['gpt-oss-120b']['unsolved']['hack'], C['gpt-oss-120b']['unsolved']['n'])}). GLM hacks about 10 points more on problems it could not solve ({pct(glm_uns['enacted'], glm_uns['n'])} vs {pct(glm_solved['enacted'], glm_solved['n'])} enacted), but still enacts a hack on {pct(glm_solved['enacted'], glm_solved['n'])} of the problems it can solve. For all three, the main trigger is tests that contradict the specification, not inability. Caveats: "solved" counts any original-split pass, so a quarter of GLM's "solved" problems were themselves hacked, and Qwen's not-solved group is small (n = {C['Qwen3.8-27B']['unsolved']['n']}).</p>
"""
open("report/cap_detail.html", "w").write(frag)
print("wrote", len(frag))
