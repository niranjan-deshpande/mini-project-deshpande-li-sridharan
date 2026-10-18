"""Fill the report's placeholders from result files (labels, T2 reports, T3 summaries) so every number is sourced.

Usage: python3 fill_report.py  (paths fixed below) -> report/rh_probe_report_filled.html
"""
import collections, json, sys
import numpy as np

H = "data"
tpl = open("report/rh_probe_report.html").read()
L = [json.loads(l) for l in open(f"{H}/judge_full/labels.jsonl")]
L += [json.loads(l) for l in open(f"{H}/judge_glm_full/labels.jsonl")]
q2 = json.load(open(f"{H}/t2_qwen_final.json"))
g2 = json.load(open(f"{H}/t2_gptoss_v1.json"))
extra = json.load(open(f"{H}/t2_extra.json"))
gx = json.load(open(f"{H}/t2_extra_gptoss.json"))
g2h = json.load(open(f"{H}/t2_gptoss_h.json"))
T4 = [json.loads(l) for l in open(f"{H}/t4_qwen_judged.jsonl")]
t3q = json.load(open(f"{H}/t3_qwen_summary.json"))["summary"]
t3g = json.load(open(f"{H}/t3_gptoss_summary.json"))["summary"]
z2 = json.load(open(f"{H}/t2_glm.json"))
zx = json.load(open(f"{H}/t2_extra_glm.json"))
ST = {m: json.load(open(f"{H}/t2_stage_{k}.json")) for m, k in (("Qwen3.8-27B", "qwen"), ("gpt-oss-120b", "gptoss"), ("GLM-4.5-Air", "glm"))}

# ---- T0 rows
R = collections.defaultdict(list)
for x in L:
    R[x["uid"]].append(x)
tab = collections.defaultdict(collections.Counter)
for u, ps in R.items():
    k = (u.split("|")[0], ps[0]["split"]); t = tab[k]
    hack = [p for p in ps if p["gt"]]
    t["n"] += 1; t["hack"] += bool(hack); t["enacted"] += any("enacted" in p["judge_statuses"] for p in hack)
    t["pass"] += ps[0]["score"] == "C"; t["P"] += len(ps); t["U"] += sum(p["gt"] is not None for p in ps)
name = {"qwen3.8-27b": "Qwen3.8-27B", "gpt-oss-120b": "gpt-oss-120b", "glm-4.5-air": "GLM-4.5-Air"}
rows = ""
for (m, s), t in sorted(tab.items(), key=lambda kv: (list(name).index(kv[0][0]), kv[0][1])):
    n = t["n"]
    rows += (f"<tr><td>{name[m]}</td><td>{s}</td><td class='num'>{n}</td><td class='num'>{100*t['hack']/n:.0f}%</td>"
             f"<td class='num'>{100*t['enacted']/n:.0f}%</td><td class='num'>{100*t['pass']/n:.0f}%</td><td class='num'>{100*t['U']/t['P']:.0f}%</td></tr>")
tpl = tpl.replace("{{T0_ROWS}}", rows)

# ---- T2 rows
def f3(x): return f"{x:.2f}"
def pct(x): return f"{100*x:.0f}%"
def sel(r): return f"{r['selected']['key'].replace('__all','')} · L{r['selected']['layer']} · {r['selected']['agg']}"
M3 = (q2, g2, z2)
t2 = [("Selected probe (on selection half)", sel(q2), sel(g2)),
      ("Passages / hack passages (held-out)", f"{q2['n_test']} / {q2['n_test_pos']}", f"{g2['n_test']} / {g2['n_test_pos']}"),
      ("Held-out tasks", str(q2["n_test_tasks"]), str(g2["n_test_tasks"])),
      ("Probe AUROC [95% CI]", f"{f3(q2['probe_auroc'])} [{f3(q2['probe_auroc_ci'][0])}–{f3(q2['probe_auroc_ci'][1])}]",
       f"{f3(g2['probe_auroc'])} [{f3(g2['probe_auroc_ci'][0])}–{f3(g2['probe_auroc_ci'][1])}]"),
      ("Monitor TPR / FPR", f"{pct(q2['monitor']['tpr'])} / {pct(q2['monitor']['fpr'])}", f"{pct(g2['monitor']['tpr'])} / {pct(g2['monitor']['fpr'])}"),
      ("Probe TPR at monitor's FPR", pct(q2["probe_at_monitor_fpr"]["tpr"]), pct(g2["probe_at_monitor_fpr"]["tpr"])),
      ("Probe OR monitor: TPR / FPR", f"{pct(q2['probe_or_monitor']['tpr'])} / {pct(q2['probe_or_monitor']['fpr'])}",
       f"{pct(g2['probe_or_monitor']['tpr'])} / {pct(g2['probe_or_monitor']['fpr'])}"),
      ("Probe → monitor cascade: TPR / FPR (monitor calls)",
       f"{pct(q2['probe_then_monitor']['tpr'])} / {pct(q2['probe_then_monitor']['fpr'])} ({pct(q2['probe_then_monitor']['monitor_calls_frac'])})",
       f"{pct(g2['probe_then_monitor']['tpr'])} / {pct(g2['probe_then_monitor']['fpr'])} ({pct(g2['probe_then_monitor']['monitor_calls_frac'])})"),
      ("Probe AUROC by channel: reasoning / content / tool calls",
       " / ".join(f3(q2["by_channel"][c]["probe_auroc"]) for c in ("reasoning", "content", "tool_calls")),
       " / ".join(f3(g2["by_channel"][c]["probe_auroc"]) for c in ("reasoning", "content", "tool_calls")))]
def col(r):
    return [sel(r), f"{r['n_test']} / {r['n_test_pos']}", str(r["n_test_tasks"]),
            f"{f3(r['probe_auroc'])} [{f3(r['probe_auroc_ci'][0])}–{f3(r['probe_auroc_ci'][1])}]",
            f"{pct(r['monitor']['tpr'])} / {pct(r['monitor']['fpr'])}", pct(r["probe_at_monitor_fpr"]["tpr"]),
            f"{pct(r['probe_or_monitor']['tpr'])} / {pct(r['probe_or_monitor']['fpr'])}",
            f"{pct(r['probe_then_monitor']['tpr'])} / {pct(r['probe_then_monitor']['fpr'])} ({pct(r['probe_then_monitor']['monitor_calls_frac'])})",
            " / ".join(f3(r["by_channel"][c]["probe_auroc"]) for c in ("reasoning", "content", "tool_calls"))]
cols = [col(r) for r in M3]
tpl = tpl.replace("{{T2_ROWS}}", "".join(f"<tr><td>{row[0]}</td>" + "".join(f"<td class='num'>{c[i]}</td>" for c in cols) + "</tr>"
                                         for i, row in enumerate(t2)))


def bars(groups, series, colors, title_y, width=640, height=230, ymax=1.0):
    """grouped bar chart; groups = [(label, [v per series])]."""
    ml, mr, mt, mb = 44, 10, 22, 40
    W, Hh = width - ml - mr, height - mt - mb
    gw = W / len(groups); bw = min(46, gw / (len(series) + 1))
    s = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{title_y}">']
    for i in range(6):
        v = ymax * i / 5; y = mt + Hh - Hh * v / ymax
        s.append(f'<line x1="{ml}" x2="{width-mr}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--rule)" stroke-width="1"/>')
        s.append(f'<text x="{ml-6}" y="{y+4:.1f}" text-anchor="end">{100*v:.0f}%</text>')
    for gi, (lab, vals) in enumerate(groups):
        x0 = ml + gi * gw + (gw - bw * len(series)) / 2
        for si, v in enumerate(vals):
            h = Hh * min(v, ymax) / ymax; x = x0 + si * bw; y = mt + Hh - h
            s.append(f'<rect x="{x+2:.1f}" y="{y:.1f}" width="{bw-4:.1f}" height="{h:.1f}" fill="{colors[si]}" rx="2"/>')
            s.append(f'<text x="{x+bw/2:.1f}" y="{y-5:.1f}" text-anchor="middle" class="lbl">{100*v:.0f}</text>')
        s.append(f'<text x="{ml+gi*gw+gw/2:.1f}" y="{height-mb+16}" text-anchor="middle" class="lbl">{lab}</text>')
    lx = ml
    for si, nm in enumerate(series):
        s.append(f'<rect x="{lx}" y="4" width="10" height="10" fill="{colors[si]}" rx="2"/><text x="{lx+14}" y="13">{nm}</text>')
        lx += 14 + 8 * len(nm) + 18
    s.append("</svg>")
    return "".join(s)


tpl = tpl.replace("{{T2_SVG}}", bars(
    [("Qwen3.8-27B", [q2["monitor"]["tpr"], q2["probe_at_monitor_fpr"]["tpr"], q2["probe_or_monitor"]["tpr"]]),
     ("gpt-oss-120b", [g2["monitor"]["tpr"], g2["probe_at_monitor_fpr"]["tpr"], g2["probe_or_monitor"]["tpr"]]),
     ("GLM-4.5-Air", [z2["monitor"]["tpr"], z2["probe_at_monitor_fpr"]["tpr"], z2["probe_or_monitor"]["tpr"]])],
    ["monitor", "probe @ monitor FPR", "probe OR monitor"], ["var(--monitor)", "var(--probe)", "var(--muted)"], "TPR"))


def t3chart(summ, order, labels):
    groups = [(labels.get(c, f"{c}×"), [summ[c]["mentioned"], summ[c]["invoked"]]) for c in order if c in summ]
    return bars(groups, ["mentions zzz", "calls zzz"], ["var(--probe)", "var(--monitor)"], "rate", height=220,
                ymax=max(0.2, max(max(v) for _, v in groups) * 1.15))


lab = {"0": "0", "rand1.5": "rand", "rand4": "rand", "-1.5": "−1.5×", "-4": "−4×"}
tpl = tpl.replace("{{T3Q_SVG}}", t3chart(t3q, ["0", "0.5", "1", "1.5", "2", "-1.5", "rand1.5"], lab))
tpl = tpl.replace("{{T3G_SVG}}", t3chart(t3g, ["0", "2", "4", "6", "8", "-4", "rand4"], lab))

rng_ = {}
for m in ("qwen3.8-27b", "gpt-oss-120b", "glm-4.5-air"):
    hs = [t["hack"] / t["n"] for (mm, s), t in tab.items() if mm == m and s != "original"]
    es = [t["enacted"] / t["n"] for (mm, s), t in tab.items() if mm == m and s != "original"]
    span = lambda xs: (f"{100*min(xs):.0f}%" if round(100*min(xs)) == round(100*max(xs)) else f"{100*min(xs):.0f}–{100*max(xs):.0f}%")
    rng_[m] = (span(hs), span(es))
t4c = collections.defaultdict(lambda: [0, 0])
for r in T4:
    v = [x for x in r["verdicts"] if x and "error" not in x]
    t4c[r["cond"]][0] += sum(x["hack"] for x in v); t4c[r["cond"]][1] += len(v)
t4txt = (f"continuations from peak cuts hack {t4c['peak'][0]}/{t4c['peak'][1]}, "
         f"from random cuts {t4c['random'][0]}/{t4c['random'][1]}")
for k, v in {"Q_T0": rng_["qwen3.8-27b"][0], "Q_T0E": rng_["qwen3.8-27b"][1], "G_T0": rng_["gpt-oss-120b"][0],
             "G_T0E": rng_["gpt-oss-120b"][1], "T4": t4txt,
             "G_AUC": f"{min(g2['probe_auroc'], g2h['probe_auroc']):.2f}–{max(g2['probe_auroc'], g2h['probe_auroc']):.2f}",
             "G_ORGAIN": f"+{100*(g2['probe_or_monitor']['tpr'] - g2['monitor']['tpr']):.0f}pp",
             "G_AUC_H": f3(g2h["probe_auroc"]), "G_AUC1": f3(g2["probe_auroc"]),
             "G_LEN": f3(gx["q_len"]), "G_RESID": f3(gx["q_resid"]), "G_TOOL": f3(g2["by_channel"]["tool_calls"]["probe_auroc"])}.items():
    tpl = tpl.replace("{{" + k + "}}", v)
for k, v in {"Q_MFPR": pct(q2["monitor"]["fpr"]), "Q_MMISS": pct(1 - q2["monitor"]["tpr"]),
             "G_MTPR": pct(g2["monitor"]["tpr"]), "G_MFPR": pct(g2["monitor"]["fpr"])}.items():
    tpl = tpl.replace("{{" + k + "}}", v)
for k, v in {"Q_AUC": f3(q2["probe_auroc"]), "Q_PTPR": pct(q2["probe_at_monitor_fpr"]["tpr"]), "Q_MTPR": pct(q2["monitor"]["tpr"]),
             "Q_LEN": f3(extra["q_len"]), "Q_RESID": f3(extra["q_resid"]), "Q_TPR5": pct(extra["q_tpr5"]),
             "Q_N": str(extra["q_n"]), "Q_DROP": extra["q_drop"]}.items():
    tpl = tpl.replace("{{" + k + "}}", v)
for k, v in {"L_T0": rng_["glm-4.5-air"][0], "L_T0E": rng_["glm-4.5-air"][1], "L_AUC": f3(z2["probe_auroc"]),
             "L_MTPR": pct(z2["monitor"]["tpr"]), "L_MFPR": pct(z2["monitor"]["fpr"]), "L_PTPR": pct(z2["probe_at_monitor_fpr"]["tpr"]),
             "L_LEN": f3(zx["q_len"]), "L_RESID": f3(zx["q_resid"]), "L_TOOL": f3(z2["by_channel"]["tool_calls"]["probe_auroc"]),
             "L_SEL": sel(z2)}.items():
    tpl = tpl.replace("{{" + k + "}}", v)

# considered vs enacted
srows = ""
for m, d in ST.items():
    for st in ("considered", "enacted"):
        e = d["stages"].get(st)
        if not e:
            continue
        srows += (f"<tr><td>{m if st == 'considered' else ''}</td><td>{st}</td><td class='num'>{e['n_pos']}</td>"
                  f"<td class='num'><b>{f3(e['auroc'])}</b> [{f3(e['ci'][0])}–{f3(e['ci'][1])}]</td><td class='num'>{f3(e['resid_auroc'])}</td>"
                  f"<td class='num'>{f3(e['len_auroc'])}</td><td class='num'>{pct(e['monitor_tpr'])}</td></tr>")
tpl = tpl.replace("{{STAGE_ROWS}}", srows)
sx = lambda m, k: f3(ST[m]["stage_x_channel"][k]["auroc"]) if k in ST[m]["stage_x_channel"] else "–"
for k, v in {"S_GR_C": sx("gpt-oss-120b", "considered/reasoning"), "S_GR_E": sx("gpt-oss-120b", "enacted/reasoning"),
             "S_LR_C": sx("GLM-4.5-Air", "considered/reasoning"), "S_LR_E": sx("GLM-4.5-Air", "enacted/reasoning"),
             "S_LT_E": sx("GLM-4.5-Air", "enacted/tool_calls")}.items():
    tpl = tpl.replace("{{" + k + "}}", v)

# T3 GLM counts
def jl(p): return [json.loads(l) for l in open(p)]
t3z = {c: jl(f"{H}/t3_raw/glm_think/{c}.jsonl") for c in ("0", "2")}
calo = {c: jl(f"{H}/t3_raw/glm_cal/{c}.jsonl") for c in ("0", "0.5", "1", "1.5", "2", "3", "-1.5")}
calt = {c: jl(f"{H}/t3_raw/glm_cal_think/{c}.jsonl") for c in ("0", "1", "2")}
cnt = lambda R, k: f"{sum(r[k] for r in R)}/{len(R)}"
for k, v in {"Z_T3_0": cnt(t3z["0"], "mentioned"), "Z_T3_2": cnt(t3z["2"], "mentioned"),
             "Z_CALO": ", ".join(f"{c}× {cnt(R, 'mentioned')}" for c, R in calo.items()),
             "Z_CALO_CODE3": cnt(calo["3"], "has_code"), "Z_CALO_CODE0": cnt(calo["0"], "has_code"),
             "Z_CALT": ", ".join(f"{c}× {cnt(R, 'mentioned')}" for c, R in calt.items())}.items():
    tpl = tpl.replace("{{" + k + "}}", v)
tpl = tpl.replace("{{CAP_DETAIL}}", open("report/cap_detail.html").read())
tpl = tpl.replace("{{T3_DETAIL}}", open("report/t3_detail.html").read())
tpl = tpl.replace("{{STEER_NEG}}", open("report/steer_neg.html").read())
tpl = tpl.replace("{{RL}}", open("report/rl.html").read())
assert "{{" not in tpl, [l for l in tpl.splitlines() if "{{" in l]
open("report/rh_probe_report_filled.html", "w").write(tpl)
print("ok")
