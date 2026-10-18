"""Task B: why our logit lens of Agastya's turn L40 vector doesn't show his reported 捷径/省事/何必/绕过.

CPU only. Compares our session-1 readout of `agastya__turn` (data/gpu/s1/lenses.json) with his own saved lens
(../../reproduction/multi-model/results/dom/qwen/logit_lens.json, lens[key][L]['top'], top-15, head(norm(v)) in build_dom.py),
and finds every (vector, layer) of his where the reported tokens appear. Output: results/lenses/lens_discrepancy.json
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AG = json.load(open(ROOT.parents[1] / "reproduction/multi-model/results/dom/qwen/logit_lens.json"))["lens"]
OURS = json.load(open(ROOT / "data/gpu/s1/lenses.json"))["logit"]["agastya__turn"]
T3 = json.load(open(ROOT.parents[1] / "reproduction/multi-model/results/t3_qwen_summary.json"))
REPORTED = ["捷径", "省事", "何必", "绕过", " bypass", " shortcut"]

ours = [t for t, *_ in OURS["top"]][:15]
his = AG["turn__all"]["40"]["top"]
hits = {w: sorted([[k, int(L), v["top"].index(w)] for k in AG for L, v in AG[k].items() if w in v["top"]], key=lambda x: (x[0], x[1]))
        for w in REPORTED}
res = {"ours_agastya_turn_L40_top15": ours, "his_turn_all_L40_top15": his,
       "overlap_top15": len(set(ours) & set(his)), "same_top10_order": ours[:9] == his[:9],
       "reported_tokens_in_any_turn_all_layer": {w: [h for h in hits[w] if h[0] == "turn__all"] for w in REPORTED},
       "reported_token_locations": {w: {"n": len(h), "keys": sorted({x[0] for x in h}),
                                        "layers_last_all": [x[1] for x in h if x[0] == "last__all"]} for w, h in hits.items()},
       "report_quotes_last_token_layers": {L: AG["last__all"][str(L)]["top"] for L in (47, 51, 55, 63)},
       "t3_steered_vector": {"key": T3["key"], "layer": T3["layer"], "lens_top15": AG[T3["key"]][str(T3["layer"])]["top"]},
       "detection_probe": "turn__all L40, max-pool (report T2)"}
(ROOT / "results/lenses").mkdir(parents=True, exist_ok=True)
json.dump(res, open(ROOT / "results/lenses/lens_discrepancy.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
