"""CPU check: every assembled SWE pair renders with the Qwen template and its span maps to tokens inside the turn."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from transformers import AutoTokenizer
from rhdecomp.gen.tokcheck import locate

tok = AutoTokenizer.from_pretrained(str(ROOT / "data/qwen_tok"))
for path in sys.argv[1:]:
    bad, n, lens = [], 0, []
    for l in open(path):
        p = json.loads(l)
        if "hack" not in p or p["checks_failed"]:
            continue
        n += 1
        for side in ("hack", "honest"):
            r = locate(tok, p[side])
            if r is None:
                bad.append((p["family"], p["meta"]["i"], side))
            else:
                lens.append(len(r[0]))
    print(f"{path}: {n} passing pairs, token-locate failures {len(bad)} {bad[:5]}, tokens/branch mean {sum(lens)/max(1,len(lens)):.0f} max {max(lens or [0])}")
