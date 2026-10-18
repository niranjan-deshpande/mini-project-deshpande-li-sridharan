"""Fetch the paper's 100 story topics (Appendix 'Dataset generation') into topics.json. Run once; not run by the agent.

Usage: python prompts/emotions/_fetch_topics.py [--html saved_index.html]
Source: https://transformer-circuits.pub/2026/emotions/index.html (Sofroniew et al. 2026). The list lives between the
heading sentence ending 'stories and dialogues datasets.' and 'Emotional stories prompt.'.
"""
import argparse, html, json, re, urllib.request
from pathlib import Path

URL = "https://transformer-circuits.pub/2026/emotions/index.html"
ap = argparse.ArgumentParser()
ap.add_argument("--html", help="use a saved copy of the page instead of downloading (the page is ~40 MB)")
a = ap.parse_args()
raw = Path(a.html).read_text(errors="ignore") if a.html else urllib.request.urlopen(URL, timeout=120).read().decode("utf-8", "ignore")
raw = re.sub(r"<script.*?</script>|<style.*?</style>", "", raw, flags=re.S)
txt = html.unescape(re.sub(r"<[^>]+>", "\n", raw))
seg = txt[txt.index("stories and dialogues datasets."):txt.index("Emotional stories prompt.")]
topics = [l.strip() for l in seg.split("\n")[1:] if l.strip()]
assert len(topics) == 100 and len(set(topics)) == 100, len(topics)
out = Path(__file__).with_name("topics.json")
json.dump({"source": URL, "topics": [{"topic_id": f"t{i:03d}", "topic": t} for i, t in enumerate(topics)]}, open(out, "w"), indent=1)
print(f"wrote {len(topics)} topics to {out}")
