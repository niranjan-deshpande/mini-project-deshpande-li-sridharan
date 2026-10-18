#!/bin/bash
# Final analysis once judging is done: T0 table, T2 per model, extras, report fill.
set -e
cd /Users/agastyasridharan/j-lens/rh_probe
python3 t0_summary.py data/judge_full/labels.jsonl | tee data/t0_final.txt
python3 t2_analyze.py --scores data/t2/qwen_part1 data/t2/qwen_part2 data/t2/qwen_part3 --labels data/judge_full/labels.jsonl --out data/t2_qwen_final.json --boot 1000 > data/t2_qwen_final.txt
python3 t2_analyze.py --scores data/t2/gptoss --labels data/judge_full/labels.jsonl --out data/t2_gptoss_v1.json --boot 1000 > data/t2_gptoss_v1.txt
python3 t2_analyze.py --scores data/t2/gptoss_h --labels data/judge_full/labels.jsonl --out data/t2_gptoss_h.json --boot 1000 > data/t2_gptoss_h.txt
python3 t2_extra.py data/t2_extra.json data/t2_qwen_final.json data/t2/qwen_part1 data/t2/qwen_part2 data/t2/qwen_part3
python3 t2_extra.py data/t2_extra_gptoss.json data/t2_gptoss_v1.json data/t2/gptoss
python3 t0_capability.py data/t0_capability.json
python3 cap_section.py
python3 t3_section.py
python3 steer_section.py
python3 rl_section.py
python3 fill_report.py
head -3 data/t2_qwen_final.txt; head -3 data/t2_gptoss_v1.txt
