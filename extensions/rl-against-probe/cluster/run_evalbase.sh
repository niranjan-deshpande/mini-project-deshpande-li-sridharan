#!/bin/bash
cd /data/agastyas/rh_rl
venv/bin/python sample.py ib_eval_tasks.jsonl p0_eval_lenient.jsonl --mode lenient --effort medium --n 4 --workers 192 > logs/p0_eval_lenient.log 2>&1
venv/bin/python sample.py ib_eval_tasks.jsonl p0_eval_strict.jsonl --mode strict --effort medium --n 4 --workers 192 > logs/p0_eval_strict.log 2>&1
