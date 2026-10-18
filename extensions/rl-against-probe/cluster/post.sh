#!/bin/bash
# CPU only: after training ends (or the watchdog fires), write per-arm progress summaries to runs/summary.txt.
cd /data/agastyas/rh_rl
until grep -q "trainers exited" logs/run_resume.log 2>/dev/null || grep -q "watchdog done" logs/watchdog.log 2>/dev/null; do sleep 60; done
sleep 30
{ date; for r in armA armB; do venv/bin/python armstat.py runs/$r --block 5; ls runs/$r | grep ckpt; echo; done; } > runs/summary.txt 2>&1
