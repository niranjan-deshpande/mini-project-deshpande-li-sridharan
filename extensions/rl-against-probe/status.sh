#!/bin/bash
# One status snapshot: training progress per arm, eval chain, GPU use, judge loops.
for i in $(seq 1 10); do
  ssh -o BatchMode=yes -o ConnectTimeout=20 ${RL_HOST:?set RL_HOST=user@gpu-host} 'cd /data/agastyas/rh_rl; date -u
    cat logs/run_resume2.log; cat logs/run_eval.log 2>/dev/null
    for r in armA armB; do venv/bin/python armstat.py runs/$r --tau 770537 --block 5 | head -n 2; venv/bin/python armstat.py runs/$r --tau 770537 --block 5 | tail -n 2; grep -c "Traceback" runs/$r/train.log | sed "s/^/$r tracebacks: /"; done
    ls runs/eval 2>/dev/null; grep "\"eval\"" runs/eval/*.log 2>/dev/null | cut -c1-250
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed -n 5,8p' < /dev/null && break
  sleep 60
done
cd /Users/agastyasridharan/j-lens/rh_rl
echo "judge_rl: $(tail -n 1 data/judge_rl.log)"; echo "judge_eval: $(tail -n 1 data/judge_eval.log 2>/dev/null)"
python3 judge_rl_stats.py | grep -A20 "armA\|armB" | grep -v "^--" | awk 'NR<=2 || /armB/ {print} /^ *[0-9]/ {last=$0} END{}' >/dev/null
python3 judge_rl_stats.py | tail -n 30
