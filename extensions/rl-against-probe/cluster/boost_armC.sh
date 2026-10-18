#!/bin/bash
# Waits for the cross-scoring to finish (GPUs 0-3 free), then moves arm C from 2 samplers to 6 without losing work:
# wait for C's next 5-step checkpoint AND that step's log line (the trainer saves the checkpoint before it writes the
# step's samples / log), stop the old chain (script first, so it cannot mark C DONE), its trainer and samplers, and
# start run_armC2.sh, which resumes from that checkpoint. Only our own processes are touched (owner + command match).
cd /data/agastyas/rh_rl
R=runs/armC
while [ ! -f runs/score/DONE ]; do sleep 30; done
echo "scoring done $(date -u); waiting for the next armC checkpoint"
last=$(ls -d $R/ckpt_[0-9]* 2>/dev/null | sed 's/.*ckpt_//' | sort -n | tail -n 1); last=${last:-0}
while true; do
  [ -f $R/DONE ] && { echo "armC already done; nothing to boost"; exit 0; }
  n=$(ls -d $R/ckpt_[0-9]* 2>/dev/null | sed 's/.*ckpt_//' | sort -n | tail -n 1); n=${n:-0}
  if [ "$n" -gt "$last" ] && [ "$(wc -l < $R/log.jsonl)" -ge "$n" ] && [ -f $R/samples/step_$(printf %04d $((n - 1))).jsonl ]; then break; fi
  sleep 5
done
[ "$n" -ge 95 ] && { echo "checkpoint $n is too close to the end; not restarting"; exit 0; }
echo "armC checkpoint $n complete (log has $(wc -l < $R/log.jsonl) steps); switching to 6 samplers $(date -u)"
tmux kill-session -t rl_C
for pid in $(ps -u agastyas -o pid=,args= | awk '/rl_train_gptoss.py/ && /--rundir runs\/armC / {print $1}'); do kill $pid; done
sleep 20
for pid in $(ps -u agastyas -o pid=,args= | awk '/rl_train_gptoss.py/ && /--rundir runs\/armC / {print $1}'); do kill -9 $pid; done
tmux kill-session -t rl_vllm_C0; tmux kill-session -t rl_vllm_C1
# keep exactly the checkpointed steps in the log (a step after the checkpoint may have been logged meanwhile)
head -n $n $R/log.jsonl > $R/log.tmp && mv $R/log.tmp $R/log.jsonl
mkdir -p $R/samples_superseded; for f in $R/samples/step_*.jsonl; do s=$((10#$(basename $f .jsonl | sed 's/step_//'))); [ $s -ge $n ] && mv $f $R/samples_superseded/; done
sleep 10
tmux new-session -d -s rl_C2 "./run_armC2.sh > logs/run_armC2.log 2>&1"
echo "run_armC2 started $(date -u)"
