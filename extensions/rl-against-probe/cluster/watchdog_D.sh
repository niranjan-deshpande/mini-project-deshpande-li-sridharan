#!/bin/bash
# Frees the GPUs used by arm D no matter how the run ends, independent of any outside connection (runs in its own tmux).
#  - finished: runs/eval/DONE_D, or runs/armD/DONE without ckpt_final_100 (trainer failed, eval skipped) -> clean up, exit
#  - stalled:  no progress file (training log, step samples, eval outputs, chain log) modified for STALL minutes -> clean up
#  - deadline: HARD_DEADLINE (UTC epoch) passed -> clean up
# Clean-up touches only our own run: the arm-D tmux sessions, then any GPU process owned by agastyas whose working
# directory is /data/agastyas/rh_rl (the trainer, eval scorers and vLLM servers all run from there).
cd /data/agastyas/rh_rl
STALL=${STALL:-45}; HARD_DEADLINE=${HARD_DEADLINE:?}
log() { echo "$(date -u) $*"; }
cleanup() {
  log "cleanup: $1"
  for s in rl_D rl_vllm_D0 rl_vllm_D1 rl_vllm_D2 rl_vllm_D3 rl_vllm_D4 rl_vllm_D5 rl_evalvllm_D1 rl_evalvllm_D2 rl_evalvllm_D3 rl_evalvllm_D4; do
    tmux kill-session -t $s 2>/dev/null && log "killed tmux $s"
  done
  for sig in TERM KILL; do
    sleep 20
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
      [ "$(ps -o user= -p $p 2>/dev/null)" = "agastyas" ] || continue
      [ "$(readlink /proc/$p/cwd)" = "/data/agastyas/rh_rl" ] || continue
      kill -$sig $p 2>/dev/null && log "kill -$sig $p ($(ps -o args= -p $p 2>/dev/null | cut -c1-60))"
    done
  done
  sleep 10
  log "GPU memory after cleanup: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
  exit 0
}
newest() { stat -c %Y runs/armD/log.jsonl runs/armD/samples/* runs/armD/ckpt_* logs/run_armD.log runs/eval/armD_*.log runs/eval/armD_*/*.jsonl 2>/dev/null | sort -n | tail -n 1; }
log "watchdog up: stall ${STALL} min, deadline $(date -u -d @$HARD_DEADLINE)"
while true; do
  [ -f runs/eval/DONE_D ] && cleanup "eval finished"
  [ -f runs/armD/DONE ] && [ ! -d runs/armD/ckpt_final_100 ] && { sleep 60; cleanup "trainer ended without ckpt_final_100"; }
  now=$(date +%s); last=$(newest); last=${last:-$now}
  [ $now -gt $HARD_DEADLINE ] && cleanup "hard deadline passed"
  [ $((now - last)) -gt $((STALL * 60)) ] && cleanup "no progress for $(((now - last) / 60)) min"
  sleep 60
done
