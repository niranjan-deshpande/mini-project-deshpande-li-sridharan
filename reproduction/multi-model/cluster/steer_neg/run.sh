# chain: 4 servers (one per GPU/condition) -> wait healthy -> 4 clients in parallel -> stop servers -> DONE marker
cd /data/agastyas/rh_probe
PY=/data/agastyas/latekey/vllm-env/bin/python
CONDS="0 neg2 neg3 rand3"
i=0
for c in $CONDS; do
  GPU=$((4+i)) PORT=$((8300+i)) COND=$c nohup bash steer_neg/serve.sh > steer_neg/serve_$c.log 2>&1 &
  echo $! > steer_neg/serve_$c.pid; i=$((i+1))
done
i=0
for c in $CONDS; do
  until curl -sf localhost:$((8300+i))/health >/dev/null; do
    sleep 15
    kill -0 $(cat steer_neg/serve_$c.pid) 2>/dev/null || { echo "server $c died $(date)" > steer_neg/FAILED; exit 1; }
  done; i=$((i+1))
done
for c in neg2 neg3 rand3; do grep -q "\[steer\] layer 33" steer_neg/serve_$c.log || { echo "no steer line in $c" > steer_neg/FAILED; exit 1; }; done
echo "servers up $(date)"
i=0; CP=""
for c in $CONDS; do $PY steer_neg/client.py $((8300+i)) $c 2 > steer_neg/client_$c.log 2>&1 & CP="$CP $!"; i=$((i+1)); done
wait $CP
for c in $CONDS; do kill $(cat steer_neg/serve_$c.pid) 2>/dev/null; done
echo "ALL DONE $(date)" > steer_neg/DONE
