#!/usr/bin/env bash
# Kalibracja lokalnych sędziów (prompt v4) na odpowiedziach organizatorów.
# Każdy model: start vLLM na maszynie GPU (rola judge) -> tunel SSH -> calibrate.py grade --as <nazwa>-v4.
# Użycie: eval/judge_calib_bench.sh HOST "nazwa=org/model[=flagi vllm]" ...
set -uo pipefail
cd "$(dirname "$0")/.."
HOST=$1; shift
PY=${PY:-../../../.venv/bin/python}
RUNS="org-gemma-4-12b-vision org-qwen-3-5-9b-vision org-ministral-3-14b-text org-llava-bielik-11b-text org-qwen3-vl-2b-instruct-vision org-gemma-3-1b org-qwen3-vl-2b-instruct"
ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -N -L 8000:127.0.0.1:8000 "$HOST" & TUN=$!
trap 'kill $TUN 2>/dev/null' EXIT
sleep 3
for spec in "$@"; do
  NAME=${spec%%=*}; rest=${spec#*=}; M=${rest%%=*}; X=""; [[ "$rest" == *=* ]] && X=${rest#*=}
  echo "=== $NAME ($M) $(date +%H:%M:%S)"
  until ssh -o BatchMode=yes "$HOST" "grep -qE '(DONE|FAIL) $M( |\$)' /scratch/logs/download.log" < /dev/null; do sleep 30; done
  ssh -o BatchMode=yes "$HOST" "grep -q 'FAIL $M' /scratch/logs/download.log" < /dev/null && { echo "  FAIL download"; continue; }
  ssh -o BatchMode=yes "$HOST" "cd ~/llmhack && EXTRA='$X' bash gpu/serve.sh judge $M 8000 0.92" < /dev/null
  ok=0; for i in $(seq 1 120); do sleep 10
    curl -sf -m 3 localhost:8000/health >/dev/null && { ok=1; break; }
    ssh -o BatchMode=yes "$HOST" 'grep -q "Engine core initialization failed" /scratch/logs/judge.log' < /dev/null && break
  done
  if [ $ok != 1 ]; then echo "  FAIL serve"; ssh -o BatchMode=yes "$HOST" 'grep -E "Error" /scratch/logs/judge.log | grep -v raise | tail -3 | cut -c1-250' < /dev/null; continue; fi
  echo "  up po $((i*10)) s"; t0=$(date +%s)
  $PY eval/calibrate.py grade --judge local-v4 --as "$NAME-v4" --judge-url http://localhost:8000/v1 \
      --desc results/org-opisy/opisy_2023.jsonl --desc-all --runs $RUNS --concurrency 16 2>&1 | grep -E "zapisano|błęd|Error" | tail -3
  echo "  czas oceny: $(( $(date +%s) - t0 )) s"
done
ssh -o BatchMode=yes "$HOST" 'pgrep -f "^/scratch/vllm/bin/python.*vllm serve" | xargs -r kill' < /dev/null
echo "=== CALIB DONE $(date +%H:%M:%S)"
