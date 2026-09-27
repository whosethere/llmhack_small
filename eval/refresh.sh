#!/usr/bin/env bash
# Ściąga odpowiedzi z serwera GPU, ocenia brakujące sędzią GPT-6 Sol, przebudowuje dashboard.
# Użycie: eval/refresh.sh [host]   (domyślnie aktualna sesja Forgehand)
set -uo pipefail
cd "$(dirname "$0")/.."
HOST=${1:-root@34.201.126.37}
rsync -az --include='*/' --include='answers*.jsonl' --include='config.json' --exclude='*' "$HOST:/workspace/llmhack/results/" results/
for d in results/*/; do
  run=$(basename "$d"); [ -f "$d/answers.jsonl" ] || continue
  n_ans=$(wc -l < "$d/answers.jsonl"); n_j=$(cat "$d/judgments_gpt-6-sol.jsonl" 2>/dev/null | wc -l)
  if [ "$n_j" -lt "$n_ans" ]; then
    echo "== ocena $run ($n_j/$n_ans)"
    .venv/bin/python eval/run.py --run "$run" --phase judge --judge gpt-6-sol --concurrency 8 2>&1 | grep -E "ERROR|UWAGA|średnia"
  fi
done
.venv/bin/python eval/dashboard.py
