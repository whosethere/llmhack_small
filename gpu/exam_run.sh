#!/usr/bin/env bash
# Egzamin jednym poleceniem (na serwerze GPU, bez internetu): solver API -> runner -> walidacja -> answers.json.
# Użycie: bash gpu/exam_run.sh <katalog_paczki> <nazwa_wyniku> [strategia: vlm-text|base] [opcje solvera...]
#   bash gpu/exam_run.sh submission/history-2023-mock-v1 mock-bielik-vlm            # po gpu/exam_serve.sh
#   bash gpu/exam_run.sh submission/history-2023-mock-v1 mock-base base             # model bazowy (serwer :8000)
# Wynik: results/exam/<nazwa_wyniku>/answers.json (+ responses.jsonl, run.json, solver.log).
set -uo pipefail
cd "$(dirname "$0")/.."
EXAM=${1:?katalog paczki}; NAME=${2:?nazwa wyniku}; STRATEGY=${3:-vlm-text}
shift $(( $# < 3 ? $# : 3 ))
PY=${PY:-/scratch/vllm/bin/python}; [ -x "$PY" ] || PY=python3
PORT=${SOLVER_PORT:-$((9000 + RANDOM % 900))}; OUT=results/exam/$NAME
mkdir -p "$OUT"
$PY exam/solver_server.py --strategy "$STRATEGY" --port "$PORT" --cache-dir "$OUT/cache" "$@" > "$OUT/solver.log" 2>&1 &
SOLVER=$!
trap 'kill $SOLVER 2>/dev/null' EXIT
# solver ładuje indeks obrazów (pHash + DINOv2, ~15 s) i cache OCR: czekamy do 180 s, dopóki proces żyje
for i in $(seq 1 900); do curl -sf -m 2 "localhost:$PORT/health" >/dev/null && break; kill -0 $SOLVER 2>/dev/null || break; sleep 0.2; done
curl -sf -m 2 "localhost:$PORT/health" >/dev/null || { echo "solver nie wstał:"; cat "$OUT/solver.log"; exit 1; }
# ITEMS="1 5.2 26" ogranicza przebieg do wybranych zadań (testy A/B); REDO=1 rozwiązuje je od nowa
$PY exam/run_exam.py --exam "$EXAM" --solver "http://localhost:$PORT" --out "$OUT" --concurrency "${CONCURRENCY:-8}" ${ITEMS:+--items $ITEMS} ${REDO:+--redo} ${DEADLINE:+--deadline $DEADLINE}
