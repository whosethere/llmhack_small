#!/usr/bin/env bash
# KATEGORIA „MAŁY” (wagi <= 2 GB): Bielik 1.5B v3 Q8_0 (1,7 GB, llama.cpp w Dockerze, :8002) + harness tekstowy -> answers.json.
# Użycie: bash gpu/exam_small.sh <katalog_paczki> <nazwa_wyniku>      np. bash gpu/exam_small.sh submission/history-2023-mock-v1 small-final
# Konfiguracja docelowa (wybrana na mockach 2023/2025, sędzia GPT-6 Sol v4; goły model 18-19/60):
#   - goły prompt jak w benchmarku (--text-only: bez nagłówka i instrukcji formatu; mały model odpowiada pełniej, literę wyciąga harness),
#   - ilustracje: OCR tesseract na powiększeniu 2x + identyfikacja kopii obrazów z Wikipedii (pHash + DINOv2) jako podpis,
#   - retrieval (gpu/serve_rag.sh, :8010) TYLKO do wypracowania (treść tematu jako zapytanie, k=4); w zadaniach fragmenty nie pomagały,
#   - wypracowanie modułem exam/essay_small.py (temat aspektowy, szablonowy wstęp/zakończenie, akapit na aspekt z notatkami, porządki).
# Wyniki (średnia z 3 przebiegów): 2023 mock 20,7/60 (34%), 2025 mock 19,7/60 (33%); goły model 18-19. Zmienne: RAG (1), RAG_SCOPE (essay|open|all), OCR (tesseract|none), TEXT_HEADERS (1 = etykiety ŹRÓDŁO/POLECENIE), CONCURRENCY (8).
set -uo pipefail
cd "$(dirname "$0")/.."
EXAM=${1:?paczka}; NAME=${2:?nazwa}
export HF_HOME=${HF_HOME:-$HOME/models/hf} HF_HUB_OFFLINE=1 PY=${PY:-/scratch/vllm/bin/python}
RAG=${RAG:-1}; RAG_ARGS=()
if [ "$RAG" = 1 ]; then
  bash gpu/serve_rag.sh >/dev/null 2>&1 || true
  curl -sf -m 3 localhost:${RAG_PORT:-8010}/ >/dev/null && RAG_ARGS=(--rag-url http://localhost:${RAG_PORT:-8010} --rag-k "${RAG_K:-3}" --rag-min-score "${RAG_MIN:-0.5}" --rag-query "${RAG_QUERY:-model}" --rag-scope "${RAG_SCOPE:-essay}") \
    || echo "UWAGA: retrieval nie odpowiada, wypracowanie bez notatek"
fi
IMG_ARGS=(); for d in ${IMG_INDEX:-$HOME/wiki/img_dino $HOME/wiki/img_en}; do [ -f "$d/phash.npy" ] && IMG_ARGS+=("$d"); done
[ ${#IMG_ARGS[@]} -gt 0 ] && IMG_ARGS=(--img-index "$(IFS=,; echo "${IMG_ARGS[*]}")") || echo "UWAGA: brak indeksu obrazów"
OCR=${OCR:-tesseract}; [ "$OCR" = tesseract ] && ! command -v tesseract >/dev/null && { echo "UWAGA: brak tesseracta"; OCR=none; }
bash gpu/serve_bielik.sh || exit 1
t0=$(date +%s)
CONCURRENCY=${CONCURRENCY:-8} DEADLINE=${DEADLINE:-900} bash gpu/exam_run.sh "$EXAM" "$NAME" single-vlm --text-only ${TEXT_HEADERS:+--text-headers} --think off \
  --llm-url http://localhost:8002/v1 --llm-model solver --ocr "$OCR" --ocr-scale "${OCR_SCALE:-2}" --img-scale none \
  --max-tokens "${MAXTOK:-1200}" --essay-max-tokens "${ESSAYTOK:-2000}" --temperature "${TEMP:-0}" --essay-rag-k "${ESSAY_RAG_K:-4}" "${RAG_ARGS[@]}" "${IMG_ARGS[@]}"
echo "czas: $(( $(date +%s) - t0 )) s"
[ -f "results/exam/$NAME/answers.json" ] || { echo "BŁĄD: brak answers.json"; tail -20 "results/exam/$NAME/solver.log"; exit 1; }
python3 exam/validate.py --exam "$EXAM" "results/exam/$NAME/answers.json" && echo "-> results/exam/$NAME/answers.json"
