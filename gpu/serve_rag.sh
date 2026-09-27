#!/usr/bin/env bash
# Serwer retrievalu (Wikipedia PL + EN-historia, e5 + reranker bge-m3) na 127.0.0.1:8010, offline.
# Użycie: bash gpu/serve_rag.sh   (zmienne: WIKI (~/wiki), PY (/scratch/vllm/bin/python), RAG_PORT (8010),
#   MASK=plik.npy ogranicza wyszukiwanie do fragmentów z maski (rag/build_mask.py); RESTART=1 zabija działający serwer)
set -uo pipefail
cd "$(dirname "$0")/.."
WIKI=${WIKI:-$HOME/wiki}; PY=${PY:-/scratch/vllm/bin/python}; PORT=${RAG_PORT:-8010}
export HF_HOME=${HF_HOME:-$HOME/models/hf} HF_HUB_OFFLINE=1
if [ "${RESTART:-0}" = 1 ]; then pkill -f "rag/retrieve_server.py" 2>/dev/null; sleep 3; fi
if curl -sf -m 3 "localhost:$PORT/" >/dev/null; then echo "retrieval już działa na :$PORT: $(curl -s localhost:$PORT/)"; exit 0; fi
mkdir -p ~/logs
(setsid nohup "$PY" rag/retrieve_server.py --chunks "$WIKI/chunks,$WIKI/en_chunks" --emb "$WIKI/emb_e5,$WIKI/en_emb_e5" --port "$PORT" --rerank ${MASK:+--mask "$MASK"} > ~/logs/retrieve_server.log 2>&1 < /dev/null &)
for i in $(seq 1 90); do
  sleep 5
  curl -sf -m 3 "localhost:$PORT/" >/dev/null && { echo "retrieval up ($((i*5))s): $(curl -s localhost:$PORT/)"; exit 0; }
done
echo "retrieval nie wstał; log: ~/logs/retrieve_server.log"; exit 1
