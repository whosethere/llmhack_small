#!/usr/bin/env bash
# Pobiera wszystko do kategorii „mały”: Bielik 1.5B Q8_0 (1,7 GB) + modele retrievalu (nie liczą się do limitu 2 GB) + obraz llama.cpp.
set -uo pipefail
export HF_HOME=${HF_HOME:-$HOME/models/hf}
PY=${PY:-/scratch/vllm/bin/python}; HF=$(dirname "$PY")/hf; [ -x "$HF" ] || HF=hf
$HF download speakleash/Bielik-1.5B-v3.0-Instruct-GGUF Bielik-1.5B-v3.0-Instruct.Q8_0.gguf
for m in intfloat/multilingual-e5-large BAAI/bge-reranker-v2-m3 facebook/dinov2-base; do $HF download $m >/dev/null && echo "DONE $m"; done
sudo docker pull ghcr.io/ggml-org/llama.cpp:server-cuda
echo "Indeks Wikipedii (fragmenty, embeddingi e5, pHash obrazów): rag/README.md; OCR: apt install tesseract-ocr tesseract-ocr-pol"
