#!/usr/bin/env bash
# Bielik 1.5B Q8_0 GGUF w llama.cpp (Docker, CUDA) na 127.0.0.1:8002; 8 slotów po 8k kontekstu. Log: sudo docker logs bielik
set -uo pipefail
HUB=$HOME/models/hf/hub; NP=${NP:-8}; CTX=$((8192*NP))
G=models--speakleash--Bielik-1.5B-v3.0-Instruct-GGUF/snapshots/99d72cc757be4f8b958018f12d6898bad5c19c9e/Bielik-1.5B-v3.0-Instruct.Q8_0.gguf
if curl -sf -m 3 localhost:8002/health >/dev/null; then echo "bielik już działa na :8002"; exit 0; fi
sudo docker rm -f bielik >/dev/null 2>&1
sudo docker run -d --name bielik --gpus all -p 127.0.0.1:8002:8000 -v "$HUB":/hf:ro ghcr.io/ggml-org/llama.cpp:server-cuda \
  -m /hf/$G --alias solver --host 0.0.0.0 --port 8000 -ngl 99 -c $CTX --parallel $NP -b 2048 -ub 2048 --flash-attn on --jinja --temp 0 >/dev/null
for i in $(seq 1 60); do sleep 3; curl -sf -m 3 localhost:8002/health >/dev/null && { echo "bielik up ($((i*3))s)"; exit 0; }; done
echo "bielik nie wstał"; sudo docker logs --tail 20 bielik; exit 1
