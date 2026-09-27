#!/usr/bin/env python3
"""Łączy listy trafień (np. e5 + bm25), deduplikuje i przelicza rerankerem (bge-reranker-v2-m3) -> top-k.
  /scratch/vllm/bin/python rag/rerank.py --inputs data/rag/e5_k30.jsonl data/rag/bm25_k30.jsonl --k 8 --out data/rag/hyb_rr_k8.jsonl"""
import argparse, json, os, torch
from sentence_transformers import CrossEncoder
ap = argparse.ArgumentParser(); ap.add_argument("--inputs", nargs="+", required=True); ap.add_argument("--k", type=int, default=8)
ap.add_argument("--model", default="BAAI/bge-reranker-v2-m3"); ap.add_argument("--out", required=True); ap.add_argument("--query-field", default="query")
a = ap.parse_args()
runs = [{json.loads(l)["id"]: json.loads(l) for l in open(p)} for p in a.inputs]
ce = CrossEncoder(a.model, device="cuda", model_kwargs={"torch_dtype": torch.float16}, max_length=1024)
with open(a.out, "w", encoding="utf-8") as f:
    for qid in runs[0]:
        q = runs[0][qid][a.query_field]; seen = {}; 
        for r in runs:
            for h in r.get(qid, {}).get("hits", []): seen.setdefault(h["text"], h)
        cands = list(seen.values())
        scores = ce.predict([(q, c["text"]) for c in cands], batch_size=32, show_progress_bar=False)
        order = sorted(range(len(cands)), key=lambda i: -float(scores[i]))[: a.k]
        hits = [{**cands[i], "score": round(float(scores[i]), 4)} for i in order]
        f.write(json.dumps({"id": qid, "query": q, "hits": hits}, ensure_ascii=False) + "\n")
print("zapisano", a.out)
