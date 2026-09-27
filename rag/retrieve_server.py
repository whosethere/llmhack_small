#!/usr/bin/env python3
"""Lokalny serwis retrievalu (offline): indeks e5 na GPU + reranker; POST /search {"queries": [...], "k": 5, "min_score": 0.0}
  /scratch/vllm/bin/python rag/retrieve_server.py --chunks ~/wiki/en_chunks --emb ~/wiki/en_emb_e5 --port 8010 [--rerank]
Odpowiedź: {"hits": [{"title","section","kind","text","score","query"}]} (deduplikowane, posortowane po score rerankera lub cos)."""
import argparse, json, os, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from retrieve import load_index
from sentence_transformers import SentenceTransformer, CrossEncoder

ap = argparse.ArgumentParser(); ap.add_argument("--chunks", required=True); ap.add_argument("--emb", required=True)
ap.add_argument("--model", default="intfloat/multilingual-e5-large"); ap.add_argument("--rerank", action="store_true"); ap.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3")
ap.add_argument("--port", type=int, default=8010); ap.add_argument("--cand", type=int, default=30)
ap.add_argument("--mask", help="plik .npy bool (rag/build_mask.py): w wyszukiwaniu biorą udział tylko fragmenty z True")
a = ap.parse_args()
import pyarrow as pa
Ms, Es = [], []
for cdir, edir in zip(a.chunks.split(","), a.emb.split(",")):
    m, e = load_index(os.path.expanduser(cdir), os.path.expanduser(edir)); Ms.append(m); Es.append(e)
M = pa.concat_tables(Ms); E = torch.cat(Es)
titles, sections, kinds, texts = [M.column(c).to_pylist() for c in ("title", "section", "kind", "text")]
enc = SentenceTransformer(a.model, device="cuda", model_kwargs={"torch_dtype": torch.float16}); prefix = "query: " if "e5" in a.model else ""
ce = CrossEncoder(a.reranker, device="cuda", model_kwargs={"torch_dtype": torch.float16}, max_length=1024) if a.rerank else None
MASK_PEN = None
if a.mask:
    mk = np.load(os.path.expanduser(a.mask))
    assert mk.shape[0] == E.shape[0], f"maska ma {mk.shape[0]} pozycji, indeks {E.shape[0]}"
    MASK_PEN = torch.where(torch.from_numpy(mk).cuda(), 0.0, float("-inf")).half()  # dodawane do cos: wykluczone -> -inf
    print(f"maska: {int(mk.sum())}/{len(mk)} fragmentów aktywnych ({a.mask})", flush=True)
lock = threading.Lock()
print(f"retrieval: {E.shape[0]} kawałków, rerank={a.rerank}, port {a.port}", flush=True)

def search(queries, k, min_score):
    with lock:
        qe = enc.encode([prefix + q for q in queries], normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False).half()
        sc = qe @ E.T
        if MASK_PEN is not None: sc = sc + MASK_PEN
        v, ix = torch.topk(sc.float(), k=a.cand if ce else k, dim=1)
        cands = {}
        for qi, (vs, ixs) in enumerate(zip(v.tolist(), ix.tolist())):
            for s, j in zip(vs, ixs):
                if j not in cands or cands[j]["score"] < s: cands[j] = {"title": titles[j], "section": sections[j], "kind": kinds[j], "text": texts[j], "score": s, "query": queries[qi]}
        hits = list(cands.values())
        if ce and hits:
            sc = ce.predict([(h["query"], h["text"]) for h in hits], batch_size=32, show_progress_bar=False)
            for h, s in zip(hits, sc): h["score"] = float(torch.sigmoid(torch.tensor(float(s))))  # 0..1
        hits = sorted(hits, key=lambda h: -h["score"])
        return [dict(h, score=round(h["score"], 4)) for h in hits if h["score"] >= min_score][:k]

class H(BaseHTTPRequestHandler):
    def log_message(self, *x): pass
    def do_GET(self): self._send(200, {"ok": True, "n": int(E.shape[0]), "rerank": a.rerank, "mask": a.mask})
    def do_POST(self):
        try:
            b = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            qs = [q for q in b.get("queries", []) if isinstance(q, str) and q.strip()][:5]
            self._send(200, {"hits": search(qs, int(b.get("k", 5)), float(b.get("min_score", 0.0))) if qs else []})
        except Exception as e:
            self._send(500, {"error": f"{type(e).__name__}: {e}"})
    def _send(self, code, obj):
        d = json.dumps(obj, ensure_ascii=False).encode(); self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(d))); self.end_headers(); self.wfile.write(d)

ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
