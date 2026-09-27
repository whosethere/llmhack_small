#!/usr/bin/env python3
"""Embedding kawałków (parquet) modelem sentence-transformers -> npy fp16 + ids.
  /scratch/vllm/bin/python rag/embed_text.py --chunks ~/wiki/chunks --out ~/wiki/emb_e5 --model intfloat/multilingual-e5-large
Każdy part-XX.parquet -> part-XX.npy (float16, znormalizowane). e5 wymaga prefiksu "passage: " / "query: "."""
import argparse, glob, os, time
import numpy as np, pyarrow.parquet as pq, torch
from sentence_transformers import SentenceTransformer
ap = argparse.ArgumentParser(); ap.add_argument("--chunks", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--model", default="intfloat/multilingual-e5-large"); ap.add_argument("--bs", type=int, default=256); ap.add_argument("--maxlen", type=int, default=512)
a = ap.parse_args(); os.makedirs(os.path.expanduser(a.out), exist_ok=True)
m = SentenceTransformer(a.model, device="cuda", model_kwargs={"torch_dtype": torch.float16}); m.max_seq_length = a.maxlen
prefix = "passage: " if "e5" in a.model else ""
files = sorted(glob.glob(os.path.join(os.path.expanduser(a.chunks), "*.parquet"))); t0 = time.time(); total = 0
for f in files:
    outp = os.path.join(os.path.expanduser(a.out), os.path.basename(f).replace(".parquet", ".npy"))
    if os.path.exists(outp): continue
    texts = [prefix + t for t in pq.read_table(f, columns=["text"]).column("text").to_pylist()]
    # sortowanie po długości = mniej paddingu
    order = np.argsort([len(t) for t in texts]); emb = np.zeros((len(texts), m.get_sentence_embedding_dimension()), dtype=np.float16)
    e = m.encode([texts[i] for i in order], batch_size=a.bs, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    emb[order] = e.astype(np.float16); np.save(outp, emb); total += len(texts)
    print(f"{os.path.basename(f)}: {len(texts)} kawałków, razem {total}, {total/(time.time()-t0):.0f}/s", flush=True)
print("EMBED DONE", total)
