#!/usr/bin/env python3
"""Indeks BM25 (bm25s) nad kawałkami Wikipedii; tokenizacja: małe litery, alfanumeryczne, prosty stemming polski (obcięcie końcówek).
  python3 rag/bm25_build.py --chunks ~/wiki/chunks --out ~/wiki/bm25
  python3 rag/bm25_build.py --search --out ~/wiki/bm25 --chunks ~/wiki/chunks --questions data/questions.jsonl --years ... --k 30 --outfile data/rag/bm25_k30.jsonl"""
import argparse, glob, json, os, re, sys, time
import bm25s, numpy as np, pyarrow.parquet as pq
TOK = re.compile(r"[a-ząćęłńóśźż0-9]+")
def stem(w):  # zgrubny stemmer: obcina typowe polskie końcówki fleksyjne
    for suf in ("ego","emu","ach","ami","ach","owi","iem","em","om","ów","ie","ia","ią","ie","iu","ę","ą","a","e","i","o","u","y"):
        if len(w) > 5 and w.endswith(suf): return w[:-len(suf)]
    return w
def tok(text): return [stem(w) for w in TOK.findall(text.lower()) if len(w) > 1]
sys.path.insert(0, os.path.dirname(__file__))
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--chunks", required=True); ap.add_argument("--out", required=True); ap.add_argument("--search", action="store_true")
    ap.add_argument("--questions"); ap.add_argument("--years", nargs="*"); ap.add_argument("--k", type=int, default=30); ap.add_argument("--outfile"); ap.add_argument("--query-mode", default="full")
    a = ap.parse_args(); out = os.path.expanduser(a.out); chunks = os.path.expanduser(a.chunks)
    files = sorted(glob.glob(os.path.join(chunks, "*.parquet")))
    if not a.search:
        t0 = time.time(); texts = []
        for f in files: texts += pq.read_table(f, columns=["text"]).column("text").to_pylist()
        print(f"kawałków {len(texts)}, wczytane {time.time()-t0:.0f}s", flush=True)
        corpus_tokens = bm25s.tokenize(texts, stopwords=None, stemmer=None, lower=True, token_pattern=r"[a-ząćęłńóśźż0-9]+", show_progress=False)
        print(f"tokenizacja {time.time()-t0:.0f}s", flush=True)
        idx = bm25s.BM25(k1=1.2, b=0.75); idx.index(corpus_tokens, show_progress=False)
        idx.save(out); print(f"BM25 zbudowany i zapisany w {out}, {time.time()-t0:.0f}s", flush=True)
    else:
        from retrieve import build_query
        idx = bm25s.BM25.load(out, mmap=True)
        metas = []
        for f in files: metas.append(pq.read_table(f, columns=["title", "section", "kind", "text"]))
        import pyarrow as pa; M = pa.concat_tables(metas); titles, sections, kinds, txts = [M.column(c).to_pylist() for c in ("title", "section", "kind", "text")]
        items = [json.loads(l) for l in open(a.questions)]
        if a.years: items = [q for q in items if q["id"].split("_z")[0] in set(a.years)]
        queries = [build_query(q, 600, a.query_mode) for q in items]
        qt = bm25s.tokenize(queries, stopwords=None, lower=True, token_pattern=r"[a-ząćęłńóśźż0-9]+", show_progress=False)
        t0 = time.time(); res, sc = idx.retrieve(qt, k=a.k, show_progress=False); print(f"bm25 {len(items)} zapytań {time.time()-t0:.1f}s")
        with open(a.outfile, "w", encoding="utf-8") as f:
            for q, qq, r, s in zip(items, queries, res, sc):
                hits = [{"title": titles[j], "section": sections[j], "kind": kinds[j], "text": txts[j], "score": float(v)} for j, v in zip(r, s)]
                f.write(json.dumps({"id": q["id"], "query": qq, "hits": hits}, ensure_ascii=False) + "\n")
        print("zapisano", a.outfile)
