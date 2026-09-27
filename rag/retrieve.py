#!/usr/bin/env python3
"""Retrieval z indeksu Wikipedii dla zadań z data/questions.jsonl (albo paczki egzaminu exam.json).
  /scratch/vllm/bin/python rag/retrieve.py --chunks ~/wiki/chunks --emb ~/wiki/emb_e5 --model intfloat/multilingual-e5-large \
      --questions data/questions.jsonl --years 2023_f2023 2024_f2023 2025_f2023 2026_f2023 --k 8 --out data/rag/e5_k8.jsonl
Zapytanie = polecenie (+ skrócony kontekst źródłowy). Wynik: {id, query, hits:[{title, section, kind, text, score}]}.
Indeks trzymany na GPU w fp16 (2.3M x 1024 = ~4.7 GB), podobieństwo = iloczyn skalarny (wektory znormalizowane)."""
import argparse, glob, json, os, re, time
import numpy as np, pyarrow.parquet as pq, torch
from sentence_transformers import SentenceTransformer

def load_index(chunks_dir, emb_dir):
    metas, embs = [], []
    for f in sorted(glob.glob(os.path.join(chunks_dir, "*.parquet"))):
        npy = os.path.join(emb_dir, os.path.basename(f).replace(".parquet", ".npy"))
        if not os.path.exists(npy): continue
        t = pq.read_table(f, columns=["title", "section", "kind", "text"])
        metas.append(t); embs.append(np.load(npy))
    E = torch.from_numpy(np.concatenate(embs)).cuda()  # fp16
    import pyarrow as pa
    M = pa.concat_tables(metas)
    return M, E

VERB = re.compile(r"^(Rozstrzygnij|Podaj|Wyjaśnij|Oceń|Wymień|Dokończ|Uzupełnij|Sformułuj|Porównaj|Przedstaw|Określ|Wskaż|Zaznacz|Napisz|Na podstawie|Przyporządkuj|Uporządkuj|Scharakteryzuj|Uzasadnij|Zadanie zawiera)", re.I)
HEAD = re.compile(r"^(Źródło \d\.|Fragment|Tekst|Mapa|Ilustracja|Fotografia|Plakat|Rysunek|Tabela|Wykres|Plan)", re.I)

def build_query(q, max_ctx=600, mode="full"):
    tresc = re.sub(r"\n{2,}", "\n", q["tresc"]).strip()
    ctx = (q.get("kontekst") or "").strip()
    if mode == "polecenie":
        # tylko polecenie + nagłówki/podpisy źródeł (bez długich cytatów, które dominują zapytanie)
        lines = (ctx + "\n" + tresc).split("\n"); keep = []
        for i, ln in enumerate(lines):
            l = ln.strip()
            if not l: continue
            if VERB.match(l) or HEAD.match(l) or (i > 0 and VERB.match(lines[i-1].strip()) and len(l) < 200):
                keep.append(l)
        # opcje zamknięte (A./B./1./2.) i tematy eseju też są cenne
        keep += [l.strip() for l in tresc.split("\n") if re.match(r"^([A-D]\.|\d\.)\s", l.strip())]
        return "\n".join(dict.fromkeys(keep))[:1000] or tresc[:600]
    text = tresc if len(tresc) <= 1200 else tresc[:1200]
    if ctx: text = ctx[:max_ctx] + "\n" + text
    return text

def search(model, E, texts, k, prefix, bs=64):
    out_scores, out_idx = [], []
    for i in range(0, len(texts), bs):
        qe = model.encode([prefix + t for t in texts[i:i+bs]], normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False).half()
        s = qe @ E.T  # (b, N)
        v, ix = torch.topk(s.float(), k=k, dim=1)
        out_scores += v.cpu().tolist(); out_idx += ix.cpu().tolist()
    return out_scores, out_idx

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True); ap.add_argument("--emb", required=True); ap.add_argument("--model", default="intfloat/multilingual-e5-large")
    ap.add_argument("--questions", default="data/questions.jsonl"); ap.add_argument("--years", nargs="*"); ap.add_argument("--exam", help="zamiast --questions: paczka exam.json")
    ap.add_argument("--k", type=int, default=8); ap.add_argument("--out", required=True); ap.add_argument("--max-ctx", type=int, default=600)
    ap.add_argument("--query-mode", choices=["full", "polecenie"], default="full")
    ap.add_argument("--hyde", help="answers.jsonl z wcześniejszego runu: szkic odpowiedzi doklejany do zapytania (HyDE)")
    a = ap.parse_args()
    t0 = time.time(); M, E = load_index(os.path.expanduser(a.chunks), os.path.expanduser(a.emb))
    print(f"indeks: {E.shape[0]} kawałków, {E.shape[1]} wym., {time.time()-t0:.0f}s", flush=True)
    if a.exam:
        ex = json.load(open(a.exam)); items = [{"id": it["id"], "tresc": it["question"], "kontekst": it.get("source_text", "")} for it in ex["items"]]
    else:
        items = [json.loads(l) for l in open(a.questions)]
        if a.years: items = [q for q in items if q["id"].split("_z")[0] in set(a.years)]
    model = SentenceTransformer(a.model, device="cuda", model_kwargs={"torch_dtype": torch.float16})
    prefix = "query: " if "e5" in a.model else ""
    queries = [build_query(q, a.max_ctx, a.query_mode) for q in items]
    if a.hyde:
        drafts = {json.loads(l)["id"]: json.loads(l)["odpowiedz"] for l in open(a.hyde)}
        queries = [qt + "\nSzkic odpowiedzi: " + drafts.get(q["id"], "")[:700] for q, qt in zip(items, queries)]
    t0 = time.time(); scores, idx = search(model, E, queries, a.k, prefix)
    print(f"retrieval {len(items)} zapytań w {time.time()-t0:.1f}s", flush=True)
    titles, sections, kinds, texts = [M.column(c).to_pylist() for c in ("title", "section", "kind", "text")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for q, qt, sc, ix in zip(items, queries, scores, idx):
            hits = [{"title": titles[j], "section": sections[j], "kind": kinds[j], "text": texts[j], "score": round(s, 4)} for s, j in zip(sc, ix)]
            f.write(json.dumps({"id": q["id"], "query": qt, "hits": hits}, ensure_ascii=False) + "\n")
    print("zapisano", a.out)
