#!/usr/bin/env python3
"""Artykuły (parquet ze skanu ZIM) -> kawałki do RAG.
  python3 rag/chunk_wiki.py --articles ~/wiki/articles --out ~/wiki/chunks --target 512
Kawałek = "Tytuł artykułu > sekcja\n" + tekst ~target znaków, cięty na granicach zdań; krótkie akapity łączone.
Osobne kawałki: infobox (skrócony) i podpisy obrazów (tytuł + podpis) - w nich często są daty/autorzy dzieł.
Wynik: chunks/part-XX.parquet {cid, title, section, text, kind}; kind = text|infobox|caption"""
import argparse, glob, json, os, re
from multiprocessing import Pool
import pyarrow as pa, pyarrow.parquet as pq
SENT = re.compile(r"(?<=[.!?…])\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ„\"(\d])")

def split_text(text, target, hard):
    out = []
    for para in [p for p in text.split("\n") if p.strip()]:
        if len(para) <= hard: out.append(para); continue
        cur = ""
        for s in SENT.split(para):
            if len(cur) + len(s) + 1 > hard and cur: out.append(cur); cur = s
            else: cur = (cur + " " + s).strip()
        if cur: out.append(cur)
    # łączenie krótkich sąsiadów do ~target
    merged, cur = [], ""
    for p in out:
        if p.startswith("## "): 
            if cur: merged.append(cur); cur = ""
            merged.append(p); continue
        if len(cur) + len(p) + 1 <= target or not cur: cur = (cur + "\n" + p).strip()
        else: merged.append(cur); cur = p
    if cur: merged.append(cur)
    return merged

def process(args):
    path, out, target, hard = args
    t = pq.read_table(path).to_pylist(); rows = []
    for r in t:
        title = r["title"]; section = "Wstęp"
        for piece in split_text(r["text"], target, hard):
            if piece.startswith("## "): section = piece[3:].strip(); continue
            if len(piece) < 40: continue
            head = title if section == "Wstęp" else f"{title} > {section}"
            rows.append({"title": title, "section": section, "text": f"{head}\n{piece}", "kind": "text"})
        if r["infobox"] and len(r["infobox"]) > 40:
            rows.append({"title": title, "section": "infobox", "text": f"{title} (infobox)\n{r['infobox'][:1500]}", "kind": "infobox"})
        caps = [im["caption"] for im in json.loads(r["images"]) if im.get("caption") and 15 <= len(im["caption"]) <= 400]
        if caps:
            rows.append({"title": title, "section": "ilustracje", "text": f"{title} (ilustracje)\n" + "\n".join(dict.fromkeys(caps))[:1500], "kind": "caption"})
    if rows:
        pq.write_table(pa.table({k: [x[k] for x in rows] for k in rows[0]}), os.path.join(out, os.path.basename(path)), compression="zstd")
    return len(rows)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--articles", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=512); ap.add_argument("--hard", type=int, default=900); ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(os.path.expanduser(a.articles), "*.parquet")))
    with Pool(a.workers) as p: n = p.map(process, [(f, os.path.expanduser(a.out), a.target, a.hard) for f in files])
    print(f"kawałków: {sum(n)} z {len(files)} plików")
