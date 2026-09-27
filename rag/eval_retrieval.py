#!/usr/bin/env python3
"""Jakość retrievalu bez sędziego: czy top-k fragmentów zawiera terminy z klucza CKE (nazwy własne, daty, liczby).
  python rag/eval_retrieval.py data/rag/e5_k8.jsonl [data/rag/inny.jsonl ...] [--k 5]
Miara: hit@k = odsetek zadań otwartych (bez eseju), dla których >= połowa terminów klucza występuje w którymś z top-k fragmentów."""
import argparse, json, re, sys
ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--k", type=int, default=8); ap.add_argument("--show", type=int, default=0)
a = ap.parse_args()
qs = {json.loads(l)["id"]: json.loads(l) for l in open("data/questions.jsonl")}
STOP = set("Rozstrzygnięcie Uzasadnienie Przykładowe Przykładowa Odpowiedź Odpowiedzi Nazwa Cecha Nazwisko Wyjaśnienie Argument Podobieństwo Różnica Tak Nie Fragment Źródło Zdający Np".split())
def terms(key):
    t = set(re.findall(r"\b[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]{3,}(?:\s[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]{2,})?\b", key)) | set(re.findall(r"\b1\d{3}\b|\b20\d\d\b", key))
    return {x for x in t if x.split()[0] not in STOP}
def score(path, k):
    hits = {json.loads(l)["id"]: json.loads(l) for l in open(path)}
    tot = ok = 0; per = {}
    for i, q in qs.items():
        if i not in hits or q["typ"] != "otwarte": continue
        T = terms(q["klucz"]["rozwiazanie"] or "")
        if len(T) < 2: continue
        blob = " ".join(h["text"] for h in hits[i]["hits"][:k]).lower()
        found = sum(1 for t in T if t.lower() in blob); tot += 1; ok += found >= len(T) / 2; per[i] = (found, len(T))
    return ok, tot, per
for f in a.files:
    for k in (1, 3, 5, a.k):
        ok, tot, per = score(f, k); print(f"{f}: hit@{k} = {ok}/{tot} ({100*ok/tot:.0f}%)")
    if a.show:
        ok, tot, per = score(f, a.k)
        for i, (fo, n) in sorted(per.items())[: a.show]: print(f"   {i}: {fo}/{n} terminów klucza w top-{a.k}")
