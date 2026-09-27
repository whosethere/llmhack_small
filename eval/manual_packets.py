#!/usr/bin/env python3
"""Wypisuje pakiety do ręcznej oceny (dla agenta lub człowieka): zadanie, zasady CKE, wzorzec, opis obrazu, odpowiedź.
  python eval/manual_packets.py --run exam-A-v1pl [--from 1 --to 6] [--ids 5.2 26]
Wypracowanie: zamiast surowego klucza dostajesz rubrykę CKE i liczbę wyrazów."""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval")); import prompts
from run import clean_zasady

def jl(p): return {json.loads(l)["id"]: json.loads(l) for l in open(p, encoding="utf-8")} if Path(p).exists() else {}
ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--from", dest="a", type=int, default=1)
ap.add_argument("--to", dest="b", type=int, default=10**6); ap.add_argument("--ids", nargs="*"); ap.add_argument("--desc", default="results/org-opisy/opisy_2023.jsonl")
a = ap.parse_args()
qs = jl(ROOT / "data/questions.jsonl"); ans = jl(ROOT / "results" / a.run / "answers.jsonl"); desc = jl(ROOT / a.desc)
def order(i): n = i.split("_z")[1].split("."); return (int(n[0]), int(n[1]) if len(n) > 1 else 0)
ids = sorted((i for i in ans if i in qs), key=order)
if a.ids: ids = [i for i in ids if i.split("_z")[1] in a.ids]
for n, i in enumerate(ids, 1):
    if not a.a <= n <= a.b: continue
    q, x = qs[i], ans[i]; k = q["klucz"] or {}; essay = q["typ"] == "wypracowanie"
    print(f"\n{'='*100}\n[{n}/{len(ids)}] {i}  typ={q['typ']}  maks={q['punkty_max']} pkt")
    if q.get("kontekst"): print(f"--- KONTEKST (wspólne źródła) ---\n{q['kontekst']}")
    print(f"--- TREŚĆ ZADANIA ---\n{q['tresc']}")
    if i in desc: print(f"--- OPIS ŹRÓDŁA GRAFICZNEGO (od organizatorów) ---\n{desc[i]['opis']}")
    elif q.get("wymaga_obrazu"): print(f"--- OBRAZ: brak opisu; obejrzyj plik(i): {', '.join(q.get('obrazy', []))} ---")
    if essay:
        print(f"--- KRYTERIA (rubryka CKE) ---\n{prompts.ESSAY_RUBRIC_CKE}")
    else:
        print(f"--- ZASADY OCENIANIA (CKE) ---\n{clean_zasady(k.get('zasady_oceniania') or k.get('klucz_raw',''))}")
        print(f"--- ODPOWIEDŹ WZORCOWA / PRZYKŁADOWA (CKE) ---\n{k.get('rozwiazanie') or '(brak w kluczu)'}")
    odp = x.get("odpowiedz") or "(BRAK ODPOWIEDZI)"
    print(f"--- ODPOWIEDŹ ZDAJĄCEGO{f' ({len(odp.split())} wyrazów)' if essay else ''} ---\n{odp}")
