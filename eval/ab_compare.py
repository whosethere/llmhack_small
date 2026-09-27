"""Porównanie runów egzaminu (format organizatorów) per zadanie po ocenie sędzią.
  python eval/ab_compare.py --base exam-a100-gemma4-thinkon-ocr-b3000-rag exam-a100-gemma4-final-dryrun --test exam-ab-scale-1 exam-ab-scale-2 [--judge gpt-6-sol-v4-opis]
Wypisuje sumy na wspólnym zbiorze zadań (tylko te, które są we wszystkich runach testowych) i zadania, gdzie średnie się różnią."""
import argparse, json
from pathlib import Path

def load(run, judge):
    p = Path("results") / run / f"judgments_{judge}.jsonl"
    return {json.loads(l)["id"]: json.loads(l) for l in open(p, encoding="utf-8")}

def key(i): return [float(x) for x in i.split("_z")[1].split(".")]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="+", required=True); ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--judge", default="gpt-6-sol-v4-opis")
    ap.add_argument("--only", nargs="*", help="tylko te zadania (krótkie id, np. 1 14.1)")
    a = ap.parse_args()
    B = [load(r, a.judge) for r in a.base]; T = [load(r, a.judge) for r in a.test]
    ids = sorted(set.intersection(*(set(t) for t in T)), key=key)
    if a.only: ids = [i for i in ids if i.split("_z")[1] in set(a.only)]
    mx = sum(T[0][i]["punkty_max"] for i in ids)
    def tot(d): return sum(d[i]["punkty"] for i in ids if i in d)
    print(f"zadań: {len(ids)}, max {mx}")
    for r, d in zip(a.base, B): print(f"  base {r:45} {tot(d):3}")
    for r, d in zip(a.test, T): print(f"  test {r:45} {tot(d):3}")
    mb = sum(tot(d) for d in B) / len(B); mt = sum(tot(d) for d in T) / len(T)
    print(f"średnia base {mb:.1f} vs test {mt:.1f} (Δ {mt-mb:+.1f})")
    print("zadanie | base... | test... (max)")
    for i in ids:
        b = [d[i]["punkty"] for d in B if i in d]; t = [d[i]["punkty"] for d in T]
        if b and sum(b)/len(b) != sum(t)/len(t):
            print(f"  {i.split('_z')[1]:5} | {' '.join(map(str,b)):7} | {' '.join(map(str,t)):7} ({T[0][i]['punkty_max']})")

if __name__ == "__main__":
    main()
