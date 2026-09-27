#!/usr/bin/env python3
"""Porównanie sędziów na runach: python eval/compare_judges.py --runs exam-A-v1pl exam-B-v2pl --judges claude-manual gpt-6-sol-v4-opis <nowy>
Pierwszy sędzia jest wzorcem (zgodność ocen liczona względem niego)."""
import argparse, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
def jl(p): return {json.loads(l)["id"]: json.loads(l) for l in open(p, encoding="utf-8")} if Path(p).exists() else {}
ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--judges", nargs="+", required=True); a = ap.parse_args()
ref = a.judges[0]
print("| run | " + " | ".join(a.judges) + " | esej (" + "/".join(a.judges) + ") | zgodne z " + ref + " |"); print("|---" * (len(a.judges) + 3) + "|")
tot = {j: [0, 0] for j in a.judges[1:]}
for r in a.runs:
    d = {j: jl(ROOT / "results" / r / f"judgments_{j}.jsonl") for j in a.judges}
    ids = [i for i in d[ref] if all(i in d[j] for j in a.judges)]
    if not ids: print(f"| {r} | brak wspólnych ocen |"); continue
    mx = sum(d[ref][i]["punkty_max"] for i in ids); es = [i for i in ids if d[ref][i]["punkty_max"] >= 10]
    sums = [f"{sum(d[j][i]['punkty'] for i in ids)}/{mx}" for j in a.judges]
    essay = "/".join(str(d[j][es[0]]["punkty"]) if es else "-" for j in a.judges)
    agree = []
    for j in a.judges[1:]:
        ag = sum(d[j][i]["punkty"] == d[ref][i]["punkty"] for i in ids); tot[j][0] += ag; tot[j][1] += len(ids); agree.append(f"{j}: {ag}/{len(ids)}")
    print(f"| {r} | " + " | ".join(sums) + f" | {essay} | " + "; ".join(agree) + " |")
print("razem: " + "; ".join(f"{j} {v[0]}/{v[1]} ({100*v[0]/v[1]:.0f}%)" for j, v in tot.items() if v[1]))
