#!/usr/bin/env python3
"""Porównanie kilku runów: python eval/report.py run1 run2 ..."""
import json, sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
qs = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data/questions.jsonl")}
runs = sys.argv[1:] or [p.name for p in (ROOT / "results").iterdir() if (p / "judgments.jsonl").exists()]
table = defaultdict(dict)
for run in runs:
    got = defaultdict(lambda: [0, 0])
    for l in open(ROOT / "results" / run / "judgments.jsonl"):
        j = json.loads(l); y = j["id"].split("_z")[0]
        got[y][0] += j["punkty"]; got[y][1] += j["punkty_max"]
    for y, (p, m) in got.items(): table[y][run] = 100 * p / m
years = sorted(table)
print("| rocznik | " + " | ".join(runs) + " |"); print("|---" * (len(runs) + 1) + "|")
for y in years: print(f"| {y} | " + " | ".join(f"{table[y].get(r, float('nan')):.0f}%" for r in runs) + " |")
print("| **średnia** | " + " | ".join(f"**{sum(table[y][r] for y in years if r in table[y]) / max(1, sum(r in table[y] for y in years)):.1f}%**" for r in runs) + " |")
