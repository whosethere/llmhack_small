#!/usr/bin/env python3
"""Zgodność sędziów z oceną referencyjną (domyślnie claude-manual). python eval/agreement.py <run> [ref]"""
import json, sys
from pathlib import Path
run = sys.argv[1]; ref = sys.argv[2] if len(sys.argv) > 2 else "claude-manual"
d = Path("results") / run
load = lambda p: {json.loads(l)["id"]: json.loads(l) for l in open(p)}
R = load(d / f"judgments_{ref}.jsonl")
print(f"| sędzia | suma sędzia | suma {ref} | max | zgodne zadania | różnica bezwzgl. | różnica bez eseju |")
print("|---|---|---|---|---|---|---|")
for f in sorted(d.glob("judgments_*.jsonl")):
    name = f.stem[len("judgments_"):]
    if name == ref: continue
    J = load(f); ids = [i for i in R if i in J]
    if not ids: continue
    agree = sum(J[i]["punkty"] == R[i]["punkty"] for i in ids)
    mad = sum(abs(J[i]["punkty"] - R[i]["punkty"]) for i in ids)
    mad_ne = sum(abs(J[i]["punkty"] - R[i]["punkty"]) for i in ids if R[i]["punkty_max"] < 10)
    print(f"| {name} | {sum(J[i]['punkty'] for i in ids)} | {sum(R[i]['punkty'] for i in ids)} | {sum(R[i]['punkty_max'] for i in ids)} | {agree}/{len(ids)} | {mad} | {mad_ne} |")
if len(sys.argv) > 3:
    J = load(d / f"judgments_{sys.argv[3]}.jsonl")
    for i in R:
        if i in J and J[i]["punkty"] != R[i]["punkty"]:
            print(f"\n{i}: {sys.argv[3]}={J[i]['punkty']} {ref}={R[i]['punkty']} /{R[i]['punkty_max']}\n  {sys.argv[3]}: {J[i]['uzasadnienie'][:260]}\n  {ref}: {R[i]['uzasadnienie'][:260]}")
