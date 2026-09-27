#!/usr/bin/env python3
"""Dopisuje ręczną ocenę zadania, którego sędzia API nie ocenił (np. filtr treści Azure).
python eval/manual_fill.py <run> <judge> <id> <punkty> "<uzasadnienie>" """
import json, sys
from pathlib import Path
run, judge, qid, pts, why = sys.argv[1:6]
q = next(json.loads(l) for l in open("data/questions.jsonl") if json.loads(l)["id"] == qid)
rec = {"id": qid, "punkty": int(pts), "punkty_max": q["punkty_max"], "uzasadnienie": f"[ręcznie, Claude: sędzia API odrzucił zadanie] {why}"}
with open(Path("results") / run / f"judgments_{judge}.jsonl", "a") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
ef = Path("results") / run / f"judge_errors_{judge}.jsonl"
if ef.exists():
    rest = [l for l in open(ef) if json.loads(l)["id"] != qid]
    ef.write_text("".join(rest)) if rest else ef.unlink()
print("ok", rec)
