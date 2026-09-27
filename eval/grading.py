#!/usr/bin/env python3
"""Ocenianie odpowiedzi przez zewnętrznych sędziów (agentów, ludzi, inne modele) - wspólny format.

Pakiet do oceny (jedno zadanie, samowystarczalny, BEZ nazwy modelu i ocen innych sędziów):
  {"id", "typ", "punkty_max", "zadanie", "zadanie_z_obrazem", "zasady_oceniania",
   "rozwiazanie_wzorcowe", "odpowiedz_zdajacego", "liczba_wyrazow"}
Ocena (to zwraca sędzia):
  {"id": "...", "punkty": <int 0..punkty_max>, "uzasadnienie": "<1-2 zdania po polsku>"}

CLI:
  python eval/grading.py status [--judge NAZWA]
  python eval/grading.py export --run RUN --judge NAZWA [--years 2025_f2023 2026_f2023] [--all] --out plik.jsonl
  python eval/grading.py import --run RUN --judge NAZWA plik.jsonl
  python eval/grading.py instructions            # reguły sędziego (do wklejenia agentowi)
Oceny trafiają do results/RUN/judgments_NAZWA.jsonl (widoczne w viewer/ i eval/agreement.py).
"""
import argparse, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
import prompts

JUDGE_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{1,39}$")
RESERVED = {"gpt-6-sol", "gpt-6-luna", "qwen3-32b-v1", "claude-manual"}  # zapisywane przez nasze skrypty


def _jsonl(p):
    out = {}
    if Path(p).exists():
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r = json.loads(line); out[r["id"]] = r
    return out


def questions():
    return _jsonl(ROOT / "data/questions.jsonl")


def runs():
    return sorted(d.name for d in (ROOT / "results").iterdir() if (d / "answers.jsonl").exists())


def check_judge_name(name, allow_reserved=False):
    if not JUDGE_NAME.match(name or ""):
        raise ValueError("nazwa sędziego: 2-40 znaków, małe litery, cyfry, . _ - (np. opus-4-agent)")
    if name in RESERVED and not allow_reserved:
        raise ValueError(f"nazwa '{name}' jest zarezerwowana dla naszych skryptów, wybierz inną")


def packet(q, a):
    k = q.get("klucz") or {}
    zad = ((q["kontekst"] + "\n\n") if q.get("kontekst") else "") + q["tresc"]
    odp = a.get("odpowiedz", "")
    return {"id": q["id"], "typ": q["typ"], "punkty_max": q["punkty_max"], "zadanie": zad,
            "zadanie_z_obrazem": bool(q.get("wymaga_obrazu")),
            "zasady_oceniania": k.get("zasady_oceniania") or k.get("klucz_raw", ""),
            "rozwiazanie_wzorcowe": k.get("rozwiazanie") or "(brak w kluczu - oceń według zasad)",
            "odpowiedz_zdajacego": odp or "(brak odpowiedzi)", "liczba_wyrazow": len(odp.split())}


def tasks(run, judge, years=None, only_missing=True, limit=None):
    """Pakiety do oceny dla runu. only_missing: pomija zadania już ocenione przez tego sędziego."""
    qs = questions()
    ans = _jsonl(ROOT / "results" / run / "answers.jsonl")
    done = _jsonl(ROOT / "results" / run / f"judgments_{judge}.jsonl") if judge else {}
    ids = [i for i in ans if i in qs and qs[i].get("klucz") and (not years or i.split("_z")[0] in years)]
    todo = [i for i in ids if not (only_missing and i in done)]
    def order(i):
        n = i.split("_z")[1].split(".")
        return (i.split("_z")[0], int(n[0]), int(n[1]) if len(n) > 1 else 0)
    todo.sort(key=order)
    return [packet(qs[i], ans[i]) for i in (todo[:limit] if limit else todo)], len(todo), len(ids)


def save(run, judge, grades, allow_reserved=False):
    """Waliduje i dopisuje oceny. Zwraca (zapisane, błędy). Ponowna ocena tego samego id nadpisuje poprzednią."""
    check_judge_name(judge, allow_reserved)
    if run not in runs():
        raise ValueError(f"nieznany run '{run}'; dostępne: {', '.join(runs())}")
    qs = questions()
    ans = _jsonl(ROOT / "results" / run / "answers.jsonl")
    ok, errors = [], []
    for g in grades:
        gid = g.get("id")
        if gid not in ans or gid not in qs:
            errors.append({"id": gid, "error": "brak takiego zadania w tym runie"}); continue
        mx = qs[gid]["punkty_max"]
        p = g.get("punkty")
        if isinstance(p, bool) or not isinstance(p, int) or not 0 <= p <= mx:
            errors.append({"id": gid, "error": f"punkty muszą być liczbą całkowitą 0..{mx}"}); continue
        why = str(g.get("uzasadnienie", "")).strip()
        if not why:
            errors.append({"id": gid, "error": "brak uzasadnienia"}); continue
        ok.append({"id": gid, "punkty": p, "punkty_max": mx, "uzasadnienie": why[:2000]})
    if ok:
        with open(ROOT / "results" / run / f"judgments_{judge}.jsonl", "a", encoding="utf-8") as f:
            for r in ok:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(ok), errors


def status(judge=None):
    qs = questions()
    out = []
    for r in runs():
        ans = _jsonl(ROOT / "results" / r / "answers.jsonl")
        n = sum(1 for i in ans if i in qs and qs[i].get("klucz"))
        judges = {f.stem[len("judgments_"):]: len(_jsonl(f)) for f in (ROOT / "results" / r).glob("judgments_*.jsonl")}
        out.append({"run": r, "zadan": n, "ocenione": {j: c for j, c in judges.items() if not judge or j == judge}})
    return out


def instructions():
    return prompts.JUDGE_SYSTEM


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("status"); s.add_argument("--judge")
    e = sp.add_parser("export"); e.add_argument("--run", required=True); e.add_argument("--judge", required=True)
    e.add_argument("--years", nargs="*"); e.add_argument("--all", action="store_true", help="także już ocenione"); e.add_argument("--out", required=True)
    i = sp.add_parser("import"); i.add_argument("--run", required=True); i.add_argument("--judge", required=True); i.add_argument("file")
    sp.add_parser("instructions")
    a = ap.parse_args()
    if a.cmd == "status":
        for r in status(a.judge): print(f"{r['run']:24s} zadań {r['zadan']:4d}  ocenione: {r['ocenione']}")
    elif a.cmd == "export":
        check_judge_name(a.judge, allow_reserved=True)
        pk, left, total = tasks(a.run, a.judge, a.years, only_missing=not a.all)
        with open(a.out, "w", encoding="utf-8") as f:
            for p in pk: f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"{len(pk)} pakietów -> {a.out} (run {a.run}: {total} zadań, do oceny przez {a.judge}: {left})")
    elif a.cmd == "import":
        grades = [json.loads(l) for l in open(a.file, encoding="utf-8") if l.strip()]
        n, errs = save(a.run, a.judge, grades)
        print(f"zapisano {n} ocen do results/{a.run}/judgments_{a.judge}.jsonl")
        for e in errs: print("  BŁĄD", e)
        sys.exit(1 if errs else 0)
    elif a.cmd == "instructions":
        print(instructions())


if __name__ == "__main__":
    main()
