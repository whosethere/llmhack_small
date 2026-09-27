#!/usr/bin/env python3
"""Import przebiegów organizatorów (benchmark CKE 2023, strona z metodologią) jako naszych runów do kalibracji sędziego.

Dla każdego ich przebiegu tworzy results/org-<run>/:
  answers.jsonl          ich odpowiedź końcowa (final_answer; oceniana jest tylko ona), id w naszym formacie 2023_f2023_zX
  judgments_org.jsonl    ich ocena (points, reason) jako sędzia "org"
  config.json            model, tryb wejścia, źródło
Brakujące pliki pobiera ze strony do --src (domyślnie prezentacje/site, poza gitem). Dane są cudze: results/org-* jest w .gitignore.

  python eval/import_org.py [--src DIR] [--with-pilots]
"""
import argparse, json, re, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://warsaw-matura-method.ania-olchowik.chatgpt.site"

def fetch(url, dest):
    if dest.exists(): return True
    try:
        data = urllib.request.urlopen(url, timeout=60).read()
    except Exception:
        return False
    dest.write_bytes(data); return True

def descriptions(src, out):
    """Stałe opisy obrazów organizatorów z ich przebiegu tekstowego: bloki „[Opis źródła …]” między nagłówkiem a poleceniem.
    -> results/org-opisy/opisy_2023.jsonl {id, opis}; używane przez sędziego (eval/calibrate.py --desc)."""
    g = json.load(open(src / "ministral-3-14b-text.json"))
    exam = {i["id"]: i["question"] for i in json.load(open(src / "astra-images.exam.json"))["items"]}
    rows = []
    for x in g["answers"]:
        u = x["prompt"]
        head = re.match(r"Zadanie [\d.]+ \(\d+ pkt\)\n\n", u)
        body = u[head.end():] if head else u
        q = exam[x["id"]].strip()
        if body.rstrip().endswith(q): body = body.rstrip()[: -len(q)]
        blocks, cur = [], None
        for line in body.split("\n"):
            if line.startswith("[Opis źródła"):
                cur = [line]; blocks.append(cur)
            elif cur is not None and re.match(r"(Źródło|Fragment|Tekst) \d", line):
                cur = None
            elif cur is not None:
                cur.append(line)
        if blocks:
            rows.append({"id": f"2023_f2023_z{x['id']}", "opis": "\n\n".join("\n".join(b).strip() for b in blocks)})
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"opisy obrazów: {len(rows)} zadań -> {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "prezentacje/site"))
    ap.add_argument("--with-pilots", action="store_true", help="dołącz 5-zadaniowe pilotaże budżetu (budget-pilot-*)")
    ap.add_argument("--descriptions", action="store_true", help="tylko wytnij stałe opisy obrazów 2023 do results/org-opisy/")
    a = ap.parse_args()
    if a.descriptions:
        return descriptions(Path(a.src), ROOT / "results/org-opisy/opisy_2023.jsonl")
    src = Path(a.src); src.mkdir(parents=True, exist_ok=True)
    fetch(f"{BASE}/index.html", src / "index.html")
    pages = sorted(set(re.findall(r'href="([a-z0-9-]+)\.html"', (src / "index.html").read_text(encoding="utf-8"))))
    for p in pages:  # strona przebiegu linkuje <run>.json i <run>.reviews.json
        if fetch(f"{BASE}/{p}.html", src / f"{p}.html"):
            for j in set(re.findall(r'href="([a-z0-9_-]+(?:\.reviews)?\.json)"', (src / f"{p}.html").read_text(encoding="utf-8"))):
                fetch(f"{BASE}/{j}", src / j)
    qmax = {q["id"]: q["punkty_max"] for q in map(json.loads, open(ROOT / "data/questions.jsonl"))}
    n_runs = 0
    for rf in sorted(src.glob("*.reviews.json")):
        name = rf.name[: -len(".reviews.json")]
        if name.startswith("budget-pilot") and not a.with_pilots: continue
        g = json.load(open(src / f"{name}.json")); reviews = json.load(open(rf))
        ans = {x["id"]: x for x in g["answers"]}
        out = ROOT / "results" / f"org-{name}"; out.mkdir(parents=True, exist_ok=True)
        A, J, pts, mx = [], [], 0, 0
        for r in reviews:
            x = ans[r["id"]]; qid = f"2023_f2023_z{r['id']}"
            assert qid in qmax, qid
            text = x["final_answer"] if "final_answer" in x else x.get("answer", "")
            A.append({"id": qid, "odpowiedz": (text or "").strip(), "sek": x.get("generation_seconds"), "org_id": r["id"]})
            J.append({"id": qid, "punkty": r["points"], "punkty_max": qmax[qid], "uzasadnienie": r["reason"]})
            pts += r["points"]; mx += qmax[qid]
        for fn, rows in (("answers.jsonl", A), ("judgments_org.jsonl", J)):
            with open(out / fn, "w", encoding="utf-8") as f:
                for row in rows: f.write(json.dumps(row, ensure_ascii=False) + "\n")
        json.dump({"run": out.name, "model_id": g.get("model_id") or g.get("model"), "source": f"{BASE}/{name}.json",
                   "input_mode": g.get("input_modality") or g.get("input_mode") or g.get("scope"),
                   "org_points": pts, "max_points": mx, "note": "odpowiedzi i oceny organizatorów (benchmark CKE 2023)"},
                  open(out / "config.json", "w"), indent=2, ensure_ascii=False)
        n_runs += 1
        print(f"  {out.name:40s} {len(A):2d} zadań  org: {pts}/{mx}")
    print(f"zaimportowano {n_runs} przebiegów do results/org-*")

if __name__ == "__main__":
    main()
