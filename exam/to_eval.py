#!/usr/bin/env python3
"""answers.json z egzaminu -> run w naszym formacie (results/<run>/answers.jsonl), żeby ocenić go lokalnie
naszym sędzią (eval/run.py --phase judge). Działa tylko dla arkuszy, do których mamy klucz CKE w data/questions.jsonl.

  python exam/to_eval.py --exam submission/history-2023-mock-v1 --answers results/exam/NAZWA/answers.json --run exam-NAZWA
  python eval/run.py --run exam-NAZWA --phase judge --years 2023_f2023
"""
import argparse, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import common as C

ROOT = Path(__file__).resolve().parent.parent
KNOWN = {"MHIP-R0-100-2305": "2023_f2023", "MHIP-R0-100-2405": "2024_f2023", "MHIP-R0-100-2505": "2025_f2023", "MHIP-R0-100-2605": "2026_f2023"}  # source_exam_id paczki -> rocznik w data/questions.jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exam", required=True); ap.add_argument("--answers", required=True); ap.add_argument("--run", required=True)
    a = ap.parse_args()
    exam, _, _ = C.load_exam(a.exam)
    src = exam.get("source_exam_id") or ""
    year = KNOWN.get(src) or (src if re.fullmatch(r"\d{4}_f\d{4}", src) else None)  # paczki z scripts/cke_to_exam.py
    if not year:
        sys.exit(f"brak klucza dla arkusza {exam.get('source_exam_id')!r}: lokalna ocena niemożliwa")
    ans = json.loads(Path(a.answers).read_text(encoding="utf-8"))
    out = ROOT / "results" / a.run; out.mkdir(parents=True, exist_ok=True)
    with open(out / "answers.jsonl", "w", encoding="utf-8") as f:
        for x in ans["answers"]:
            f.write(json.dumps({"id": f"{year}_z{x['id']}", "odpowiedz": x["answer"], "sek": None}, ensure_ascii=False) + "\n")
    json.dump({"run": a.run, "source": str(a.answers), "exam_id": exam["exam_id"], "note": "odpowiedzi z harnessu egzaminu"},
              open(out / "config.json", "w"), indent=2, ensure_ascii=False)
    print(f"{len(ans['answers'])} odpowiedzi -> {out}/answers.jsonl; ocena: python eval/run.py --run {a.run} --phase judge --years {year}")


if __name__ == "__main__":
    main()
