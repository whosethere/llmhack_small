#!/usr/bin/env python3
"""Walidacja answers.json przed wysłaniem: reguły strony zgłoszeń (błędy) + nasze kontrole jakości (ostrzeżenia).

  python exam/validate.py --exam submission/history-2023-mock-v1 results/exam/NAZWA/answers.json
Kod wyjścia 1, gdy plik zostałby odrzucony.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exam", required=True)
    ap.add_argument("answers")
    a = ap.parse_args()
    exam, _, template_ids = C.load_exam(a.exam)
    raw = Path(a.answers).read_bytes()
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        sys.exit(f"BŁĄD: to nie jest poprawny JSON w UTF-8: {e}")
    errors, warnings = C.validate_answers(obj, exam, template_ids)
    for w in warnings: print(f"UWAGA {w}")
    for e in errors: print(f"BŁĄD {e}")
    print(f"{a.answers}: {len(raw)} B, błędy: {len(errors)}, ostrzeżenia: {len(warnings)} -> "
          f"{'do poprawy' if errors else 'format OK'}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
