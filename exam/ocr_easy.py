#!/usr/bin/env python3
"""EasyOCR (pl+en, GPU) dla listy plików obrazów; uruchamiany w osobnym środowisku (EASYOCR_PY), żeby nie ruszać venv z vLLM.
  /scratch/ocr/bin/python exam/ocr_easy.py <plik.png> [<plik2.png> ...]   -> JSONL {"path", "text"} na stdout
Tekst: linie w kolejności czytania (góra->dół, lewo->prawo), tylko rozpoznania z pewnością >= 0.3."""
import json, sys

def main():
    import easyocr
    reader = easyocr.Reader(["pl", "en"], gpu=True, verbose=False)
    for path in sys.argv[1:]:
        try:
            res = reader.readtext(path, detail=1, paragraph=False)
            rows = [(min(p[1] for p in box), min(p[0] for p in box), txt) for box, txt, conf in res if conf >= 0.3]
            rows.sort(key=lambda r: (round(r[0] / 25), r[1]))  # wiersze co ~25 px, w wierszu od lewej
            text = "\n".join(r[2] for r in rows)
        except Exception as e:
            text = ""; sys.stderr.write(f"easyocr {path}: {e}\n")
        print(json.dumps({"path": path, "text": text}, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
