"""OCR ilustracji z paczki egzaminu jako krok przed VLM: odczytane napisy trafiają do promptu opisywacza do weryfikacji.
Silniki: tesseract (CLI, pol+eng) albo kraken (CLI, model domyślny "basic"). Działa offline. Wynik cache'owany po sha256.

  python exam/ocr.py --engine tesseract --exam submission/history-2023-mock-v1        # wypisz OCR wszystkich obrazów
"""
import argparse, hashlib, json, os, re, subprocess, sys, tempfile, threading
from pathlib import Path

KRAKEN_BIN = os.environ.get("KRAKEN_BIN", "/scratch/kraken/bin/kraken")
EASYOCR_PY = os.environ.get("EASYOCR_PY", "/scratch/ocr/bin/python")  # osobny venv z easyocr (exam/ocr_easy.py)
KRAKEN_MODEL = os.environ.get("KRAKEN_MODEL", "")  # pusty = pierwszy pobrany model (kraken get 10.5281/zenodo.2577813 -> en_best.mlmodel)


def kraken_model():
    if KRAKEN_MODEL: return KRAKEN_MODEL
    import glob
    found = glob.glob(os.path.expanduser("~/.local/share/htrmopo/*/*.mlmodel")) + glob.glob(os.path.expanduser("~/.config/kraken/*.mlmodel"))
    if not found: raise RuntimeError("brak modelu krakena: uruchom `kraken get 10.5281/zenodo.2577813` albo ustaw KRAKEN_MODEL")
    return found[0]


def run_tesseract(png_path):
    """tesseract <png> stdout -l pol+eng --psm 11 (sparse text: napisy rozrzucone po mapie/ilustracji)."""
    out = subprocess.run(["tesseract", str(png_path), "stdout", "-l", "pol+eng", "--psm", "11"],
                         capture_output=True, text=True, timeout=300)
    if out.returncode != 0:
        raise RuntimeError(f"tesseract: {out.stderr[:200]}")
    return out.stdout


def run_kraken(png_path):
    """kraken -i <png> <txt> binarize segment ocr [-m model]; wersja podstawowa, bez własnych modeli."""
    with tempfile.TemporaryDirectory() as td:
        txt = Path(td) / "out.txt"
        cmd = [KRAKEN_BIN, "-i", str(png_path), str(txt), "binarize", "segment", "ocr", "-m", kraken_model()]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if out.returncode != 0 or not txt.exists():
            raise RuntimeError(f"kraken: {(out.stderr or out.stdout)[-300:]}")
        return txt.read_text(encoding="utf-8", errors="replace")


def run_easyocr_batch(paths):
    """Jeden proces EasyOCR dla wielu plików (model ładuje się ~10 s): -> {ścieżka: tekst}."""
    cmd = [EASYOCR_PY, str(Path(__file__).parent / "ocr_easy.py"), *map(str, paths)]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if out.returncode != 0:
        raise RuntimeError(f"easyocr: {out.stderr[-300:]}")
    res = {}
    for l in out.stdout.splitlines():
        if l.startswith("{"):
            r = json.loads(l); res[r["path"]] = r["text"]
    return res


def run_easyocr(png_path):
    return run_easyocr_batch([png_path]).get(str(png_path), "")


ENGINES = {"tesseract": run_tesseract, "kraken": run_kraken, "easyocr": run_easyocr}


def clean(text, min_len=2):
    """Zostawia linie z co najmniej jednym słowem 2+ liter/cyfr; usuwa śmieci OCR (pojedyncze znaki, ramki).
    Zwraca '' gdy obraz nie zawiera czytelnego tekstu (zdjęcia, obrazy): za mało sensownych wyrazów."""
    lines = []
    for l in text.splitlines():
        l = re.sub(r"[|_~^`]+", " ", l).strip()
        if re.search(r"[A-Za-zÀ-ž0-9]{%d,}" % min_len, l):
            lines.append(re.sub(r"\s{2,}", " ", l))
    out = "\n".join(dict.fromkeys(lines))
    # sensowne wyrazy: 4+ liter bez wielkich w środku (OCR-owe „śmieci” to krótkie mieszane ciągi) albo liczby 4-cyfrowe (daty)
    good = re.findall(r"\b(?:[A-ZŁŚŻŹĆŃÓĘĄ]?[a-złśżźćńóęą]{3,}|[A-ZŁŚŻŹĆŃÓĘĄ]{4,}|\d{4})\b", out)
    return out if len(good) >= 4 else ""


def upscale_png(data, scale):
    """Powiększenie Lanczos + skala szarości + autokontrast przed OCR (tesseract lepiej czyta napisy >= ~30 px wysokości)."""
    if scale <= 1.01: return data
    import io
    from PIL import Image, ImageOps
    im = Image.open(io.BytesIO(data)).convert("L")
    im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    im = ImageOps.autocontrast(im, cutoff=1)
    buf = io.BytesIO(); im.save(buf, "PNG", compress_level=1)
    return buf.getvalue()


class OCR:
    def __init__(self, engine, cache_dir=None, scale=1.0):
        if engine not in ENGINES: raise ValueError(f"nieznany silnik OCR: {engine}")
        self.engine, self.cache, self.scale = engine, {}, float(scale)
        self.lock = threading.Lock()  # jeden proces OCR naraz: równoległe tesseracty (każdy wielowątkowy) dławią CPU
        tag = f"_x{self.scale:g}" if self.scale > 1.01 else ""
        self.cache_file = Path(cache_dir) / f"ocr_{engine}{tag}.jsonl" if cache_dir else None
        if self.cache_file and self.cache_file.exists():
            for l in open(self.cache_file, encoding="utf-8"):
                r = json.loads(l); self.cache[r["sha256"]] = r["text"]

    def prefetch(self, items):
        """items: [(bajty PNG, sha256)] -> OCR wsadowo (easyocr: jeden proces), wynik do cache. Inne silniki: po kolei."""
        todo = [(d, h or hashlib.sha256(d).hexdigest()) for d, h in items]
        todo = [(d, h) for d, h in todo if h not in self.cache]
        if not todo: return
        if self.engine != "easyocr":
            for d, h in todo: self.read_bytes(d, h)
            return
        with self.lock, tempfile.TemporaryDirectory() as td:
            paths = []
            for d, h in todo:
                p = Path(td) / f"{h}.png"; p.write_bytes(upscale_png(d, self.scale)); paths.append(p)
            res = run_easyocr_batch(paths)
            for d, h in todo:
                self._store(h, clean(res.get(str(Path(td) / f"{h}.png"), "")))

    def _store(self, sha256, text):
        self.cache[sha256] = text
        if self.cache_file:
            with open(self.cache_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({"sha256": sha256, "text": text}, ensure_ascii=False) + "\n")

    def read_bytes(self, data, sha256=None):
        """OCR dla surowych bajtów PNG (z żądania /solve); -> oczyszczony tekst ('' gdy nic nie odczytano)."""
        sha256 = sha256 or hashlib.sha256(data).hexdigest()
        with self.lock:
            if sha256 in self.cache: return self.cache[sha256]
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                f.write(upscale_png(data, self.scale)); path = f.name
            try:
                text = clean(ENGINES[self.engine](path))
            finally:
                os.unlink(path)
            self._store(sha256, text)
            return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=list(ENGINES), required=True); ap.add_argument("--exam", required=True)
    ap.add_argument("--out", help="zapis {path, sha256, text} jako JSONL")
    ap.add_argument("--scale", type=float, default=1.0, help="powiększenie obrazu przed OCR, np. 2")
    ap.add_argument("--cache-dir", help="zapis do cache (ten sam katalog co --ocr-cache solvera): wstępny OCR całej paczki przed egzaminem")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).parent)); import common as C
    exam, root, _ = C.load_exam(a.exam)
    if a.cache_dir: Path(a.cache_dir).mkdir(parents=True, exist_ok=True)
    ocr = OCR(a.engine, cache_dir=a.cache_dir, scale=a.scale)
    ocr.prefetch([((root / im["path"]).read_bytes(), im["sha256"]) for it in exam["items"] for im in it["images"]])
    seen, rows = set(), []
    for it in exam["items"]:
        for im in it["images"]:
            if im["sha256"] in seen: continue
            seen.add(im["sha256"])
            text = ocr.read_bytes((root / im["path"]).read_bytes(), im["sha256"])
            rows.append({"path": im["path"], "sha256": im["sha256"], "text": text})
            print(f"===== {im['path']} ({len(text)} zn.)\n{text[:600]}")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
