"""Wspólne dla runnera i solvera egzaminu w formacie organizatorów (separate-text-and-images-v1).

Paczka egzaminu: exam.json + images/*.png + answers-template.json (+ README.md).
Tu: wczytanie i kontrola paczki, rodzaje answer_format, sprowadzanie odpowiedzi zamkniętych do wymaganej
składni, walidacja answers.json (reguły ze strony zgłoszeń + nasze ostrzeżenia).
Tylko biblioteka standardowa: działa offline na każdej maszynie.
"""
import base64, hashlib, json, re, sys
from pathlib import Path
from urllib.parse import urlparse

MAX_FILE_BYTES = 1024 * 1024     # answers.json: maks. 1 MiB (UTF-8)
MAX_ANSWER_CHARS = 100_000       # maks. długość jednej odpowiedzi
IMG_MARK = re.compile(r"\[Obraz: ([^\]]+)\]")
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def ensure_local(url, allow_remote=False):
    """Egzamin idzie bez internetu: runner i solver rozmawiają tylko z localhost (chyba że --allow-remote)."""
    host = urlparse(url).hostname or ""
    if not allow_remote and host not in LOCAL_HOSTS:
        raise SystemExit(f"adres {url} nie jest lokalny; egzamin działa offline (użyj --allow-remote tylko do testów)")


def load_exam(path, strict=False):
    """Katalog paczki albo ścieżka do exam.json -> (exam, root, template_ids).
    Domyślnie łagodnie: brakujący obraz lub zła suma sha256 to ostrzeżenie (obraz pomijany / używany mimo sumy),
    szablon odpowiedzi ma pierwszeństwo przy ustalaniu listy id. strict=True przerywa jak dawniej (do kontroli paczki)."""
    p = Path(path)
    root = p if p.is_dir() else p.parent
    exam = json.loads((root / "exam.json").read_text(encoding="utf-8"))
    errors, warnings = [], []
    items = exam.get("items") or []
    for it in items:  # brakujące pola nie mogą wywrócić egzaminu
        it.setdefault("id", str(it.get("id", "")))
        it["id"] = str(it["id"])
        it.setdefault("group", it["id"].split(".")[0])
        it.setdefault("max_points", 1); it.setdefault("question", ""); it.setdefault("source_text", "")
        it.setdefault("answer_format", ""); it["images"] = it.get("images") or []
    ids = [it["id"] for it in items]
    if len(set(ids)) != len(ids): errors.append("powtórzone id w exam.json")
    for it in items:
        keep = []
        for im in it["images"]:
            f = root / (im.get("path") or "")
            if not im.get("path") or not f.exists():
                warnings.append(f"{it['id']}: brak pliku {im.get('path')!r} (zadanie idzie bez tego obrazu)"); continue
            digest = hashlib.sha256(f.read_bytes()).hexdigest()
            if im.get("sha256") and digest != im["sha256"]:
                warnings.append(f"{it['id']}: suma sha256 nie zgadza się dla {im['path']} (obraz użyty mimo to)")
            im["sha256"] = digest
            keep.append(im)
        it["images"] = keep
    template_ids = ids
    tpl = root / "answers-template.json"
    if tpl.exists():
        t = json.loads(tpl.read_text(encoding="utf-8"))
        template_ids = [str(a["id"]) for a in t.get("answers", [])] or ids
        if t.get("exam_id") and t.get("exam_id") != exam.get("exam_id"):
            warnings.append(f"exam_id w answers-template.json ({t.get('exam_id')!r}) różny od exam.json ({exam.get('exam_id')!r}); używam szablonu")
            exam["exam_id"] = t["exam_id"]
        if sorted(template_ids) != sorted(ids):
            warnings.append(f"id w answers-template.json różne od pytań: brak pytań dla {sorted(set(template_ids) - set(ids))}, "
                            f"pytania bez miejsca w szablonie {sorted(set(ids) - set(template_ids))}")
    exam.setdefault("exam_id", "unknown"); exam["items"] = items
    if strict: errors += warnings
    if errors:
        raise ValueError("paczka egzaminu niepoprawna:\n  " + "\n  ".join(dict.fromkeys(errors)))
    for w in dict.fromkeys(warnings): print(f"  UWAGA paczka: {w}", file=sys.stderr)
    return exam, root, template_ids


def image_payload(root, im):
    """Obraz do żądania /solve: ścieżka, suma i treść PNG w base64 (solver nie musi mieć dostępu do plików)."""
    data = (Path(root) / im["path"]).read_bytes()
    mime = "image/jpeg" if data[:3] == b"\xff\xd8\xff" else "image/webp" if data[8:12] == b"WEBP" else "image/png"
    return {"path": im["path"], "sha256": im.get("sha256"), "source_page": im.get("source_page"),
            "mime": mime, "data_b64": base64.b64encode(data).decode()}


# ---- answer_format ----

def option_letters(question):
    """Litery odpowiedzi wypisane w poleceniu („A. …”, „B. …”); domyślnie A–D."""
    found = sorted(set(re.findall(r"(?m)^\s*([A-F])\.\s", question)))
    return found or list("ABCD")


def question_labels(question):
    """Etykiety do wypełnienia na końcu polecenia, np. ["Rozstrzygnięcie:", "Uzasadnienie:"] albo ["Nazwisko:", "Wyjaśnienie:"]."""
    labels = []
    for line in reversed((question or "").strip().split("\n")):
        if re.fullmatch(r"\s*[^\n:]{1,40}:\s*", line):
            labels.append(line.strip())
        else:
            break
    return labels[::-1]


def is_essay(item):
    """Wypracowanie rozpoznawane po treści, nie po numerze zadania: answer_format mówi o tekście/wypracowaniu,
    polecenie ma tematy do wyboru i minimum wyrazów, albo zadanie jest warte >= 10 pkt."""
    f = (item.get("answer_format") or "").lower(); q = (item.get("question") or "").lower()
    try: pts = float(item.get("max_points") or 0)
    except (TypeError, ValueError): pts = 0
    return (f.startswith("jeden tekst") or "wypracowani" in f or "wypracowani" in q
            or ("tematy" in q and re.search(r"minimum\s+\d+\s+wyraz", q)) or pts >= 10)


def format_spec(item):
    """Rodzaj odpowiedzi wynikający z answer_format:
      {"kind": "essay"} | {"kind": "open", "labels": [...]} | {"kind": "letter", "letters": [...]}
      | {"kind": "pairs", "pairs": [(etykieta, "PF"|"letter"|"digit"), ...], "letters": [...]}
    Wartości w answer_format to wyłącznie przykład składni, nie odpowiedź."""
    f = (item.get("answer_format") or "").strip()
    q = item.get("question") or ""
    letters = option_letters(q)
    if is_essay(item):
        return {"kind": "essay"}
    if re.fullmatch(r"[A-F]", f):
        return {"kind": "letter", "letters": letters}
    lines = [l for l in f.split("\n") if l.strip()]
    m = [re.fullmatch(r"\s*([0-9]{1,2}(?:\.[0-9])?|[A-Za-z])\s*:\s*(\S+)\s*", l) for l in lines]
    if lines and all(m):
        kind = lambda v: "PF" if v in ("P", "F") else "letter" if re.fullmatch(r"[A-F]", v) else "digit" if v.isdigit() else None
        pairs = [(x.group(1), kind(x.group(2))) for x in m]
        if all(k for _, k in pairs):
            return {"kind": "pairs", "pairs": pairs, "letters": letters}
    spec = {"kind": "open", "labels": question_labels(item.get("question", ""))}
    if f and not f.lower().startswith("tekst"):  # wzór, którego nie rozumiemy: model dostanie go dosłownie jako składnię
        spec["raw_format"] = f
    return spec


def format_instruction(spec):
    """Opis wymaganej formy odpowiedzi dla modelu; celowo bez przykładowych wartości z answer_format."""
    k = spec["kind"]
    if k == "letter":
        return f"Odpowiedz wyłącznie jedną literą ({', '.join(spec['letters'])}), bez uzasadnienia."
    if k == "pairs":
        what = {"PF": "P albo F", "letter": "litera (" + "/".join(spec["letters"]) + ")", "digit": "numer"}
        lines = "\n".join(f"{lab}: <{what[v]}>" for lab, v in spec["pairs"])
        return f"Odpowiedz wyłącznie w {len(spec['pairs'])} liniach, dokładnie w tej formie, bez uzasadnień:\n{lines}"
    if k == "essay":
        return "Zacznij od linii „Temat nr X”, potem całe wypracowanie (co najmniej 350 wyrazów)."
    if spec.get("raw_format"):
        return ("Odpowiedz dokładnie w tej formie (to wzór składni; wartości w nim są przykładowe, nie poprawne odpowiedzi):\n"
                + spec["raw_format"])
    if spec.get("labels"):
        return "Odpowiedz, wypełniając kolejno etykiety z polecenia, każdą w osobnej linii: " + " ".join(spec["labels"])
    return "Odpowiedz zwięźle, podając wszystkie elementy wymagane w poleceniu."


LET = "A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż"


def _value(seg, kind, letters):
    """Jedna jednoznaczna wartość w fragmencie tekstu albo None."""
    if kind == "PF":
        tok = set(re.findall(rf"(?<![{LET}])([PF])(?![{LET}])", seg))
        if not tok:
            tok = {x for x, w in (("P", r"(?i)prawd"), ("F", r"(?i)fałsz")) if re.search(w, seg)}
    elif kind == "letter":
        tok = set(re.findall(rf"(?<![{LET}])([{''.join(letters)}])(?![{LET}])", seg))
    else:
        tok = set(re.findall(r"(?<!\d)(\d{1,2})(?!\d)", seg))
    return tok.pop() if len(tok) == 1 else None


def normalize_closed(text, spec):
    """Odpowiedź modelu -> dokładna składnia answer_format (np. "1: P\\n2: F\\n3: P" albo "C").
    None, gdy nie da się jej jednoznacznie odczytać (wtedy solver prosi model o sam format)."""
    t = re.sub(r"[*_`]", "", text or "").strip()
    if spec["kind"] == "letter":
        m = re.search(rf"(?i)odpowied[zź]\w*\s*[:\-–]?\s*([{''.join(spec['letters'])}])(?![{LET}])", t)
        if m: return m.group(1).upper()
        first = t.split("\n")[0]
        v = _value(first, "letter", spec["letters"]) or _value(t, "letter", spec["letters"])
        return v
    if spec["kind"] != "pairs":
        return None
    labels = [lab for lab, _ in spec["pairs"]]
    # fragmenty zaczynające się od etykiety na początku linii: „1: P”, „1. P”, „A – 3”, „**2)** F”
    pat = r"(?m)^\s*(?:\(|\[)?(" + "|".join(re.escape(l) for l in labels) + r")\s*[.:)\]–\-]"
    pos = [(m.start(), m.end(), m.group(1)) for m in re.finditer(pat, t)]
    segs = {}
    for i, (s, e, lab) in enumerate(pos):
        segs.setdefault(lab, t[e:pos[i + 1][0] if i + 1 < len(pos) else len(t)])
    out = []
    if all(l in segs for l in labels):
        for lab, kind in spec["pairs"]:
            v = _value(segs[lab], kind, spec.get("letters", list("ABCD")))
            if v is None: break
            out.append(f"{lab}: {v}")
        if len(out) == len(labels):
            return "\n".join(out)
    # bez etykiet: dokładnie tyle wartości, ile pozycji, np. „F, P, P” albo „P F P”
    kinds = {k for _, k in spec["pairs"]}
    if len(kinds) == 1 and len(t) <= 40:
        kind = kinds.pop()
        rx = {"PF": rf"(?<![{LET}])([PF])(?![{LET}])", "letter": rf"(?<![{LET}])([A-F])(?![{LET}])", "digit": r"(?<!\d)(\d{1,2})(?!\d)"}[kind]
        vals = re.findall(rx, t)
        if len(vals) == len(labels):
            return "\n".join(f"{lab}: {v}" for lab, v in zip(labels, vals))
    return None


def closed_guess(spec):
    """Awaryjna odpowiedź dla zadania zamkniętego, gdy solver zawiódł: pusta odpowiedź to pewne 0, zgadnięcie ma wartość oczekiwaną > 0."""
    if spec["kind"] == "letter":
        return spec["letters"][0]
    if spec["kind"] == "pairs":
        first = {"PF": "P", "letter": spec.get("letters", ["A"])[0], "digit": "1"}
        return "\n".join(f"{lab}: {first[kind]}" for lab, kind in spec["pairs"])
    return ""


def _norm_label(s):
    return re.sub(r"[*_.]", "", (s or "").lower())


def missing_labels(text, labels):
    """Etykiety z polecenia (np. „Rozstrzygnięcie:”, „Cecha:” dwa razy), których brakuje w odpowiedzi."""
    t, need = _norm_label(text), {}
    for l in labels:
        k = _norm_label(l).rstrip(": ").strip()
        need[k] = need.get(k, 0) + 1
    return [k for k, n in need.items() if len(re.findall(re.escape(k) + r"\s*:", t)) < n]


PREAMBLE = re.compile(r"(?i)^\s*(oto\s+(moja\s+)?odpowied[zź][^\n]*?:|odpowied[zź]\s+na\s+zadanie[^\n]*?:)[ \t]*\n*")


def clean_answer(text):
    """Porządki bez zmiany treści: bez markdownu (**, __, nagłówki #) i wstępów typu „Oto odpowiedź na zadanie:”."""
    t = re.sub(r"\*\*|__", "", text or "")
    t = re.sub(r"(?m)^#{1,6}[ \t]*", "", t)
    t = PREAMBLE.sub("", t)
    t = re.sub(r"[ \t]+\n", "\n", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def essay_ok(text):
    """(czy_ok, liczba_wyrazów, czy_jest_numer_tematu) dla wypracowania."""
    words = len((text or "").split())
    topic = bool(re.search(r"(?i)\btemat\w*\s*(nr\.?|numer)?\s*[1-3]\b", (text or "")[:200]))
    return words >= 300 and topic, words, topic


# ---- walidacja answers.json ----

def validate_answers(obj, exam, template_ids=None):
    """-> (błędy, ostrzeżenia). Błędy = reguły strony zgłoszeń (plik zostałby odrzucony);
    ostrzeżenia = nasze kontrole jakości (puste odpowiedzi, składnia zamkniętych, wypracowanie)."""
    errors, warnings = [], []
    ids = template_ids or [it["id"] for it in exam["items"]]
    items = {it["id"]: it for it in exam["items"]}
    if not isinstance(obj, dict) or set(obj) != {"exam_id", "answers"}:
        return ["na górze pliku muszą być dokładnie pola exam_id i answers"], warnings
    if obj["exam_id"] != exam["exam_id"]:
        errors.append(f"exam_id {obj['exam_id']!r} różny od {exam['exam_id']!r}")
    if not isinstance(obj["answers"], list):
        return errors + ["answers musi być listą"], warnings
    seen = []
    for a in obj["answers"]:
        if not isinstance(a, dict) or set(a) != {"id", "answer"}:
            errors.append(f"wpis {a!r:.60} musi mieć dokładnie pola id i answer"); continue
        if not isinstance(a["id"], str): errors.append(f"id {a['id']!r} musi być stringiem")
        if not isinstance(a["answer"], str): errors.append(f"odpowiedź {a['id']!r} musi być stringiem")
        elif len(a["answer"]) > MAX_ANSWER_CHARS: errors.append(f"odpowiedź {a['id']} ma ponad {MAX_ANSWER_CHARS} znaków")
        seen.append(a["id"])
    dup = sorted({i for i in seen if seen.count(i) > 1})
    if dup: errors.append(f"powtórzone id: {dup}")
    missing = [i for i in ids if i not in seen]
    extra = [i for i in seen if i not in ids]
    if missing: errors.append(f"brak odpowiedzi dla id: {missing}")
    if extra: errors.append(f"nieznane id: {extra}")
    size = len(json.dumps(obj, ensure_ascii=False).encode("utf-8"))
    if size > MAX_FILE_BYTES: errors.append(f"plik ma {size} B, limit {MAX_FILE_BYTES} B")
    for a in obj["answers"]:
        if not isinstance(a, dict) or not isinstance(a.get("answer"), str): continue
        if a.get("id") not in items:
            if not a["answer"].strip(): warnings.append(f"{a['id']}: pusta odpowiedź (id z szablonu bez pytania w exam.json)")
            continue
        text, spec = a["answer"], format_spec(items[a["id"]])
        if not text.strip():
            warnings.append(f"{a['id']}: pusta odpowiedź"); continue
        if "<think>" in text or "</think>" in text:
            warnings.append(f"{a['id']}: ślad rozumowania (<think>) w odpowiedzi")
        if spec["kind"] in ("letter", "pairs") and normalize_closed(text, spec) != text.strip():
            warnings.append(f"{a['id']}: odpowiedź zamknięta nie jest w składni answer_format: {text[:60]!r}")
        if spec["kind"] == "essay":
            ok, words, topic = essay_ok(text)
            if not ok: warnings.append(f"{a['id']}: wypracowanie {words} wyrazów, numer tematu: {'jest' if topic else 'brak'}")
        if spec["kind"] == "open" and spec.get("labels"):
            miss = missing_labels(text, spec["labels"])
            if miss: warnings.append(f"{a['id']}: brak etykiet z polecenia: {miss}")
        if spec["kind"] in ("open", "essay") and ("**" in text or re.search(r"(?m)^#{1,6} ", text)):
            warnings.append(f"{a['id']}: formatowanie markdown w odpowiedzi")
    return errors, warnings
