#!/usr/bin/env python3
"""Laboratorium promptów wypracowania dla małego modelu (Bielik 1.5B): każdy wariant na wszystkich tematach arkuszy f2023.

  python small_model/essay_lab.py gen --variants base simple staged simple_rag staged_rag [--temperature 0] [--rep 1.05]
  python small_model/essay_lab.py judge --variants ...          # sędzia GPT-6 Sol v4 (eval/run.py --phase judge)
  python small_model/essay_lab.py report --variants ...

Wyniki: results/essaylab-<wariant>[-t<N>]/answers.jsonl (id = zadanie wypracowania z danego rocznika, "Temat nr N" w tekście)
+ essays.jsonl z debugiem (plan, notatki, liczba wyrazów). Model: OpenAI API (vLLM) pod --llm-url, retrieval pod --rag-url.
"""
import argparse, json, re, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval")); sys.path.insert(0, str(ROOT / "exam"))
import prompts as EP  # eval/prompts.py: ESSAY_SYSTEM = obecny prompt harnessu (kontrola)

YEARS = ["2023_f2023", "2024_f2023", "2025_f2023", "2026_f2023"]
JUDGE = "gpt-6-sol-v4"

# ---------- tematy ----------

def load_essays():
    out = {}
    for l in (ROOT / "data/questions.jsonl").open(encoding="utf-8"):
        q = json.loads(l)
        if q["punkty_max"] == 15 and q["rok"] and q["id"].split("_z")[0] in YEARS:
            out[q["id"].split("_z")[0]] = q
    return out


def parse_topics(tresc):
    """{n: pełna treść tematu} z polecenia (trzy tematy przed WYPRACOWANIE)."""
    body = tresc.split("WYPRACOWANIE")[0]
    parts = re.split(r"(?m)^\s*([123])\.\s+", body)
    return {int(parts[i]): re.sub(r"\s+", " ", parts[i + 1]).strip() for i in range(1, len(parts) - 1, 2)}


def thesis(topic):
    return topic.split("Zajmij stanowisko")[0].strip().rstrip(".") + "."


def elements(topic):
    """('aspekty', [3 przymiotniki]) albo ('wybór', 'trzech wybranych władców z tego okresu')."""
    m = re.search(r"aspekt(?:y|ów)?:?\s*([^.]+)", topic)
    if m and "," in m.group(1):
        raw = re.split(r",\s*|\s+i\s+|\s+oraz\s+", m.group(1).strip().rstrip("."))
        return "aspekty", [a.strip(" -") for a in raw if a.strip()][:3]
    m = re.search(r"(trz(?:ech|y)\s+wybran\w+[^.]*)", topic)
    return "wybór", (m.group(1).strip() if m else "trzy wybrane przykłady")


def decl(adj):  # przymiotnik w narzędniku: militarny -> militarnym, społeczno-gospodarczy -> społeczno-gospodarczym
    return adj[:-1] + "ym" if adj.endswith("y") else adj[:-1] + "im" if adj.endswith("i") else adj


def words(t):
    return len((t or "").split())


def clean(t):
    t = re.sub(r"\*\*|__|^#+ ", "", t or "", flags=re.M)
    t = re.sub(r"(?im)^\s*(wstęp|rozwinięcie|zakończenie|plan|akapit \d)\s*:?\s*$", "", t)
    t = re.sub(r"(?i)^\s*temat\s*(nr\.?)?\s*\d\s*[.:]?\s*\n+", "", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()

# ---------- model i retrieval ----------

class LLM:
    def __init__(self, url, model, temperature=0.0, rep=1.05):
        self.url, self.model, self.t, self.rep = url.rstrip("/"), model, temperature, rep
        self.calls = 0

    def __call__(self, system, user, max_tokens, temperature=None):
        body = {"model": self.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "max_tokens": max_tokens, "temperature": self.t if temperature is None else temperature, "repetition_penalty": self.rep}
        req = urllib.request.Request(self.url + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            self.calls += 1
            return (json.load(r)["choices"][0]["message"].get("content") or "").strip()


def retrieve(rag_url, queries, k=8, min_score=0.3):
    body = json.dumps({"queries": queries, "k": k, "min_score": min_score}).encode()
    req = urllib.request.Request(rag_url.rstrip("/") + "/search", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r).get("hits", [])


def notes_for(rag_url, topic, kind, els, per=900, cap=4200, mode="multi", k=8):
    th = thesis(topic)
    if mode == "task":   # cała treść tematu jako jedno zapytanie
        qs = [topic]
    else:
        qs = [th] + ([f"{th.rstrip('.')} – aspekt {a}" for a in els] if kind == "aspekty" else [f"{th.rstrip('.')} – {els}"])
    hits = retrieve(rag_url, qs, k=k)
    frags, used = [], 0
    for h in hits:
        t = h["text"][:per]
        if used + len(t) > cap: break
        frags.append(t); used += len(t)
    return ("\n###\n".join(frags), [{"title": h["title"], "score": h["score"], "q": h["query"][-40:]} for h in hits])

# ---------- warianty ----------

SYS_SIMPLE = """Jesteś maturzystą. Piszesz wypracowanie z historii po polsku pełnymi zdaniami, w akapitach. Nie używasz list, punktów, numeracji, myślników ani nagłówków (np. „Aspekt militarny:”). Podajesz tylko fakty, których jesteś pewien (daty, postacie, wydarzenia); nie zmyślasz szczegółów i trzymasz się okresu, którego dotyczy temat."""

NOTES = """Notatki z encyklopedii (mogą pomóc; korzystaj tylko z tych, które pasują do tematu, nie przepisuj ich dosłownie):
{notes}
---
"""


def structure_lines(kind, els):
    """Opis struktury bez numeracji (numerowana instrukcja skłania mały model do pisania listy zamiast tekstu)."""
    if kind == "aspekty":
        a = els
        return (f"Wypracowanie ma pięć akapitów. Wstęp mówi wprost, czy zgadzasz się z tezą, i zapowiada trzy aspekty: {a[0]}, {a[1]} i {a[2]}. "
                f"Potem trzy akapity, po jednym na aspekt: akapit o aspekcie {decl(a[0])} zaczyna się od słów „W aspekcie {decl(a[0])}”, "
                f"akapit o aspekcie {decl(a[1])} od słów „W aspekcie {decl(a[1])}”, akapit o aspekcie {decl(a[2])} od słów „W aspekcie {decl(a[2])}”; "
                f"w każdym z nich podajesz dwa lub trzy konkretne fakty i piszesz, jak potwierdzają lub podważają tezę. "
                f"Zakończenie zawiera wniosek i powtarza stanowisko.")
    return (f"Wypracowanie ma pięć akapitów. Wstęp mówi wprost, czy zgadzasz się z tezą, i wymienia {els}, które omówisz. "
            f"Potem trzy akapity, po jednym na każdy z wybranych przykładów: pierwsze zdanie akapitu nazywa przykład, dalej dwa lub trzy konkretne fakty "
            f"i zdanie, jak potwierdzają lub podważają tezę. Zakończenie zawiera wniosek i powtarza stanowisko.")


BULLET = re.compile(r"(?m)^\s*(?:[-•*]|\d+[.)])\s+")


def prose_fix(llm, text, debug, max_tokens=1500):
    """Lista punktów zamiast tekstu -> model przepisuje własne notatki jako ciągłe wypracowanie (małe modele robią to dobrze)."""
    if len(BULLET.findall(text)) < 4: return text
    u = f"Poniżej notatki do wypracowania. Przepisz je jako ciągły tekst wypracowania w pięciu akapitach (400 wyrazów), pełnymi zdaniami, bez list, numeracji i nagłówków. Zachowaj stanowisko wobec tezy i wszystkie fakty; nie dodawaj nowych.\n\nNotatki:\n{text}"
    out = clean(llm(SYS_SIMPLE, u, max_tokens))
    debug["prose_fix"] = {"bullets": len(BULLET.findall(text)), "before": words(text), "after": words(out), "still_bullets": len(BULLET.findall(out))}
    return out if words(out) >= 0.6 * words(text) and len(BULLET.findall(out)) < 4 else text


def add_paragraph(llm, topic, kind, els, text, debug, i):
    """Za krótko: dopisz akapit o kolejnym elemencie (wstawiany przed zakończeniem) zamiast prosić o „dłuższą wersję” (mały model pisze wtedy tyle samo)."""
    e = els[i % 3] if kind == "aspekty" else None
    what = f"o aspekcie {decl(e)} (zacznij od słów „Warto dodać, że w aspekcie {decl(e)}”)" if e else "z dodatkowym przykładem potwierdzającym twoje stanowisko (nazwij go w pierwszym zdaniu)"
    u = f"Temat wypracowania: {topic}\n\nTwoje wypracowanie:\n{text[-2500:]}\n\nDopisz jeden dodatkowy akapit (90–110 wyrazów) {what}: dwa lub trzy konkretne fakty i zdanie, jak odnoszą się do tezy. Pełnymi zdaniami, bez listy. Napisz tylko ten akapit."
    p = clean(llm(SYS_SIMPLE, u, 350))
    debug.setdefault("added", []).append(words(p))
    if words(p) < 40: return text
    paras = text.split("\n\n")
    return "\n\n".join(paras[:-1] + [p, paras[-1]]) if len(paras) >= 2 else text + "\n\n" + p


def ensure_len(llm, system, user, text, lo, debug, key, max_tokens):
    if words(text) >= lo: return text
    again = clean(llm(system, user + f"\n\nTwój tekst ma {words(text)} wyrazów, a potrzeba co najmniej {lo}. Napisz go jeszcze raz w całości, dłuższy, dodając konkretne fakty w każdym akapicie.", max_tokens))
    debug[key] = {"before": words(text), "after": words(again)}
    return again if words(again) > words(text) else text


def v_base(llm, q, n, rag_url, debug):
    """Kontrola: obecny prompt harnessu, model sam wybiera temat (n ignorowane)."""
    txt = llm(EP.ESSAY_SYSTEM, q["tresc"] + "\n\nOdpowiedź:", 2000)
    m = re.search(r"Temat\s*nr\s*([123])", txt); debug["chosen"] = int(m.group(1)) if m else None
    return txt.strip()


def v_simple(llm, q, n, rag_url, debug, rag=False):
    topic = parse_topics(q["tresc"])[n]; kind, els = elements(topic)
    pre = ""
    if rag:
        notes, hits = notes_for(rag_url, topic, kind, els); debug["rag"] = hits
        pre = NOTES.format(notes=notes) if notes else ""
    user = f"{pre}Temat wypracowania: {topic}\n\nNapisz wypracowanie o długości 400–450 wyrazów. {structure_lines(kind, els)} Napisz tylko tekst wypracowania, pełnymi zdaniami."
    text = prose_fix(llm, clean(llm(SYS_SIMPLE, user, 1400)), debug)
    if len(BULLET.findall(text)) >= 4: text = debullet(text); debug["debullet"] = True
    for i in range(2):
        if words(text) >= 320: break
        text = add_paragraph(llm, topic, kind, els, text, debug, i)
    return f"Temat nr {n}\n\n{text}"


def debullet(text):
    """Ostateczność: punktory mechanicznie na zdania, sąsiednie punkty sklejone w akapit (spójność ucierpi mniej niż przy liście)."""
    out, buf = [], []
    for line in text.split("\n"):
        if BULLET.match(line):
            buf.append(BULLET.sub("", line).strip().rstrip(".") + ".")
        else:
            if buf: out.append(" ".join(buf)); buf = []
            out.append(line)
    if buf: out.append(" ".join(buf))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def v_staged(llm, q, n, rag_url, debug, rag=False, rag_mode="multi", rag_k=8):
    """Bez planu: wstęp -> akapit na element (3) -> zakończenie, każdy osobnym krótkim zapytaniem; wybór przykładów osobnym mini-pytaniem."""
    topic = parse_topics(q["tresc"])[n]; kind, els = elements(topic); th = thesis(topic)
    pre = ""
    if rag:
        notes, hits = notes_for(rag_url, topic, kind, els, per=1200 if rag_mode == "task" else 900, cap=4800, mode=rag_mode, k=rag_k); debug["rag"] = hits
        pre = NOTES.format(notes=notes) if notes else ""
    if kind == "aspekty":
        names = els
    else:  # np. „trzech wybranych władców z tego okresu”: model podaje tylko nazwy, o których wie najwięcej
        raw = llm("Odpowiadasz po polsku, tylko listą nazw po przecinku, bez komentarza.", f"{pre}Temat wypracowania: {topic}\n\nPodaj {els}, o których wiesz najwięcej pewnych faktów i które pasują do okresu tematu. Tylko trzy nazwy po przecinku.", 60)
        names = [x.strip(" .\n-") for x in re.split(r",|;|\n", raw) if x.strip(" .\n-")][:3]
        names = (names + ["pierwszy przykład", "drugi przykład", "trzeci przykład"])[:3]
        debug["names"] = names
    raw = clean(llm(SYS_SIMPLE, f"{pre}Temat wypracowania: {topic}\n\nNapisz jednym zdaniem, czy zgadzasz się z tezą (zgadzam się / nie zgadzam się / zgadzam się częściowo) i dlaczego.", 80))
    m = re.search(r"[^.\n]*(zgadzam|nie zgadzam)[^.\n]*\.", raw, re.I)   # tylko jedno zdanie ze stanowiskiem (z notatkami model potrafi zamiast tego napisać konspekt)
    stance = m.group(0).strip() if m else f"Zgadzam się z tezą, że {th[0].lower() + th[1:]}"
    debug["stance"] = stance
    head = f"{pre}Temat wypracowania: {topic}\nStanowisko: {stance}\n\n"
    lst = ", ".join(names)
    intro = clean(llm(SYS_SIMPLE, head + f"Napisz WSTĘP wypracowania (60–80 wyrazów, pełnymi zdaniami): w pierwszym zdaniu stanowisko wobec tezy, potem zapowiedź trzech elementów: {lst}. Tylko tekst wstępu.", 250))
    paras = []
    for i, e in enumerate(names, 1):
        lead = f"W aspekcie {decl(e)}" if kind == "aspekty" else e
        u = head + f"Napisz jeden akapit wypracowania (90–120 wyrazów, pełnymi zdaniami, bez listy) o elemencie: {e}. Zacznij od słów „{lead}”, podaj dwa lub trzy konkretne fakty (daty, postacie, wydarzenia) i zakończ zdaniem, jak ten element potwierdza lub podważa tezę. Tylko tekst akapitu."
        p = clean(llm(SYS_SIMPLE, u, 350))
        if len(BULLET.findall(p)) >= 2: p = debullet(p)
        if words(p) < 50: p = ensure_len(llm, SYS_SIMPLE, u, p, 70, debug, f"extend_p{i}", 450)
        paras.append(p)
    outro = clean(llm(SYS_SIMPLE, head + f"Elementy omówione w rozwinięciu: {lst}.\n\nNapisz ZAKOŃCZENIE wypracowania (50–70 wyrazów, pełnymi zdaniami): wniosek z trzech elementów i powtórzenie stanowiska, bez nowych faktów. Tylko tekst zakończenia.", 200))
    debug["parts_words"] = [words(intro)] + [words(p) for p in paras] + [words(outro)]
    text = "\n\n".join([intro] + paras + [outro])
    for i in range(2):
        if words(text) >= 320: break
        text = add_paragraph(llm, topic, kind, els, text, debug, i)
    return f"Temat nr {n}\n\n{text}"


def trim_words(text, limit):
    """Ucina akapit na granicy zdania po ~limit wyrazach (dłuższy akapit = więcej okazji do błędów, a punktów za długość nie ma)."""
    if words(text) <= limit: return text
    out, n = [], 0
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if n >= limit: break
        out.append(sent); n += words(sent)
    return " ".join(out)


def v_staged2(llm, q, n, rag_url, debug):
    """staged_rag z notatkami osobno do każdego elementu, faktami tylko z notatek i akapitami przyciętymi do ~120 wyrazów."""
    topic = parse_topics(q["tresc"])[n]; kind, els = elements(topic); th = thesis(topic)
    if kind == "aspekty":
        names = els
    else:
        raw = llm("Odpowiadasz po polsku, tylko listą nazw po przecinku, bez komentarza.", f"Temat wypracowania: {topic}\n\nPodaj {els}, o których wiesz najwięcej pewnych faktów i które pasują do okresu tematu. Tylko trzy nazwy po przecinku.", 60)
        names = [x.strip(" .\n-") for x in re.split(r",|;|\n", raw) if x.strip(" .\n-")][:3]
        names = (names + ["pierwszy przykład", "drugi przykład", "trzeci przykład"])[:3]
        debug["names"] = names
    # notatki: osobne zapytanie na element (temat + element), po 3 fragmenty; ogólne do wstępu
    per_el, hits_dbg = {}, []
    for e in names:
        hits = retrieve(rag_url, [f"{th.rstrip('.')} – {e}", e if kind != "aspekty" else f"{th.rstrip('.')} aspekt {e}"], k=3, min_score=0.3)
        per_el[e] = "\n###\n".join(h["text"][:800] for h in hits); hits_dbg += [{"el": e, "title": h["title"], "score": h["score"]} for h in hits]
    debug["rag"] = hits_dbg
    sys_fact = SYS_SIMPLE + " Fakty (daty, nazwiska, wydarzenia) bierzesz wyłącznie z podanych notatek; jeśli notatki czegoś nie zawierają, piszesz ogólnie, bez wymyślania dat i nazwisk."
    raw = clean(llm(SYS_SIMPLE, f"Temat wypracowania: {topic}\n\nNapisz jednym zdaniem, czy zgadzasz się z tezą (zgadzam się / nie zgadzam się / zgadzam się częściowo) i dlaczego.", 80))
    m = re.search(r"[^.\n]*(zgadzam|nie zgadzam)[^.\n]*\.", raw, re.I)
    stance = m.group(0).strip() if m else f"Zgadzam się z tezą, że {th[0].lower() + th[1:]}"
    debug["stance"] = stance
    lst = ", ".join(names)
    intro = trim_words(clean(llm(SYS_SIMPLE, f"Temat wypracowania: {topic}\nStanowisko: {stance}\n\nNapisz WSTĘP wypracowania (60–80 wyrazów, pełnymi zdaniami): w pierwszym zdaniu stanowisko wobec tezy, potem zapowiedź trzech elementów: {lst}. Bez dat i nazwisk. Tylko tekst wstępu.", 220)), 90)
    paras = []
    for i, e in enumerate(names, 1):
        lead = f"W aspekcie {decl(e)}" if kind == "aspekty" else e
        notes = NOTES.format(notes=per_el[e]) if per_el[e] else ""
        u = f"{notes}Temat wypracowania: {topic}\nStanowisko: {stance}\n\nNapisz jeden akapit wypracowania (90–110 wyrazów, pełnymi zdaniami, bez listy) o elemencie: {e}. Zacznij od słów „{lead}”, podaj dwa lub trzy konkretne fakty Z NOTATEK i zakończ zdaniem, jak ten element potwierdza lub podważa tezę. Tylko tekst akapitu."
        p = clean(llm(sys_fact, u, 320))
        if len(BULLET.findall(p)) >= 2: p = debullet(p)
        if words(p) < 50: p = ensure_len(llm, sys_fact, u, p, 70, debug, f"extend_p{i}", 400)
        paras.append(trim_words(p, 125))
    outro = trim_words(clean(llm(SYS_SIMPLE, f"Temat wypracowania: {topic}\nStanowisko: {stance}\nElementy omówione w rozwinięciu: {lst}.\n\nNapisz ZAKOŃCZENIE wypracowania (50–70 wyrazów, pełnymi zdaniami): wniosek z trzech elementów i powtórzenie stanowiska, bez nowych faktów. Tylko tekst zakończenia.", 180)), 80)
    debug["parts_words"] = [words(intro)] + [words(p) for p in paras] + [words(outro)]
    text = "\n\n".join([intro] + paras + [outro])
    for i in range(2):
        if words(text) >= 320: break
        text = add_paragraph(llm, topic, kind, els, text, debug, i)
    return f"Temat nr {n}\n\n{text}"


HEADING = re.compile(r"(?im)^\s*(aspekt[^:\n]{0,50}|podsumowanie|wnioski?|wstęp|zakończenie|rozwinięcie|argumenty?|stanowisko|teza)\s*:?\s*$")
INLINE_HEADING = re.compile(r"(?i)\s*(?:aspekt [^.:\n]{0,40}|podsumowanie|wnioski?)\s*:\s*")


def polish(text):
    """Porządki po małym modelu: nagłówki w środku tekstu, powtórzone zdania, urwane ostatnie zdanie. Zachowuje „Temat nr N”."""
    m = re.match(r"\s*(Temat nr \d)\s*\n+", text); head = m.group(1) + "\n\n" if m else ""
    body = text[m.end():] if m else text
    body = HEADING.sub("", body); body = INLINE_HEADING.sub(" ", body)
    paras, seen = [], set()
    for para in re.split(r"\n\s*\n", body):
        sents = []
        for sent in re.split(r"(?<=[.!?])\s+", para.strip()):
            key = re.sub(r"\W+", " ", sent.lower()).strip()[:70]
            if len(key) < 15 or key in seen: continue   # powtórka albo śmieć
            seen.add(key); sents.append(sent.strip())
        if sents: paras.append(" ".join(sents))
    body = "\n\n".join(paras).strip()
    if body and body[-1] not in ".!?»”\"":   # urwane ostatnie zdanie (limit tokenów)
        cut = max(body.rfind(". "), body.rfind(".\n"), body.rfind("!"), body.rfind("?"))
        if cut > 0.6 * len(body): body = body[:cut + 1]
    return head + body


def polish_runs(a):
    for v in a.variants:
        for n in [1, 2, 3]:
            src = ROOT / "results" / run_name(v, n, True, a.tag); dst = ROOT / "results" / run_name(v, n, True, a.tag + a.out_tag)
            dst.mkdir(parents=True, exist_ok=True)
            with (src / "answers.jsonl").open(encoding="utf-8") as f, (dst / "answers.jsonl").open("w", encoding="utf-8") as g, (dst / "essays.jsonl").open("w", encoding="utf-8") as h:
                for l in f:
                    a1 = json.loads(l); t = polish(a1["odpowiedz"])
                    if words(t) < 300 <= words(a1["odpowiedz"]): t = a1["odpowiedz"]   # porządki nie mogą zepchnąć poniżej 300 wyrazów
                    g.write(json.dumps({**a1, "odpowiedz": t}, ensure_ascii=False) + "\n")
                    h.write(json.dumps({"id": a1["id"], "topic": n, "words": words(t), "debug": {"before": words(a1["odpowiedz"])}}, ensure_ascii=False) + "\n")
                    print(f"{v}{a.tag}-t{n} {a1['id'][:4]}: {words(a1['odpowiedz'])} -> {words(t)} wyr.")


# ---------- tipsy: zdania-fakty per aspekt, instrukcja na końcu, few-shot, filtr lat/nazwisk po generacji ----------

YEAR = re.compile(r"\b(\d{3,4})\b")
FEWSHOT = """Przykład dobrego akapitu (inny temat):
W aspekcie gospodarczym panowanie Kazimierza Wielkiego wyraźnie wzmocniło państwo. Król lokował wiele miast, w 1364 roku założył Akademię Krakowską, a statutami wiślicko-piotrkowskimi uporządkował prawo. Wprowadzenie grosza krakowskiego i ochrona kupców ożywiły handel, a dochody skarbu wzrosły. Te działania pokazują, że władca konsekwentnie budował gospodarcze podstawy państwa, co potwierdza tezę o jego wybitnej roli."""


def fact_sentences(hits, topic, limit=5, max_words=30):
    """Z fragmentów Wikipedii zostają tylko zdania z rokiem albo nazwą własną, po jednym fakcie, <= max_words wyrazów."""
    out, seen = [], set()
    for h in hits:
        for sent in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", h["text"])):
            sent = sent.strip()
            if not (12 <= words(sent) <= max_words) or sent[-1] not in ".!?": continue
            has_year = bool(YEAR.search(sent)); has_name = bool(re.search(r"(?<!^)(?<![.!?] )\b[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]{2,}", sent))
            if not (has_year or has_name): continue
            key = sent.lower()[:60]
            if key in seen: continue
            seen.add(key); out.append(sent)
            if len(out) >= limit: return out
    return out


def allowed_tokens(text):
    yrs = set(YEAR.findall(text)); names = set(w.lower() for w in re.findall(r"\b[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]{2,}\b", text))
    return yrs, names


def filter_paragraph(p, notes_text, topic, mode, debug, key):
    """Zdania z rokiem/nazwiskiem spoza notatek i tematu wylatują (sędzia liczy każdy błąd: 3–5 błędów = -2 pkt)."""
    if mode == "none": return p
    yrs, names = allowed_tokens(notes_text + " " + topic + " " + FEWSHOT.split("\n", 1)[0])
    kept, dropped = [], []
    for sent in re.split(r"(?<=[.!?])\s+", p):
        bad = any(y not in yrs and 800 <= int(y) <= 2030 for y in YEAR.findall(sent))
        if not bad and mode == "names":
            toks = re.findall(r"(?<![.!?] )\b[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]{2,}\b", sent[1:])   # nie pierwsze słowo zdania
            bad = any(t.lower() not in names and t.lower()[:5] not in {n[:5] for n in names} for t in toks)
        (dropped if bad else kept).append(sent)
    debug.setdefault(key, []).append({"dropped": len(dropped), "kept": len(kept)})
    out = " ".join(kept)
    return out if words(out) >= 40 else p   # nie zostawiaj pustego akapitu


def v_facts(llm, q, n, rag_url, debug, filt="none", fewshot=False, nfacts=5, min_score=0.3):
    topic = parse_topics(q["tresc"])[n]; kind, els = elements(topic); th = thesis(topic).rstrip(".")
    if kind == "aspekty": names = els
    else:
        raw = llm("Odpowiadasz po polsku, tylko listą nazw po przecinku, bez komentarza.", f"Temat wypracowania: {topic}\n\nPodaj {els}, o których wiesz najwięcej pewnych faktów i które pasują do okresu tematu. Tylko trzy nazwy po przecinku.", 60)
        names = ([x.strip(" .\n-") for x in re.split(r",|;|\n", raw) if x.strip(" .\n-")][:3] + ["pierwszy przykład", "drugi przykład", "trzeci przykład"])[:3]
        debug["names"] = names
    stance = f"Zgadzam się z tezą: „{th}”"
    if kind == "aspekty":
        intro = f"{stance}. W wypracowaniu uzasadnię to stanowisko, omawiając kolejno trzy aspekty: {els[0]}, {els[1]} i {els[2]}."
        outro = f"Podsumowując, analiza aspektu {els[0][:-1]}ego, {els[1][:-1]}ego i {els[2][:-1]}ego prowadzi do wniosku, że teza „{th}” jest słuszna. Dlatego podtrzymuję stanowisko zajęte we wstępie."
    else:
        intro = f"{stance}. W wypracowaniu uzasadnię to stanowisko, omawiając kolejno trzy przykłady: {names[0]}, {names[1]} i {names[2]}."
        outro = f"Podsumowując, omówione przykłady ({names[0]}, {names[1]} i {names[2]}) prowadzą do wniosku, że teza „{th}” jest słuszna. Dlatego podtrzymuję stanowisko zajęte we wstępie."
    paras, used = [], []
    for i, e in enumerate(names, 1):
        hits = retrieve(rag_url, [f"{th} – {e}", f"{th}"], k=6, min_score=min_score) if rag_url else []
        facts = fact_sentences(hits, topic, limit=nfacts); notes_text = " ".join(facts)
        debug.setdefault("facts", []).append({"el": e, "n": len(facts), "hits": [h["title"] for h in hits][:4]})
        lead = f"W aspekcie {decl(e)}" if kind == "aspekty" else e
        block = ("Fakty z encyklopedii do tego akapitu (korzystaj tylko z nich):\n" + "\n".join(f"{k + 1}. {f}" for k, f in enumerate(facts)) + "\n\n") if facts else ""
        usedtxt = ("Fakty już użyte we wcześniejszych akapitach (nie powtarzaj): " + "; ".join(used[-6:]) + "\n\n") if used else ""
        # kolejność: notatki -> (few-shot) -> teza/stanowisko -> instrukcja NA KOŃCU (efekt świeżości u małych modeli)
        u = block + (FEWSHOT + "\n\n" if fewshot else "") + usedtxt + f"Temat: {topic}\nStanowisko: {stance}.\n\nNapisz jeden akapit wypracowania (90–120 wyrazów, pełnymi zdaniami, bez listy) o elemencie: {e}. Zacznij od słów „{lead}”, użyj dwóch lub trzech faktów z listy powyżej i zakończ zdaniem, jak ten element potwierdza tezę. Tylko tekst akapitu."
        p = clean(llm(SYS_SIMPLE, u, 320))
        if len(BULLET.findall(p)) >= 2: p = debullet(p)
        if words(p) < 50:
            p2 = clean(llm(SYS_SIMPLE, u, 500)); p = p2 if words(p2) > words(p) else p
        p = filter_paragraph(p, notes_text, topic, filt, debug, "filter")
        p = trim_words(p, 130); paras.append(p)
        used += [f"{y}" for y in YEAR.findall(p)][:4]
    text = f"{intro}\n\n" + "\n\n".join(paras) + f"\n\n{outro}"
    for i in range(2):   # < 300 wyrazów: jeszcze jeden akapit
        if words(text) >= 320: break
        text = add_paragraph(llm, topic, kind, els, text, debug, i)
    debug["parts_words"] = [words(intro)] + [words(p) for p in paras] + [words(outro)]
    return f"Temat nr {n}\n\n{text}"


# ---------- krótki prompt z szablonem trzech części (wstęp: nawiązanie do tematu i stanowisko; rozwinięcie: wszystkie wątki z tematu; zakończenie: podsumowanie) ----------
SYS_TPL = """Jesteś maturzystą. Piszesz wypracowanie z historii po polsku, ciągłym tekstem w akapitach, bez list i nagłówków. Podajesz tylko fakty, których jesteś pewien."""


def v_tpl(llm, q, n, rag_url, debug, facts=False):
    topic = parse_topics(q["tresc"])[n]; kind, els = elements(topic); th = thesis(topic).rstrip(".")
    threads = ", ".join(els) if kind == "aspekty" else els
    block = ""
    if facts and rag_url:
        hits = retrieve(rag_url, [th] + ([f"{th} – {e}" for e in els] if kind == "aspekty" else []), k=6, min_score=0.3)
        fs = fact_sentences(hits, topic, limit=6); debug["facts"] = len(fs)
        if fs: block = "Fakty, z których możesz skorzystać:\n" + "\n".join(f"{i + 1}. {f}" for i, f in enumerate(fs)) + "\n\n"
    u = (block + f"Temat wypracowania: {topic}\n\nNapisz wypracowanie (co najmniej 350 wyrazów) w trzech częściach:\n"
         f"wstęp: zajmij stanowisko wobec tezy i nawiąż ogólnie do tematu;\n"
         f"rozwinięcie: rozwiń wszystkie wątki wymienione w temacie ({threads}), każdy w osobnym akapicie z konkretnymi faktami;\n"
         f"zakończenie: podsumowanie argumentów i wniosek.\nTylko tekst wypracowania.")
    debug["prompt_words"] = words(u)
    text = clean(llm(SYS_TPL, u, 1500))
    if len(BULLET.findall(text)) >= 4: text = debullet(text)
    if facts: text, _ = (lambda r: (r, None))(text)   # bez filtra lat: test samego promptu
    for i in range(2):
        if words(text) >= 320: break
        text = add_paragraph(llm, topic, kind, els, text, debug, i)
    return f"Temat nr {n}\n\n{text}"


VARIANTS = {
    "base": (v_base, False),
    "simple": (lambda *a: v_simple(*a, rag=False), True),
    "simple_rag": (lambda *a: v_simple(*a, rag=True), True),
    "staged": (lambda *a: v_staged(*a, rag=False), True),
    "staged_rag": (lambda *a: v_staged(*a, rag=True), True),
    "staged_rag2": (v_staged2, True),
    "staged_rag_task4": (lambda *a: v_staged(*a, rag=True, rag_mode="task", rag_k=4), True),   # treść tematu jako zapytanie, k=4
    "staged_rag_k4": (lambda *a: v_staged(*a, rag=True, rag_mode="multi", rag_k=4), True),
    "facts5": (lambda *a: v_facts(*a, filt="none"), True),            # zdania-fakty per aspekt, instrukcja na końcu, szablonowy wstęp/zakończenie
    "facts5_years": (lambda *a: v_facts(*a, filt="years"), True),     # + filtr: zdania z rokiem spoza notatek wylatują
    "facts5_names": (lambda *a: v_facts(*a, filt="names"), True),     # + filtr lat i nazw własnych
    "facts5_fs": (lambda *a: v_facts(*a, filt="years", fewshot=True), True),   # + wzorcowy akapit
    "facts5_years_m05": (lambda *a: v_facts(*a, filt="years", min_score=0.5), True),   # próg rerankera 0.5 zamiast 0.3
    "facts3_years": (lambda *a: v_facts(*a, filt="years", nfacts=3), True),            # 3 fakty na aspekt zamiast 5
    "tpl": (lambda *a: v_tpl(*a, facts=False), True),         # krótki prompt: szablon wstęp/rozwinięcie/zakończenie, jeden strzał
    "tpl_facts": (lambda *a: v_tpl(*a, facts=True), True),    # jw. + 6 zdań-faktów z retrievalu
}

# ---------- przebieg ----------

def run_name(variant, n, per_topic, tag):
    return f"essaylab-{variant}{tag}" + (f"-t{n}" if per_topic else "")


def gen(a):
    essays = load_essays(); llm = LLM(a.llm_url, a.llm_model, a.temperature, a.rep)
    jobs = []
    for v in a.variants:
        fn, per_topic = VARIANTS[v]
        for year, q in essays.items():
            for n in ([1, 2, 3] if per_topic else [0]):
                jobs.append((v, per_topic, year, q, n))

    def work(job):
        v, per_topic, year, q, n = job; fn = VARIANTS[v][0]; debug = {}; t0 = time.time()
        try:
            text = fn(llm, q, n, a.rag_url, debug)
            if v != "base" and not a.no_polish:
                t2 = polish(text); text = t2 if words(t2) >= 300 or words(text) < 300 else text
        except Exception as e:
            text, debug["error"] = "", str(e)[:300]
        return job, text, debug, time.time() - t0

    with ThreadPoolExecutor(a.concurrency) as ex:
        for (v, per_topic, year, q, n), text, debug, dt in ex.map(work, jobs):
            out = ROOT / "results" / run_name(v, n, per_topic, a.tag); out.mkdir(parents=True, exist_ok=True)
            with (out / "answers.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"id": q["id"], "odpowiedz": text, "sek": round(dt, 1)}, ensure_ascii=False) + "\n")
            with (out / "essays.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"id": q["id"], "topic": n, "words": words(text), "sek": round(dt, 1), "debug": debug}, ensure_ascii=False) + "\n")
            print(f"{v:11s} {year} t{n}  {words(text):4d} wyr.  {dt:5.1f}s  {debug.get('error', '')}", flush=True)
    print("wywołań modelu:", llm.calls)


def runs_of(a):
    return [run_name(v, n, VARIANTS[v][1], a.tag) for v in a.variants for n in ([1, 2, 3] if VARIANTS[v][1] else [0])]


def judge(a):
    py = ROOT / ".venv/bin/python"
    cmds = [[str(py), str(ROOT / "eval/run.py"), "--run", r, "--phase", "judge", "--judge", JUDGE, "--years", *YEARS, "--concurrency", "4"] for r in runs_of(a)]
    with ThreadPoolExecutor(4) as ex:
        for r, res in zip(runs_of(a), ex.map(lambda c: subprocess.run(c, capture_output=True, text=True), cmds)):
            tail = (res.stdout.strip().splitlines() or [""])[-1]
            print(f"{r:28s} rc={res.returncode} {tail[:100]}" + (f" ERR {res.stderr.strip().splitlines()[-1][:120]}" if res.returncode else ""))


def report(a):
    rows = {}
    for r in runs_of(a):
        d = ROOT / "results" / r
        if not (d / f"judgments_{JUDGE}.jsonl").exists(): continue
        ess = {json.loads(l)["id"]: json.loads(l) for l in (d / "essays.jsonl").open(encoding="utf-8")} if (d / "essays.jsonl").exists() else {}
        for l in (d / f"judgments_{JUDGE}.jsonl").open(encoding="utf-8"):
            j = json.loads(l); e = ess.get(j["id"], {}); v = r.replace("essaylab-", "")
            sk = j.get("skladowe") or {}
            rows.setdefault(v, {})[j["id"].split("_z")[0]] = (j["punkty"], e.get("words"), sk.get("elementy"), sk.get("bledy"), sk.get("spojnosc"), (j.get("uzasadnienie") or "")[:160])
    print(f"{'wariant':22s} " + " ".join(f"{y[:4]:>5s}" for y in YEARS) + "   śr.   słowa  el.śr  błędy  spójn")
    for v, per in rows.items():
        pts = [per[y][0] if y in per else None for y in YEARS]
        ws = [per[y][1] for y in YEARS if y in per and per[y][1]]
        els = [sum(per[y][2]) for y in YEARS if y in per and per[y][2]]; bl = [per[y][3] for y in YEARS if y in per and per[y][3] is not None]; sp = [per[y][4] for y in YEARS if y in per and per[y][4] is not None]
        avg = lambda xs: sum(xs) / max(1, len(xs))
        print(f"{v:22s} " + " ".join(f"{p:5}" if p is not None else "    -" for p in pts) + f"  {avg([p for p in pts if p is not None]):5.1f}  {avg(ws):6.0f}  {avg(els):5.1f}  {avg(bl):5.1f}  {avg(sp):5.1f}")
    if a.verbose:
        for v, per in rows.items():
            for y, (p, w, el, b, s, u) in per.items(): print(f"[{v} {y}] {p} pkt, {w} wyr., el={el} błędy={b} spójn={s}: {u}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "judge", "report", "polish"]); ap.add_argument("--out-tag", default="_CL")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS)); ap.add_argument("--tag", default="")
    ap.add_argument("--llm-url", default="http://localhost:18000/v1"); ap.add_argument("--llm-model", default="solver")
    ap.add_argument("--rag-url", default="http://localhost:18010")
    ap.add_argument("--temperature", type=float, default=0.0); ap.add_argument("--rep", type=float, default=1.05)
    ap.add_argument("--concurrency", type=int, default=6); ap.add_argument("--verbose", action="store_true"); ap.add_argument("--no-polish", action="store_true")
    a = ap.parse_args()
    {"gen": gen, "judge": judge, "report": report, "polish": polish_runs}[a.cmd](a)
