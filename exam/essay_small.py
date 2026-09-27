"""Wypracowanie dla małego modelu tekstowego (Bielik 1.5B, --text-only). Wnioski z small_model/essay_lab.py (12 tematów, sędzia Sol v4):
obecny prompt: 0.8 pkt/esej (<300 wyrazów, temat nie ten, co wybrany); ten moduł (wariant „facts5_years”): 4.0 pkt/esej, błędy 6 -> 3,
z regułą wyboru tematu 5.5 na 4 arkuszach (wyrocznia 5.5).
  - temat wybieramy sami (choose_topic): tylko tematy z trzema aspektami (tematy „trzech wybranych władców/wydarzeń” ~1.3 pkt:
    model bierze przykłady spoza ram czasowych), z nich najpierw nowożytne/nowoczesne (od XVIII w.: więcej faktów ogólnych w Wikipedii,
    mniej wymyślonych dat), potem bez nazwiska/nazwy własnej w tezie, potem pierwszy; model przy temp 0 i tak „wybiera” zawsze temat 1;
  - wstęp i zakończenie z szablonu (bez faktów = bez błędów, pewna struktura = 3 pkt za spójność), model pisze tylko trzy akapity,
    po jednym na aspekt; do każdego akapitu osobny retrieval (teza + aspekt) i tylko ZDANIA-FAKTY (rok albo nazwa własna,
    12–30 wyrazów, 5 na aspekt, lista numerowana), instrukcja na końcu promptu (mały model wykonuje to, co ostatnie);
  - po generacji: zdania z rokiem spoza notatek i tematu wylatują (3–5 błędów = -2 pkt u sędziego), akapit przycięty do ~130 wyrazów;
  - temperatura 0 (0.3/0.6 dawały więcej błędów); fragmenty w zwykłych zadaniach nie pomagają (próg 0.5 szkodzi), retrieval tylko tu.
Użycie: EssaySmall(chat, rag_url).write(question) -> tekst zaczynający się od „Temat nr N”; chat(system, user, max_tokens) -> str.
"""
import json, re, urllib.request

SYSTEM = """Jesteś maturzystą. Piszesz wypracowanie z historii po polsku pełnymi zdaniami, w akapitach. Nie używasz list, punktów, numeracji, myślników ani nagłówków. Podajesz tylko fakty, których jesteś pewien (daty, postacie, wydarzenia); nie zmyślasz szczegółów i trzymasz się okresu, którego dotyczy temat."""
BULLET = re.compile(r"(?m)^\s*(?:[-•*]|\d+[.)])\s+")
HEADING = re.compile(r"(?im)^\s*(aspekt[^:\n]{0,50}|podsumowanie|wnioski?|wstęp|zakończenie|rozwinięcie|argumenty?|stanowisko|teza)\s*:?\s*$")
INLINE_HEADING = re.compile(r"(?i)\s*(?:aspekt [^.:\n]{0,40}|podsumowanie|wnioski?)\s*:\s*")
YEAR = re.compile(r"\b(\d{3,4})\b")
NAME = re.compile(r"\b[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]{2,}\b")
ROMAN = {"I": 1, "V": 5, "X": 10}
MODERN_WORDS = ("rewolucj", "wojna światowa", "wojny światowej", "komunist", "zimna wojna", "zimnej wojny", "belle", "przemysłow", "niepodległ", "okupac", "prl", "solidarno")


def words(t):
    return len((t or "").split())


def parse_topics(question):
    """{n: treść tematu} z polecenia (tematy 1.–3. przed „WYPRACOWANIE”/„BRUDNOPIS”)."""
    body = re.split(r"WYPRACOWANIE|BRUDNOPIS", question)[0]
    parts = re.split(r"(?m)^\s*([123])\.\s+", body)
    return {int(parts[i]): re.sub(r"\s+", " ", parts[i + 1]).strip() for i in range(1, len(parts) - 1, 2)}


def thesis(topic):
    return topic.split("Zajmij stanowisko")[0].strip().rstrip(".")


def elements(topic):
    """('aspekty', [trzy przymiotniki]) albo ('wybór', 'trzech wybranych władców z tego okresu')."""
    m = re.search(r"aspekt(?:y|ów)?:?\s*([^.]+)", topic)
    if m and "," in m.group(1):
        raw = re.split(r",\s*|\s+i\s+|\s+oraz\s+", m.group(1).strip().rstrip("."))
        return "aspekty", [re.sub(r"\s*-\s*", "-", a.strip(" -")) for a in raw if a.strip()][:3]
    m = re.search(r"(trz(?:ech|y)\s+wybran\w+[^.]*)", topic)
    return "wybór", (m.group(1).strip() if m else "trzy wybrane przykłady")


def roman(s):
    total, prev = 0, 0
    for ch in reversed(s):
        v = ROMAN.get(ch, 0); total += -v if v < prev else v; prev = max(prev, v)
    return total


def is_modern(th):
    """Okres od XVIII wieku (rok >= 1700, wiek >= XVIII albo słowa-klucze epoki nowoczesnej)."""
    yrs = [int(y) for y in YEAR.findall(th) if 800 <= int(y) <= 2030]
    if yrs and max(yrs) >= 1700: return True
    cents = [roman(c) for c in re.findall(r"\b([IVX]{1,5})\s*(?:wiek|w\.)", th)]
    if cents and max(cents) >= 18: return True
    return any(w in th.lower() for w in MODERN_WORDS)


def has_proper(th):
    """Nazwa własna w tezie poza pierwszym słowem (władca, państwo, miejsce): tematy o osobach wychodzą gorzej."""
    return bool(NAME.search(th[1:]))


def choose_topic(topics):
    """Aspektowy > nowożytny/nowoczesny > bez nazwy własnej > niższy numer."""
    def score(n):
        t = topics[n]; th = thesis(t)
        return ((elements(t)[0] == "aspekty"), is_modern(th), not has_proper(th), -n)
    return max(topics, key=score) if topics else 1


def decl(adj):  # narzędnik: militarny -> militarnym, społeczno-gospodarczy -> społeczno-gospodarczym
    return adj[:-1] + "ym" if adj.endswith("y") else adj[:-1] + "im" if adj.endswith("i") else adj


def gen(adj):  # dopełniacz: militarny -> militarnego
    return adj[:-1] + "ego" if adj.endswith("y") else adj[:-1] + "iego" if adj.endswith("i") else adj


def clean(t):
    t = re.sub(r"\*\*|__|^#+ ", "", t or "", flags=re.M)
    t = re.sub(r"(?i)^\s*temat\s*(nr\.?)?\s*\d\s*[.:]?\s*\n+", "", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def debullet(text):
    out, buf = [], []
    for line in text.split("\n"):
        if BULLET.match(line): buf.append(BULLET.sub("", line).strip().rstrip(".") + ".")
        else:
            if buf: out.append(" ".join(buf)); buf = []
            out.append(line)
    if buf: out.append(" ".join(buf))
    return "\n".join(out).strip()


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def fact_sentences(hits, limit=5, max_words=30):
    """Z fragmentów Wikipedii zostają tylko zdania z rokiem albo nazwą własną (nie na początku zdania), 12–30 wyrazów, bez powtórek."""
    out, seen = [], set()
    for h in hits:
        for sent in sentences(re.sub(r"\s+", " ", h["text"])):
            if not (12 <= words(sent) <= max_words) or sent[-1] not in ".!?": continue
            if not (YEAR.search(sent) or NAME.search(sent[1:])): continue
            key = sent.lower()[:60]
            if key in seen: continue
            seen.add(key); out.append(sent)
            if len(out) >= limit: return out
    return out


def year_filter(par, allowed_text):
    """Zdania z rokiem (800–2030), którego nie ma w notatkach ani temacie, wylatują; za mało zostaje = zostaw oryginał."""
    yrs = set(YEAR.findall(allowed_text)); kept = []
    for s in sentences(par):
        if any(y not in yrs and 800 <= int(y) <= 2030 for y in YEAR.findall(s)): continue
        kept.append(s)
    out = " ".join(kept)
    return (out, True) if words(out) >= 40 else (par, False)


def trim_words(text, limit):
    if words(text) <= limit: return text
    out, n = [], 0
    for s in sentences(text):
        if n >= limit: break
        out.append(s); n += words(s)
    return " ".join(out)


def polish(body):
    """Nagłówki w środku tekstu, powtórzone zdania, urwane ostatnie zdanie (limit tokenów)."""
    body = HEADING.sub("", body); body = INLINE_HEADING.sub(" ", body)
    paras, seen = [], set()
    for para in re.split(r"\n\s*\n", body):
        sents = []
        for sent in sentences(para):
            key = re.sub(r"\W+", " ", sent.lower()).strip()[:70]
            if len(key) < 15 or key in seen: continue
            seen.add(key); sents.append(sent)
        if sents: paras.append(" ".join(sents))
    body = "\n\n".join(paras).strip()
    if body and body[-1] not in ".!?»”\"":
        cut = max(body.rfind(". "), body.rfind(".\n"), body.rfind("!"), body.rfind("?"))
        if cut > 0.6 * len(body): body = body[:cut + 1]
    return body


class EssaySmall:
    def __init__(self, chat, rag_url=None, rag_k=6, rag_min_score=0.3, n_facts=5, para_tokens=320):
        self.chat, self.rag_url, self.k, self.min_score, self.n_facts, self.para_tokens = chat, rag_url, rag_k, rag_min_score, n_facts, para_tokens

    def facts(self, th, element, debug):
        """Zdania-fakty do jednego aspektu: retrieval (teza + aspekt, sama teza), potem filtr zdań."""
        if not self.rag_url: return []
        try:
            body = json.dumps({"queries": [f"{th} – {element}", th], "k": self.k, "min_score": self.min_score}).encode()
            with urllib.request.urlopen(urllib.request.Request(self.rag_url.rstrip("/") + "/search", data=body, headers={"Content-Type": "application/json"}), timeout=120) as r:
                hits = json.load(r).get("hits", [])
            facts = fact_sentences(hits, self.n_facts)
            debug.setdefault("essay_facts", []).append({"el": element, "hits": [h["title"] for h in hits][:4], "n": len(facts)})
            return facts
        except Exception as e:  # retrieval to tylko pomoc
            debug["essay_rag_error"] = f"{type(e).__name__}: {str(e)[:120]}"; return []

    def examples(self, topic, els, debug):
        """Temat „trzech wybranych …” (tylko gdy nie ma tematu aspektowego): model podaje same nazwy."""
        raw = self.chat("Odpowiadasz po polsku, tylko listą nazw po przecinku, bez komentarza.",
                        f"Temat wypracowania: {topic}\n\nPodaj {els}, o których wiesz najwięcej pewnych faktów i które pasują do okresu tematu. Tylko trzy nazwy po przecinku.", 60)
        names = [x.strip(" .\n-") for x in re.split(r",|;|\n", raw or "") if x.strip(" .\n-")][:3]
        names = (names + ["pierwszy przykład", "drugi przykład", "trzeci przykład"])[:3]
        debug["essay_names"] = names
        return names

    def paragraph(self, topic, th, stance, kind, element, used, debug):
        facts = self.facts(th, element, debug); notes = " ".join(facts)
        lead = f"W aspekcie {decl(element)}" if kind == "aspekty" else element
        block = ("Fakty z encyklopedii do tego akapitu (korzystaj tylko z nich):\n" + "\n".join(f"{i + 1}. {f}" for i, f in enumerate(facts)) + "\n\n") if facts else ""
        usedtxt = ("Fakty już użyte we wcześniejszych akapitach (nie powtarzaj): " + "; ".join(used[-6:]) + "\n\n") if used else ""
        # kolejność: notatki -> użyte fakty -> temat i stanowisko -> instrukcja NA KOŃCU (efekt świeżości u małych modeli)
        u = block + usedtxt + f"Temat: {topic}\nStanowisko: {stance}.\n\nNapisz jeden akapit wypracowania (90–120 wyrazów, pełnymi zdaniami, bez listy) o elemencie: {element}. Zacznij od słów „{lead}”, użyj dwóch lub trzech faktów z listy powyżej i zakończ zdaniem, jak ten element potwierdza tezę. Tylko tekst akapitu."
        p = clean(self.chat(SYSTEM, u, self.para_tokens))
        if len(BULLET.findall(p)) >= 2: p = debullet(p)
        if words(p) < 50:
            p2 = clean(self.chat(SYSTEM, u, self.para_tokens * 2)); p = p2 if words(p2) > words(p) else p
        p, filtered = year_filter(p, notes + " " + topic)
        debug.setdefault("essay_filter", []).append(filtered)
        return trim_words(p, 130)

    def write(self, question, debug=None):
        debug = debug if debug is not None else {}
        topics = parse_topics(question) or {1: re.sub(r"\s+", " ", question)[:600]}
        n = choose_topic(topics); topic = topics[n]; kind, els = elements(topic); th = thesis(topic)
        debug["essay_topic"] = {"n": n, "kind": kind, "elements": els, "modern": is_modern(th), "proper": has_proper(th)}
        names = els if kind == "aspekty" else self.examples(topic, els, debug)
        stance = f"Zgadzam się z tezą: „{th}”"
        if kind == "aspekty":
            intro = f"{stance}. W wypracowaniu uzasadnię to stanowisko, omawiając kolejno trzy aspekty: {els[0]}, {els[1]} i {els[2]}."
            outro = f"Podsumowując, analiza aspektu {gen(els[0])}, {gen(els[1])} i {gen(els[2])} prowadzi do wniosku, że teza „{th}” jest słuszna. Dlatego podtrzymuję stanowisko zajęte we wstępie."
        else:
            intro = f"{stance}. W wypracowaniu uzasadnię to stanowisko, omawiając kolejno trzy przykłady: {names[0]}, {names[1]} i {names[2]}."
            outro = f"Podsumowując, omówione przykłady ({names[0]}, {names[1]} i {names[2]}) prowadzą do wniosku, że teza „{th}” jest słuszna. Dlatego podtrzymuję stanowisko zajęte we wstępie."
        paras, used = [], []
        for e in names:
            p = self.paragraph(topic, th, stance, kind, e, used, debug)
            paras.append(p); used += YEAR.findall(p)[:4]
        body = polish("\n\n".join(paras))
        text = f"{intro}\n\n{body}\n\n{outro}"
        for i in range(2):  # poniżej 300 wyrazów = 0 pkt za spójność: akapit uzupełniający o kolejnym elemencie
            if words(text) >= 320: break
            e = names[i % 3]; lead = f"Warto dodać, że w aspekcie {decl(e)}" if kind == "aspekty" else f"Wracając do przykładu {e}"
            u = f"Temat: {topic}\nStanowisko: {stance}.\n\nNapisz jeszcze jeden akapit wypracowania (90–110 wyrazów, pełnymi zdaniami, bez listy) o elemencie: {e}, z innymi faktami niż dotąd. Zacznij od słów „{lead}”. Tylko tekst akapitu."
            p = clean(self.chat(SYSTEM, u, self.para_tokens)); p, _ = year_filter(p, topic)
            if words(p) >= 40: paras.append(trim_words(p, 130)); body = polish("\n\n".join(paras)); text = f"{intro}\n\n{body}\n\n{outro}"
            debug.setdefault("essay_added", []).append(words(p))
        debug["essay_parts_words"] = [words(intro)] + [words(p) for p in paras] + [words(outro)]
        return f"Temat nr {n}\n\n{text}"
