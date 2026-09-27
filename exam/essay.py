"""Wypracowanie maturalne (zadanie 26): wybór tematu przez model (dosłowne przepisanie, sprawdzane jako podciąg polecenia),
struktura wstęp / rozwinięcie w 3 elementach / zakończenie, dwa warianty pisania:
  single  - jeden prompt z pełną strukturą i limitami słów
  staged  - plan (stanowisko + 3 elementy z faktami) -> wstęp -> rozwinięcie -> zakończenie, każdy etap osobnym zapytaniem
Zwraca gotowy tekst zaczynający się od "Temat nr N". Używane przez solver_server (single-vlm, vlm-text) i eval/essay_test.py.
"""
import re

TOPIC_SYSTEM = "Przepisujesz fragment tekstu dosłownie. Nie parafrazujesz, nie skracasz, nie komentujesz."
TOPIC_USER = """Poniżej treść zadania z trzema tematami wypracowania. Przepisz DOSŁOWNIE, bez zmian, skrótów i komentarza, pełną treść tematu nr {n}: od pierwszego słowa po numerze do końca tego tematu (wraz ze zdaniem „Zajmij stanowisko…”). Nie dopisuj nic więcej.

{question}"""

CHOOSE_SYSTEM = "Jesteś maturzystą wybierającym temat wypracowania z historii. Odpowiadasz wyłącznie jedną cyfrą."
CHOOSE_USER = """Poniżej zadanie z trzema tematami wypracowania. Wybierz temat, do którego znasz NAJWIĘCEJ pewnych faktów (dat, nazwisk, wydarzeń, dokumentów) mieszczących się w jego ramach czasowych i który potrafisz omówić we wszystkich trzech wymaganych elementach. Odpowiedz tylko cyfrą 1, 2 albo 3.

{question}"""

ESSAY_SYSTEM = """Jesteś maturzystą piszącym wypracowanie z historii na egzaminie maturalnym (poziom rozszerzony, CKE). Piszesz po polsku, ciągłym tekstem w akapitach, bez nagłówków, bez list, bez markdownu. Nie wypisujesz planu ani komentarzy, tylko gotowy tekst.
Zasady faktów: podajesz tylko fakty, których jesteś pewien; przy niepewnej dacie używasz ogólniejszego określenia (np. „w latach 60.”) zamiast zgadywać. Każdy błąd faktograficzny obniża ocenę. Trzymasz się ram czasowych tematu."""

SINGLE_USER = """Temat nr {n}: {topic}

Napisz wypracowanie o długości {total} wyrazów według tej struktury:
1. WSTĘP ({intro} wyrazów): w pierwszym zdaniu jednoznaczne stanowisko wobec tezy (zgadzam się / nie zgadzam się / zgadzam się częściowo, bez wahania), ramy czasowe i terytorialne tematu, zapowiedź trzech elementów, które omówisz.
2. ROZWINIĘCIE ({body} wyrazów, trzy akapity, po jednym na każdy z trzech elementów wskazanych w temacie). W każdym akapicie: co najmniej dwa konkretne fakty (daty, nazwiska, nazwy wydarzeń, dokumentów, instytucji) i wyraźne zdanie, jak ten element potwierdza lub podważa tezę.
3. ZAKOŃCZENIE ({outro} wyrazów): wniosek wynikający z trzech akapitów, powtórzenie stanowiska ze wstępu, bez nowych faktów.

Zacznij od linii „Temat nr {n}”, potem tekst."""

PLAN_SYSTEM = """Jesteś maturzystą przygotowującym plan wypracowania z historii (poziom rozszerzony, CKE). Odpowiadasz po polsku, zwięźle, w punktach. Podajesz tylko fakty, których jesteś pewien, mieszczące się w ramach czasowych tematu."""

PLAN_USER = """Temat nr {n}: {topic}

Przygotuj plan wypracowania (tylko plan, bez tekstu wypracowania):
STANOWISKO: jedno zdanie (zgadzam się / nie zgadzam się / zgadzam się częściowo).
ELEMENT 1: nazwa elementu wskazanego w temacie; 2–3 konkretne fakty (daty, nazwiska, wydarzenia), których jesteś pewien; jedno zdanie, jak element odnosi się do tezy.
ELEMENT 2: jak wyżej.
ELEMENT 3: jak wyżej.
Elementy muszą mieścić się w ramach czasowych tematu. Jeśli temat każe wybrać (np. trzech władców, trzy wydarzenia), wybierz te, o których wiesz najwięcej."""

INTRO_USER = """Temat nr {n}: {topic}

Plan:
{plan}

Napisz WSTĘP wypracowania ({intro} wyrazów): w pierwszym zdaniu jednoznaczne stanowisko wobec tezy, ramy czasowe i terytorialne, zapowiedź trzech elementów z planu. Tylko wstęp, bez nagłówka."""

BODY_USER = """Temat nr {n}: {topic}

Plan:
{plan}

Wstęp (już napisany):
{intro}

Napisz ROZWINIĘCIE ({body} wyrazów): trzy akapity, po jednym na każdy element z planu, w kolejności z planu. W każdym akapicie co najmniej dwa konkretne fakty z planu i wyraźne zdanie, jak ten element potwierdza lub podważa tezę. Nie powtarzaj wstępu, nie pisz zakończenia. Tylko trzy akapity, bez nagłówków."""

OUTRO_USER = """Temat nr {n}: {topic}

Wstęp i rozwinięcie (już napisane):
{intro}

{body}

Napisz ZAKOŃCZENIE ({outro} wyrazów): wniosek wynikający z trzech akapitów rozwinięcia i powtórzenie stanowiska ze wstępu. Bez nowych faktów. Tylko zakończenie, bez nagłówka."""

EXTEND_USER = """Ten fragment ma {words} wyrazów, a powinien mieć {target}. Napisz go jeszcze raz w całości, rozbudowując argumentację o kolejne pewne fakty i ich znaczenie dla tezy. Nie zmieniaj stanowiska. Tylko tekst fragmentu."""


def _norm(s):
    return re.sub(r"\s+", " ", s or "").strip()


def _clean(text):
    t = re.sub(r"\*\*|__|#+ ", "", text or "")
    t = re.sub(r"(?im)^\s*(wstęp|rozwinięcie|zakończenie|plan)\s*:?\s*$", "", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def words(text):
    return len((text or "").split())


class EssayWriter:
    """chat(system, user, max_tokens) -> tekst; parametry długości w wyrazach."""

    def __init__(self, chat, n_topic=1, total=(380, 420), intro=(50, 60), body=(270, 300), outro=(50, 60), essay_max_tokens=3000, small_tokens=800):
        self.chat, self.n, self.tot, self.intro, self.body, self.outro, self.max_tokens = chat, n_topic, total, intro, body, outro, essay_max_tokens
        self.small_tokens = small_tokens  # etapy krótkie (wybór, temat, wstęp, zakończenie); z reasoningiem ustaw >= 3000

    def choose_topic(self, question, debug=None):
        """Model wskazuje numer tematu, który zna najlepiej; przy niejednoznacznej odpowiedzi zostaje self.n."""
        txt = self.chat(CHOOSE_SYSTEM, CHOOSE_USER.format(question=question), self.small_tokens)
        m = re.search(r"[123]", (txt or "").strip()[-20:] or txt or "")
        n = int(m.group(0)) if m else self.n
        if debug is not None: debug["topic_choice"] = {"raw": (txt or "")[-60:], "n": n}
        return n

    def extract_topic(self, question, debug=None):
        """Model przepisuje temat nr n dosłownie; sprawdzamy, że wynik jest podciągiem polecenia (po normalizacji spacji)."""
        q_norm = _norm(question)
        for attempt in range(2):
            txt = _norm(self.chat(TOPIC_SYSTEM, TOPIC_USER.format(n=self.n, question=question), self.small_tokens))
            txt = re.sub(r"^(temat\s*(nr\.?)?\s*)?\d\s*[.):]\s*", "", txt, flags=re.I).strip()
            if len(txt) > 40 and txt in q_norm:
                if debug is not None: debug["topic_extract"] = "ok" if attempt == 0 else "ok_retry"
                return txt
        if debug is not None: debug["topic_extract"] = "fallback_full_question"
        return question  # nie udało się: model dostaje całe polecenie, jak dotąd

    def _rng(self, r):
        return f"{r[0]}–{r[1]}"

    def _ensure_len(self, system, user, text, lo, debug_key, debug):
        """Za krótki fragment: jedna prośba o rozbudowanie; przyjmujemy, jeśli dłuższy. Pusty tekst: ponowna próba z 2x budżetem."""
        if not text.strip():
            text = _clean(self.chat(system, user, self.max_tokens * 2))
            if debug is not None: debug[debug_key + "_retry_empty"] = words(text)
        if words(text) >= lo: return text
        target = f"{lo}+"
        again = _clean(self.chat(system, user + "\n\n" + EXTEND_USER.format(words=words(text), target=target), self.max_tokens))
        if debug is not None: debug[debug_key] = {"before": words(text), "after": words(again)}
        return again if words(again) > words(text) else text

    def write_single(self, topic, debug=None):
        user = SINGLE_USER.format(n=self.n, topic=topic, total=self._rng(self.tot), intro=self._rng(self.intro), body=self._rng(self.body), outro=self._rng(self.outro))
        text = _clean(self.chat(ESSAY_SYSTEM, user, self.max_tokens))
        text = self._ensure_len(ESSAY_SYSTEM, user, text, 300 + 40, "extend", debug)
        return self._with_header(text)

    def write_staged(self, topic, debug=None):
        plan = _clean(self.chat(PLAN_SYSTEM, PLAN_USER.format(n=self.n, topic=topic), max(800, self.small_tokens)))
        u = INTRO_USER.format(n=self.n, topic=topic, plan=plan, intro=self._rng(self.intro))
        intro = self._ensure_len(ESSAY_SYSTEM, u, _clean(self.chat(ESSAY_SYSTEM, u, self.small_tokens)), self.intro[0] - 10, "extend_intro", debug)
        u = BODY_USER.format(n=self.n, topic=topic, plan=plan, intro=intro, body=self._rng(self.body))
        body = self._ensure_len(ESSAY_SYSTEM, u, _clean(self.chat(ESSAY_SYSTEM, u, self.max_tokens)), self.body[0] - 30, "extend_body", debug)
        u = OUTRO_USER.format(n=self.n, topic=topic, intro=intro, body=body, outro=self._rng(self.outro))
        outro = self._ensure_len(ESSAY_SYSTEM, u, _clean(self.chat(ESSAY_SYSTEM, u, self.small_tokens)), self.outro[0] - 10, "extend_outro", debug)
        if debug is not None:
            debug["plan"] = plan; debug["parts_words"] = [words(intro), words(body), words(outro)]
        return self._with_header(f"{intro}\n\n{body}\n\n{outro}")

    def _with_header(self, text):
        text = re.sub(r"(?i)^\s*temat\s*(nr\.?)?\s*\d\s*[.:]?\s*\n+", "", text).strip()
        return f"Temat nr {self.n}\n\n{text}"

    def write(self, question, mode="single", debug=None):
        if mode == "auto":  # wybór tematu przez model, potem jeden prompt
            self.n = self.choose_topic(question, debug); mode = "single"
        topic = self.extract_topic(question, debug)
        if debug is not None: debug["topic"] = topic[:300]
        return self.write_staged(topic, debug) if mode == "staged" else self.write_single(topic, debug)
