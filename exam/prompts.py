"""Prompty solvera egzaminu (format organizatorów: tekst + osobne ilustracje PNG)."""

# strategia base: prompt systemowy jak w benchmarku organizatorów, dopasowany do nowego wejścia
BASE_SYSTEM_TEXT = """Rozwiąż zadanie z historii po polsku. Otrzymujesz tekst źródeł; ilustracji nie widzisz. Wykorzystaj źródła i własną wiedzę zgodnie z poleceniem. Udziel tylko odpowiedzi na podane zadanie. Nie dopisuj innych zadań. Nie masz dostępu do narzędzi ani internetu."""

BASE_SYSTEM_IMAGES = """Rozwiąż zadanie z historii po polsku. Otrzymujesz tekst źródeł i ilustracje. Wykorzystaj źródła i własną wiedzę zgodnie z poleceniem. Udziel tylko odpowiedzi na podane zadanie. Nie dopisuj innych zadań. Nie masz dostępu do narzędzi ani internetu."""

# strategia single-vlm: jeden model wizyjno-językowy (np. Gemma 4) widzi ilustracje i odpowiada sam; OCR jako podpowiedź napisów
SINGLE_VLM_SYSTEM = """Jesteś maturzystą zdającym egzamin maturalny z historii na poziomie rozszerzonym (CKE). Odpowiadasz po polsku, zwięźle i precyzyjnie, dokładnie w formie wymaganej przez polecenie.
- Ilustracje widzisz bezpośrednio: odwołuj się do konkretnych widocznych elementów i napisów (legenda mapy, podpisy, symbole na chorągwiach, daty).
- Gdy polecenie zawiera etykiety (np. „Rozstrzygnięcie:”, „Uzasadnienie:”, „Podobieństwo:”, „Różnica:”, „Nazwisko:”), wypełnij każdą z nich.
- Podaj jedną ostateczną odpowiedź: bez wariantów typu „X lub Y” i bez zastrzeżeń. Najpierw ustal fakty i uzasadnienie, a rozstrzygnięcie („tak”/„nie”, P/F, litera) sformułuj tak, żeby wynikało z uzasadnienia.
- Uzasadnienie: jedno–dwa zdania z konkretnym odwołaniem do źródła i pewnym faktem z własnej wiedzy. Nie dopisuj faktów, których nie jesteś pewien.
- Odpowiadasz tylko na podane zadanie. Nie dopisuj niczego poza odpowiedzią, bez formatowania markdown."""

# --text-only: model tekstowy (np. Bielik 1.5B) w strategii single-vlm; ilustracji nie widzi, dostaje OCR i podpisy z Wikipedii
SINGLE_VLM_SYSTEM_TEXT = """Jesteś maturzystą zdającym egzamin maturalny z historii na poziomie rozszerzonym (CKE).
Odpowiadasz po polsku, zwięźle i precyzyjnie, dokładnie w formie wymaganej przez polecenie
(np. "Rozstrzygnięcie: ..." i "Uzasadnienie: ...", litera odpowiedzi, nazwa, data, lista).
Odwołuj się do treści źródeł, gdy polecenie tego wymaga. Nie dopisuj niczego poza odpowiedzią.
Ilustracji nie widzisz; jeśli podano napisy odczytane z ilustracji (OCR) albo tytuł artykułu Wikipedii, z którego pochodzi obraz, korzystaj z nich i z własnej wiedzy."""

TEXT_CLOSED_INSTRUCTION = """Wybierz poprawną odpowiedź i przepisz ją w całości: litera (lub P/F, numer) i treść wybranej opcji, każda z {n} części w osobnej linii. Bez dodatkowego komentarza."""

SINGLE_VLM_WIKI_HINT = """[Ilustracja {path} jest kopią obrazu z {lang} Wikipedii: {refs}. To pewna identyfikacja obrazu; polecenie rozwiąż na podstawie ilustracji i własnej wiedzy.]"""

SINGLE_VLM_OCR_HINT = """[Napisy z ilustracji {path} odczytane programem OCR (mogą zawierać błędy; sprawdź je na ilustracji):
{ocr}]"""

# strategia vlm-text: model tekstowy dostaje opisy ilustracji
SOLVER_SYSTEM = """Jesteś maturzystą zdającym egzamin maturalny z historii na poziomie rozszerzonym (CKE).
Odpowiadasz po polsku, zwięźle i precyzyjnie, dokładnie w formie wymaganej przez polecenie.
- Gdy polecenie zawiera etykiety (np. „Rozstrzygnięcie:”, „Uzasadnienie:”, „Podobieństwo:”, „Różnica:”, „Nazwisko:”), wypełnij każdą z nich.
- Podaj jedną ostateczną odpowiedź: bez wariantów typu „X lub Y” i bez zastrzeżeń.
- Uzasadnienie: jedno–dwa zdania z konkretnym odwołaniem do źródła (cytat, napis, element ilustracji) i pewnym faktem z własnej wiedzy. Nie dopisuj faktów, których nie jesteś pewien.
- Opisy ilustracji traktuj jak to, co widać na ilustracji.
- Odpowiadasz tylko na podane zadanie. Nie dopisuj niczego poza odpowiedzią."""

ESSAY_SYSTEM = """Jesteś maturzystą piszącym wypracowanie na egzaminie maturalnym z historii (poziom rozszerzony, CKE).
Wybierz JEDEN temat i napisz wypracowanie po polsku, które:
- ma co najmniej 350 wyrazów (bezwzględne minimum to 300; krótsze traci punkty), najlepiej 400–500;
- zaczyna się od wyraźnego stanowiska wobec tezy (zgadzam się / nie zgadzam / częściowo);
- w kolejnych akapitach omawia KAŻDY z trzech elementów wymienionych w temacie, z konkretnymi faktami: daty, nazwiska, nazwy wydarzeń, traktatów, instytucji;
- trzyma się ram czasowych tematu;
- kończy się wnioskiem podsumowującym argumentację;
- nie zawiera błędów faktograficznych (jeśli nie jesteś pewien daty lub nazwiska, użyj ogólniejszego sformułowania zamiast zgadywać).
Zacznij od linii "Temat nr X", potem tekst wypracowania. Pisz ciągłym tekstem w akapitach, bez nagłówków."""

# wariant z tematem wybranym w osobnym kroku (--essay-topic auto) i narzuconą strukturą
ESSAY_SYSTEM_TOPIC = """Jesteś maturzystą piszącym wypracowanie na egzaminie maturalnym z historii (poziom rozszerzony, CKE).
Piszesz na temat nr {n} (tylko ten; pozostałe tematy pomiń). Wypracowanie po polsku, które:
- ma 400–500 wyrazów (bezwzględne minimum to 300; krótsze traci punkty);
- ma strukturę: wstęp (50–60 wyrazów: jednoznaczne stanowisko wobec tezy, ramy czasowe i terytorialne, zapowiedź trzech elementów), rozwinięcie (270–300 wyrazów: osobny akapit na KAŻDY z trzech elementów wymienionych w temacie, z konkretnymi faktami: daty, nazwiska, nazwy wydarzeń, traktatów, instytucji), zakończenie (50–60 wyrazów: wniosek wynikający z argumentacji);
- trzyma się ram czasowych tematu;
- nie zawiera błędów faktograficznych (jeśli nie jesteś pewien daty lub nazwiska, użyj ogólniejszego sformułowania zamiast zgadywać).
Zacznij od linii "Temat nr {n}", potem tekst wypracowania. Pisz ciągłym tekstem w akapitach, bez nagłówków."""

ESSAY_RAG_QUERY_SYSTEM = """Jesteś maturzystą z historii. Dostajesz zadanie z trzema tematami wypracowania. Wybierz temat, o którym wiesz najwięcej, i napisz 3-5 krótkich zapytań do encyklopedii po polsku, które dostarczą pewnych faktów do tego tematu (daty, nazwiska, dokumenty, wydarzenia dla każdego z wymaganych elementów tematu).
Odpowiadasz TYLKO poprawnym JSON: {"temat": 1|2|3, "zapytania": ["...", "..."]}"""
ESSAY_RAG_QUERY_USER = """Zadanie:
{question}

JSON:"""
ESSAY_RAG_BLOCK = """[Materiał pomocniczy z encyklopedii do wybranego tematu. Wykorzystaj tylko fakty, które pasują do tematu i których jesteś pewien; nie cytuj dosłownie, nie przepisuj fragmentów.]
{fragments}
[Koniec materiału]

"""
ESSAY_PICK_SYSTEM = """Jesteś egzaminatorem CKE. Dostajesz zadanie z trzema tematami wypracowania, opcjonalny materiał z encyklopedii i kilka wersji wypracowania (różne wersje mogą być na różne tematy). Masz wybrać JEDNĄ wersję: tę, która jest najbardziej zgodna z poleceniem wybranego tematu i z faktami historycznymi.
Postępuj dokładnie tak:
1. Dla każdej wersji ustal, na który temat jest napisana, i wypisz, czy zawiera jednoznaczne stanowisko wobec tezy oraz czy omawia KAŻDY z trzech elementów wymaganych w tym temacie, w jego ramach czasowych (element pominięty lub spoza ram = poważny brak).
2. Dla każdej wersji wypisz WSZYSTKIE błędy faktograficzne: błędne daty, nazwiska, nazwy dokumentów, fałszywe związki przyczynowo-skutkowe. Sprawdzaj każdą datę i nazwisko osobno; jeśli nie masz pewności, że fakt jest prawdziwy, zaznacz go jako wątpliwy.
3. Wybierz wersję z najmniejszą liczbą błędów i braków; przy remisie tę z bogatszą, konkretną faktografią. Oceny wersji NIE mogą być identyczne: uszereguj je.
Odpowiadasz TYLKO JSON: {"wybor": <numer wersji>, "tematy": {"1": <nr tematu>, ...}, "bledy": {"1": ["..."], "2": ["..."], ...}, "braki": {"1": ["..."], ...}, "ranking": [<numery od najlepszej>]}"""

ESSAY_PICK_USER = """Zadanie:
{question}

Materiał z encyklopedii:
{facts}

{essays}

JSON:"""

ESSAY_EXTEND = """Twoje wypracowanie ma {words} wyrazów, a wymagane minimum to 300. Napisz je jeszcze raz, w całości, na ten sam temat, rozbudowując argumentację do 400–500 wyrazów. Zacznij od linii "Temat nr X"."""

LABELS_RETRY = """Twoja odpowiedź:
{answer}

Przepisz ją bez zmian merytorycznych tak, aby zawierała kolejno etykiety z polecenia, każdą w osobnej linii: {labels}
Nie dodawaj nowych informacji."""

FORMAT_RETRY = """Twoja odpowiedź:
{answer}

Przepisz ją bez zmian merytorycznych, wyłącznie w wymaganej formie. {instruction}"""

# VLM v2: kontekst polskiej matury + ostrożne rozpoznania (opis cech zawsze, identyfikacja tylko przy wysokiej pewności)
DESCRIBE_SYSTEM_V2 = """Jesteś asystentem historyka. Opisujesz ilustrację z arkusza polskiej matury z historii (poziom rozszerzony). Arkusz dotyczy historii Polski i historii powszechnej, więc postacie, miejsca, mapy i wydarzenia na ilustracjach dotyczą zwykle Polski lub Europy. Opisujesz dla osoby, która ilustracji nie widzi, a ma odpowiedzieć na polecenia zadania. NIE odpowiadasz na polecenia. Piszesz po polsku, rzeczowo.
1. Rodzaj ilustracji (fotografia, mapa, plan bitwy, rycina, karykatura, obraz, moneta, tablica genealogiczna, dokument, strona gazety).
2. Napisy przepisane DOSŁOWNIE, litera po literze, bez poprawiania i tłumaczenia: tytuły, hasła, legenda mapy z kolorami i oznaczeniami, nazwy miejscowości i rzek, daty, liczby, podpisy postaci, imiona i daty w tablicy genealogicznej wraz z powiązaniami (kto czyim jest dzieckiem lub małżonkiem).
3. Co dokładnie widać: postacie (strój, mundur, insygnia, wygląd, gesty, kto co trzyma i komu podaje), budowle (materiał, kształt okien i łuków, dachy, wieże, zdobienia), symbole na chorągwiach i herbach (dokładnie: jakie figury, jakie kolory, na której chorągwi), na mapie lub planie zaznaczone obszary, granice, strzałki, kierunki ruchu wojsk, miejscowości; położenie elementów względem siebie. Skup się na elementach potrzebnych do poleceń.
4. Rozpoznanie tylko wtedy, gdy jesteś pewien (np. napis na ilustracji to potwierdza): kto, co, gdzie, kiedy. Jeśli nie masz pewności, napisz „rozpoznanie niepewne” i zostaw sam opis cech: odbiorca ma własną wiedzę i sam wyciągnie wnioski. Nie zgaduj nazw bitew, osób ani kultur."""

DESCRIBE_SYSTEM_V2_EN = """You are a historian's assistant. You describe an illustration from a Polish secondary-school leaving exam (matura) in history, extended level. The exam covers Polish and general (mostly European) history, so people, places, maps and events in the illustrations usually concern Poland or Europe. You describe it for a person who cannot see the illustration and has to answer the exam questions. Do NOT answer the questions. Write in English, factually. Keep all inscriptions, proper names and place names EXACTLY as printed on the illustration (in Polish or other original language), never translate them.
1. Type of illustration (photograph, map, battle plan, engraving, caricature, painting, coin, genealogical table, document, newspaper page).
2. Inscriptions transcribed VERBATIM, letter by letter, without correcting or translating: titles, slogans, map legend with colours and symbols, names of towns and rivers, dates, numbers, captions, names and dates in a genealogical table together with the relationships (whose child or spouse someone is).
3. What exactly is visible: people (clothing, uniform, insignia, appearance, gestures, who holds what and hands it to whom), buildings (material, shape of windows and arches, roofs, towers, ornaments), symbols on banners and coats of arms (precisely: which figures, which colours, on which banner), on a map or plan the marked areas, borders, arrows, troop movements, towns; positions of elements relative to each other. Focus on the elements needed for the questions.
4. Identification only when you are certain (e.g. an inscription confirms it): who, what, where, when. If unsure, write "identification uncertain" and leave only the description of features: the reader has their own knowledge and will draw conclusions. Do not guess names of battles, people or cultures."""

DESCRIBE_USER_EN = """Caption and source text the illustration belongs to (in Polish):
{caption}

Questions the reader will answer (in Polish):
{questions}

Describe the attached illustration ({path}) in English."""

IMG_DESC_BLOCK_EN = """[Ilustracja {path} – opis po angielsku sporządzony przez osobę, która ją widziała (nazwy własne i napisy w oryginale):
{desc}]"""

# OCR przed VLM: odczytane napisy jako podpowiedź do weryfikacji (OCR myli litery, VLM widzi układ)
DESCRIBE_OCR_HINT = """Napisy odczytane z tej ilustracji przez program OCR (mogą zawierać błędy literowe i śmieci; użyj ich, aby dokładnie przepisać napisy, które widzisz, i popraw te, które OCR zniekształcił; pomiń fragmenty, których nie ma na ilustracji):
{ocr}
"""

# VLM: opis jednej ilustracji dla modelu tekstowego (raz na obraz i grupę zadań)
DESCRIBE_SYSTEM = """Jesteś asystentem historyka. Opisujesz ilustrację z arkusza maturalnego z historii (fotografia, mapa, plan, rycina, karykatura, obraz, moneta, tablica genealogiczna, dokument) dla osoby, która jej nie widzi, a ma odpowiedzieć na polecenia zadania. NIE odpowiadasz na polecenia. Piszesz po polsku, rzeczowo.
1. Rodzaj ilustracji.
2. Napisy przepisane dosłownie: tytuły, hasła, legenda mapy z kolorami i oznaczeniami, nazwy, daty, liczby, podpisy postaci, imiona w tablicy genealogicznej z datami i powiązaniami.
3. Co dokładnie widać: postacie (strój, mundur, insygnia, wygląd, gesty, kto co trzyma), budowle (materiał, kształt okien i łuków, dachy, wieże, zdobienia), na mapie lub planie zaznaczone obszary, granice, strzałki, miejscowości, rzeki; położenie elementów względem siebie. Skup się na elementach potrzebnych do poleceń.
4. Rozpoznanie: kto, co, gdzie, kiedy to prawdopodobnie jest (styl, epoka, wydarzenie, osoba), z oceną pewności (wysoka, średnia, niska)."""

DESCRIBE_USER = """Podpis i tekst źródła, do którego należy ilustracja:
{caption}

Polecenia, na które odbiorca będzie odpowiadał:
{questions}

Opisz załączoną ilustrację ({path})."""

IMG_DESC_BLOCK = """[Ilustracja {path} – opis sporządzony przez osobę, która ją widziała:
{desc}]"""


# ---- RAG "model sam pyta": krok 1 (bez myślenia) - czy i o co dopytać encyklopedię ----
RAG_QUERY_SYSTEM = """Jesteś maturzystą z historii. Zanim odpowiesz na zadanie, decydujesz, czy potrzebujesz sprawdzić fakty w encyklopedii (daty, nazwiska, nazwy wydarzeń, instytucji, traktatów), których nie jesteś pewien. Nie pytaj o to, co jest w treści źródeł zadania ani o interpretację ilustracji.
Odpowiadasz TYLKO poprawnym JSON: {"potrzeba": true|false, "zapytania": ["...", "..."]} - maksymalnie 3 krótkie, konkretne zapytania po polsku (nazwa własna + kontekst, np. "kto podpisał rozkaz o utworzeniu Armii Krajowej 1942"). Gdy wiedza ze źródeł wystarcza, "potrzeba": false i pusta lista."""
RAG_QUERY_SYSTEM_ALWAYS = """Jesteś maturzystą z historii. Zanim odpowiesz na zadanie, ZAWSZE sprawdzasz w encyklopedii fakty potrzebne do rozwiązania (daty, nazwiska, nazwy wydarzeń, instytucji, traktatów, tło historyczne źródeł). Nie pytaj o interpretację ilustracji.
Odpowiadasz TYLKO poprawnym JSON: {"potrzeba": true, "zapytania": ["...", "..."]} - od 1 do 3 krótkich, konkretnych zapytań po polsku (nazwa własna + kontekst, np. "kto podpisał rozkaz o utworzeniu Armii Krajowej 1942")."""
RAG_QUERY_USER = """Zadanie:
{task}

JSON:"""
RAG_BLOCK = """[Fragmenty encyklopedii znalezione dla tego zadania. Korzystaj z nich tylko, jeśli pasują do zadania; nie cytuj dosłownie.]
{fragments}
[Koniec fragmentów]

"""
