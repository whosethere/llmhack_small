# llmhack-small: model ≤ 2 GB zdający maturę z historii

Kategoria „mały” hackathonu Warsaw Model Trainers (25–27.09.2026): limit **2 GB wag modelu** (retrieval i reranker nie liczą się).
Model: **`speakleash/Bielik-1.5B-v3.0-Instruct` w Q8_0 (GGUF, 1,70 GB)**, serwowany llama.cpp (Docker, CUDA), tylko tekst.
Wydzielone z repo [llmhack](https://github.com/whosethere/llmhack) (rozwiązanie główne: Gemma 4 12B, 45–51/60).

## Wynik (sędzia GPT-6 Sol, prompt v4 zgodny z ocenami organizatorów w 96%)

Konfiguracja docelowa `gpu/exam_small.sh`, średnia z 3 przebiegów na mock (rozrzut identycznej konfiguracji ±2 pkt: llama.cpp
z 8 slotami nie jest deterministyczne, sędzia też):

| arkusz (mock = arkusz CKE w formacie organizatorów) | zamknięte | otwarte z obrazem | otwarte tekstowe | wypracowanie | **razem /60** | **%** | przebiegi (rozrzut) |
|---|---|---|---|---|---|---|---|
| 2023 | 2,0 / 3 | 4,3 / 26 | 9,3 / 16 | 5,0 / 15 | **20,7** | **34,4%** | 3 (18–22) |
| 2024 | 0,0 / 4 | 5,5 / 31 | 5,0 / 10 | 2,5 / 15 | **13,0** | **21,7%** | 2 (12–14) |
| 2025 | 3,0 / 8 | 8,3 / 24 | 3,0 / 13 | 5,3 / 15 | **19,7** | **32,8%** | 3 (18–22) |
| 2026 | 3,0 / 6 | 7,0 / 29 | 2,0 / 10 | 6,5 / 15 | **18,5** | **30,8%** | 2 (17–20) |
| **średnia 4 arkuszy** | | | | 4,8 / 15 | **18,0** | **30,0%** | |

Arkusz 2024 jest najtrudniejszy dla modelu tekstowego (30 z 40 zadań z ilustracją, 31 pkt) i ma temat wypracowania o osobie (Piłsudski);
zadanie 21 (kadr z „Triumfu woli”) odrzuca filtr treści sędziego i jest oceniane ręcznie (0 pkt).

Goły model (prompt benchmarkowy, bez niczego): 2023 18–19/60, wypracowanie 0–1/15. Cały zysk to wypracowanie.

## Co robi harness (`exam/solver_server.py --text-only`)

1. **Goły prompt** jak w benchmarku (`źródło + polecenie + „Odpowiedź:”`), bez nagłówka „Zadanie X (n pkt)” i bez instrukcji
   formatu: te dwie rzeczy kosztowały model 1.5B ~7 pkt (odpowiedzi 2,5× krótsze, w zamkniętych zgadywał literę). Literę z pełnej
   odpowiedzi wyciąga harness (`normalize_closed`). Etykiety `ŹRÓDŁO:/POLECENIE:` (`--text-headers`): bez różnicy.
2. **Ilustracje** (model ich nie widzi): OCR tesseract na powiększeniu 2× jako podpowiedź + identyfikacja kopii obrazów z Wikipedii
   PL/EN (pHash ≤ 10 potwierdzony DINOv2 ≥ 0,65) jako tytuł artykułu i podpis. Trafienia rzadkie (2023: 2 z 19, 2025: 0).
3. **Retrieval tylko do wypracowania.** Fragmenty Wikipedii w zwykłych zadaniach nie pomagają (model pyta, k=3: ±0; treść zadania
   jako zapytanie z progiem rerankera 0,5: **−7 pkt**, próg 0,7 = brak fragmentów).
4. **Wypracowanie** (`exam/essay_small.py`, wybrane spośród 17 wariantów w `small_model/essay_lab.py` na 12 tematach CKE 2023–2026):
   - temat wybiera reguła: tylko tematy z trzema aspektami (tematy „trzech wybranych władców” ~1,3 pkt: przykłady spoza ram
     czasowych) → od XVIII w. → bez nazwy własnej w tezie → niższy numer (5,5 pkt/esej vs 3,3 dla „pierwszy aspektowy”);
   - wstęp i zakończenie z szablonu (bez faktów = bez błędów, 3 pkt za spójność), model pisze tylko trzy akapity, po jednym na aspekt;
   - do każdego akapitu osobny retrieval (teza + aspekt, e5-large + bge-reranker, próg 0,3) i **tylko zdania-fakty** (rok albo nazwa
     własna, 12–30 wyrazów, 5 na aspekt, lista numerowana), instrukcja na końcu promptu;
   - po generacji: zdania z rokiem spoza notatek wylatują (błędy 6,0 → 3,2 na esej), przycięcie do ~130 wyrazów, porządki
     (nagłówki, powtórzenia, urwane zdanie), dopisanie akapitu, gdy < 320 wyrazów;
   - temperatura 0, `repeat_penalty 1.05`. Jeden prompt na cały esej: 2,1–2,5 pkt; krokowo z notatkami: 4,0; z regułą tematu 5,5.

Odrzucone po testach: Llama 3.2 1B (2/60, nie zna polskiego), Qwen3-4B-2507 i Gemma 3 4B (19/60, za duże), Bielik 4.5B (4,9 GB, za duży),
temperatura 0,3/0,6 (więcej błędów), 3 fakty na aspekt, próg rerankera 0,5, few-shot, RAG w zadaniach otwartych.

## Uruchomienie (maszyna z GPU, Docker, tesseract-ocr-pol, indeks Wikipedii wg `rag/README.md`)

```bash
bash gpu/download_small.sh                                     # Bielik Q8_0, e5, reranker, DINOv2, obraz llama.cpp
bash gpu/serve_rag.sh                                          # retrieval :8010 (Wikipedia PL+EN)
bash gpu/exam_small.sh submission/history-2023-mock-v1 nazwa   # -> results/exam/nazwa/answers.json (~40 s)
python exam/to_eval.py --exam submission/history-2023-mock-v1 --answers results/exam/nazwa/answers.json --run exam-nazwa
python eval/calibrate.py grade --judge gpt-6-sol-v4 --as gpt-6-sol-v4-opis --desc results/org-opisy/opisy_2023.jsonl --desc-all --runs exam-nazwa
```

Zmienne `exam_small.sh`: `TEMP` (0; 0,2 i 0,4 dawały 18 i 17/60 na 2023), `RAG` (1), `RAG_SCOPE` (essay|open|all), `OCR` (tesseract|none), `TEXT_HEADERS`, `ESSAY_RAG_K` (4), `CONCURRENCY` (8).
Laboratorium eseju: `python small_model/essay_lab.py gen|judge|report --variants facts5_years ...` (model pod `--llm-url`, retrieval pod `--rag-url`).

## Struktura

```
exam/        harness egzaminu (solver API, essay_small.py, OCR, walidacja, to_eval)
eval/        sędzia (GPT-6 Sol v4 przez Forgehand), kalibracja, raport
rag/         indeks Wikipedii (ZIM -> fragmenty -> e5 -> serwer z rerankerem), pHash obrazów
gpu/         exam_small.sh, serve_bielik.sh (llama.cpp), serve_rag.sh, exam_run.sh, download_small.sh
small_model/ essay_lab.py: warianty promptu wypracowania i ich oceny
results/     odpowiedzi i oceny: exam/final-*, exam-final-* (docelowe), raw-*, hdr-* (A/B), essaylab-* (17 wariantów), small-bielik-* (goły model)
submission/  exam.json paczek mock (obrazy: z paczek organizatorów)
data/        questions.jsonl (zadania CKE 2015–2026 z kluczami)
```
