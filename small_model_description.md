# Opis zgłoszenia – kategoria „mały” (do formularza organizatorów)

## Model (pole „Models / Quantization”)

- **Link:** https://huggingface.co/speakleash/Bielik-1.5B-v3.0-Instruct-GGUF
- **Kwantyzacja:** Q8_0 (oficjalna kwantyzacja SpeakLeash) – plik `Bielik-1.5B-v3.0-Instruct.Q8_0.gguf`, **1 699 568 288 B = 1,70 GB (1,58 GiB)**,
  rewizja `99d72cc757be4f8b958018f12d6898bad5c19c9e`, licencja Apache-2.0. Poniżej limitu 2 GB.
- Model bazowy tej kwantyzacji: https://huggingface.co/speakleash/Bielik-1.5B-v3.0-Instruct (bf16, 3,19 GB – za duży na limit, stąd Q8_0).
- Jeden model tekstowy (ilustracji nie widzi), bez fine-tuningu i bez LoRA. Serwowany przez llama.cpp (Docker `ghcr.io/ggml-org/llama.cpp:server-cuda`),
  temperatura 0, `repeat_penalty 1.05`, kontekst 8192.
- Modele pomocnicze, nie generują odpowiedzi (retrieval i reranker nie liczą się do limitu): `intfloat/multilingual-e5-large` (embeddingi, 2,2 GB),
  `BAAI/bge-reranker-v2-m3` (reranker, 2,3 GB), `facebook/dinov2-base` (weryfikacja kopii ilustracji z Wikipedii, 0,35 GB), Tesseract OCR (pol).
- Baza wiedzy (nie liczy się do wag): fragmenty polskiej Wikipedii i angielskiej Wikipedii historii ze zrzutów Kiwix (CC BY-SA), offline;
  używana wyłącznie do wypracowania.

## Model bazowy (wiersz „Base model”)

Ten sam plik `speakleash/Bielik-1.5B-v3.0-Instruct-GGUF` (Q8_0) odpytany bezpośrednio promptem benchmarkowym: tylko tekst zadania, bez OCR,
bez retrievalu, bez normalizacji odpowiedzi. Na arkuszu 2023 (sędzia zgodny z organizatorami): **18–19/60 bazowy vs 20,7/60 rozwiązanie**
(średnia z 3 przebiegów; wypracowanie 0–1 → 5 pkt).

## Krótki opis rozwiązania (pole „Project” / opis)

llmhack-small – Bielik 1.5B Q8_0 (1,7 GB) + harness tekstowy: goły prompt maturalny (bez nagłówków i instrukcji formatu, które szkodzą
małemu modelowi), ilustracje przez OCR (tesseract, powiększenie 2×) i identyfikację kopii obrazów z Wikipedii (pHash + DINOv2), sprowadzanie
odpowiedzi zamkniętych do składni answer_format, wypracowanie pisane krokowo: temat wybierany regułą (tematy aspektowe, od XVIII w.), szablonowy
wstęp i zakończenie, akapit na aspekt z 5 zdaniami-faktami z Wikipedii (retrieval tylko tu), filtr zdań z datami spoza notatek, kontrola długości.
Ewaluacja własnym sędzią GPT-6 skalibrowanym do ocen organizatorów (96% zgodności). Repo: llmhack_small (wydzielone z github.com/whosethere/llmhack).

## Kategorie

„Mały ale wartki” (wagi ≤ 2 GB).

## Wyniki własne (sędzia skalibrowany, średnie z 2–3 przebiegów; rozrzut ±2)

2023: 20,7/60 · 2024: 13,0/60 · 2025: 19,7/60 · 2026: 18,5/60 (najlepsze pojedyncze przebiegi: 22/60 na 2023 i 2025).
