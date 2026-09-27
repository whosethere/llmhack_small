# Harness egzaminu (format organizatorów)

Paczka egzaminu (`exam.json` + `images/*.png` + `answers-template.json`) -> pytanie po pytaniu przez
**solver API** -> `answers.json` do wysłania na stronie zgłoszeń. Wszystko działa **offline**: runner i solver
rozmawiają tylko z `localhost`, modele muszą być pobrane wcześniej. Tylko biblioteka standardowa Pythona.

```
exam.json + images/ ─► run_exam.py ── POST /solve (1 pytanie, JSON) ─► solver_server.py ─► vLLM (localhost)
                         └─► answers.json, responses.jsonl, run.json
```

## Na serwerze GPU

```bash
bash gpu/setup.sh speakleash/Bielik-11B-v3.0-Instruct-awq cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit  # raz, z internetem
bash gpu/exam_serve.sh                                  # Bielik :8000 + Qwen3-VL :8001 naraz, offline
bash gpu/exam_run.sh submission/history-2023-mock-v1 mock-bielik-vlm            # strategia vlm-text
python3 exam/validate.py --exam submission/history-2023-mock-v1 results/exam/mock-bielik-vlm/answers.json
```

Model bazowy (do liczenia postępu): `bash gpu/serve.sh solver <model>` i
`bash gpu/exam_run.sh <paczka> <nazwa> base` (dla modelu wizyjnego dodaj `--multimodal`).

## Kontrakt solver API

`POST /solve`:
```json
{"exam_id": "history-2023-mock-v1", "instructions": "…",
 "item": {"id": "5.2", "group": 5, "max_points": 1, "question": "…", "source_text": "…", "answer_format": "…",
          "images": [{"path": "images/Z05-S1.png", "sha256": "…", "mime": "image/png", "data_b64": "…"}]},
 "context": {"group_questions": [{"id": "5.1", "question": "…"}, …]}}
```
Odpowiedź: `{"id": "5.2", "answer": "…", "pipeline": "vlm-text:solver+vlm", "ms": 5230, "debug": {…}}`.
`debug` (opisy ilustracji, surowy tekst modelu, poprawki formatu) trafia tylko do `responses.jsonl`, nigdy do `answers.json`.
`GET /health`, `GET /info`. Nową strategię dodaje się w `solver_server.py`; runner się nie zmienia.

## Co robi przetwarzanie

- `load_exam`: sprawdza sumy sha256 obrazów i zgodność id z `answers-template.json`; nic nie jest na sztywno
  (finał ma własne `exam_id` i szablon).
- Znaczniki `[Obraz: images/…]` w `source_text`: w `vlm-text` zastępowane opisem z VLM (jeden opis na obraz i grupę
  zadań, zapisywany w `cache/`), w `base --multimodal` obraz trafia dokładnie w miejsce znacznika.
- `answer_format`: otwarte (model wypełnia etykiety z polecenia, np. „Rozstrzygnięcie:”), zamknięte (`1: P…`, `A`,
  `A: 1…`, `1: A…`; model dostaje samą składnię bez przykładowych wartości, odpowiedź jest sprowadzana do dokładnej
  składni, przy niejednoznacznej solver prosi o sam format), wypracowanie (`Temat nr X`, krótsze niż 350 wyrazów
  jest dogenerowywane).
- `validate.py`: reguły strony zgłoszeń (błędy) i nasze kontrole (puste, składnia zamkniętych, wypracowanie, `<think>`).
- `to_eval.py`: `answers.json` próbnego egzaminu -> `results/<run>/` do oceny naszym sędzią (`eval/run.py --phase judge`).

## Test bez GPU

`python3 exam/tests/e2e_fake.py --exam submission/history-2023-mock-v1` – atrapy vLLM, obie strategie, walidacja.
