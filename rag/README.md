# RAG: Wikipedia (PL + EN historia) dla solvera

Baza wiedzy budowana ze zrzutów ZIM (Kiwix), nie ma jej w repozytorium. Solver sam decyduje, czy i o co pytać
(`exam/solver_server.py`, `rag_prefix`): model dostaje zadanie, zwraca JSON `{"potrzeba": bool, "zapytania": [...]}`
(maks. 3 zapytania po polsku), a fragmenty z retrievalu trafiają przed źródło zadania.

## Budowa indeksu (raz, z internetem; potem wszystko offline)

```bash
# 1. zrzuty ZIM (Kiwix): wikipedia_pl_all_maxi (22.7 GB) i wikipedia_en_history_maxi -> ~/wiki/
wget -P ~/wiki https://download.kiwix.org/zim/wikipedia/wikipedia_pl_all_maxi_2026-08.zim
wget -P ~/wiki https://download.kiwix.org/zim/wikipedia/wikipedia_en_history_maxi_2026-07.zim
# 2. artykuły -> akapity (pomija sekcje typu Przypisy/Bibliografia), fragmenty po ~512 znaków + infoboksy + podpisy
ZIM=~/wiki/wikipedia_pl_all_maxi_2026-08.zim python rag/zim_scan.py ~/wiki/articles.parquet
python rag/chunk_wiki.py ~/wiki/articles.parquet ~/wiki/chunks          # 8.77 mln fragmentów (PL)
# 3. embeddingi intfloat/multilingual-e5-large, fp16, shardy .npy (PL ~2 h na A100, 17 GB)
python rag/embed_text.py ~/wiki/chunks ~/wiki/emb_e5
# to samo dla EN historii -> ~/wiki/en_chunks, ~/wiki/en_emb_e5 (981 tys. fragmentów)
```

## Serwer retrievalu (egzamin)

`bash gpu/serve_rag.sh` uruchamia `rag/retrieve_server.py` na 127.0.0.1:8010: oba indeksy w pamięci GPU,
kosinus top-30 (e5), reranker `BAAI/bge-reranker-v2-m3` (sigmoid 0..1), próg `min_score`. `POST /search`
`{"queries": [...], "k": 4, "min_score": 0.5}` -> `{"hits": [{"title", "text", "score"}, ...]}`.

## Pozostałe skrypty

- `retrieve.py`, `rerank.py`, `eval_retrieval.py`: retrieval wsadowy dla `data/questions.jsonl` i hit@k na frazach z klucza
  (e5 25%, + reranker 31%); `bm25_build.py`: indeks leksykalny (bm25s), nieużywany w finale.
- `embed_images.py`, `image_search.py`: indeks DINOv2 obrazów z Wikipedii (1,4 mln) i dopasowanie ilustracji z arkusza;
  na arkuszu 2023 trafiły 3 z 19 ilustracji, odłożone.

Modele: `intfloat/multilingual-e5-large` (2,2 GB), `BAAI/bge-reranker-v2-m3` (2,3 GB); oba poniżej 8 GB, nie są solverem.
