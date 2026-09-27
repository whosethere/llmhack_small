#!/usr/bin/env python3
"""Buduje dashboard/index.html z results/*/ (+ wyniki organizatorów z prezentacji).
python eval/dashboard.py   -> potem publikacja dashboard/index.html jako Artifact (ten sam plik = ten sam URL)."""
import json, statistics, datetime
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIMIT_GB = 8.0

# rozmiar wag na dysku (GB, z HF API) i opisy runów
RUNS_META = {
    "qwen3-8b-base":        {"label": "Qwen3-8B bf16", "model": "Qwen/Qwen3-8B", "size": 16.4, "variant": "bazowy, prompt v1",
                             "note": "Pierwszy baseline: stary prompt eseju, wagi bf16 ponad limitem."},
    "qwen3-8b-awq":         {"label": "Qwen3-8B AWQ", "model": "Qwen/Qwen3-8B-AWQ", "size": 6.10, "variant": "bazowy"},
    "bielik-11b-v3-awq":    {"label": "Bielik-11B v3.0 AWQ", "model": "speakleash/Bielik-11B-v3.0-Instruct-awq", "size": 6.19, "variant": "bazowy"},
    "bielik-11b-v2.6-awq":  {"label": "Bielik-11B v2.6 AWQ", "model": "speakleash/Bielik-11B-v2.6-Instruct-AWQ", "size": 6.19, "variant": "bazowy"},
    "gemma4-12b-qat-REF":   {"label": "Gemma 4 12B QAT", "model": "google/gemma-4-12B-it-qat-w4a16-ct", "size": 10.26, "variant": "punkt odniesienia",
                             "note": "Ponad limitem 8 GB (10.3 GB). Tylko dla porównania: u organizatorów 76.7% w trybie obrazów, tu bez obrazów i bez trybu myślenia."},
    "ornith-35b-a3b-fp8":   {"label": "Ornith-1.5 35B-A3B FP8", "model": "ornith-ai/Ornith-1.5-35B-A3B-FP8", "size": 39.4, "variant": "punkt odniesienia",
                             "note": "MoE 35B (3B aktywnych), ponad limitem: najmniejszy GGUF (IQ2_XXS) ma 10.3 GB. Bez trybu myślenia."},
    "tr-ornith-35b-a3b-fp8": {"label": "Ornith-1.5 35B-A3B FP8", "model": "ornith-ai/Ornith-1.5-35B-A3B-FP8", "size": 39.4, "variant": "tłumaczenie PL→EN→PL",
                             "note": "MoE 35B (3B aktywnych), ponad limitem. Pytania i odpowiedzi tłumaczone przez TranslateGemma 4B. Bez trybu myślenia."},
    "gemma4-12b-q4+img":        {"label": "Gemma 4 12B QAT q4_0 (GGUF)", "model": "google/gemma-4-12B-it-qat-q4_0-gguf", "size": 7.16, "variant": "obrazy stron, myślenie",
                                 "note": "Jeden model tekst+obraz przez llama.cpp; strony arkusza jako obrazy w prompcie; tryb myślenia."},
    "gemma4-12b-q4+img+nothink": {"label": "Gemma 4 12B QAT q4_0 (GGUF)", "model": "google/gemma-4-12B-it-qat-q4_0-gguf", "size": 7.16, "variant": "obrazy stron, bez myślenia"},
    "gemma4-12b-q4+img+enrag-k8-sc": {"label": "Gemma 4 12B QAT q4_0 (GGUF)", "model": "google/gemma-4-12B-it-qat-q4_0-gguf", "size": 7.16, "variant": "obrazy + RAG en-wiki k8",
                                 "note": "RAG z angielskiej Wikipedii historycznej (e5), 8 fragmentów, bez RAG w zadaniach zamkniętych."},
    "bielik-v3+vl+rag-e5-k8-sc": {"label": "Bielik-11B v3.0 AWQ + Qwen3-VL", "model": "speakleash/Bielik-11B-v3.0-Instruct-awq", "size": 6.19, "variant": "opisy obrazów + RAG pl-wiki k8",
                                 "note": "RAG z polskiej Wikipedii (e5), 8 fragmentów, bez RAG w zamkniętych. Wg Sola v4 gorzej niż bez RAG."},
    "gpt-6-luna-api":       {"label": "GPT-6 Luna (API)", "model": "gpt-6-luna", "size": None, "variant": "API, bez rozumowania",
                             "note": "Punkt odniesienia: duży model przez API Forgehand, nie spełnia zasad konkursu (offline, ≤8 GB)."},
    "tr-qwen3-8b-awq":      {"label": "Qwen3-8B AWQ", "model": "Qwen/Qwen3-8B-AWQ", "size": 6.10, "variant": "tłumaczenie PL→EN→PL",
                             "note": "Pytania i odpowiedzi tłumaczone przez TranslateGemma 4B (8.6 GB bf16, do finału potrzebna wersja ≤8 GB)."},
    "tr-bielik-11b-v3-awq": {"label": "Bielik-11B v3.0 AWQ", "model": "speakleash/Bielik-11B-v3.0-Instruct-awq", "size": 6.19, "variant": "tłumaczenie PL→EN→PL",
                             "note": "Pytania i odpowiedzi tłumaczone przez TranslateGemma 4B (8.6 GB bf16, do finału potrzebna wersja ≤8 GB)."},
}
JUDGE_LABEL = {"gpt-6-sol": "GPT-6 Sol", "gpt-6-luna": "GPT-6 Luna", "qwen3-32b-v1": "Qwen3-32B (prompt v1)", "claude-manual": "Claude (ręcznie)"}

# Prezentacja organizatorów: arkusz CKE maj 2023 (nasz 2023_f2023). Tekst: /55 (bez zad. 7, 8, 15; opisy obrazów), obrazy: /60.
# (model, parametry B, [M]=multimodal, tekst%, obrazy%, (zamk. tekst, zamk. obr.), (otw. tekst, otw. obr.), (esej tekst, esej obr.))
ORG = [
    ("Gemma 3 1B", 1.00, False, 14.5, None, (4, None), (4, None), (0, None)),
    ("Bielik 1.5B", 1.60, False, 27.3, None, (5, None), (9, None), (1, None)),
    ("Qwen3 1.7B", 1.72, False, 23.6, None, (4, None), (8, None), (1, None)),
    ("SmolLM3 3B", 3.08, False, 18.2, None, (5, None), (5, None), (0, None)),
    ("Llama 3.2 3B", 3.21, False, 18.2, None, (3, None), (7, None), (0, None)),
    ("Phi-4 mini 3.8B", 3.84, False, 18.2, None, (5, None), (5, None), (0, None)),
    ("Qwen3 4B", 4.02, False, 40.0, None, (7, None), (12, None), (3, None)),
    ("Bielik 4.5B", 4.76, False, 41.8, None, (6, None), (15, None), (2, None)),
    ("Qwen3 VL 2B Instruct", 2.13, True, 10.9, 15.0, (3, 5), (3, 4), (0, 0)),
    ("Qwen3 VL 2B Thinking", 2.13, True, 12.7, 6.7, (3, 1), (4, 3), (0, 0)),
    ("Gemma 3 4B", 4.30, True, 36.4, 26.7, (10, 4), (7, 9), (3, 3)),
    ("PLLuM 4B", 4.30, True, 30.9, 21.7, (5, 3), (12, 9), (0, 1)),
    ("InternVL3.5 8B", 8.53, True, None, 36.7, (None, 5), (None, 16), (None, 1)),
    ("Qwen3.5 9B", 9.65, True, None, 60.0, (None, 5), (None, 23), (None, 8)),
    ("LLaVA-Bielik 11B", 11.62, True, 52.7, 23.3, (5, 6), (21, 8), (3, 0)),
    ("Gemma 4 12B", 11.96, True, None, 76.7, (None, 10), (None, 24), (None, 12)),
    ("LLaVA-PLLuM 12B", 12.71, True, 34.5, 23.3, (1, 3), (18, 11), (0, 0)),
    ("Ministral 3 14B", 13.95, True, 60.0, 48.3, (5, 7), (25, 19), (3, 3)),
]

def load(p):
    return [json.loads(l) for l in open(p)] if p.exists() else []

def main():
    qs = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data/questions.jsonl")}
    runs = []
    for d in sorted((ROOT / "results").iterdir()):
        if not d.is_dir() or not (d / "answers.jsonl").exists(): continue
        meta = RUNS_META.get(d.name, {"label": d.name, "model": "?", "size": None, "variant": "?"})
        ans = {a["id"]: a for a in load(d / "answers.jsonl")}
        judges = {}
        for jf in sorted(d.glob("judgments_*.jsonl")):
            jn = jf.stem[len("judgments_"):]
            js = {j["id"]: j for j in load(jf)}
            years = defaultdict(lambda: {"p": 0, "m": 0, "n": 0, "cat": defaultdict(lambda: [0, 0])})
            for qid, j in js.items():
                q = qs.get(qid)
                if not q: continue
                y = years[qid.split("_z")[0]]
                y["p"] += j["punkty"]; y["m"] += q["punkty_max"]; y["n"] += 1
                c = y["cat"][q["typ"]]; c[0] += j["punkty"]; c[1] += q["punkty_max"]
            judges[jn] = {"label": JUDGE_LABEL.get(jn, jn), "years": {k: {"p": v["p"], "m": v["m"], "n": v["n"], "cat": dict(v["cat"])} for k, v in years.items()},
                          "items": {qid: [j["punkty"], j["uzasadnienie"]] for qid, j in js.items()},
                          "errors": len(load(d / f"judge_errors_{jn}.jsonl"))}
        secs = [a.get("sek", 0) + a.get("sek_tr", 0) for a in ans.values()]
        runs.append({"id": d.name, **meta, "legal": meta.get("size") is not None and meta["size"] <= LIMIT_GB,
                     "n_answers": len(ans), "median_s": round(statistics.median(secs), 1) if secs else None,
                     "judges": judges,
                     "answers": {qid: {"a": a["odpowiedz"][:2500], "en": a.get("odpowiedz_en", "")[:2500]} for qid, a in ans.items()}})
    questions = {qid: {"typ": q["typ"], "max": q["punkty_max"], "img": q["wymaga_obrazu"], "t": q["tresc"][-600:],
                       "key": (q["klucz"]["rozwiazanie"] or "")[:800]}
                 for qid, q in qs.items() if q["formula"] == "2023" and any(qid in r["answers"] for r in runs)}
    org = [{"name": n, "params": p, "mm": mm, "text": t, "img": i, "closed": c, "open": o, "essay": e} for n, p, mm, t, i, c, o, e in ORG]
    data = {"generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "runs": runs, "questions": questions, "org": org, "limit": LIMIT_GB}
    tpl = (ROOT / "dashboard/template.html").read_text()
    out = tpl.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    (ROOT / "dashboard/index.html").write_text(out)
    print(f"dashboard/index.html: {len(runs)} runów, {len(questions)} pytań, {len(out)/1024:.0f} KB")

if __name__ == "__main__":
    main()
