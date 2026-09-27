#!/usr/bin/env python3
"""Test pisania wypracowania: N przebiegów każdego wariantu (single, staged) na modelu z lokalnego API, zapis jako runy do oceny.

  python eval/essay_test.py --exam submission/history-2023-mock-v1 --llm-url http://localhost:8002/v1 --llm-model gemma \
      --modes single staged --repeats 3 --out-prefix essay-gemma
Wynik: results/<prefix>-<mode>-<k>/answers.jsonl (tylko zadanie 26) + essays.json z debugiem; ocena: eval/calibrate.py grade --runs ...
"""
import argparse, json, re, sys, time, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "exam"))
from essay import EssayWriter, words

ROOT = Path(__file__).resolve().parent.parent


def make_chat(url, model, temperature):
    def chat(system, user, max_tokens):
        body = json.dumps({"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                           "max_tokens": max_tokens, "temperature": temperature}).encode()
        req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            msg = json.load(r)["choices"][0]["message"]
            txt = re.sub(r"<think>.*?</think>\s*", "", msg.get("content") or "", flags=re.S).strip()
            return txt  # pusty content (myślenie zjadło budżet) -> pusty string, nigdy rozumowanie
    return chat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exam", required=True); ap.add_argument("--llm-url", required=True); ap.add_argument("--llm-model", required=True)
    ap.add_argument("--modes", nargs="+", default=["single", "staged"]); ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--topic", type=int, default=1); ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out-prefix", required=True); ap.add_argument("--year", default="2023_f2023")
    ap.add_argument("--essay-max-tokens", type=int, default=3000); ap.add_argument("--small-tokens", type=int, default=800)
    a = ap.parse_args()
    exam = json.loads((Path(a.exam) / "exam.json").read_text(encoding="utf-8"))
    item = next(it for it in exam["items"] if it["id"] == "26")
    writer = EssayWriter(make_chat(a.llm_url, a.llm_model, a.temperature), n_topic=a.topic, essay_max_tokens=a.essay_max_tokens, small_tokens=a.small_tokens)
    for mode in a.modes:
        for k in range(1, a.repeats + 1):
            run = f"{a.out_prefix}-{mode}-{k}"; out = ROOT / "results" / run; out.mkdir(parents=True, exist_ok=True)
            t, dbg = time.time(), {}
            text = writer.write(item["question"], mode, dbg)
            sec = round(time.time() - t, 1)
            (out / "answers.jsonl").write_text(json.dumps({"id": f"{a.year}_z26", "odpowiedz": text, "sek": sec}, ensure_ascii=False) + "\n", encoding="utf-8")
            (out / "essays.json").write_text(json.dumps({"mode": mode, "k": k, "sec": sec, "words": words(text), "debug": dbg, "text": text}, ensure_ascii=False, indent=2), encoding="utf-8")
            json.dump({"run": run, "model_id": a.llm_model, "essay_mode": mode, "note": "tylko wypracowanie (eval/essay_test.py)"}, open(out / "config.json", "w"), ensure_ascii=False, indent=2)
            print(f"{run}: {words(text)} wyrazów w {sec}s | temat nr {writer.n}: {dbg.get('topic_extract')} | {dbg.get('parts_words', '')}")
            writer.n = a.topic  # tryb auto zmienia numer tematu; przywróć przed kolejnym przebiegiem


if __name__ == "__main__":
    main()
