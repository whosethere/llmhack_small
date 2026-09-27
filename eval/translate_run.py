#!/usr/bin/env python3
"""Wariant tłumaczeniowy: PL -> EN (TranslateGemma) -> odpowiedź po angielsku -> PL (TranslateGemma) -> sędzia.

Fazy (każda wymaga innego modelu na :8000, więc uruchamiane osobno):
  tr_in   : TranslateGemma tłumaczy kontekst+treść pytań -> data/questions_en.jsonl (wspólne dla wszystkich solverów)
  solve   : solver odpowiada po angielsku -> results/<run>/answers_en.jsonl
  tr_out  : TranslateGemma tłumaczy odpowiedzi na PL -> results/<run>/answers.jsonl (format standardowy)
Ocena: python eval/run.py --run <run> --phase judge --judge gpt-6-sol
"""
import argparse, asyncio, json, re, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from openai import AsyncOpenAI
import prompts
from run import load_jsonl, key_of, ROOT

QEN = ROOT / "data" / "questions_en.jsonl"

def chunks(text, limit=1400):
    """Dzieli tekst na kawałki <= limit znaków po granicach linii (TranslateGemma: kontekst 2K tokenów)."""
    out, cur = [], ""
    for ln in text.split("\n"):
        while len(ln) > limit:  # bardzo długa linia -> tnij po zdaniach
            cut = ln.rfind(". ", 0, limit); cut = cut + 1 if cut > 200 else limit
            if cur: out.append(cur); cur = ""
            out.append(ln[:cut]); ln = ln[cut:].lstrip()
        if len(cur) + len(ln) + 1 > limit and cur:
            out.append(cur); cur = ln
        else:
            cur = (cur + "\n" + ln) if cur else ln
    if cur: out.append(cur)
    return out

async def translate(client, text, src, tgt, sem):
    if not text.strip(): return text
    parts = []
    for ch in chunks(text):
        async with sem:
            for attempt in range(4):
                try:
                    r = await client.chat.completions.create(model="solver", temperature=0, max_tokens=2048,
                        messages=[{"role": "user", "content": f"<<<source>>>{src}<<<target>>>{tgt}<<<text>>>{ch}"}])
                    parts.append((r.choices[0].message.content or "").strip()); break
                except Exception:
                    if attempt == 3: raise
                    await asyncio.sleep(3)
    return "\n".join(parts)

async def solve_en(client, q, sem, max_tokens):
    ctx = (q["kontekst_en"] + "\n") if q["kontekst_en"] else ""
    note = prompts.IMG_NOTE_EN if q["wymaga_obrazu"] else ""
    user = note + ctx + q["tresc_en"] + "\n\nAnswer:"
    essay = q["typ"] == "wypracowanie"
    async with sem:
        t = time.time()
        r = await client.chat.completions.create(model="solver", temperature=0.0, max_tokens=3500 if essay else max_tokens,
            messages=[{"role": "system", "content": prompts.ESSAY_SYSTEM_EN if essay else prompts.SOLVER_SYSTEM_EN},
                      {"role": "user", "content": user}])
    txt = re.sub(r"<think>.*?</think>\s*", "", r.choices[0].message.content or "", flags=re.S).strip()
    return {"id": q["id"], "odpowiedz_en": txt, "sek": round(time.time() - t, 1)}

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["tr_in", "solve", "tr_out"])
    ap.add_argument("--run"); ap.add_argument("--model-id")
    ap.add_argument("--url", default="http://localhost:8000/v1")
    ap.add_argument("--years", nargs="*"); ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=1500)
    a = ap.parse_args()
    client = AsyncOpenAI(base_url=a.url, api_key="EMPTY", timeout=600)
    sem = asyncio.Semaphore(a.concurrency)
    qs = [q for q in load_jsonl(ROOT / "data/questions.jsonl") if q["formula"] == "2023" and q["klucz"]]
    if a.years: qs = [q for q in qs if key_of(q) in set(a.years)]

    if a.phase == "tr_in":
        done = {x["id"]: x for x in load_jsonl(QEN)}
        todo = [q for q in qs if q["id"] not in done]
        print(f"tr_in: {len(todo)} pytań do przetłumaczenia")
        async def one(q):
            k, t = await asyncio.gather(translate(client, q["kontekst"], "pl", "en", sem), translate(client, q["tresc"], "pl", "en", sem))
            return {**q, "kontekst_en": k, "tresc_en": t}
        with open(QEN, "a") as f:
            for c in asyncio.as_completed([one(q) for q in todo]):
                r = await c; f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
        print("tr_in done:", len(load_jsonl(QEN)))
        return

    out = ROOT / "results" / a.run; out.mkdir(parents=True, exist_ok=True)
    if a.phase == "solve":
        json.dump({**vars(a), "variant": "translate PL->EN->PL (TranslateGemma 4B)"}, open(out / "config.json", "w"), indent=2)
        qen = {x["id"]: x for x in load_jsonl(QEN)}
        done = {x["id"] for x in load_jsonl(out / "answers_en.jsonl")}
        todo = [qen[q["id"]] for q in qs if q["id"] in qen and q["id"] not in done]
        print(f"solve: {len(todo)}")
        with open(out / "answers_en.jsonl", "a") as f:
            for c in asyncio.as_completed([solve_en(client, q, sem, a.max_tokens) for q in todo]):
                r = await c; f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
        print("solve done")
    elif a.phase == "tr_out":
        en = {x["id"]: x for x in load_jsonl(out / "answers_en.jsonl")}
        done = {x["id"] for x in load_jsonl(out / "answers.jsonl")}
        todo = [x for i, x in en.items() if i not in done]
        print(f"tr_out: {len(todo)}")
        async def back(x):
            t = time.time(); pl = await translate(client, x["odpowiedz_en"], "en", "pl", sem)
            return {"id": x["id"], "odpowiedz": pl, "odpowiedz_en": x["odpowiedz_en"], "sek": x["sek"], "sek_tr": round(time.time() - t, 1)}
        with open(out / "answers.jsonl", "a") as f:
            for c in asyncio.as_completed([back(x) for x in todo]):
                r = await c; f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
        print("tr_out done")

if __name__ == "__main__":
    asyncio.run(main())
