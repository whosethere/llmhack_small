#!/usr/bin/env python3
"""Opisywacz obrazów: VLM (<=8GB) ogląda stronę arkusza + polecenie i opisuje źródła graficzne.
Opis trafia potem do solvera wiedzowego (eval/run.py --img-desc TAG).

Jeden opis na zadanie (z3.1 i z3.2 dzielą obraz -> klucz 2025_f2023_z3), z treścią wszystkich podpunktów.
Przykład (na serwerze, VLM na :8000 przez gpu/serve.sh vlm <model>):
  /scratch/vllm/bin/python eval/describe.py --tag qwen3vl8b --years 2025_f2023 2026_f2023
Wynik: data/img_desc/<tag>.jsonl  {grupa, opis, obrazy, sek}  (wznawialne)
"""
import argparse, asyncio, base64, json, re, sys, time
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from openai import AsyncOpenAI
import prompts
from run import ROOT, load_jsonl, key_of

def group_of(qid):
    return qid.split(".")[0]

def img_part(path):
    b64 = base64.b64encode((ROOT / path).read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}

async def describe(client, model, grp, qs, sem, max_tokens):
    imgs = sorted({p for q in qs for p in q["obrazy"]})
    ctx = qs[0]["kontekst"] or ""
    polecenia = "\n\n".join(f"[{q['id'].split('_z')[1]}] {q['tresc']}" for q in qs)
    text = prompts.DESCRIBE_USER.format(kontekst=ctx, polecenia=polecenia)
    content = [img_part(p) for p in imgs] + [{"type": "text", "text": text}]
    async with sem:
        t = time.time()
        r = await client.chat.completions.create(model=model, temperature=0.0, max_tokens=max_tokens,
            messages=[{"role": "system", "content": prompts.DESCRIBE_SYSTEM}, {"role": "user", "content": content}])
    txt = re.sub(r"<think>.*?</think>\s*", "", r.choices[0].message.content or "", flags=re.S).strip()
    return {"grupa": grp, "opis": txt, "obrazy": imgs, "sek": round(time.time() - t, 1)}

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="nazwa zestawu opisów, np. qwen3vl8b")
    ap.add_argument("--url", default="http://localhost:8000/v1"); ap.add_argument("--model", default="vlm")
    ap.add_argument("--years", nargs="*", help="np. 2025_f2023; domyślnie wszystkie f2023")
    ap.add_argument("--concurrency", type=int, default=8); ap.add_argument("--max-tokens", type=int, default=1200)
    a = ap.parse_args()

    qs = load_jsonl(ROOT / "data/questions.jsonl")
    years = set(a.years) if a.years else {key_of(q) for q in qs if q["formula"] == "2023"}
    groups = defaultdict(list)
    for q in qs:
        if key_of(q) in years and q["obrazy"] and q["typ"] != "wypracowanie":
            groups[group_of(q["id"])].append(q)
    out = ROOT / "data/img_desc" / f"{a.tag}.jsonl"; out.parent.mkdir(exist_ok=True)
    done = {x["grupa"] for x in load_jsonl(out)}
    todo = [g for g in groups if g not in done]
    print(f"[{a.tag}] grup z obrazem: {len(groups)}, do opisania: {len(todo)}")
    client = AsyncOpenAI(base_url=a.url, api_key="EMPTY")
    sem = asyncio.Semaphore(a.concurrency)
    with open(out, "a") as f:
        for coro in asyncio.as_completed([describe(client, a.model, g, groups[g], sem, a.max_tokens) for g in todo]):
            r = await coro; f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            print(f"  {r['grupa']} ({r['sek']}s, {len(r['opis'].split())} słów)")

if __name__ == "__main__":
    asyncio.run(main())
