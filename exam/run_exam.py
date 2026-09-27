#!/usr/bin/env python3
"""Runner egzaminu: czyta paczkę organizatorów (exam.json + images/ + answers-template.json) i odpytuje
solver API pytanie po pytaniu. Działa offline (solver na localhost). Tylko biblioteka standardowa.

  python exam/run_exam.py --exam submission/history-2023-mock-v1 --solver http://localhost:9000 --out results/exam/NAZWA

Pliki w --out:
  answers.json      do wysłania: tylko exam_id i answers (kolejność szablonu, puste "" gdy brak odpowiedzi)
  responses.jsonl   odpowiedzi solvera z debug (opisy ilustracji, surowy tekst, poprawki formatu); wznawianie
  run.json          metadane: exam_id, /info solvera, czasy, wynik walidacji
Ponowne uruchomienie pomija zadania już rozwiązane (--redo: od nowa, --items: tylko wybrane).
"""
import argparse, json, sys, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import common as C


def http_json(url, payload=None, timeout=900):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        body = e.read()[:500].decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {e.code}: {body}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exam", required=True, help="katalog paczki (z exam.json) albo ścieżka do exam.json")
    ap.add_argument("--solver", default="http://localhost:9000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--items", nargs="*", help="tylko te id, np. 1 5.2 26")
    ap.add_argument("--redo", action="store_true", help="rozwiąż od nowa (z --items tylko te zadania, reszta zostaje)")
    ap.add_argument("--timeout", type=int, default=900, help="sekundy na jedno zadanie")
    ap.add_argument("--deadline", type=int, default=0, help="limit sekund na cały egzamin; solver dostaje time_left i po przekroczeniu odpowiada w trybie szybkim (bez myślenia)")
    ap.add_argument("--fast-after", type=int, default=0, help="po tylu sekundach od startu solver ma odpowiadać w trybie szybkim (0 = deadline - 240)")
    ap.add_argument("--no-guess", action="store_true", help="nie wpisuj awaryjnych odpowiedzi w zamkniętych, gdy solver zawiódł")
    ap.add_argument("--allow-remote", action="store_true", help="tylko do testów: solver spoza localhost")
    a = ap.parse_args()

    C.ensure_local(a.solver, a.allow_remote)
    exam, root, template_ids = C.load_exam(a.exam)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    info = http_json(a.solver.rstrip("/") + "/info", timeout=30)
    print(f"[{exam['exam_id']}] {len(exam['items'])} zadań, solver: {info.get('strategy')} "
          f"llm={info.get('llm', {}).get('model')} vlm={(info.get('vlm') or {}).get('model')}")

    resp_file = out / "responses.jsonl"
    done = {}
    if resp_file.exists():
        for l in open(resp_file, encoding="utf-8"):
            r = json.loads(l)
            if r.get("ok"): done[r["id"]] = r
    if a.redo:  # od nowa tylko wskazane zadania (--items), bez --items wszystkie
        for i in (a.items or list(done)): done.pop(i, None)
    groups = {}
    for it in exam["items"]:
        groups.setdefault(it.get("group"), []).append({"id": it["id"], "question": it.get("question", "")})
    todo = [it for it in exam["items"] if it["id"] not in done and (not a.items or it["id"] in a.items)]
    print(f"  zapisane: {len(done)}, do rozwiązania: {len(todo)}")

    lock, t0 = threading.Lock(), time.time()

    fast_after = a.fast_after or (max(60, a.deadline - 240) if a.deadline else 0)

    def solve(it):
        images = []
        for im in it["images"]:
            try: images.append(C.image_payload(root, im))
            except Exception as e: print(f"  {it['id']:6s} UWAGA obraz {im.get('path')}: {e}", flush=True)
        elapsed = time.time() - t0
        payload = {"exam_id": exam["exam_id"], "instructions": exam.get("instructions", ""),
                   "item": {**{k: v for k, v in it.items() if k != "images"}, "images": images},
                   "context": {"group_questions": groups[it.get("group")]},
                   "time_left": (a.deadline - elapsed) if a.deadline else None,
                   "fast": bool(fast_after and elapsed > fast_after)}
        t = time.time()
        try:
            timeout = a.timeout if not a.deadline else max(60, min(a.timeout, a.deadline - elapsed))
            r = http_json(a.solver.rstrip("/") + "/solve", payload, timeout)
            if r.get("id") != it["id"] or not isinstance(r.get("answer"), str):
                raise RuntimeError(f"niepoprawna odpowiedź solvera: {str(r)[:200]}")
            return {**r, "ok": True, "sec": round(time.time() - t, 1)}
        except Exception as e:
            return {"id": it["id"], "ok": False, "error": str(e)[:500], "sec": round(time.time() - t, 1)}

    with ThreadPoolExecutor(max_workers=a.concurrency) as pool, open(resp_file, "a", encoding="utf-8") as f:
        for fut in as_completed([pool.submit(solve, it) for it in todo]):
            r = fut.result()
            with lock:
                f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
                if r["ok"]: done[r["id"]] = r
            status = f"{r['sec']:5.1f}s" if r["ok"] else f"BŁĄD {r['error'][:120]}"
            print(f"  {r['id']:6s} {status}", flush=True)

    # druga próba dla zadań, które zawiodły (po kolei, w trybie szybkim), potem awaryjne zgadywanie w zamkniętych
    failed_once = [it for it in todo if it["id"] not in done]
    if failed_once:
        print(f"  ponawiam {len(failed_once)} zadań w trybie szybkim (równolegle)", flush=True)
        fast_after = 1
        with ThreadPoolExecutor(max_workers=a.concurrency) as pool, open(resp_file, "a", encoding="utf-8") as f:
            for fut in as_completed([pool.submit(solve, it) for it in failed_once]):
                r = fut.result()
                with lock:
                    f.write(json.dumps({**r, "retry": True}, ensure_ascii=False) + "\n"); f.flush()
                    if r["ok"]: done[r["id"]] = r
                print(f"  {r['id']:6s} " + (f"{r['sec']:5.1f}s (2. próba)" if r["ok"] else f"BŁĄD {r['error'][:120]}"), flush=True)
    guessed = []
    items_by_id = {it["id"]: it for it in exam["items"]}
    if not a.no_guess:
        for i in template_ids:
            if i in done or i not in items_by_id: continue
            g = C.closed_guess(C.format_spec(items_by_id[i]))
            if g: done[i] = {"id": i, "answer": g, "ok": True, "guess": True}; guessed.append(i)
    answers = {"exam_id": exam["exam_id"],
               "answers": [{"id": i, "answer": done[i]["answer"] if i in done else ""} for i in template_ids]}
    errors, warnings = C.validate_answers(answers, exam, template_ids)
    (out / "answers.json").write_text(json.dumps(answers, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failed = [i for i in template_ids if i not in done]
    meta = {"exam_id": exam["exam_id"], "solver": info, "seconds": round(time.time() - t0, 1),
            "answered": len(template_ids) - len(failed), "failed": failed, "guessed": guessed, "errors": errors, "warnings": warnings}
    (out / "run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n[{exam['exam_id']}] odpowiedzi: {meta['answered']}/{len(template_ids)} w {meta['seconds']}s -> {out / 'answers.json'}")
    for w in warnings: print(f"  UWAGA {w}")
    for e in errors: print(f"  BŁĄD {e}")
    if guessed: print(f"  awaryjnie zgadnięte (solver zawiódł): {guessed}")
    if failed: print(f"  niezrobione (puste w answers.json, uruchom ponownie): {failed}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
