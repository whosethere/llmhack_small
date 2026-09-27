#!/usr/bin/env python3
"""Runner ewaluacji: solver (model <=8GB) -> judge (duży model) -> raport per rocznik.

Przykład:
  python eval/run.py --run qwen3-8b-base --solver-url http://localhost:8000/v1 --solver-model Qwen/Qwen3-8B \
      --judge-url http://localhost:8001/v1 --judge-model Qwen/Qwen3-32B-AWQ --years 2023_f2023 2024_f2023
Wyniki: results/<run>/answers.jsonl, judgments.jsonl, report.md (wznawialne - pomija już policzone).
"""
import argparse, asyncio, json, re, sys, time
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from openai import AsyncOpenAI
import prompts

ROOT = Path(__file__).resolve().parent.parent

def load_jsonl(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []

def key_of(r):
    return r["id"].split("_z")[0]

async def solve(client, model, q, sem, temperature, max_tokens, img_desc=None, provider="local", reasoning=None, rag=None, rag_k=5, rag_skip_closed=False, images=False):
    ctx = (q["kontekst"] + "\n") if q["kontekst"] else ""
    if rag and q["id"] in rag and not (rag_skip_closed and q["typ"] == "zamkniete"):
        hits = rag[q["id"]]["hits"][:rag_k][::-1]  # odwrócone: najlepszy ostatni, najbliżej zadania
        frag = "\n###\n".join(h["text"] for h in hits)
        ctx = prompts.RAG_NOTE.format(fragmenty=frag) + ctx
    desc = (img_desc or {}).get(q["id"].split(".")[0])
    if desc and "BRAK ŹRÓDŁA GRAFICZNEGO" not in desc[:60]:
        note = prompts.IMG_DESC_NOTE.format(opis=desc)
    elif desc:
        note = ""  # VLM stwierdził, że na stronie nie ma grafiki
    else:
        note = prompts.IMG_NOTE if (q["wymaga_obrazu"] and not images) else ""
    user = note + prompts.SOLVER_USER.format(kontekst=ctx, tresc=q["tresc"])
    essay = q["typ"] == "wypracowanie"
    system = prompts.ESSAY_SYSTEM if essay else prompts.SOLVER_SYSTEM
    content = user
    if images and q.get("obrazy") and not essay:  # multimodalnie: strony arkusza z ilustracjami przed tekstem zadania
        import base64
        parts = []
        for pth in q["obrazy"][:4]:
            fp = ROOT / pth
            if fp.exists():
                parts.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(fp.read_bytes()).decode()}})
        if parts:
            content = parts + [{"type": "text", "text": prompts.IMG_PAGE_NOTE + user}]
    async with sem:
        t = time.time()
        r = await chat(client, provider, model, [{"role": "system", "content": system}, {"role": "user", "content": content}],
                       max_out=max(max_tokens, 3000) if essay else max_tokens, reasoning=reasoning, temperature=temperature)
    txt = r.choices[0].message.content or ""
    txt = re.sub(r"<think>.*?</think>\s*", "", txt, flags=re.S)  # Qwen3 thinking
    if not txt.strip() and provider == "local":  # myślenie zjadło limit -> powtórka z większym limitem, a potem bez myślenia
        global THINK
        for think, mx in ((THINK, 8000), (False, max_tokens)):
            old = THINK; THINK = think
            try:
                async with sem:
                    r = await chat(client, provider, model, [{"role": "system", "content": system}, {"role": "user", "content": content}], max_out=mx, reasoning=reasoning, temperature=temperature)
            finally:
                THINK = old
            txt = re.sub(r"<think>.*?</think>\s*", "", r.choices[0].message.content or "", flags=re.S)
            if txt.strip(): break
    return {"id": q["id"], "odpowiedz": txt.strip(), "sek": round(time.time() - t, 1), "prompt": user}

THINK = None  # --think on/off dla solvera lokalnego (chat_template_kwargs.enable_thinking)
TOK = re.compile(r"\b([PF]|[A-D]|[1-4])\b")

def closed_check(q, ans):
    """Deterministyczna ocena zamkniętych: klucz składa się WYŁĄCZNIE z tokenów P/F, A-D, 1-4
    (plus separatory). Zwraca punkty przy pełnej zgodności, None gdy nie da się rozstrzygnąć (-> sędzia LLM)."""
    key = (q["klucz"]["rozwiazanie"] or "").split("---")[0]
    kt = TOK.findall(key)
    rest = re.sub(r"[PFA-D1-4]", "", "".join(TOK.findall(key)))  # sanity
    leftover = re.sub(r"[\s\-–.,;:()\[\]/|]|\b([PF]|[A-D]|[1-4])\b", "", key)
    if not kt or leftover or q["typ"] == "wypracowanie":
        return None
    at = TOK.findall(ans["odpowiedz"])
    if len(at) >= len(kt) and (at[-len(kt):] == kt or at[:len(kt)] == kt):
        return q["punkty_max"]
    return None

NUMW = {"jedno": 1, "jedna": 1, "jeden": 1, "dwa": 2, "dwie": 2, "trzy": 3, "cztery": 4, "pięć": 5}
LET = "A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż"

def pf_choices(txt, n):
    """Wybory P/F dla stwierdzeń 1..n z odpowiedzi w formie listy numerowanej; None, gdy nie da się ich jednoznacznie odczytać."""
    pos = [(m.start(), int(m.group(1))) for m in re.finditer(r"(?m)^[\s*#]*([1-9])\s*[.)]", txt)]
    segs = {}
    for i, (p, num) in enumerate(pos):
        segs.setdefault(num, txt[p:pos[i + 1][0] if i + 1 < len(pos) else len(txt)])
    if sorted(segs)[:n] != list(range(1, n + 1)): return None
    out = []
    for k in range(1, n + 1):
        seg = re.sub(r"^[\s*#]*\d\s*[.)]", "", segs[k])
        tok = set(re.findall(rf"(?<![{LET}])([PF])(?![{LET}])", seg))
        if not tok:  # słownie: prawda / fałsz
            tok = {x for x, w in (("P", r"(?i)prawd"), ("F", r"(?i)fałsz")) if re.search(w, seg)}
        if len(tok) != 1: return None
        out.append(tok.pop())
    return out

def closed_partial(q, ans):
    """Prawda/fałsz z punktacją częściową (np. 2 pkt – trzy prawidłowe wskazania, 1 pkt – dwa): liczy trafienia według progów klucza.
    None, gdy klucz nie jest listą P/F albo odpowiedzi nie da się jednoznacznie odczytać (wtedy ocenia sędzia)."""
    k = q["klucz"]
    rozw = (k["rozwiazanie"] or "").split("---")[0]
    if not re.fullmatch(r"[\s\dPF.,;:()–-]*", rozw): return None  # klucz musi być samą listą „nr – P/F”
    pairs = re.findall(r"(\d+)\s*[.)]?\s*[–-]\s*([PF])\b", rozw)
    if [int(n) for n, _ in pairs] != list(range(1, len(pairs) + 1)): return None
    key = [v for _, v in pairs]
    th = {NUMW[w]: int(p) for p, w in re.findall(r"(\d)\s*pkt\s*[–-][^\n]*?\b(" + "|".join(NUMW) + r")\s+prawidłow", k["zasady_oceniania"] or "")}
    if len(key) < 2 or not th or q["typ"] == "wypracowanie": return None
    c = pf_choices(ans["odpowiedz"], len(key))
    return None if c is None else th.get(sum(x == y for x, y in zip(c, key)), 0)

JUDGES = {  # nazwa -> (provider, model, reasoning_effort[, wersja promptu; domyślnie v1])
    "gpt-6-sol": ("forgehand", "gpt-6-sol", "low"),
    "gpt-6-luna": ("forgehand", "gpt-6-luna", "none"),
    "qwen3-32b": ("local", "judge", None),
    "gpt-6-sol-v2": ("forgehand", "gpt-6-sol", "low", "v2"),    # zasady CKE + reguły organizatorów (eval/calibrate.py)
    "gpt-6-luna-v2": ("forgehand", "gpt-6-luna", "none", "v2"),
    "gpt-6-sol-v3": ("forgehand", "gpt-6-sol", "low", "v3"),    # v2 + reguły z kalibracji (poboczne błędy, minimalne odpowiedzi)
    "gpt-6-sol-v4": ("forgehand", "gpt-6-sol", "low", "v4"),    # v3 + ocena wprost napisanej odpowiedzi i sedna pytania
    "local-v4": ("local", "judge", None, "v4"),
    "kimi-k3-v4": ("kimi", "kimi-k3", None, "v4"),              # Moonshot API; klucz: KIMI_API_KEY albo ~/.config/llmhack/kimi_key                 # dowolny model na vLLM (--judge-url), prompt v4; nazwa pliku przez --as
}
USAGE = {"calls": 0, "in": 0, "out": 0}  # tokeny zużyte przez sędziów API w tym procesie

def clean_zasady(z):
    """Obcina wymagania egzaminacyjne doklejone przez parser przed właściwymi zasadami („N pkt – ...”)."""
    m = re.search(r"(?m)^\s*\d+\s*pkt\s*[–—-]", z or "")
    return z[m.start():] if m else (z or "")

def essay_points(j, words):
    """Punkty wypracowania ze składowych wg zasad CKE; None, gdy składowe są niekompletne."""
    try:
        el = [max(0, min(4, int(x))) for x in j["elementy"]][:3]
        errs, b = max(0, int(j["bledy"])), max(0, min(3, int(j["spojnosc"])))
    except Exception:
        return None
    ded = 0 if errs == 0 else 1 if errs <= 2 else 2 if errs <= 5 else 3
    return max(0, sum(el) - ded) + (b if words >= 300 else 0)

def forgehand_client():
    import os
    tok = os.environ.get("FORGEHAND_TOKEN") or json.load(open(Path.home() / ".config/forgehand/config.json"))["token"]
    team = os.environ.get("FORGEHAND_TEAM_ID", "01a0d9da-82b5-7478-a0c6-b4aa47e4c2ec")
    return AsyncOpenAI(api_key=tok, base_url=f"https://app.forgehand.app/api/v1/teams/{team}/llm/v1", max_retries=0, timeout=180)

def kimi_client():
    import os
    key = os.environ.get("KIMI_API_KEY") or (Path.home() / ".config/llmhack/kimi_key").read_text().strip()
    return AsyncOpenAI(api_key=key, base_url="https://api.moonshot.ai/v1", max_retries=0, timeout=300)

async def chat(client, provider, model, messages, max_out, reasoning=None, temperature=0.0):
    """Jedno wywołanie chat z retry (429/5xx). Forgehand: max_completion_tokens + reasoning_effort, bez temperature."""
    for attempt in range(6):
        try:
            if provider == "forgehand":
                kw = {"max_completion_tokens": max_out}
                if reasoning: kw["reasoning_effort"] = reasoning
                r = await client.chat.completions.create(model=model, messages=messages, **kw)
            elif provider == "kimi":  # Moonshot: modele z rozumowaniem przyjmują tylko temperature=1 (pomijamy parametr)
                r = await client.chat.completions.create(model=model, messages=messages, max_tokens=max_out)
            else:
                kw = {}
                if THINK is not None: kw["extra_body"] = {"chat_template_kwargs": {"enable_thinking": THINK}}
                r = await client.chat.completions.create(model=model, messages=messages, max_tokens=max_out, temperature=temperature, **kw)
            u = getattr(r, "usage", None)
            USAGE["calls"] += 1
            if u: USAGE["in"] += u.prompt_tokens or 0; USAGE["out"] += u.completion_tokens or 0
            return r
        except Exception as e:
            code = getattr(e, "status_code", None)
            if code in (400, 401, 403) or attempt == 5: raise
            await asyncio.sleep(2 ** attempt + 1)

def parse_judge_json(raw):
    """JSON sędziego; naprawia polskie cytaty zamknięte zwykłym cudzysłowem („Galicja" -> „Galicja”), częste u Kimi."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(re.sub(r'„([^"„”\n]*)"', r'„\1”', raw))

async def judge(client, model, q, ans, sem, provider="local", reasoning=None, version="v1", desc=None):
    """desc: opis źródła graficznego dla sędziego (sędzia API nie widzi obrazów)."""
    if not (ans["odpowiedz"] or "").strip():  # brak odpowiedzi końcowej = 0 pkt, bez wywołania API
        return {"id": q["id"], "punkty": 0, "punkty_max": q["punkty_max"], "uzasadnienie": "BRAK ODPOWIEDZI"}
    det = closed_check(q, ans)
    if det is not None:
        return {"id": q["id"], "punkty": det, "punkty_max": q["punkty_max"], "uzasadnienie": "DET: zgodne z kluczem"}
    part = closed_partial(q, ans)
    if part is not None:
        return {"id": q["id"], "punkty": part, "punkty_max": q["punkty_max"], "uzasadnienie": "DET: prawda/fałsz, trafienia wg progów klucza"}
    ctx = (q["kontekst"] + "\n") if q["kontekst"] else ""
    if desc: ctx += "\n" + prompts.JUDGE_DESC.format(opis=desc) + "\n"
    k = q["klucz"]
    essay = q["typ"] == "wypracowanie"
    words = len(ans["odpowiedz"].split())
    meta = f" (liczba wyrazów: {words})" if essay else ""
    if version in ("v2", "v3", "v4"):
        system = {"v2": prompts.JUDGE_SYSTEM_V2, "v3": prompts.JUDGE_SYSTEM_V3, "v4": prompts.JUDGE_SYSTEM_V4}[version]
        zasady = prompts.ESSAY_RUBRIC_CKE if essay else clean_zasady(k["zasady_oceniania"] or k["klucz_raw"])
        rozw = "(nie ma jednego wzorca - oceń według kryteriów)" if essay else (k["rozwiazanie"] or "(brak - oceń wg zasad)")
    else:
        system, zasady = prompts.JUDGE_SYSTEM, k["zasady_oceniania"] or k["klucz_raw"]
        rozw = k["rozwiazanie"] or "(brak - oceń wg zasad)"
    user = prompts.JUDGE_USER.format(punkty_max=q["punkty_max"], kontekst=ctx, tresc=q["tresc"],
        zasady=zasady, rozwiazanie=rozw, odpowiedz=ans["odpowiedz"] or "(brak odpowiedzi)", meta=meta)
    if version in ("v2", "v3", "v4") and essay:
        user += "\n\n" + prompts.ESSAY_JSON_V2
    async with sem:
        r = await chat(client, provider, model, [{"role": "system", "content": system}, {"role": "user", "content": user}],
                       max_out=1500 if provider == "forgehand" else 8000, reasoning=reasoning)  # lokalnie: modele z rozumowaniem (gpt-oss) potrzebują zapasu
    txt = r.choices[0].message.content or ""
    txt = re.sub(r"<think>.*?</think>\s*", "", txt, flags=re.S)
    m = re.search(r"\{.*\}", txt, re.S)
    extra = {}
    try:
        j = parse_judge_json(m.group(0)); pts = int(j["punkty"]); uz = str(j.get("uzasadnienie", ""))
        if version in ("v2", "v3", "v4") and essay:
            ep = essay_points(j, words)
            extra = {"skladowe": {x: j.get(x) for x in ("temat", "elementy", "bledy", "spojnosc")}, "punkty_sedziego": pts}
            if ep is not None: pts = ep
    except Exception:
        pts, uz = 0, "PARSE_ERROR: " + txt[:200]
    pts = max(0, min(q["punkty_max"], pts))
    return {"id": q["id"], "punkty": pts, "punkty_max": q["punkty_max"], "uzasadnienie": uz, **extra}

def report(qs, js, out):
    jm = {j["id"]: j for j in js}
    rows = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for q in qs:
        j = jm.get(q["id"])
        if not j: continue
        for k in ("razem", q["typ"]):
            rows[key_of(q)][k][0] += j["punkty"]; rows[key_of(q)][k][1] += q["punkty_max"]
    def pct(a): return f"{100*a[0]/a[1]:.0f}%" if a[1] else "-"
    lines = ["| rocznik | punkty | % | zamknięte | otwarte | wypracowanie |", "|---|---|---|---|---|---|"]
    form = defaultdict(list)
    for key in sorted(rows):
        r = rows[key]; lines.append(f"| {key} | {r['razem'][0]}/{r['razem'][1]} | {pct(r['razem'])} | {pct(r['zamkniete'])} | {pct(r['otwarte'])} | {pct(r['wypracowanie'])} |")
        form[key.split("_f")[1]].append(100 * r["razem"][0] / r["razem"][1])
    for f, v in sorted(form.items()):
        lines.append(f"| **średnia f{f}** ({len(v)} roczników) | | **{sum(v)/len(v):.1f}%** | | | |")
    txt = "\n".join(lines); Path(out).write_text(txt + "\n"); print(txt)

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--phase", choices=["solve", "judge", "all"], default="all", help="solve: tylko odpowiedzi; judge: tylko ocena zapisanych odpowiedzi")
    ap.add_argument("--solver-url", default="http://localhost:8000/v1"); ap.add_argument("--solver-model", default="solver")
    ap.add_argument("--solver-api", choices=["local", "forgehand"], default="local",
                    help="forgehand: solver przez proxy Forgehand (--solver-model gpt-6-luna / gpt-6-sol), punkt odniesienia")
    ap.add_argument("--solver-reasoning", default="none", help="reasoning_effort dla --solver-api forgehand (none/low/medium/high)")
    ap.add_argument("--judge", default="gpt-6-sol", help="gpt-6-sol | gpt-6-luna | qwen3-32b (lokalny vLLM na --judge-url)")
    ap.add_argument("--judge-url", default="http://localhost:8000/v1", help="dla własnego sędziego: base URL API zgodnego z OpenAI")
    ap.add_argument("--judge-model", help="dla własnego sędziego: nazwa modelu w tym API")
    ap.add_argument("--judge-key-env", help="dla własnego sędziego: zmienna środowiskowa z kluczem API")
    ap.add_argument("--model-id", help="nazwa modelu HF do raportu/dashboardu (np. Qwen/Qwen3-8B-AWQ)")
    ap.add_argument("--years", nargs="*", help="np. 2025_f2023; domyślnie wszystkie f2023")
    ap.add_argument("--img-desc", help="tag opisów obrazów z eval/describe.py (data/img_desc/<tag>.jsonl) wstawianych do promptu")
    ap.add_argument("--rag", help="plik z rag/retrieve.py (data/rag/<tag>.jsonl): fragmenty Wikipedii w prompcie")
    ap.add_argument("--rag-k", type=int, default=5, help="ile fragmentów do promptu")
    ap.add_argument("--rag-skip-closed", action="store_true", help="bez fragmentów RAG w zadaniach zamkniętych (psują format odpowiedzi)")
    ap.add_argument("--images", action="store_true", help="model multimodalny: strony arkusza (data/img) jako obrazy w prompcie")
    ap.add_argument("--think", choices=["on", "off"], help="tryb myślenia solvera (Gemma 4 / Qwen3): domyślnie ustawienie serwera")
    ap.add_argument("--skip-images", action="store_true", help="pomiń zadania z wymaga_obrazu")
    ap.add_argument("--limit", type=int); ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0); ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--api-key", default="EMPTY")
    a = ap.parse_args()
    global THINK
    if a.think: THINK = a.think == "on"

    qs = load_jsonl(ROOT / "data/questions.jsonl")
    years = set(a.years) if a.years else {key_of(q) for q in qs if q["formula"] == "2023"}
    qs = [q for q in qs if key_of(q) in years and q["klucz"]]
    if a.skip_images: qs = [q for q in qs if not q["wymaga_obrazu"]]
    if a.limit: qs = qs[: a.limit]
    out = ROOT / "results" / a.run; out.mkdir(parents=True, exist_ok=True)
    if a.phase != "judge" or not (out / "config.json").exists():
        json.dump(vars(a), open(out / "config.json", "w"), indent=2, ensure_ascii=False)

    sem = asyncio.Semaphore(a.concurrency)
    solver = forgehand_client() if a.solver_api == "forgehand" else AsyncOpenAI(base_url=a.solver_url, api_key=a.api_key)
    sprov, sreason = ("forgehand", a.solver_reasoning) if a.solver_api == "forgehand" else ("local", None)
    if a.judge in JUDGES:
        jprov, jmodel, jreason = JUDGES[a.judge][:3]
        jver = JUDGES[a.judge][3] if len(JUDGES[a.judge]) > 3 else "v1"
    else:  # własny sędzia: dowolne API zgodne z OpenAI (vLLM, OpenRouter, Anthropic /v1 compat, ...)
        import os, grading
        grading.check_judge_name(a.judge)
        if not a.judge_model: sys.exit("własny sędzia wymaga --judge-model i --judge-url")
        jprov, jmodel, jreason, jver = "local", a.judge_model, None, "v1"
        if a.judge_key_env: a.api_key = os.environ[a.judge_key_env]
    judge_c = None if a.phase == "solve" else (forgehand_client() if jprov == "forgehand" else AsyncOpenAI(base_url=a.judge_url, api_key=a.api_key))

    img_desc = None
    if a.img_desc:
        dp = ROOT / "data/img_desc" / f"{a.img_desc}.jsonl"
        if not dp.exists(): sys.exit(f"brak {dp} - uruchom eval/describe.py --tag {a.img_desc}")
        img_desc = {x["grupa"]: x["opis"] for x in load_jsonl(dp)}
    rag = None
    if a.rag:
        rp = Path(a.rag) if Path(a.rag).exists() else ROOT / "data/rag" / f"{a.rag}.jsonl"
        if not rp.exists(): sys.exit(f"brak pliku RAG {rp}")
        rag = {x["id"]: x for x in load_jsonl(rp)}
    ans = {x["id"]: x for x in load_jsonl(out / "answers.jsonl")}
    todo = [q for q in qs if q["id"] not in ans] if a.phase != "judge" else []
    print(f"[{a.run}] zadań: {len(qs)}, do rozwiązania: {len(todo)}")
    async def safe_solve(q):
        try:
            return await solve(solver, a.solver_model, q, sem, a.temperature, a.max_tokens, img_desc, sprov, sreason, rag, a.rag_k, a.rag_skip_closed, a.images)
        except Exception as e:  # np. filtr treści Azure przy solverze API - zadanie zostaje bez odpowiedzi, nie 0 pkt
            return {"id": q["id"], "error": f"{type(e).__name__}: {str(e)[:200]}"}
    solve_err = []
    with open(out / "answers.jsonl", "a") as f:
        for coro in asyncio.as_completed([safe_solve(q) for q in todo]):
            r = await coro
            if "error" in r:
                solve_err.append(r); print(f"  SOLVE ERROR {r['id']}: {r['error'][:120]}"); continue
            ans[r["id"]] = r; f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            print(f"  solved {r['id']} ({r['sek']}s)")
    if solve_err:
        with open(out / "solve_errors.jsonl", "w") as f:
            for e in solve_err: f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"[{a.run}] UWAGA: {len(solve_err)} zadań bez odpowiedzi, szczegóły w solve_errors.jsonl")

    if a.phase == "solve":
        print(f"[{a.run}] odpowiedzi zapisane: {len(ans)}; oceń: --phase judge"); return
    jf = out / f"judgments_{a.judge}.jsonl"
    js = {x["id"]: x for x in load_jsonl(jf)}
    todo = [q for q in qs if q["id"] not in js and q["id"] in ans]
    print(f"[{a.run}] do oceny: {len(todo)}")
    async def safe_judge(q):
        try:
            return await judge(judge_c, jmodel, q, ans[q["id"]], sem, jprov, jreason, jver)
        except Exception as e:  # np. filtr treści Azure (400) - nie przerywaj całego przebiegu
            return {"id": q["id"], "error": f"{type(e).__name__}: {str(e)[:200]}"}
    errors = []
    with open(jf, "a") as f:
        for coro in asyncio.as_completed([safe_judge(q) for q in todo]):
            r = await coro
            if "error" in r:
                errors.append(r); print(f"  ERROR {r['id']}: {r['error'][:120]}"); continue
            js[r["id"]] = r; f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            print(f"  judged {r['id']}: {r['punkty']}/{r['punkty_max']}")
    if errors:
        with open(out / f"judge_errors_{a.judge}.jsonl", "w") as f:
            for e in errors: f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"[{a.run}] UWAGA: {len(errors)} zadań bez oceny ({a.judge}), szczegóły w judge_errors_{a.judge}.jsonl")

    report(qs, list(js.values()), out / f"report_{a.judge}.md")

if __name__ == "__main__":
    asyncio.run(main())
