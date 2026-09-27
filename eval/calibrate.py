#!/usr/bin/env python3
"""Kalibracja naszego sędziego względem ocen organizatorów (sędzia "org" w results/org-*, import: eval/import_org.py).

  python eval/calibrate.py plan   --judge gpt-6-luna-v2 [--panel | --runs org-a org-b]   # liczba wywołań API i szacunek kosztu
  python eval/calibrate.py grade  --judge gpt-6-luna-v2 [--panel | --runs ...] [--max-calls N]
  python eval/calibrate.py report --judges gpt-6-luna gpt-6-luna-v2 [--panel | --runs ...] [--diffs plik.md]
  python eval/calibrate.py balance

Oszczędzanie kredytu: puste odpowiedzi i zamknięte zgodne z kluczem oceniamy bez API (tak jak organizatorzy),
oceny są wznawialne (plik judgments_<sędzia>.jsonl w każdym runie), --max-calls to twardy limit wywołań,
--panel ogranicza ocenę do 9 przebiegów z całego zakresu wyników (tańszy zestaw dla Sola).
Podział do strojenia bez przeuczenia: zadania o nieparzystym numerze = "tune", parzystym = "val".
"""
import argparse, asyncio, json, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import run as R

ROOT = R.ROOT
# 9 przebiegów od 60/60 do ~30%: tekst i obrazy, w tym modele na bazie Bielika
PANEL = ["org-astra-images", "org-gemma-4-12b-vision", "org-qwen-3-5-9b-vision", "org-ministral-3-14b-text",
         "org-llava-bielik-11b-text", "org-bielik-4-5b", "org-internvl-3-5-8b-vision", "org-gemma-3-4b-text", "org-pllum-4b"]
# średni koszt wywołania z rejestru Forgehand (26.09): Sol 5.71 USD / 1410, Luna 0.018 USD / 111
COST = {"gpt-6-sol": 0.0041, "gpt-6-luna": 0.00016}
ESSAY_MULT = {"v1": 3.0, "v2": 2.0, "v3": 2.0, "v4": 2.0}  # wypracowanie: długi klucz i długa odpowiedź

ITEM_OK = lambda qid: True  # filtr --items (np. z19 z5.2)
DESC = {}  # --desc: opisy obrazów dla sędziego {id: opis}

def load(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []

def questions():
    return {q["id"]: q for q in map(json.loads, open(ROOT / "data/questions.jsonl"))}

def pick_runs(a):
    if a.runs: return a.runs
    if a.panel: return PANEL
    return sorted(d.name for d in (ROOT / "results").iterdir() if d.name.startswith("org-") and (d / "judgments_org.jsonl").exists())

def spec(judge):
    s = R.JUDGES[judge]
    return s[0], s[1], s[2], (s[3] if len(s) > 3 else "v1")

def balance():
    import urllib.request
    tok = json.load(open(Path.home() / ".config/forgehand/config.json"))["token"]
    team = "01a0d9da-82b5-7478-a0c6-b4aa47e4c2ec"
    req = urllib.request.Request(f"https://app.forgehand.app/api/v1/teams/{team}/llm/usage", headers={"Authorization": "Bearer " + tok})
    d = json.load(urllib.request.urlopen(req, timeout=30))
    return d["availableMicroUsd"] / 1e6

def todo_items(qs, runs, judge):  # judge = nazwa pliku ocen (--as albo --judge)
    """-> lista (run, q, ans, rodzaj) do oceny; rodzaj: free (bez API) albo api."""
    out = []
    for run in runs:
        d = ROOT / "results" / run
        done = {j["id"] for j in load(d / f"judgments_{judge}.jsonl")}
        for x in load(d / "answers.jsonl"):
            if x["id"] in done or not ITEM_OK(x["id"]): continue
            q = qs[x["id"]]
            free = not x["odpowiedz"].strip() or R.closed_check(q, x) is not None or R.closed_partial(q, x) is not None
            out.append((run, q, x, "free" if free else "api"))
    return out

def cmd_plan(a, qs):
    prov, model, _, ver = spec(a.judge)
    items = todo_items(qs, pick_runs(a), a.out or a.judge)
    api = [i for i in items if i[3] == "api"]
    essays = sum(1 for i in api if i[1]["typ"] == "wypracowanie")
    est = COST.get(model, 0) * (len(api) - essays + essays * ESSAY_MULT[ver])
    print(f"[{a.out or a.judge}] do oceny: {len(items)} (bez API: {len(items) - len(api)}, przez API: {len(api)}, w tym wypracowań: {essays})"
          f" -> szacunek ~{est:.2f} USD")
    return api, est

async def cmd_grade(a, qs):
    prov, model, reason, ver = spec(a.judge)
    name = a.out or a.judge
    items = todo_items(qs, pick_runs(a), name)
    n_api = sum(1 for i in items if i[3] == "api")
    if a.max_calls is not None and n_api > a.max_calls:
        keep, n = [], 0
        for i in items:
            if i[3] == "api":
                if n >= a.max_calls: continue
                n += 1
            keep.append(i)
        items = keep
        print(f"limit --max-calls {a.max_calls}: oceniam {n} z {n_api} wywołań API")
    cmd_plan(a, qs)
    before = balance() if prov == "forgehand" else None
    client = (R.forgehand_client() if prov == "forgehand" else R.kimi_client() if prov == "kimi"
              else R.AsyncOpenAI(base_url=a.judge_url, api_key="EMPTY"))
    sem = asyncio.Semaphore(a.concurrency)
    files = {run: open(ROOT / "results" / run / f"judgments_{name}.jsonl", "a", encoding="utf-8") for run in {i[0] for i in items}}
    errors = defaultdict(list)

    async def one(run, q, x):
        try:
            return run, await R.judge(client, model, q, x, sem, prov, reason, ver, DESC.get(q["id"]))
        except Exception as e:  # np. filtr treści Azure (400)
            return run, {"id": q["id"], "error": f"{type(e).__name__}: {str(e)[:200]}"}

    done = 0
    for coro in asyncio.as_completed([one(run, q, x) for run, q, x, _ in items]):
        run, r = await coro
        if "error" in r: errors[run].append(r); continue
        files[run].write(json.dumps(r, ensure_ascii=False) + "\n"); files[run].flush()
        done += 1
        if done % 50 == 0: print(f"  ocenione: {done}/{len(items)}")
    for f in files.values(): f.close()
    for run, errs in errors.items():
        with open(ROOT / "results" / run / f"judge_errors_{name}.jsonl", "w") as f:
            for e in errs: f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"[{name}] zapisano {done} ocen, błędy: {sum(len(v) for v in errors.values())}; "
          f"API: {R.USAGE['calls']} wywołań, {R.USAGE['in']} tok. wej., {R.USAGE['out']} tok. wyj.")
    if before is not None:
        await asyncio.sleep(3)
        print(f"saldo: {before:.4f} -> {balance():.4f} USD (wydano ~{before - balance():.4f} USD; rezerwacje mogą się jeszcze rozliczać)")

def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]: j += 1
        for k in range(i, j + 1): r[order[k]] = (i + j) / 2
        i = j + 1
    return r

def spearman(x, y):
    rx, ry = ranks(x), ranks(y); n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** .5; vy = sum((b - my) ** 2 for b in ry) ** .5
    return cov / (vx * vy) if vx and vy else float("nan")

def cmd_report(a, qs):
    runs = pick_runs(a)
    org = {run: {j["id"]: j for j in load(ROOT / "results" / run / "judgments_org.jsonl") if ITEM_OK(j["id"])} for run in runs}
    ans = {run: {x["id"]: x for x in load(ROOT / "results" / run / "answers.jsonl")} for run in runs}
    split = lambda qid: "tune" if qs[qid]["zadanie"] % 2 else "val"
    print(f"przebiegi: {len(runs)}; wzorzec: oceny organizatorów\n")
    print("| sędzia | ocen | zgodne | zamkn. | otwarte | wypr. | tune | val | śr. różnica/zad. | śr. |różnica| | wypr. różnica |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    tot = {}
    for jn in a.judges:
        J = {run: {j["id"]: j for j in load(ROOT / "results" / run / f"judgments_{jn}.jsonl")} for run in runs}
        c = defaultdict(lambda: [0, 0]); diff, adiff, ediff = [], [], []
        tot[jn] = {}
        for run in runs:
            ids = [i for i in org[run] if i in J[run]]
            if len(ids) == len(org[run]):  # sumy tylko dla kompletnie ocenionych przebiegów
                tot[jn][run] = (sum(J[run][i]["punkty"] for i in ids), sum(org[run][i]["punkty"] for i in ids), sum(org[run][i]["punkty_max"] for i in ids))
            for i in ids:
                o, p = org[run][i]["punkty"], J[run][i]["punkty"]; ok = o == p
                for k in ("all", qs[i]["typ"], split(i)): c[k][0] += ok; c[k][1] += 1
                diff.append(p - o); adiff.append(abs(p - o))
                if qs[i]["typ"] == "wypracowanie": ediff.append(p - o)
        pc = lambda k: f"{100 * c[k][0] / c[k][1]:.0f}% ({c[k][1]})" if c[k][1] else "-"
        m = lambda v: f"{sum(v) / len(v):+.2f}" if v else "-"
        print(f"| {jn} | {c['all'][1]} | {pc('all')} | {pc('zamkniete')} | {pc('otwarte')} | {pc('wypracowanie')} | {pc('tune')} | {pc('val')} "
              f"| {m(diff)} | {sum(adiff) / max(1, len(adiff)):.2f} | {m(ediff)} |")
    if a.items:  # zgodność i różnica punktów osobno dla każdego zadania
        print("\n| zadanie | max | ocen | suma org | " + " | ".join(f"{j}: zgodne / suma" for j in a.judges) + " |")
        print("|---|---|---|---|" + "---|" * len(a.judges))
        Js = {j: {run: {x["id"]: x for x in load(ROOT / "results" / run / f"judgments_{j}.jsonl")} for run in runs} for j in a.judges}
        for qid in sorted({i for run in runs for i in org[run]}, key=lambda i: (qs[i]["zadanie"], i)):
            ids = [(run, qid) for run in runs if qid in org[run] and all(qid in Js[j][run] for j in a.judges)]
            so = sum(org[r][i]["punkty"] for r, i in ids)
            cells = []
            for j in a.judges:
                agree = sum(Js[j][r][i]["punkty"] == org[r][i]["punkty"] for r, i in ids)
                sj = sum(Js[j][r][i]["punkty"] for r, i in ids)
                cells.append(f"{agree}/{len(ids)} / {sj} ({sj - so:+d})")
            print(f"| {qid.split('_')[-1]} | {qs[qid]['punkty_max']} | {len(ids)} | {so} | " + " | ".join(cells) + " |")
    full = [r for r in runs if all(r in tot[j] for j in a.judges)]
    if full:
        print("\n| przebieg | max | org | " + " | ".join(a.judges) + " |")
        print("|---|---|---|" + "---|" * len(a.judges))
        for r in sorted(full, key=lambda r: -tot[a.judges[0]][r][1] / tot[a.judges[0]][r][2]):
            _, o, mx = tot[a.judges[0]][r]
            print(f"| {r[4:]} | {mx} | {o} | " + " | ".join(f"{tot[j][r][0]} ({tot[j][r][0] - o:+d})" for j in a.judges) + " |")
        print("\n| sędzia | przebiegi | śr. różnica sumy | śr. |różnica| sumy | maks. |różnica| | Spearman (% wyniku) | zgodna kolejność par (|Δorg| ≥ 5 pp) |")
        print("|---|---|---|---|---|---|---|")
        for j in a.judges:
            d = [tot[j][r][0] - tot[j][r][1] for r in full]
            po = [tot[j][r][1] / tot[j][r][2] for r in full]; pj = [tot[j][r][0] / tot[j][r][2] for r in full]
            pairs = [(x, y) for x in range(len(full)) for y in range(x + 1, len(full)) if abs(po[x] - po[y]) >= .05]
            same = sum((po[x] - po[y]) * (pj[x] - pj[y]) > 0 for x, y in pairs)
            print(f"| {j} | {len(full)} | {sum(d) / len(d):+.2f} | {sum(map(abs, d)) / len(d):.2f} | {max(map(abs, d))} "
                  f"| {spearman(po, pj):.3f} | {same}/{len(pairs)} |")
    if full and len(full) >= 6:  # warstwa 2: liniowa korekta sum per kategoria, ocena leave-one-out
        print("\n| sędzia | śr. |różnica| sumy: surowo | po korekcie (LOO) | śr. różnica po korekcie |")
        print("|---|---|---|---|")
        for j in a.judges:
            J = {run: {x["id"]: x for x in load(ROOT / "results" / run / f"judgments_{j}.jsonl")} for run in full}
            cat = lambda run, src: {t: sum(src[run][i]["punkty"] for i in org[run] if qs[i]["typ"] == t) for t in ("zamkniete", "otwarte", "wypracowanie")}
            ours = {r: cat(r, J) for r in full}; theirs = {r: cat(r, org) for r in full}
            raw, cor, bias = [], [], []
            for r in full:
                pred = 0
                for t in ("zamkniete", "otwarte", "wypracowanie"):
                    xs = [ours[o][t] for o in full if o != r]; ys = [theirs[o][t] for o in full if o != r]
                    mx_, my_ = sum(xs) / len(xs), sum(ys) / len(ys)
                    vx = sum((x - mx_) ** 2 for x in xs)
                    b = sum((x - mx_) * (y - my_) for x, y in zip(xs, ys)) / vx if vx else 1.0
                    pred += my_ + b * (ours[r][t] - mx_)
                o = sum(theirs[r].values())
                raw.append(abs(sum(ours[r].values()) - o)); cor.append(abs(pred - o)); bias.append(pred - o)
            print(f"| {j} | {sum(raw) / len(raw):.2f} | {sum(cor) / len(cor):.2f} | {sum(bias) / len(bias):+.2f} |")
    if a.diffs:
        jn = a.judges[-1]
        rows = []
        for run in runs:
            J = {j["id"]: j for j in load(ROOT / "results" / run / f"judgments_{jn}.jsonl")}
            for i, o in org[run].items():  # org już przefiltrowane przez --items
                if i in J and J[i]["punkty"] != o["punkty"]:
                    rows.append((abs(J[i]["punkty"] - o["punkty"]), run, i, o, J[i]))
        rows.sort(key=lambda r: (r[2], -r[0]))
        with open(a.diffs, "w", encoding="utf-8") as f:
            f.write(f"# Rozbieżności: {jn} vs org ({len(rows)})\n\n")
            for _, run, i, o, j in rows:
                txt = ans[run][i]["odpowiedz"].replace("\n", " ")
                f.write(f"## {i} [{split(i)}, {qs[i]['typ']}] {run[4:]}: {jn}={j['punkty']} org={o['punkty']} /{o['punkty_max']}\n"
                        f"- org: {o['uzasadnienie']}\n- {jn}: {j['uzasadnienie']}\n- odpowiedź: {txt[:500]}{'…' if len(txt) > 500 else ''}\n\n")
        print(f"\nrozbieżności {jn}: {len(rows)} -> {a.diffs}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "grade", "report", "balance"])
    ap.add_argument("--judge"); ap.add_argument("--judges", nargs="*")
    ap.add_argument("--as", dest="out", help="zapisz oceny pod inną nazwą (np. powtórka do pomiaru szumu sędziego)")
    ap.add_argument("--runs", nargs="*"); ap.add_argument("--panel", action="store_true")
    ap.add_argument("--max-calls", type=int); ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--judge-url", default="http://localhost:8000/v1"); ap.add_argument("--diffs")
    ap.add_argument("--items", nargs="*", help="tylko te zadania, np. z19 z5.2 (sufiks id)")
    ap.add_argument("--desc", help="plik {id, opis}: opis obrazu w prompcie sędziego; ocenia tylko zadania z opisem")
    ap.add_argument("--desc-all", action="store_true", help="z --desc: oceniaj wszystkie zadania, opis dołączaj tam, gdzie jest")
    a = ap.parse_args()
    if a.cmd == "balance": print(f"dostępne: {balance():.4f} USD"); return
    qs = questions()
    global ITEM_OK, DESC
    if a.items:
        want = set(a.items); ITEM_OK = lambda qid: qid.split("_")[-1] in want
    if a.desc:
        DESC.update({r["id"]: r["opis"] for r in load(a.desc)})
        if not a.desc_all:
            prev = ITEM_OK; ITEM_OK = lambda qid: qid in DESC and prev(qid)
    if a.cmd == "plan": cmd_plan(a, qs)
    elif a.cmd == "grade": asyncio.run(cmd_grade(a, qs))
    else: cmd_report(a, qs)

if __name__ == "__main__":
    main()
