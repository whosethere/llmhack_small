#!/usr/bin/env python3
"""Test odporności harnessu na paczkę, która odbiega od mocka (bez GPU, atrapa vLLM):
brak pliku obrazu, zła suma sha256, obraz JPEG pod nazwą .png, wielki obraz, brak pól group/images/max_points,
nietypowe answer_format (1.1: P, a: 1), inny exam_id w szablonie, id w szablonie bez pytania, pusty question.
Do tego: martwy backend (każde zadanie zawodzi -> 2. próba -> awaryjne odpowiedzi w zamkniętych) i limit czasu (tryb szybki).

  python exam/tests/mutate_test.py [--exam submission/history-2023-mock-v1]
"""
import argparse, json, shutil, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent; EXAM_DIR = HERE.parent; ROOT = EXAM_DIR.parent
sys.path.insert(0, str(EXAM_DIR))
import common as C
PY = sys.executable


def wait(url, sec=20):
    for _ in range(sec * 10):
        try: urllib.request.urlopen(url, timeout=1); return
        except Exception: time.sleep(0.1)
    raise SystemExit(f"nie wstał: {url}")


def mutate(src, dst):
    shutil.copytree(src, dst)
    exam = json.loads((dst / "exam.json").read_text(encoding="utf-8"))
    items = exam["items"]; by = {it["id"]: it for it in items}
    with_img = [it for it in items if it["images"]]
    # 1. brak pliku obrazu; 2. zła suma; 3. JPEG pod .png; 4. wielki obraz
    (dst / with_img[0]["images"][0]["path"]).unlink()
    with_img[1]["images"][0]["sha256"] = "0" * 64
    from PIL import Image
    p3 = dst / with_img[2]["images"][0]["path"]; Image.open(p3).convert("RGB").save(p3, "JPEG")
    p4 = dst / with_img[3]["images"][0]["path"]; Image.new("RGB", (5000, 4000), "white").save(p4, "PNG")
    for it in (with_img[2], with_img[3]):
        for im in it["images"]: im.pop("sha256", None)
    # 5. brakujące pola
    del with_img[4]["group"]; del with_img[5]["images"]; del with_img[6]["max_points"]
    # 6. nietypowe answer_format
    pf = [it for it in items if it["answer_format"].startswith("1: ")]
    pf[0]["answer_format"] = "1.1: P\n1.2: F\n1.3: P"; pf[1]["answer_format"] = "a: 1\nb: 2\nc: 3"
    # 7. pusty question i source
    open_it = [it for it in items if it["answer_format"].startswith("Tekst")][0]
    open_it["question"] = ""; open_it["source_text"] = ""
    # 8. wypracowanie bez numeru grupy 26: przenumeruj id
    ess = [it for it in items if C.is_essay(it)][0]; ess["id"] = "99"; ess["group"] = 99
    exam["exam_id"] = "history-final-xyz"
    (dst / "exam.json").write_text(json.dumps(exam, ensure_ascii=False), encoding="utf-8")
    # 9. szablon: inny exam_id, dodatkowe id bez pytania, wypracowanie pod nowym id
    tpl = {"exam_id": "history-final-xyz-template", "answers": [{"id": it["id"], "answer": ""} for it in items] + [{"id": "100", "answer": ""}]}
    (dst / "answers-template.json").write_text(json.dumps(tpl), encoding="utf-8")
    return exam, tpl


def run_runner(exam, port, out, *extra):
    r = subprocess.run([PY, str(EXAM_DIR / "run_exam.py"), "--exam", str(exam), "--solver", f"http://localhost:{port}",
                        "--out", str(out), "--concurrency", "6", *extra], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--exam", default=str(ROOT / "submission/history-2023-mock-v1"))
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="exam-mut-")); pkg = tmp / "pkg"
    exam, tpl = mutate(Path(a.exam), pkg)
    tids = [x["id"] for x in tpl["answers"]]
    ok = True
    def check(cond, msg):
        nonlocal ok; print(("  OK   " if cond else "  FAIL ") + msg); ok &= bool(cond)
    procs = []
    try:
        procs.append(subprocess.Popen([PY, str(HERE / "fake_vllm.py"), "--port", "18100"]))
        procs.append(subprocess.Popen([PY, str(EXAM_DIR / "solver_server.py"), "--strategy", "single-vlm", "--port", "19100",
                                       "--llm-url", "http://localhost:18100/v1", "--img-scale", "budget", "--think", "on",
                                       "--cache-dir", str(tmp / "cache")], stdout=subprocess.DEVNULL, stderr=open(tmp / "solver.log", "w")))
        procs.append(subprocess.Popen([PY, str(EXAM_DIR / "solver_server.py"), "--strategy", "single-vlm", "--port", "19101",
                                       "--llm-url", "http://localhost:18199/v1", "--rag-url", "http://localhost:18198",
                                       "--cache-dir", str(tmp / "cache2")], stdout=subprocess.DEVNULL, stderr=open(tmp / "solver2.log", "w")))
        for u in ("http://localhost:18100/health", "http://localhost:19100/health", "http://localhost:19101/health"): wait(u)

        print("paczka zmutowana, backend działa:")
        rc, log = run_runner(pkg, 19100, tmp / "out1", "--deadline", "3")
        ans = json.loads((tmp / "out1" / "answers.json").read_text(encoding="utf-8"))
        ex2, _, tids2 = C.load_exam(pkg)
        errors, warnings = C.validate_answers(ans, ex2, tids2)
        check(rc == 0 and not errors, f"runner kończy bez błędów walidacji (rc={rc}, błędy={errors})")
        check(ans["exam_id"] == "history-final-xyz-template", "exam_id wzięty z szablonu odpowiedzi")
        check([x["id"] for x in ans["answers"]] == tids, "kolejność i komplet id jak w szablonie (z id bez pytania)")
        got = {x["id"]: x["answer"] for x in ans["answers"]}
        check(got["100"] == "", "id bez pytania -> pusta odpowiedź, bez wywrotki")
        check(all(got[i].strip() for i in tids if i != "100"), "każde zadanie z pytaniem ma niepustą odpowiedź")
        check(sum(1 for w in warnings if "pusta" in w) == 1, "walidator: tylko jedna pusta odpowiedź (id 100)")
        resp = [json.loads(l) for l in open(tmp / "out1" / "responses.jsonl", encoding="utf-8")]
        check(any(r.get("debug", {}).get("fast") for r in resp), "po przekroczeniu limitu czasu solver odpowiada w trybie szybkim")
        check(all(r.get("ok") for r in resp), f"żadne zadanie nie zawiodło ({sum(1 for r in resp if not r.get('ok'))} błędów)")
        check("UWAGA paczka" in log and "brak pliku" in log and "sha256" in log, "ostrzeżenia o brakującym obrazie i złej sumie (bez przerwania)")
        spec99 = C.format_spec([it for it in ex2["items"] if it["id"] == "99"][0])
        check(spec99["kind"] == "essay", "wypracowanie rozpoznane po treści, nie po numerze 26")
        s1 = C.format_spec([it for it in ex2["items"] if it["answer_format"].startswith("1.1:")][0])
        check(s1["kind"] == "pairs" and s1["pairs"][0][0] == "1.1", "answer_format z etykietami 1.1/1.2 rozpoznany jako pary")
        check(C.normalize_closed("1.1: P\n1.2: F\n1.3: P", s1) == "1.1: P\n1.2: F\n1.3: P", "normalizacja par z etykietami 1.1")

        print("martwy backend i martwy retrieval:")
        rc, log = run_runner(pkg, 19101, tmp / "out2", "--timeout", "20")
        ans = json.loads((tmp / "out2" / "answers.json").read_text(encoding="utf-8"))
        errors, warnings = C.validate_answers(ans, ex2, tids2)
        meta = json.loads((tmp / "out2" / "run.json").read_text(encoding="utf-8"))
        closed = [it["id"] for it in ex2["items"] if C.format_spec(it)["kind"] in ("letter", "pairs")]
        got = {x["id"]: x["answer"] for x in ans["answers"]}
        check(not errors, f"answers.json nadal poprawny formalnie (błędy={errors})")
        check(set(meta["guessed"]) == set(closed), f"wszystkie zamknięte ({len(closed)}) dostały awaryjną odpowiedź")
        check(all(C.normalize_closed(got[i], C.format_spec(ex2_it)) == got[i] for ex2_it in ex2["items"] for i in [ex2_it["id"]] if i in closed),
              "awaryjne odpowiedzi są w składni answer_format")
        check("ponawiam" in log, "runner ponowił nieudane zadania")
    finally:
        for p in procs: p.kill()
    print("WYNIK:", "wszystko OK" if ok else "SĄ BŁĘDY", f"(katalog: {tmp})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
