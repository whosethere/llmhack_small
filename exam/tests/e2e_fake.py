#!/usr/bin/env python3
"""Test całego harnessu bez GPU i bez internetu: atrapy vLLM -> solver (vlm-text i base) -> runner -> walidacja.

  python exam/tests/e2e_fake.py [--exam submission/history-2023-mock-v1]
"""
import argparse, json, shutil, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXAM_DIR = HERE.parent
ROOT = EXAM_DIR.parent
sys.path.insert(0, str(EXAM_DIR))
import common as C

PY = sys.executable


def wait(url, sec=20):
    for _ in range(sec * 10):
        try:
            urllib.request.urlopen(url, timeout=1); return
        except Exception:
            time.sleep(0.1)
    raise SystemExit(f"nie wstał: {url}")


def get(url):
    return json.load(urllib.request.urlopen(url, timeout=5))


def run_runner(exam, port, out, *extra):
    r = subprocess.run([PY, str(EXAM_DIR / "run_exam.py"), "--exam", exam, "--solver", f"http://localhost:{port}",
                        "--out", str(out), "--concurrency", "6", *extra], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--exam", default=str(ROOT / "submission/history-2023-mock-v1"))
    a = ap.parse_args()
    exam, root, tids = C.load_exam(a.exam)
    tmp = Path(tempfile.mkdtemp(prefix="exam-e2e-"))
    procs = []
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  OK   " if cond else "  FAIL ") + msg)
        ok &= bool(cond)

    try:
        for port in (18000, 18001):
            procs.append(subprocess.Popen([PY, str(HERE / "fake_vllm.py"), "--port", str(port)]))
        procs.append(subprocess.Popen([PY, str(EXAM_DIR / "solver_server.py"), "--strategy", "vlm-text", "--port", "19000",
                                       "--llm-url", "http://localhost:18000/v1", "--vlm-url", "http://localhost:18001/v1",
                                       "--cache-dir", str(tmp / "cache")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        procs.append(subprocess.Popen([PY, str(EXAM_DIR / "solver_server.py"), "--strategy", "base", "--port", "19001",
                                       "--llm-url", "http://localhost:18000/v1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        for u in ("http://localhost:18000/health", "http://localhost:18001/health", "http://localhost:19000/health", "http://localhost:19001/health"):
            wait(u)

        print("vlm-text:")
        code, log = run_runner(a.exam, 19000, tmp / "vlm")
        ans = json.loads((tmp / "vlm/answers.json").read_text(encoding="utf-8"))
        errors, warnings = C.validate_answers(ans, exam, tids)
        check(code == 0 and not errors, f"answers.json poprawny (kod {code}, błędy {errors})")
        check(not warnings, f"brak ostrzeżeń ({warnings})")
        A = {x["id"]: x["answer"] for x in ans["answers"]}
        check(A["3"] == "1: P\n2: F\n3: P" and A["21"] == "1: B\n2: B" and A["2.2"] == "A: 2\nB: 3", "zamknięte w składni answer_format")
        check(A["11.2"] == "C" and A["13.2"] == "C", "jedna litera")
        check(len(A["26"].split()) > 400, f"wypracowanie dogenerowane ({len(A['26'].split())} wyrazów)")
        check(not C.missing_labels(A["6"], ["Podobieństwo:", "Różnica:"]) and not C.missing_labels(A["7"], C.question_labels(
            next(it["question"] for it in exam["items"] if it["id"] == "7"))), f"etykiety uzupełnione (6: {A['6'][:60]!r})")
        check(not any("**" in v or v.startswith("Oto") or "\n#" in v for v in A.values()), "bez markdownu i wstępów")
        check(A["4.1"].startswith("Odpowiedź"), f"porządki nie ruszają treści ({A['4.1'][:40]!r})")
        n_pairs = len({(im["sha256"], it["group"]) for it in exam["items"] for im in it["images"]})
        vlm = get("http://localhost:18001/stats")
        check(vlm["describe_calls"] == n_pairs, f"VLM: {vlm['describe_calls']} opisów na {n_pairs} par (obraz, grupa)")
        llm = get("http://localhost:18000/stats")
        check(llm["images"] == 0, "model tekstowy nie dostaje obrazów")
        resp = [json.loads(l) for l in open(tmp / "vlm/responses.jsonl", encoding="utf-8")]
        check(all("TEST" in "".join(r["debug"]["descriptions"].values()) for r in resp if r["debug"].get("descriptions")), "opisy ilustracji w debug")
        code2, log2 = run_runner(a.exam, 19000, tmp / "vlm")
        check("do rozwiązania: 0" in log2, "wznowienie: nic do zrobienia przy drugim uruchomieniu")

        print("single-vlm:")
        procs.append(subprocess.Popen([PY, str(EXAM_DIR / "solver_server.py"), "--strategy", "single-vlm", "--port", "19002",
                                       "--llm-url", "http://localhost:18001/v1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        wait("http://localhost:19002/health")
        before = get("http://localhost:18001/stats")
        code, log = run_runner(a.exam, 19002, tmp / "single")
        ans = json.loads((tmp / "single/answers.json").read_text(encoding="utf-8"))
        errors, warnings = C.validate_answers(ans, exam, tids)
        check(code == 0 and not errors, f"answers.json poprawny (kod {code}, błędy {errors})")
        after = get("http://localhost:18001/stats")
        n_img = sum(len(it["images"]) for it in exam["items"])
        check(after["images"] - before["images"] == n_img, f"model dostał wszystkie obrazy inline ({after['images'] - before['images']}/{n_img})")
        A = {x["id"]: x["answer"] for x in ans["answers"]}
        check(A["3"] == "1: P\n2: F\n3: P", "zamknięte w składni answer_format")

        print("base:")
        code, log = run_runner(a.exam, 19001, tmp / "base")
        ans = json.loads((tmp / "base/answers.json").read_text(encoding="utf-8"))
        errors, warnings = C.validate_answers(ans, exam, tids)
        check(code == 0 and not errors, "answers.json poprawny")
        check(any(w.startswith("26:") for w in warnings), "base nie dogenerowuje wypracowania (ostrzeżenie o długości)")

        print("walidator:")
        bad = {"exam_id": exam["exam_id"], "answers": [{"id": "1", "answer": 5}]}
        (tmp / "bad.json").write_text(json.dumps(bad))
        r = subprocess.run([PY, str(EXAM_DIR / "validate.py"), "--exam", a.exam, str(tmp / "bad.json")], capture_output=True, text=True)
        check(r.returncode == 1 and "musi być stringiem" in r.stdout and "brak odpowiedzi" in r.stdout, "zły plik odrzucony")
        r = subprocess.run([PY, str(EXAM_DIR / "solver_server.py"), "--strategy", "base", "--llm-url", "http://example.com/v1"],
                           capture_output=True, text=True)
        check(r.returncode != 0 and "offline" in (r.stdout + r.stderr), "solver odmawia backendu spoza localhost")
    finally:
        for p in procs: p.terminate()
        shutil.rmtree(tmp, ignore_errors=True)
    print("WYNIK:", "wszystko OK" if ok else "SĄ BŁĘDY")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
