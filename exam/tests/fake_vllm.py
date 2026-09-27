#!/usr/bin/env python3
"""Atrapa serwera vLLM (API zgodne z OpenAI) do testów harnessu bez GPU i bez internetu.
Odpowiada przewidywalnym tekstem zależnie od promptu; GET /stats zwraca liczbę wywołań i obrazów.

  python exam/tests/fake_vllm.py --port 18000
"""
import argparse, json, re, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATS = {"calls": 0, "images": 0, "describe_calls": 0}
LOCK = threading.Lock()


def text_of(content):
    return content if isinstance(content, str) else " ".join(p.get("text", "") for p in content if p.get("type") == "text")


def reply(body):
    msgs = body["messages"]
    system, last = text_of(msgs[0]["content"]), text_of(msgs[-1]["content"])
    n_img = sum(1 for m in msgs if isinstance(m["content"], list) for p in m["content"] if p.get("type") == "image_url")
    with LOCK:
        STATS["calls"] += 1; STATS["images"] += n_img
        if "Opisujesz ilustrację" in system: STATS["describe_calls"] += 1
    if "Opisujesz ilustrację" in system:
        return f"Rodzaj: ilustracja testowa (obrazów w żądaniu: {n_img}). Napis: „TEST”."
    if "Napisz je jeszcze raz" in last:
        return "Temat nr 2\n" + " ".join(["argument"] * 420)
    if "Temat nr X" in system:
        return "Temat nr 2\n" + " ".join(["argument"] * 200)  # za krótkie: solver vlm-text ma dogenerować
    if re.search(r"wyłącznie w \d+ liniach", last):
        labels = re.findall(r"(?m)^(\S+): <", last)
        if "P albo F" in last:
            return "\n".join(f"{l}. **{'PF'[i % 2]}** – krótkie uzasadnienie" for i, l in enumerate(labels))
        if "litera" in last:
            return "\n".join(f"{l}. B. jakaś odpowiedź" for l in labels)
        return "\n".join(f"{l} – {i + 2}" for i, l in enumerate(labels))
    if "jedną literą" in last:
        return "Odpowiedź: C, ponieważ tak wynika z tekstu."
    m = re.search(r"etykiety z polecenia, każdą w osobnej linii: (.+)", last)
    if m:
        labels = re.findall(r"\S[^:]*?:", m.group(1))
        if "Przepisz ją" in last or "Podobieństwo:" not in labels:  # poprawka etykiet albo zwykła odpowiedź z etykietami
            return "\n".join(f"**{l}** treść testowa" for l in labels)
        return "Oto odpowiedź na zadanie:\n\n**Oba źródła** oceniają Łokietka różnie."  # bez etykiet: solver ma poprawić
    return "Oto odpowiedź na zadanie:\n\n## Odpowiedź\n**Hanza** – treść testowa."


class H(BaseHTTPRequestHandler):
    def send(self, obj):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        self.send(STATS if self.path == "/stats" else {"ok": True})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.send({"choices": [{"message": {"role": "assistant", "content": reply(body)}}],
                   "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, required=True)
    ThreadingHTTPServer(("127.0.0.1", ap.parse_args().port), H).serve_forever()
