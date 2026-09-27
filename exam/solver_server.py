#!/usr/bin/env python3
"""Solver API egzaminu: przyjmuje jedno pytanie w JSON i zwraca odpowiedź. Działa offline:
backendy to lokalne serwery vLLM (API zgodne z OpenAI) na localhost. Tylko biblioteka standardowa.

  python exam/solver_server.py --strategy vlm-text --port 9000 \\
      --llm-url http://localhost:8000/v1 --llm-model solver --vlm-url http://localhost:8001/v1 --vlm-model vlm
  python exam/solver_server.py --strategy base --llm-url http://localhost:8000/v1 --llm-model solver [--multimodal]

Strategie:
  base      jeden model, prompt jak w benchmarku organizatorów (model bazowy do liczenia postępu);
            ilustracje tylko z --multimodal; odpowiedzi zamknięte sprowadzane do składni answer_format
  vlm-text  VLM opisuje każdą ilustrację (raz na obraz i grupę zadań), model tekstowy odpowiada z opisami;
            poprawka formatu zamkniętych i dogenerowanie za krótkiego wypracowania

POST /solve  {"exam_id", "instructions", "item": {id, group, max_points, question, source_text, answer_format,
              images: [{path, sha256, mime, data_b64}]}, "context": {"group_questions": [{id, question}]}}
          -> {"id", "answer", "pipeline", "ms", "debug": {...}}   (debug trafia tylko do logu runnera)
GET /health, GET /info
"""
import argparse, os, hashlib, json, re, sys, threading, time, urllib.error, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import common as C
import prompts as P
from ocr import OCR

VERSION = "exam-solver-1"


def strip_think(txt):
    txt = re.sub(r"<think>.*?</think>\s*", "", txt or "", flags=re.S)
    if "<think>" in txt:  # myślenie urwane przed końcem: brak odpowiedzi końcowej
        txt = txt.split("</think>")[-1] if "</think>" in txt else ""
    return txt.strip()


def chat(url, model, messages, max_tokens, temperature=0.0, timeout=600, think=None, seed=None, extra=None):
    """Jedno wywołanie /chat/completions lokalnego vLLM/llama.cpp (bez bibliotek zewnętrznych). think: None = ustawienie serwera; extra: dodatkowe pola payloadu."""
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, **(extra or {})}
    if seed is not None: payload["seed"] = int(seed)
    if think is not None: payload["chat_template_kwargs"] = {"enable_thinking": bool(think)}
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.load(r)
            msg = d["choices"][0]["message"]
            content = strip_think(msg.get("content") or "")
            # pusty content = myślenie zjadło budżet; NIE wklejamy reasoning_content (to nie jest odpowiedź)
            return content, d.get("usage") or {}
        except urllib.error.HTTPError as e:
            if e.code < 500 or attempt == 2:
                raise RuntimeError(f"backend {url}: HTTP {e.code}: {e.read()[:300]!r}")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == 2:
                raise RuntimeError(f"backend {url}: {e}")
        time.sleep(2 * (attempt + 1))


IMG_SCALE = "none"   # --img-scale: powiększanie ilustracji przed wysłaniem do modelu (exam/imgprep.py)
_IMG_CACHE, _IMG_LOCK = {}, threading.Lock()


def image_part(im):
    b64, mime = im["data_b64"], im.get("mime", "image/png")
    if IMG_SCALE != "none":
        import base64 as _b64, imgprep
        key = (im.get("sha256") or hashlib.sha256(b64.encode()).hexdigest(), IMG_SCALE)
        with _IMG_LOCK:
            if key not in _IMG_CACHE:
                try:
                    data, mime2, _ = imgprep.prepare(_b64.b64decode(b64), IMG_SCALE)
                    _IMG_CACHE[key] = (_b64.b64encode(data).decode(), mime2)
                except Exception as e:  # uszkodzony/nietypowy obraz: wysyłamy oryginał
                    sys.stderr.write(f"  UWAGA skalowanie {im.get('path')}: {e}\n"); _IMG_CACHE[key] = (b64, mime)
            b64, mime = _IMG_CACHE[key]
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def header(item):
    return f"Zadanie {item.get('id')} ({item.get('max_points', '?')} pkt)"


class ImageIndex:
    """Ilustracje będące kopiami/wycinkami obrazów z Wikipedii (PL + EN): kandydaci po pHash (rag/phash_images.py,
    odległość Hamminga <= max_dist), potwierdzenie kosinusem DINOv2 między ilustracją a kandydatem (>= min_dino).
    Kalibracja na arkuszu 2023: prawdziwe kopie 0.76-0.94, kolizje pHash <= 0.28. Kilka indeksów: katalogi po przecinku."""
    def __init__(self, paths, max_dist=10, min_dino=0.65, model="facebook/dinov2-base"):
        import numpy as np, torch
        from libzim.reader import Archive
        from transformers import AutoImageProcessor, AutoModel
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rag")); import phash_images as PH
        self.PH, self.np, self.torch, self.max_dist, self.min_dino = PH, np, torch, max_dist, min_dino
        self.idx = []
        for d in paths:
            d = os.path.expanduser(d); meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
            H = np.load(os.path.join(d, "phash.npy"))
            refs = [json.loads(l).get("refs") or [] for l in open(os.path.join(d, "refs.jsonl"), encoding="utf-8")]
            srcs = [json.loads(l)["src"] for l in open(os.path.join(d, "refs.jsonl"), encoding="utf-8")]
            self.idx.append((os.path.basename(d.rstrip("/")), H, refs, srcs, Archive(os.path.expanduser(meta["zim"]))))
        self.proc = AutoImageProcessor.from_pretrained(model); self.model = AutoModel.from_pretrained(model, dtype=torch.float16).cuda().eval()
        self.lock = threading.Lock()
        self.n = sum(len(x[1]) for x in self.idx)

    def _emb(self, ims):
        px = self.proc(images=ims, return_tensors="pt")["pixel_values"].cuda().half()
        with self.torch.no_grad(): o = self.model(pixel_values=px).pooler_output
        return self.torch.nn.functional.normalize(o.float(), dim=1)

    def lookup(self, data):
        import io
        from PIL import Image
        q = Image.open(io.BytesIO(data)).convert("RGB"); h = self.np.uint64(self.PH.phash_pil(q))
        best = None
        with self.lock:
            qe = self._emb([q])
            for name, H, refs, srcs, z in self.idx:
                d = self.PH.hamming(H, h); d[H == 0] = 99
                for j in self.np.argsort(d)[:3]:
                    if int(d[j]) > self.max_dist or not refs[j]: continue
                    try:
                        cos = float((qe @ self._emb([self.PH.load_zim_image(z, srcs[j]).convert("RGB")]).T).item())
                    except Exception:
                        continue
                    if cos >= self.min_dino and (best is None or cos > best["dino"]):
                        best = {"index": name, "dist": int(d[j]), "dino": round(cos, 3), "refs": [(t, c) for t, c in refs[j][:3]]}
        return best


class Solver:
    def __init__(self, a):
        self.a = a
        self.desc_cache, self.locks, self.guard = {}, {}, threading.Lock()
        self.img_index = None
        if a.img_index:
            try:
                self.img_index = ImageIndex([p for p in a.img_index.split(",") if p.strip()], a.img_max_dist, a.img_min_dino)
                sys.stderr.write(f"indeks obrazów: {self.img_index.n} hashy w {len(self.img_index.idx)} indeksach, DINO >= {a.img_min_dino}\n")
            except Exception as e:  # brak indeksu nie zatrzymuje egzaminu
                sys.stderr.write(f"UWAGA: indeks obrazów niedostępny ({e}); jadę bez niego\n")
        self.ocr = OCR(a.ocr, a.ocr_cache, scale=a.ocr_scale) if a.ocr != "none" else None  # cache OCR wspólny dla wszystkich przebiegów
        self.cache_file = Path(a.cache_dir) / "descriptions.jsonl" if a.cache_dir else None
        if self.cache_file and self.cache_file.exists():
            for l in open(self.cache_file, encoding="utf-8"):
                r = json.loads(l); self.desc_cache[r["key"]] = r["desc"]

    def info(self):
        return {"version": VERSION, "strategy": self.a.strategy, "multimodal": self.a.multimodal, "think": self.a.think, "rag": self.a.rag_url, "img_scale": IMG_SCALE, "img_index": bool(self.img_index),
                "describe": {"prompt": self.a.describe_prompt, "lang": self.a.describe_lang, "ocr": self.a.ocr},
                "llm": {"url": self.a.llm_url, "model": self.a.llm_model},
                "vlm": {"url": self.a.vlm_url, "model": self.a.vlm_model} if self.a.strategy in ("vlm-text", "split") else None}

    def describe_prompts(self):
        """(system, user) opisywacza wg --describe-prompt (v1 | v2) i --describe-lang (pl | en)."""
        if self.a.describe_lang == "en":
            return P.DESCRIBE_SYSTEM_V2_EN, P.DESCRIBE_USER_EN
        return (P.DESCRIBE_SYSTEM_V2 if self.a.describe_prompt == "v2" else P.DESCRIBE_SYSTEM), P.DESCRIBE_USER

    def desc_block(self):
        return P.IMG_DESC_BLOCK_EN if self.a.describe_lang == "en" else P.IMG_DESC_BLOCK

    def pipeline(self):
        return f"{self.a.strategy}:{self.a.llm_model}" + (f"+{self.a.vlm_model}" if self.a.strategy in ("vlm-text", "split") else "") + (f"+ocr:{self.a.ocr}" if self.a.ocr != "none" else "")

    # ---- opisy ilustracji (vlm-text) ----
    def describe(self, im, item, questions):
        key = im["sha256"] + ":" + hashlib.sha256("\n".join(questions).encode()).hexdigest()[:12] + f":{self.a.describe_prompt}-{self.a.describe_lang}-{self.a.ocr}"
        with self.guard:
            lock = self.locks.setdefault(key, threading.Lock())
        with lock:  # równoległe pytania z tej samej grupy czekają na jeden opis
            if key in self.desc_cache:
                return self.desc_cache[key], True
            sys_p, usr_t = self.describe_prompts()
            user = usr_t.format(caption=item.get("source_text") or "(brak)", questions="\n".join(questions), path=im["path"])
            if self.ocr:  # OCR jako podpowiedź przed opisem; pusty wynik = brak podpowiedzi
                import base64 as _b64
                txt = self.ocr.read_bytes(_b64.b64decode(im["data_b64"]), im.get("sha256"))
                if txt.strip(): user = P.DESCRIBE_OCR_HINT.format(ocr=txt[:3000]) + "\n" + user
            msgs = [{"role": "system", "content": sys_p},
                    {"role": "user", "content": [image_part(im), {"type": "text", "text": user}]}]
            desc, _ = chat(self.a.vlm_url, self.a.vlm_model, msgs, self.a.describe_max_tokens)
            self.desc_cache[key] = desc
            if self.cache_file:
                with self.guard, open(self.cache_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "path": im["path"], "desc": desc}, ensure_ascii=False) + "\n")
            return desc, False

    # ---- rozwiązywanie ----
    def solve(self, req):
        t0 = time.time()
        item = req["item"]
        spec = C.format_spec(item)
        debug = {"format": spec["kind"], "retries": 0}
        tl = req.get("time_left")
        if bool(req.get("fast")) or (tl is not None and tl < 120):
            debug["fast"] = True  # koniec czasu: bez myślenia i bez retrievalu, byle odpowiedzieć (flaga per żądanie, serwer jest wielowątkowy)
        if self.a.strategy == "base":
            answer = self.solve_base(item, spec, debug)
        elif self.a.strategy == "single-vlm":
            answer = self.solve_single_vlm(item, spec, debug)   # jeden model wizyjny (np. Gemma 4) na wszystko
        elif self.a.strategy == "split" and item["images"] and spec["kind"] != "essay":
            answer = self.solve_vlm_direct(item, spec, debug)   # zadanie z obrazem: VLM odpowiada sam, widząc ilustrację
        elif self.a.strategy == "split":
            answer = self.solve_vlm_text(item, spec, req.get("context") or {}, debug)  # bez obrazu: Bielik (opisy nie są potrzebne)
        else:
            answer = self.solve_vlm_text(item, spec, req.get("context") or {}, debug)
        return {"id": item["id"], "answer": answer, "pipeline": self.pipeline(),
                "ms": int(1000 * (time.time() - t0)), "debug": debug}

    def user_content(self, item, source, instruction, images=None):
        """Treść pytania; przy images (multimodal) obrazy trafiają w miejsca znaczników [Obraz: …]."""
        tail = f"\n\n{item['question']}\n\n{instruction}"
        if not images:
            return f"{header(item)}\n\n{source}{tail}".strip()
        by_path = {im["path"]: im for im in images}
        parts, pos, used = [{"type": "text", "text": header(item) + "\n\n"}], 0, set()
        for m in C.IMG_MARK.finditer(source):
            parts.append({"type": "text", "text": source[pos:m.end()]})
            if m.group(1) in by_path:
                parts.append(image_part(by_path[m.group(1)])); used.add(m.group(1))
            pos = m.end()
        parts += [image_part(im) for p, im in by_path.items() if p not in used]  # obraz bez znacznika
        parts.append({"type": "text", "text": source[pos:] + tail})
        return parts

    def ask(self, url, model, system, user, spec, debug, fix=True, temperature=None, seed=None):
        max_tokens = self.a.essay_max_tokens if spec["kind"] == "essay" else self.a.max_tokens
        temp = self.a.temperature if temperature is None else temperature
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        think = {"on": True, "off": False, "auto": spec["kind"] not in ("letter", "pairs"), "server": None}[self.a.think]
        if debug.get("fast"): think = False
        raw, usage = chat(url, model, msgs, max_tokens, temp, think=think, seed=seed)
        if not raw.strip():  # myślenie zjadło budżet: 2x budżet, a potem bez myślenia
            raw, usage = chat(url, model, msgs, max_tokens * 2, temp, think=think, seed=seed)
            debug["retry_empty"] = True
        if not raw.strip() and think is not False:
            raw, usage = chat(url, model, msgs, max_tokens, self.a.temperature, think=False)
            debug["retry_nothink"] = True
        debug["raw"], debug["usage"] = raw, usage
        answer = raw
        if spec["kind"] in ("letter", "pairs"):
            norm = C.normalize_closed(raw, spec)
            if norm is None and fix:  # prośba o sam format, bez zmiany treści
                debug["retries"] += 1
                retry = P.FORMAT_RETRY.format(answer=raw, instruction=C.format_instruction(spec))
                raw2, _ = chat(url, model, msgs + [{"role": "assistant", "content": raw}, {"role": "user", "content": retry}], 200, 0.0)
                debug["raw_format_retry"] = raw2
                norm = C.normalize_closed(raw2, spec)
            answer = norm if norm is not None else raw
            debug["normalized"] = norm is not None
        elif fix:  # otwarte i wypracowanie: porządki formatu, etykiety z polecenia, długość wypracowania
            answer = C.clean_answer(raw)
            if spec["kind"] == "open" and spec.get("labels") and C.missing_labels(answer, spec["labels"]):
                debug["retries"] += 1
                retry = P.LABELS_RETRY.format(answer=answer, labels=" ".join(spec["labels"]))
                raw2, _ = chat(url, model, msgs + [{"role": "assistant", "content": raw}, {"role": "user", "content": retry}], max_tokens, 0.0)
                debug["raw_labels_retry"] = raw2
                fixed = C.clean_answer(raw2)
                if not C.missing_labels(fixed, spec["labels"]): answer = fixed
                debug["labels_ok"] = not C.missing_labels(answer, spec["labels"])
            if spec["kind"] == "essay":
                words = len(answer.split())
                if words < self.a.essay_min_words:
                    debug["retries"] += 1
                    raw2, _ = chat(url, model, msgs + [{"role": "assistant", "content": raw},
                                                      {"role": "user", "content": P.ESSAY_EXTEND.format(words=words)}], max_tokens, self.a.temperature)
                    debug["raw_essay_retry"] = raw2
                    fixed = C.clean_answer(raw2)
                    if len(fixed.split()) > words: answer = fixed
        return answer.strip()

    def solve_base(self, item, spec, debug):
        mm = self.a.multimodal and item["images"]
        system = P.BASE_SYSTEM_IMAGES if mm else P.BASE_SYSTEM_TEXT
        instruction = C.format_instruction(spec) if spec["kind"] in ("letter", "pairs") else item.get("answer_format", "")
        source = item.get("source_text") or ""
        user = self.user_content(item, source, instruction, item["images"] if mm else None)
        return self.ask(self.a.llm_url, self.a.llm_model, system, user, spec, debug, fix=False)

    def solve_single_vlm(self, item, spec, debug):
        """Jeden model VLM: obrazy inline w miejscach znaczników, OCR (jeśli włączony) jako podpowiedź napisów tuż za obrazem."""
        import base64 as _b64
        source = item.get("source_text") or ""
        if self.ocr and item["images"]:
            hints = {}
            for im in item["images"]:
                try:
                    txt = self.ocr.read_bytes(_b64.b64decode(im["data_b64"]), im.get("sha256"))
                except Exception as e:  # OCR to tylko podpowiedź: awaria nie może zabrać zadania
                    debug["ocr_error"] = f"{im.get('path')}: {str(e)[:120]}"; txt = ""
                if txt.strip(): hints[im["path"]] = P.SINGLE_VLM_OCR_HINT.format(path=im["path"], ocr=txt[:2000])
            debug["ocr_hints"] = list(hints)
            source = C.IMG_MARK.sub(lambda m: m.group(0) + ("\n" + hints[m.group(1)] if m.group(1) in hints else ""), source)
        if self.img_index and item["images"]:
            wiki = {}
            for im in item["images"]:
                try:
                    hit = self.img_index.lookup(_b64.b64decode(im["data_b64"]))
                except Exception as e:
                    debug["img_index_error"] = str(e)[:120]; hit = None
                if hit:
                    refs = "; ".join(f"artykuł „{t}”" + (f", podpis: „{c}”" if c else "") for t, c in hit["refs"])
                    lang = "angielskiej" if "en" in hit["index"] else "polskiej"
                    wiki[im["path"]] = P.SINGLE_VLM_WIKI_HINT.format(path=im["path"], lang=lang, refs=refs)
                    debug.setdefault("wiki_hits", []).append({"path": im["path"], **{k: v for k, v in hit.items() if k != "refs"}})
            debug["wiki_images"] = {k: v[:200] for k, v in wiki.items()}
            if wiki:
                source = C.IMG_MARK.sub(lambda m: m.group(0) + ("\n" + wiki[m.group(1)] if m.group(1) in wiki else ""), source)
        system = P.ESSAY_SYSTEM if spec["kind"] == "essay" else (P.SINGLE_VLM_SYSTEM_TEXT if self.a.text_only else P.SINGLE_VLM_SYSTEM)
        if spec["kind"] == "essay" and self.a.essay_topic == "auto":
            system = P.ESSAY_SYSTEM_TOPIC.format(n=self.choose_essay_topic(item, debug))
        if spec["kind"] == "essay" and not debug.get("fast") and (self.a.essay_rag or self.a.essay_best_of > 1):
            return self.solve_essay(item, spec, debug, system, source)
        if self.a.rag_url and not debug.get("fast") and (self.a.rag_scope == "all" or (self.a.rag_scope == "open" and spec["kind"] not in ("letter", "pairs"))):
            source = self.rag_prefix(item, source, debug, always=self.a.rag_always and spec["kind"] != "essay") + source  # esej: bez wymuszania
        images = None if self.a.text_only else (item["images"] or None)   # model tekstowy: zostają OCR, podpisy z Wikipedii i RAG
        if self.a.text_only and spec["kind"] == "essay":   # mały model tekstowy: wypracowanie modułem essay_small (szablon + akapity z notatkami)
            return self.essay_small(item, debug)
        if self.a.text_only and spec["kind"] != "essay":   # rama promptu jak w benchmarku eval/run.py (bez nagłówka i instrukcji formatu): mały model odpowiada pełniej; literę wyciąga normalize_closed
            if self.a.text_headers:   # etykiety sekcji: mały model lepiej rozdziela źródło od polecenia (test A/B)
                user = (f"ŹRÓDŁO:\n{source}\n\n" if source.strip() else "") + f"POLECENIE:\n{item['question']}\n\nOdpowiedź:"
            else:
                user = f"{source}\n{item['question']}\n\nOdpowiedź:".strip()
        else:
            user = self.user_content(item, source, C.format_instruction(spec), images)
        return self.ask(self.a.llm_url, self.a.llm_model, system, user, spec, debug, fix=True)

    def solve_essay(self, item, spec, debug, system, source):
        """Wypracowanie: (1) opcjonalnie retrieval przed pisaniem: model wybiera temat i pisze zapytania, fragmenty trafiają
        do promptu jako materiał; (2) opcjonalnie N wersji (pierwsza deterministyczna, reszta z temperaturą i ziarnem),
        z których model sam wybiera najlepszą według rubryki CKE (stanowisko, trzy elementy, fakty, ramy czasowe, błędy)."""
        from concurrent.futures import ThreadPoolExecutor
        facts = ""
        if self.a.essay_rag and self.a.rag_url:
            try:
                q = (item.get("question") or "")[:4000]
                raw, _ = chat(self.a.llm_url, self.a.llm_model, [{"role": "system", "content": P.ESSAY_RAG_QUERY_SYSTEM},
                              {"role": "user", "content": P.ESSAY_RAG_QUERY_USER.format(question=q)}], 400, 0.0, think=False)
                m = re.search(r"\{.*\}", raw or "", re.S); j = json.loads(m.group(0)) if m else {}
                qs = [x for x in (j.get("zapytania") or []) if isinstance(x, str) and x.strip()][:5]
                debug["essay_rag"] = {"temat": j.get("temat"), "zapytania": qs}
                if qs:
                    body = json.dumps({"queries": qs, "k": self.a.essay_rag_k, "min_score": self.a.rag_min_score}).encode()
                    req = urllib.request.Request(self.a.rag_url.rstrip("/") + "/search", data=body, headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=120) as r: hits = json.load(r).get("hits") or []
                    debug["essay_rag"]["hits"] = [{"title": h["title"], "score": h["score"]} for h in hits]
                    if hits:
                        frags = "\n###\n".join(f"[{h['title']}] {h['text']}" for h in hits)
                        facts = P.ESSAY_RAG_BLOCK.format(fragments=frags)
            except Exception as e:
                debug["essay_rag_error"] = f"{type(e).__name__}: {str(e)[:150]}"
        user = self.user_content(item, facts + source, C.format_instruction(spec), None)
        n = max(1, int(self.a.essay_best_of))
        if n == 1:
            return self.ask(self.a.llm_url, self.a.llm_model, system, user, spec, debug, fix=True)
        cands = [None] * n; dbgs = [{"format": "essay", "retries": 0} for _ in range(n)]
        def gen(i):
            cands[i] = self.ask(self.a.llm_url, self.a.llm_model, system, user, spec, dbgs[i], fix=True,
                                temperature=(None if i == 0 else self.a.essay_temp), seed=(None if i == 0 else 1000 + i))
        with ThreadPoolExecutor(n) as pool: list(pool.map(gen, range(n)))
        debug["essay_candidates"] = [{"words": len((c or "").split()), "retries": d.get("retries", 0)} for c, d in zip(cands, dbgs)]
        valid = [i for i, c in enumerate(cands) if c and len(c.split()) >= 250]
        if len(valid) <= 1:
            pick = valid[0] if valid else max(range(n), key=lambda i: len((cands[i] or "").split()))
            debug["essay_pick"] = {"wybor": pick, "powod": "jedyny/najdłuższy kandydat"}; return cands[pick] or ""
        listing = "\n\n".join(f"=== WERSJA {i + 1} ===\n{cands[i]}" for i in valid)
        try:
            msgs = [{"role": "system", "content": P.ESSAY_PICK_SYSTEM},
                    {"role": "user", "content": P.ESSAY_PICK_USER.format(question=(item.get("question") or "")[:4000], facts=facts or "(brak)", essays=listing)}]
            raw, _ = chat(self.a.llm_url, self.a.llm_model, msgs, self.a.essay_max_tokens, 0.0, think=True)  # myślenie zjada tokeny: limit jak dla eseju
            m = re.search(r"\{.*\}", raw or "", re.S); j = json.loads(m.group(0)) if m else {}
            if not j:  # brak JSON (myślenie urwane): jeszcze raz bez myślenia
                raw, _ = chat(self.a.llm_url, self.a.llm_model, msgs, 1500, 0.0, think=False)
                m = re.search(r"\{.*\}", raw or "", re.S); j = json.loads(m.group(0)) if m else {}
                debug["essay_pick_nothink"] = True
            pick = int(j.get("wybor", 1)) - 1
            if pick not in valid: pick = valid[0]
            debug["essay_pick"] = {"wybor": pick, "ocena": j}
        except Exception as e:
            pick = valid[0]; debug["essay_pick"] = {"wybor": pick, "error": str(e)[:150]}
        return cands[pick]

    def essay_small(self, item, debug):
        import essay_small as ES
        penalty = {"repeat_penalty": 1.05, "repetition_penalty": 1.05}   # llama.cpp / vLLM: bez tego 1.5B przy temp 0 potrafi się zapętlić
        def ch(system, user, max_tokens):
            raw, _ = chat(self.a.llm_url, self.a.llm_model, [{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens, self.a.temperature, think=False, extra=penalty)
            return raw
        writer = ES.EssaySmall(ch, rag_url=self.a.rag_url if not debug.get("fast") else None, rag_k=self.a.essay_rag_k)
        text = writer.write(item.get("question") or "", debug)
        debug["raw"] = text[:300]; debug["usage"] = None
        return text

    def choose_essay_topic(self, item, debug):
        """Osobny krok: model (bez myślenia) wskazuje numer tematu, o którym wie najwięcej; brak cyfry = temat 1."""
        import essay as E
        q = (item.get("question") or "") + "\n" + (item.get("source_text") or "")
        raw, _ = chat(self.a.llm_url, self.a.llm_model, [{"role": "system", "content": E.CHOOSE_SYSTEM},
                      {"role": "user", "content": E.CHOOSE_USER.format(question=q)}], 20, 0.0, think=False)
        m = re.search(r"[123]", raw or "")
        n = int(m.group(0)) if m else 1
        debug["topic_choice"] = {"raw": (raw or "")[-40:], "n": n}
        return n

    def rag_prefix(self, item, source, debug, always=None):
        if always is None: always = self.a.rag_always
        """Krok 1: model (bez myślenia) mówi, czego nie wie -> zapytania; retrieval z progiem; brak trafień = brak fragmentów."""
        task = (item.get("question") or "") + "\n" + C.IMG_MARK.sub("[ilustracja]", item.get("source_text") or "")[:2500]
        try:
            if self.a.rag_query == "task":   # treść zadania jako jedno zapytanie, zawsze (mały model pisze słabe zapytania)
                qs = [task[:1500]]; debug["rag_queries"] = ["<treść zadania>"]; debug["rag_need"] = True
            else:
                raw, _ = chat(self.a.llm_url, self.a.llm_model, [{"role": "system", "content": P.RAG_QUERY_SYSTEM_ALWAYS if always else P.RAG_QUERY_SYSTEM}, {"role": "user", "content": P.RAG_QUERY_USER.format(task=task)}], 300, 0.0, think=False)
                m = re.search(r"\{.*\}", raw, re.S); j = json.loads(m.group(0)) if m else {}
                qs = [q for q in (j.get("zapytania") or []) if isinstance(q, str) and q.strip()][:3]
                debug["rag_queries"] = qs; debug["rag_need"] = bool(j.get("potrzeba"))
                if not qs or (not j.get("potrzeba") and not always): return ""
            body = json.dumps({"queries": qs, "k": self.a.rag_k, "min_score": self.a.rag_min_score}).encode()
            with urllib.request.urlopen(urllib.request.Request(self.a.rag_url.rstrip("/") + "/search", data=body, headers={"Content-Type": "application/json"}), timeout=120) as r:
                hits = json.load(r).get("hits", [])
            debug["rag_hits"] = [{"title": h["title"], "score": h["score"]} for h in hits]
            if not hits: return ""
            frags = "\n###\n".join(h["text"][:1200] for h in hits[::-1])  # najlepszy ostatni, najbliżej zadania
            return P.RAG_BLOCK.format(fragments=frags)
        except Exception as e:
            debug["rag_error"] = f"{type(e).__name__}: {str(e)[:150]}"; return ""

    def solve_vlm_direct(self, item, spec, debug):
        """Strategia split: VLM dostaje source_text z ilustracjami w miejscach znaczników i odpowiada; porządki formatu jak w vlm-text."""
        debug["solver"] = "vlm"
        system = P.SOLVER_SYSTEM.replace("Opisy ilustracji traktuj jak to, co widać na ilustracji.", "Ilustracje widzisz bezpośrednio: odwołuj się do konkretnych widocznych elementów i napisów.")
        user = self.user_content(item, item.get("source_text") or "", C.format_instruction(spec), item["images"])
        return self.ask(self.a.vlm_url, self.a.vlm_model, system, user, spec, debug, fix=True)

    def solve_vlm_text(self, item, spec, ctx, debug):
        questions = [f"[{q['id']}] {q['question']}" for q in ctx.get("group_questions") or [{"id": item["id"], "question": item["question"]}]]
        descs = {}
        for im in item["images"]:
            descs[im["path"]], cached = self.describe(im, item, questions)
        debug["descriptions"] = descs
        source = item.get("source_text") or ""
        block = self.desc_block()
        source = C.IMG_MARK.sub(lambda m: block.format(path=m.group(1), desc=descs.get(m.group(1), "(brak opisu)")), source)
        missing = [p for p in descs if f"[Obraz: {p}]" not in (item.get("source_text") or "")]
        if missing:  # obraz bez znacznika w tekście: opis na końcu źródeł
            source += "\n\n" + "\n\n".join(block.format(path=p, desc=descs[p]) for p in missing)
        system = P.ESSAY_SYSTEM if spec["kind"] == "essay" else P.SOLVER_SYSTEM
        user = self.user_content(item, source, C.format_instruction(spec))
        return self.ask(self.a.llm_url, self.a.llm_model, system, user, spec, debug, fix=True)


def make_handler(solver):
    class H(BaseHTTPRequestHandler):
        def send(self, code, obj):
            b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code); self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

        def do_GET(self):
            if self.path == "/health": return self.send(200, {"ok": True})
            if self.path == "/info": return self.send(200, solver.info())
            self.send(404, {"error": "nieznana ścieżka"})

        def do_POST(self):
            if self.path != "/solve": return self.send(404, {"error": "nieznana ścieżka"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                item = req["item"]
                if "id" not in item: raise ValueError("brak pola item.id")
                item.setdefault("question", ""); item.setdefault("max_points", 1); item["images"] = item.get("images") or []
                item.setdefault("source_text", ""); item.setdefault("answer_format", "")
            except Exception as e:
                return self.send(400, {"error": f"niepoprawne żądanie: {e}"})
            try:
                out = solver.solve(req)
                sys.stderr.write(f"  {item['id']:6s} {out['ms'] / 1000:5.1f}s {out['debug'].get('format')}\n")
                self.send(200, out)
            except Exception as e:
                sys.stderr.write(f"  {item['id']:6s} BŁĄD {e}\n")
                self.send(500, {"id": item["id"], "error": str(e)})

        def log_message(self, *args):
            pass
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", choices=["base", "vlm-text", "split", "single-vlm"], required=True)
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--llm-url", default="http://localhost:8000/v1"); ap.add_argument("--llm-model", default="solver")
    ap.add_argument("--vlm-url", default="http://localhost:8001/v1"); ap.add_argument("--vlm-model", default="vlm")
    ap.add_argument("--multimodal", action="store_true", help="base: wysyłaj ilustracje do modelu (model wizyjny)")
    ap.add_argument("--text-headers", action="store_true", help="--text-only: etykiety ŹRÓDŁO:/POLECENIE: w prompcie zadania")
    ap.add_argument("--text-only", action="store_true", help="single-vlm: nie wysyłaj ilustracji (model tekstowy, np. Bielik); OCR, indeks obrazów i RAG działają")
    ap.add_argument("--max-tokens", type=int, default=1500); ap.add_argument("--essay-max-tokens", type=int, default=3000)
    ap.add_argument("--describe-max-tokens", type=int, default=1200)
    ap.add_argument("--describe-prompt", choices=["v1", "v2"], default="v2", help="v2: kontekst polskiej matury, ostrożne rozpoznania")
    ap.add_argument("--describe-lang", choices=["pl", "en"], default="pl", help="en: VLM opisuje po angielsku, model tekstowy dostaje opis bez tłumaczenia")
    ap.add_argument("--essay-min-words", type=int, default=350, help="krótsze wypracowanie jest dogenerowywane (vlm-text); próg CKE to 300")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--rag-url", help="serwis retrievalu (rag/retrieve_server.py), np. http://localhost:8010; model sam generuje zapytania")
    ap.add_argument("--rag-always", action="store_true", help="retrieval dla każdego zadania: model zawsze pisze 1-3 zapytania (próg rerankera dalej odsiewa)")
    ap.add_argument("--rag-scope", choices=["all", "open", "essay"], default="all", help="essay: fragmenty tylko do wypracowania (essay_small); open: także zadania otwarte, zamknięte bez")
    ap.add_argument("--rag-query", choices=["model", "task"], default="model", help="model: model pisze zapytania i decyduje, czy potrzebuje; task: treść zadania jako zapytanie, zawsze")
    ap.add_argument("--rag-k", type=int, default=4); ap.add_argument("--rag-min-score", type=float, default=0.5, help="próg rerankera 0..1; poniżej = brak fragmentów")
    ap.add_argument("--think", choices=["server", "on", "off", "auto"], default="server", help="tryb myślenia (Gemma 4/Qwen3): auto = otwarte i wypracowanie z myśleniem, zamknięte bez")
    ap.add_argument("--cache-dir", help="katalog na zapisane opisy ilustracji (wznawianie bez ponownego VLM)")
    ap.add_argument("--ocr", choices=["none", "tesseract", "kraken", "easyocr"], default="none", help="OCR przed VLM: napisy jako podpowiedź do weryfikacji")
    ap.add_argument("--ocr-cache", default="results/ocr_cache", help="katalog cache OCR (sha256 obrazu -> tekst), wspólny dla przebiegów")
    ap.add_argument("--img-index", help="katalogi (po przecinku) z phash.npy + refs.jsonl + meta.json (rag/phash_images.py): ilustracje będące kopiami obrazów z Wikipedii dostają tytuł artykułu i podpis")
    ap.add_argument("--img-max-dist", type=int, default=10, help="maks. odległość Hamminga pHash dla kandydatów (potwierdzanych DINOv2)")
    ap.add_argument("--img-min-dino", type=float, default=0.65, help="min. kosinus DINOv2 ilustracja-kandydat, żeby uznać kopię")
    ap.add_argument("--ocr-scale", type=float, default=1.0, help="powiększenie obrazu (Lanczos, skala szarości) przed OCR, np. 2")
    ap.add_argument("--essay-rag", action="store_true", help="wypracowanie: retrieval przed pisaniem (model wybiera temat i pisze zapytania; fragmenty jako materiał)")
    ap.add_argument("--essay-rag-k", type=int, default=8); ap.add_argument("--essay-best-of", type=int, default=1, help="N wersji wypracowania, model wybiera najlepszą")
    ap.add_argument("--essay-temp", type=float, default=0.7, help="temperatura wersji 2..N (pierwsza deterministyczna)")
    ap.add_argument("--essay-topic", choices=["model", "auto"], default="model",
                    help="wypracowanie: model = temat wybierany w trakcie pisania; auto = osobny krok wyboru tematu (bez myślenia), potem pisanie ze strukturą")
    ap.add_argument("--allow-remote", action="store_true", help="tylko do testów: zezwól na backendy spoza localhost")
    ap.add_argument("--img-scale", choices=["none", "2x", "3x", "budget"], default="none",
                    help="powiększanie ilustracji dla modelu (Gemma: 1 token = 48x48 px, maks. 1120/obraz); budget = do ~2,45 Mpx, maks. 3x")
    a = ap.parse_args()
    global IMG_SCALE; IMG_SCALE = a.img_scale
    C.ensure_local(a.llm_url, a.allow_remote)
    if a.strategy in ("vlm-text", "split"): C.ensure_local(a.vlm_url, a.allow_remote)
    if a.cache_dir: Path(a.cache_dir).mkdir(parents=True, exist_ok=True)
    if a.ocr != "none": Path(a.ocr_cache).mkdir(parents=True, exist_ok=True)
    solver = Solver(a)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(solver))
    print(f"solver {solver.pipeline()} na http://{a.host}:{a.port} (POST /solve, GET /health, GET /info)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
