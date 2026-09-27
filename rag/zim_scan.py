#!/usr/bin/env python3
"""Skan ZIM plwiki -> parquet z artykułami (tekst, sekcje, obrazy+podpisy, infobox).
  python3 zim_scan.py --out ~/wiki/articles --workers 12 [--limit N]
Każdy worker otwiera własne Archive i przetwarza co N-ty wpis. Wynik: articles/part-XX.parquet."""
import argparse, json, os, re, sys, time
from multiprocessing import Process
ZIM = os.path.expanduser(os.environ.get("ZIM", "~/wiki/wikipedia_pl_all_maxi_2026-08.zim"))
SKIP_H2 = {"Przypisy", "Bibliografia", "Linki zewnętrzne", "Zobacz też", "Uwagi", "Literatura", "Źródła",
           "References", "Bibliography", "External links", "See also", "Notes", "Further reading", "Sources", "Citations", "Footnotes"}

def worker(wid, nw, out, limit):
    from libzim.reader import Archive
    from selectolax.parser import HTMLParser
    import pyarrow as pa, pyarrow.parquet as pq
    z = Archive(ZIM); n = z.entry_count if not limit else min(limit, z.entry_count)
    rows = []; t0 = time.time(); done = 0
    def flush(final=False):
        nonlocal rows
        if not rows: return
        tbl = pa.table({k: [r[k] for r in rows] for k in rows[0]})
        p = f"{out}/part-{wid:02d}-{done:08d}.parquet"; pq.write_table(tbl, p, compression="zstd"); rows = []
    for i in range(wid, n, nw):
        e = z._get_entry_by_id(i)
        if e.is_redirect or "/" in e.path: continue  # artykuły są w korzeniu; _assets_/_mw_ zawierają '/'
        it = e.get_item()
        if it.mimetype != "text/html": continue
        html = bytes(it.content).decode("utf-8", "ignore")
        tree = HTMLParser(html)
        body = tree.body
        if body is None: continue
        title = e.title
        # obrazy + podpisy (figure/figcaption); infobox obrazy przez alt/podpis komórki
        images = []
        for fig in body.css("figure"):
            img = fig.css_first("img"); cap = fig.css_first("figcaption")
            if img and img.attributes.get("src", "").startswith("./_assets_/"):
                images.append({"src": img.attributes["src"][2:], "caption": (cap.text(strip=True) if cap else ""),
                               "w": img.attributes.get("data-file-width"), "h": img.attributes.get("data-file-height")})
        infobox = ""
        ib = body.css_first("table.infobox")
        if ib:
            cells = []
            for tr in ib.css("tr"):
                t = re.sub(r"\s+", " ", tr.text(separator=" ", strip=True)).strip()
                if 3 < len(t) < 400: cells.append(t)
            infobox = "\n".join(cells)[:4000]
            for img in ib.css("img"):
                s = img.attributes.get("src", "")
                if s.startswith("./_assets_/"): images.append({"src": s[2:], "caption": img.attributes.get("alt", "") or "", "w": img.attributes.get("data-file-width"), "h": img.attributes.get("data-file-height")})
        # tekst: sekcje h2 -> akapity; pomiń przypisy itd.; usuń tabele/figury/nawigację
        for sel in ("figure", "table", "style", "script", "sup.reference", ".mw-editsection", ".navbox", ".hatnote", "ol.references", ".reflist", "math"):
            for nd in body.css(sel): nd.decompose()
        parts = []; cur = "Wstęp"; skip = False
        BLOCK = {"h2", "h3", "p", "blockquote", "li", "dd"}
        for nd in body.traverse(include_text=False):  # kolejność dokumentu
            tag = nd.tag
            if tag not in BLOCK: continue
            if tag in ("li", "dd", "p"):  # pomiń zagnieżdżone (li w li, p w li)
                par = nd.parent; nested = False
                while par is not None and par.tag != "body":
                    if par.tag in ("li", "dd", "p", "blockquote"): nested = True; break
                    par = par.parent
                if nested: continue
            if tag in ("h2", "h3"):
                h = nd.text(strip=True)
                if tag == "h2": cur = h; skip = h in SKIP_H2
                elif not skip: parts.append(f"\n## {h}\n")
                continue
            if skip: continue
            t = re.sub(r"[ \t]+", " ", nd.text(separator=" ", strip=True)).strip()
            if len(t) >= 20: parts.append(("• " + t) if tag in ("li", "dd") else t)
        text = "\n".join(parts)
        if len(text) < 100 and not images: continue
        rows.append({"id": i, "path": e.path, "title": title, "text": text, "infobox": infobox, "images": json.dumps(images, ensure_ascii=False)})
        done += 1
        if len(rows) >= 20000: flush()
        if wid == 0 and done % 5000 == 0: print(f"[w0] {done} art., {i}/{n} wpisów, {time.time()-t0:.0f}s", flush=True)
    flush(True); print(f"[w{wid}] done {done} art. in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.expanduser("~/wiki/articles")); ap.add_argument("--workers", type=int, default=12); ap.add_argument("--limit", type=int)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    ps = [Process(target=worker, args=(w, a.workers, a.out, a.limit)) for w in range(a.workers)]
    [p.start() for p in ps]; [p.join() for p in ps]; print("ALL DONE")
