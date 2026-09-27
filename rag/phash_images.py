#!/usr/bin/env python3
"""Dokładne/prawie dokładne duplikaty ilustracji: pHash (DCT 64 bit) dla obrazów z indeksu img_dino (refs.jsonl, ZIM).
W przeciwieństwie do DINOv2 (podobieństwo stylu: każda mapa pasuje do każdej mapy) pHash trafia tylko, gdy ilustracja
z arkusza jest kopią/wycinkiem obrazu z Wikipedii; wtedy tytuł artykułu i podpis to pewna informacja.
  /scratch/vllm/bin/python rag/phash_images.py refs --articles ~/wiki/en_articles --index ~/wiki/img_en   -> refs.jsonl (obraz -> artykuł, podpis)
  ZIM=~/wiki/wikipedia_en_history_maxi_2026-07.zim /scratch/vllm/bin/python rag/phash_images.py build --index ~/wiki/img_en [--workers 12]
  /scratch/vllm/bin/python rag/phash_images.py build --index ~/wiki/img_dino [--workers 24]   -> ~/wiki/img_dino/phash.npy
  /scratch/vllm/bin/python rag/phash_images.py query --index ~/wiki/img_dino --images submission/history-2023-mock-v1/images [--max-dist 10]
"""
import argparse, io, json, os, sys, time
import numpy as np
from multiprocessing import Pool

ZIM = os.path.expanduser(os.environ.get("ZIM", "~/wiki/wikipedia_pl_all_maxi_2026-08.zim"))
_z = None
_DCT = None


def dct_matrix(n=32):
    k = np.arange(n)[:, None]; i = np.arange(n)[None, :]
    m = np.sqrt(2 / n) * np.cos(np.pi * (2 * i + 1) * k / (2 * n)); m[0] /= np.sqrt(2)
    return m


def phash_pil(im):
    """PIL -> 64-bitowy pHash (DCT 32x32, 8x8 niskich częstotliwości, próg = mediana bez DC)."""
    global _DCT
    if _DCT is None: _DCT = dct_matrix(32)
    from PIL import Image
    g = np.asarray(im.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
    d = (_DCT @ g @ _DCT.T)[:8, :8].flatten()
    med = np.median(d[1:])
    bits = (d > med).astype(np.uint64)
    return int(np.packbits(bits.astype(np.uint8)).view(">u8")[0]) if False else int("".join("1" if b else "0" for b in bits), 2)


def _work(args):
    i, path = args
    global _z
    try:
        if _z is None:
            from libzim.reader import Archive; _z = Archive(ZIM)
        from PIL import Image
        im = load_zim_image(_z, path)
        return i, phash_pil(im)
    except Exception:
        return i, 0


def load_zim_image(z, path):
    """Obraz z ZIM; ścieżki ze skanera bywają zakodowane procentowo (%22, %D0…), a w ZIM leżą zdekodowane."""
    from urllib.parse import unquote
    from PIL import Image
    for p in (path, unquote(path), unquote(unquote(path))):
        try:
            b = bytes(z.get_entry_by_path(p).get_item().content); im = Image.open(io.BytesIO(b)); im.load(); return im
        except Exception:
            continue
    raise KeyError(path)


def gray64(im):
    from PIL import Image
    g = np.asarray(im.convert("L").resize((64, 64), Image.LANCZOS), dtype=np.float64).flatten()
    return (g - g.mean()) / (g.std() + 1e-6)


def correlation(im_a, im_b):
    """Korelacja Pearsona obrazów sprowadzonych do 64x64 w skali szarości (1.0 = ten sam obraz, wycinek ~0.5-0.8, inny < 0.4)."""
    return float((gray64(im_a) * gray64(im_b)).mean())


def hamming(a, b):
    x = np.bitwise_xor(a, b)
    # popcount 64-bit
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int64)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["refs", "build", "query"]); ap.add_argument("--index", required=True)
    ap.add_argument("--articles", help="refs: katalog parquet z rag/zim_scan.py (kolumny title, images)"); ap.add_argument("--min-w", type=int, default=150)
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2)); ap.add_argument("--images"); ap.add_argument("--max-dist", type=int, default=10)
    ap.add_argument("--out")
    a = ap.parse_args(); idx = os.path.expanduser(a.index)
    if a.cmd == "refs":  # jak w embed_images.py: obraz -> [(tytuł, podpis)], tylko obrazy o szerokości >= min-w
        import glob, pyarrow.parquet as pq
        os.makedirs(idx, exist_ok=True); refs = {}
        for f in sorted(glob.glob(os.path.join(os.path.expanduser(a.articles), "*.parquet"))):
            t = pq.read_table(f, columns=["title", "images"])
            for title, ims in zip(t.column("title").to_pylist(), t.column("images").to_pylist()):
                for im in json.loads(ims):
                    try: w = int(im.get("w") or 0)
                    except ValueError: w = 0
                    if w >= a.min_w: refs.setdefault(im["src"], []).append((title, im.get("caption", "")))
        with open(os.path.join(idx, "refs.jsonl"), "w") as f:
            for p in sorted(refs): f.write(json.dumps({"src": p, "refs": refs[p][:20]}, ensure_ascii=False) + "\n")
        print(f"refs: {len(refs)} obrazów -> {idx}/refs.jsonl"); return
    refs = [json.loads(l) for l in open(os.path.join(idx, "refs.jsonl"))]
    if a.cmd == "build":
        H = np.zeros(len(refs), dtype=np.uint64); t0 = time.time()
        with Pool(a.workers) as pool:
            for n, (i, h) in enumerate(pool.imap_unordered(_work, list(enumerate(r["src"] for r in refs)), chunksize=256)):
                H[i] = h
                if n % 50000 == 0: print(f"{n}/{len(refs)} {time.time() - t0:.0f}s", flush=True)
        np.save(os.path.join(idx, "phash.npy"), H); print(f"zapisano {len(refs)} hashy, zera (błędy): {int((H == 0).sum())}, {time.time() - t0:.0f}s")
        json.dump({"zim": ZIM, "n": len(refs)}, open(os.path.join(idx, "meta.json"), "w"))
    else:
        import glob
        from PIL import Image
        H = np.load(os.path.join(idx, "phash.npy"))
        meta = json.load(open(os.path.join(idx, "meta.json"))) if os.path.exists(os.path.join(idx, "meta.json")) else {"zim": ZIM}
        from libzim.reader import Archive; z = Archive(os.path.expanduser(meta["zim"]))
        files = sorted(glob.glob(os.path.join(a.images, "*.png")) + glob.glob(os.path.join(a.images, "*.jpg")))
        rows = []
        for f in files:
            q = Image.open(f); h = np.uint64(phash_pil(q))
            d = hamming(H, h); d[H == 0] = 99
            order = np.argsort(d)[:3]
            hits = []
            for j in order:
                try: corr = round(correlation(q, load_zim_image(z, refs[j]["src"])), 3)
                except Exception: corr = None
                hits.append({"dist": int(d[j]), "corr": corr, "src": refs[j]["src"], "refs": refs[j]["refs"][:3]})
            best = hits[0]; flag = "TRAFIENIE" if best["dist"] <= a.max_dist else ""
            print(f"{os.path.basename(f):14s} dist {best['dist']:2d} corr {best['corr']!s:6} {flag:9s} {best['refs'][0][0][:36]:36s} | {best['refs'][0][1][:50]}")
            rows.append({"image": os.path.basename(f), "hits": hits})
        if a.out:
            with open(a.out, "w") as fo:
                for r in rows: fo.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
