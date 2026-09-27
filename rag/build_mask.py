#!/usr/bin/env python3
"""Maska fragmentów dla serwera retrievalu: lista tytułów artykułów -> tablica bool w kolejności indeksu (jak load_index).
  python rag/build_mask.py --chunks ~/wiki/chunks,~/wiki/en_chunks --emb ~/wiki/emb_e5,~/wiki/en_emb_e5 \
      --titles ~/wiki/dumps/sets/cat_historia_d4.txt --apply-to 0 --out ~/wiki/masks/cat_historia_d4.npy
--apply-to: numer indeksu (0 = pierwszy z --chunks), do którego stosuje się lista; pozostałe indeksy przechodzą w całości.
Tytuły porównywane po normalizacji (podkreślenia -> spacje)."""
import argparse, glob, os
import numpy as np, pyarrow.parquet as pq


def norm(t):
    return (t or "").replace("_", " ").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True); ap.add_argument("--emb", required=True)
    ap.add_argument("--titles", required=True); ap.add_argument("--apply-to", type=int, default=0); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    allowed = {norm(l) for l in open(os.path.expanduser(a.titles), encoding="utf-8") if l.strip()}
    parts, seen_titles, kept_titles = [], set(), set()
    for i, (cdir, edir) in enumerate(zip(a.chunks.split(","), a.emb.split(","))):
        for f in sorted(glob.glob(os.path.join(os.path.expanduser(cdir), "*.parquet"))):
            if not os.path.exists(os.path.join(os.path.expanduser(edir), os.path.basename(f).replace(".parquet", ".npy"))):
                continue
            titles = pq.read_table(f, columns=["title"]).column("title").to_pylist()
            if i == a.apply_to:
                m = np.fromiter((norm(t) in allowed for t in titles), dtype=bool, count=len(titles))
                seen_titles.update(map(norm, titles)); kept_titles.update(norm(t) for t, k in zip(titles, m) if k)
            else:
                m = np.ones(len(titles), dtype=bool)
            parts.append(m)
    mask = np.concatenate(parts)
    os.makedirs(os.path.dirname(os.path.expanduser(a.out)) or ".", exist_ok=True)
    np.save(os.path.expanduser(a.out), mask)
    print(f"{a.out}: {mask.sum()}/{len(mask)} fragmentów przepuszczonych; lista {len(allowed)} tytułów, w indeksie {len(kept_titles)} "
          f"({100 * len(kept_titles) / max(1, len(allowed)):.1f}% listy), indeks ma {len(seen_titles)} tytułów")


if __name__ == "__main__":
    main()
