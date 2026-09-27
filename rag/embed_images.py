#!/usr/bin/env python3
"""Embedding obrazów z ZIM (DINOv2) -> npy fp16 + lista ścieżek. Mapę obraz->artykuł/podpis daje articles/*.parquet.
  /scratch/vllm/bin/python rag/embed_images.py --articles ~/wiki/articles --out ~/wiki/img_dino --min-w 150
Bierze tylko obrazy, do których odwołuje się jakiś artykuł (figure/infobox), o oryginalnej szerokości >= min-w."""
import argparse, glob, io, json, os, time
import numpy as np, pyarrow.parquet as pq, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel
from libzim.reader import Archive
ZIM = os.path.expanduser("~/wiki/wikipedia_pl_all_maxi_2026-08.zim")

class DS(Dataset):
    def __init__(self, paths, proc): self.paths = paths; self.proc = proc; self.z = None
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        if self.z is None: self.z = Archive(ZIM)
        try:
            b = bytes(self.z.get_entry_by_path(self.paths[i]).get_item().content)
            im = Image.open(io.BytesIO(b)).convert("RGB")
            return self.proc(images=im, return_tensors="pt")["pixel_values"][0], i
        except Exception:
            return torch.zeros(3, 224, 224), -1

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--articles", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="facebook/dinov2-base"); ap.add_argument("--min-w", type=int, default=150); ap.add_argument("--bs", type=int, default=256); ap.add_argument("--limit", type=int)
    a = ap.parse_args(); out = os.path.expanduser(a.out); os.makedirs(out, exist_ok=True)
    # obraz -> [(tytuł, podpis)]
    refs = {}
    for f in sorted(glob.glob(os.path.join(os.path.expanduser(a.articles), "*.parquet"))):
        for title, ims in zip(*[pq.read_table(f, columns=["title", "images"]).column(c).to_pylist() for c in ("title", "images")]):
            for im in json.loads(ims):
                try: w = int(im.get("w") or 0)
                except ValueError: w = 0
                if w >= a.min_w: refs.setdefault(im["src"], []).append((title, im.get("caption", "")))
    paths = sorted(refs)[: a.limit] if a.limit else sorted(refs)
    print(f"obrazów do policzenia: {len(paths)} (z {a.min_w}px+), artykułów-odwołań: {sum(len(v) for v in refs.values())}", flush=True)
    with open(os.path.join(out, "refs.jsonl"), "w") as f:
        for p in paths: f.write(json.dumps({"src": p, "refs": refs[p][:20]}, ensure_ascii=False) + "\n")
    proc = AutoImageProcessor.from_pretrained(a.model); model = AutoModel.from_pretrained(a.model, torch_dtype=torch.float16).cuda().eval()
    dl = DataLoader(DS(paths, proc), batch_size=a.bs, num_workers=12, pin_memory=True)
    emb = np.zeros((len(paths), model.config.hidden_size), dtype=np.float16); bad = 0; t0 = time.time(); n = 0
    with torch.no_grad():
        for px, idx in dl:
            o = model(pixel_values=px.cuda().half()).pooler_output  # CLS po layernorm
            o = torch.nn.functional.normalize(o.float(), dim=1).cpu().numpy().astype(np.float16)
            for j, i in enumerate(idx.tolist()):
                if i >= 0: emb[i] = o[j]
                else: bad += 1
            n += len(idx)
            if n % (a.bs * 40) == 0: print(f"{n}/{len(paths)} {n/(time.time()-t0):.0f} img/s, błędów {bad}", flush=True)
    np.save(os.path.join(out, "emb.npy"), emb); print("IMG EMBED DONE", len(paths), "błędów", bad)
