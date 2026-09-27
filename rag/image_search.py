#!/usr/bin/env python3
"""POC: ilustracja z egzaminu -> najbliższe obrazy z Wikipedii (DINOv2) -> artykuł + podpis.
  /scratch/vllm/bin/python rag/image_search.py --index ~/wiki/img_dino --images submission/history-2023-mock-v1/images --k 5 --out data/rag/img_mock2023.jsonl
Dodatkowo --self-test: bierze losowe obrazy z indeksu, przycina je i sprawdza, czy odnajdują same siebie (kalibracja progu)."""
import argparse, glob, io, json, os, random, time
import numpy as np, torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

def embed(model, proc, ims):
    px = proc(images=ims, return_tensors="pt")["pixel_values"].cuda().half()
    with torch.no_grad(): o = model(pixel_values=px).pooler_output
    return torch.nn.functional.normalize(o.float(), dim=1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--index", required=True); ap.add_argument("--images"); ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--model", default="facebook/dinov2-base"); ap.add_argument("--out"); ap.add_argument("--self-test", type=int, default=0)
    a = ap.parse_args(); idx = os.path.expanduser(a.index)
    E = torch.from_numpy(np.load(os.path.join(idx, "emb.npy"))).cuda()
    refs = [json.loads(l) for l in open(os.path.join(idx, "refs.jsonl"))]
    print(f"indeks obrazów: {E.shape[0]}", flush=True)
    proc = AutoImageProcessor.from_pretrained(a.model); model = AutoModel.from_pretrained(a.model, torch_dtype=torch.float16).cuda().eval()
    def topk(qe, k):
        s = (qe.half() @ E.T).float(); v, ix = torch.topk(s, k=k, dim=1); return v.cpu().tolist(), ix.cpu().tolist()
    if a.self_test:
        from libzim.reader import Archive
        z = Archive(os.path.expanduser("~/wiki/wikipedia_pl_all_maxi_2026-08.zim")); random.seed(0)
        picks = random.sample(range(len(refs)), a.self_test); ims = []
        for i in picks:
            im = Image.open(io.BytesIO(bytes(z.get_entry_by_path(refs[i]["src"]).get_item().content))).convert("RGB")
            w, h = im.size; ims.append(im.crop((int(w*0.1), int(h*0.1), int(w*0.85), int(h*0.9))).resize((int(w*0.6), int(h*0.6))))  # wycinek + przeskalowanie
        v, ix = topk(embed(model, proc, ims), 5)
        hit1 = sum(ix[j][0] == picks[j] for j in range(len(picks))); hit5 = sum(picks[j] in ix[j] for j in range(len(picks)))
        top1 = [v[j][0] for j in range(len(picks))]; second = [v[j][1] for j in range(len(picks))]
        print(f"self-test na {a.self_test} przyciętych obrazach: top1 {hit1}, top5 {hit5}; podobieństwo trafienia mediana {np.median(top1):.3f}, drugi wynik mediana {np.median(second):.3f}")
    if a.images:
        files = sorted(glob.glob(os.path.join(a.images, "*.png")) + glob.glob(os.path.join(a.images, "*.jpg")))
        ims = [Image.open(f).convert("RGB") for f in files]; v, ix = topk(embed(model, proc, ims), a.k)
        rows = []
        for f, vs, ixs in zip(files, v, ix):
            hits = [{"score": round(s, 3), "src": refs[j]["src"], "refs": refs[j]["refs"][:3]} for s, j in zip(vs, ixs)]
            rows.append({"image": os.path.basename(f), "hits": hits})
            best = hits[0]; print(f"{os.path.basename(f):14s} {best['score']:.3f}  {best['refs'][0][0][:40]:40s} | {best['refs'][0][1][:70]}")
        if a.out:
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            with open(a.out, "w") as f:
                for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
