"""Skalowanie ilustracji przed wysłaniem do modelu.

Gemma 4 (wariant „unified”, bez wieży wizyjnej): 1 token obrazu = łata 48x48 px, maks. 1120 tokenów na obraz,
czyli ok. 2,58 Mpx. Obrazy z paczek egzaminacyjnych mają 0,36–1,7 Mpx (156–730 tokenów), więc powiększenie
daje modelowi drobniejszy podział napisów (monety, mapy, podpisy) bez zmiany limitu kontekstu.

Tryby: none (bez zmian), 2x, 3x (stały mnożnik; serwer i tak zmniejszy do limitu),
budget (mnożnik dobrany tak, by obraz miał ~2,45 Mpx, maks. 3x). Nigdy nie pomniejszamy.
Wymaga Pillow tylko dla trybów innych niż none.
"""
import io

MODES = ("none", "2x", "3x", "budget")
TOKEN_PX = 48 * 48
MAX_TOKENS = 1120
TARGET_PX = int(0.95 * MAX_TOKENS * TOKEN_PX)  # ~2,45 Mpx, zapas na zaokrąglenia w serwerze
MAX_FACTOR = 3.0


def factor(w, h, mode):
    if mode == "none": return 1.0
    f = float(mode[0]) if mode in ("2x", "3x") else (TARGET_PX / max(1, w * h)) ** 0.5
    return max(1.0, min(f, MAX_FACTOR))


def prepare(data, mode):
    """Bajty PNG -> (bajty, mime, info). Zwraca oryginał, gdy mnożnik <= 1.01."""
    if mode not in MODES: raise ValueError(f"nieznany tryb skalowania: {mode}")
    if mode == "none": return data, "image/png", {"factor": 1.0}
    from PIL import Image
    im = Image.open(io.BytesIO(data)); w, h = im.size
    f = factor(w, h, mode)
    if f <= 1.01: return data, "image/png", {"factor": 1.0, "size": [w, h]}
    nw, nh = round(w * f), round(h * f)
    out = im.convert("RGB").resize((nw, nh), Image.LANCZOS)
    buf = io.BytesIO(); out.save(buf, "PNG", compress_level=1)
    return buf.getvalue(), "image/png", {"factor": round(f, 2), "size": [nw, nh], "tokens": nw * nh // TOKEN_PX}
