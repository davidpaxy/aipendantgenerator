import os
import glob
import base64
import mimetypes
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from google import genai

# Post-processing immagini (monocromatico argento brunito + autoframing + detail)
try:
    from PIL import Image, ImageEnhance, ImageOps, ImageFilter
    PIL_OK = True
except Exception:
    PIL_OK = False


# =============================================================================
# Config / Setup
# =============================================================================
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("Imposta la variabile d'ambiente GOOGLE_API_KEY")

IMAGE_MODEL_ID = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
ROOT_PATH = os.getenv("ROOT_PATH", "")

client = genai.Client(api_key=API_KEY)
app = FastAPI(title="NOVE25 Pendant Generator", root_path=ROOT_PATH)

# CORS (aperto in dev; restringi in produzione)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


# =============================================================================
# Models
# =============================================================================
class GenerateRequest(BaseModel):
    prompt: Optional[str] = None
    style: Optional[str] = "3d"            # "3d" | "basrelief"
    size_mm: Optional[int] = 30            # 20 | 30 | 40 | 60
    images: Optional[List[str]] = None     # Data URL; usiamo SOLO la prima


class GenerateResponse(BaseModel):
    images: List[str]  # data URLs


# =============================================================================
# Utils
# =============================================================================
def _to_data_url(raw_bytes: bytes, mime: str = "image/png") -> str:
    b64 = base64.b64encode(raw_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _data_url_to_inline(durl: str):
    if not durl or not durl.startswith("data:"):
        return None
    try:
        header, b64 = durl.split(",", 1)
    except ValueError:
        return None

    mime = "image/png"
    if ";base64" in header:
        try:
            mime = header.split("data:")[1].split(";")[0] or "image/png"
        except Exception:
            mime = "image/png"

    try:
        raw = base64.b64decode(b64)
    except Exception:
        return None

    return {"mime_type": mime, "data": raw}


# === Post-processing: forza “argento brunito” (monocromatico) =================
def _silver_brunito_bytes(raw: bytes) -> bytes:
    """
    Desatura e regola leggermente contrasto/luminosità per un look coerente
    'argento brunito'. Se PIL non è disponibile, restituisce i bytes originali.
    """
    if not PIL_OK:
        return raw
    try:
        from io import BytesIO
        im = Image.open(BytesIO(raw))
        alpha = None
        if im.mode in ("RGBA", "LA"):
            alpha = im.getchannel("A")

        # Grayscale “dolce”
        gray = ImageOps.grayscale(im).convert("RGB")
        # Contrasto e luminosità tarati per 'brunito'
        gray = ImageEnhance.Contrast(gray).enhance(1.18)   # +18% contrasto
        gray = ImageEnhance.Brightness(gray).enhance(0.94)  # -6% luce

        if alpha is not None:
            gray.putalpha(alpha)

        out = BytesIO()
        gray.save(out, format="PNG", optimize=True)
        return out.getvalue()
    except Exception:
        return raw


def _autoframe_on_black(png_bytes: bytes, target_px: int = 1024, fill_margin: float = 0.86) -> bytes:
    """
    Prende un'immagine con sfondo nero, trova il bbox del soggetto (pixel non-neri),
    centra e scala affinché il lato lungo occupi ~fill_margin del canvas quadrato.
    Restituisce PNG.
    """
    if not PIL_OK:
        return png_bytes
    from io import BytesIO
    im = Image.open(BytesIO(png_bytes)).convert("RGBA")

    # Maschera: preferisci alpha, altrimenti derivata da luminanza (soglia su nero)
    try:
        alpha = im.getchannel("A")
        mask = alpha
    except Exception:
        lum = ImageOps.grayscale(im)
        mask = lum.point(lambda x: 255 if x > 8 else 0, mode="1").convert("L")

    bbox = mask.getbbox()
    if not bbox:
        return png_bytes  # fallback: immagine vuota o tutta nera

    # Ritaglia soggetto, ridimensiona e centra su canvas quadrato
    subject = im.crop(bbox)
    w, h = subject.size
    canvas = Image.new("RGBA", (target_px, target_px), (0, 0, 0, 255))
    scale = (target_px * fill_margin) / max(w, h)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    subject = subject.resize(new_size, Image.LANCZOS)

    x = (target_px - subject.size[0]) // 2
    y = (target_px - subject.size[1]) // 2
    canvas.paste(subject, (x, y), subject)

    out = BytesIO()
    canvas.save(out, format="PNG", optimize=True)
    return out.getvalue()


def _apply_detail_enhance(png_bytes: bytes, size_mm: int) -> bytes:
    """
    Aumenta nitidezza in modo graduale in base alla dimensione. Output PNG.
    """
    if not PIL_OK:
        return png_bytes
    from io import BytesIO
    im = Image.open(BytesIO(png_bytes)).convert("RGBA")

    # Parametri in funzione della taglia
    if size_mm >= 60:
        radius, percent = 1.4, 180  # UnsharpMask
    elif size_mm >= 40:
        radius, percent = 1.2, 150
    elif size_mm >= 30:
        radius, percent = 1.0, 120
    else:
        radius, percent = 0.8, 100

    try:
        im = im.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=3))
    except Exception:
        pass

    out = BytesIO()
    im.save(out, format="PNG", optimize=True)
    return out.getvalue()


def _target_px_for_size(size_mm: int) -> int:
    """
    Canvas quadrato in px in base alla dimensione del ciondolo:
    20mm -> 896px, 30mm -> 1024px, 40mm -> 1280px, 60mm -> 1536px.
    """
    table = {20: 896, 30: 1024, 40: 1280, 60: 1536}
    return table.get(size_mm, 1024)


def _detail_clause(size: int) -> str:
    if size >= 60:
        return "Livello di dettaglio: micro-dettagli molto fini (texture metalliche leggere, micro-sfumature, piccoli raccordi e incisioni superficiali). "
    if size >= 40:
        return "Livello di dettaglio: dettagli fini ben visibili (smussi netti, superfici pulite con leggere variazioni). "
    if size >= 30:
        return "Livello di dettaglio: dettagli moderati, evita micro-texture troppo fitte. "
    return "Livello di dettaglio: semplice e leggibile, evita pattern minuti che si perdono sotto i 20 mm. "


# =============================================================================
# Contromaglia fissa da /static/contromaglia.*
# =============================================================================
def _load_contromaglia_inline():
    pattern = os.path.join(STATIC_DIR, "contromaglia.*")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[warn] contromaglia non trovata: {pattern}")
        return None
    path = files[0]
    try:
        with open(path, "rb") as f:
            raw = f.read()
        mime = mimetypes.guess_type(path)[0] or "image/png"
        return {"mime_type": mime, "data": raw}
    except Exception as e:
        print(f"[warn] caricamento contromaglia fallito: {e}")
        return None


_CONTROMAGLIA_INLINE = _load_contromaglia_inline()


# =============================================================================
# Prompt builder (rinforzato per proporzioni e camera)
# =============================================================================
def _build_pendant_prompt(user_prompt: str, style: str, size_mm: int) -> str:
    """
    Prompt testo con vincoli forti su proporzioni/ottica:
    - monocromatico argento brunito
    - contromaglia obbligatoria 8×4 mm integrata e leggermente incassata
    - rapporto dimensionale contromaglia:corpo
    - spessore percepito, margini, camera ortho-like (focale lunga)
    """
    s = (style or "3d").lower()
    size = size_mm if size_mm in (20, 30, 40, 60) else 30
    style_line = (
        "BAS-RELIEF frontale, basso rilievo (MA con volume reale; niente piattezza)."
        if s == "basrelief"
        else "FULL 3D quasi frontale (leggero tre-quarti, camera alta)."
    )

    proportions_block = (
        f"Proporzioni: ciondolo ~{size} mm di altezza (ESCLUSO anellino). "
        "Contromaglia OBBLIGATORIA identica al riferimento, ovale 8×4 mm. "
        f"Scala relativa: altezza contromaglia : altezza corpo ≈ 8 : {size}. "
        "Mantieni spessore percepito 3–4 mm con bordo sicurezza ≥1.2 mm. "
        "Assi e silhouette coerenti: niente stretching orizzontale/verticale, niente foreshortening marcato. "
        "Simmetria sull’asse verticale salvo soggetti esplicitamente asimmetrici."
    )

    framing_block = (
        "Framing: soggetto centrato; margine nero 7–10% su tutti i lati. "
        "Evita tagli e parti fuori inquadratura. "
        "Ottica: resa ortho-like con focale lunga (≈ 85–135 mm equivalente), prospettiva molto contenuta."
    )

    bans_block = (
        "Vietato: traforo/outline/wireframe/ritaglio piatto; niente scritte/numeri/smalti. "
        "Corpo SOLIDO con smussi e raccordi morbidi. "
        "Sfondo NERO pieno, luce da studio coerente."
    )

    detail_block = _detail_clause(size)

    base = (
        "Fotografia prodotto di un CIONDOLO in ARGENTO BRUNITO LUCIDO (monocromatico). "
        "Ignora qualunque colore: usa un unico metallo argento brunito. "
        f"Stile: {style_line} "
        f"{proportions_block} "
        f"{framing_block} "
        f"{bans_block} "
        f"{detail_block}"
        "Soggetto richiesto (senza lettere): "
    )
    return base + (user_prompt or "").strip()


def _build_copy_prompt(size_mm: int) -> str:
    """
    Prompt per copia fedele da immagine utente, con proporzioni e camera stabilizzate.
    """
    size = size_mm if size_mm in (20, 30, 40, 60) else 30
    proportions_block = (
        f"Proporzioni: ciondolo ~{size} mm di altezza (ESCLUSO anellino). "
        "Contromaglia OBBLIGATORIA identica al riferimento, ovale 8×4 mm. "
        f"Scala relativa: altezza contromaglia : altezza corpo ≈ 8 : {size}. "
        "Mantieni spessore percepito 3–4 mm, bordo ≥1.2 mm. "
        "Niente stretching o deformazioni prospettiche marcate; simmetria sull’asse verticale."
    )
    framing_block = (
        "Framing: soggetto centrato; margine nero 7–10% su tutti i lati. "
        "Ottica ortho-like (≈ 85–135 mm eq.) per minimizzare distorsioni."
    )
    detail_block = _detail_clause(size)

    return (
        "Fotografia prodotto di un CIONDOLO in ARGENTO BRUNITO LUCIDO (monocromatico). "
        "IGNORA COMPLETAMENTE i colori presenti nell'immagine utente: rendi tutto in un unico metallo ARGENTO BRUNITO LUCIDO. "
        "Stile: FULL 3D quasi frontale (leggero tre-quarti, camera alta). "
        f"{proportions_block} "
        f"{framing_block} "
        f"{detail_block}"
        "RIPRODUCI FEDELMENTE il soggetto dell'immagine utente senza interpretazioni né varianti. "
        "Usa ESATTAMENTE la contromaglia allegata; integrata e LEGGERMENTE INCASSATA nel corpo (affogata, raccordo morbido). "
        "Vietato: traforo/outline/wireframe/ritaglio piatto; corpo SOLIDO con smussi. "
        "Sfondo NERO pieno, luce da studio coerente. "
        "Nessuna scritta, lettering, numeri o smalti."
    )


# =============================================================================
# Endpoint principale
# =============================================================================
@app.post("/api/generate", response_model=GenerateResponse)
def generate_images(req: GenerateRequest):
    # Stile
    style = (req.style or "3d").lower()
    if style not in ("3d", "basrelief"):
        style = "3d"

    # Taglia
    try:
        size_val = int(req.size_mm or 30)
    except Exception:
        size_val = 30
    if size_val not in (20, 30, 40, 60):
        size_val = 30

    target_px = _target_px_for_size(size_val)

    # Prima (ed unica) immagine utente
    first_user_inline = None
    if req.images:
        for durl in req.images[:1]:
            inline = _data_url_to_inline(durl)
            if inline:
                first_user_inline = inline
                break

    # Prompt utente
    user_prompt_input = (req.prompt or "").strip()

    # Logica:
    # - immagine + nessun prompt -> COPIA FEDELE 3D, MONOCROMATICA argento brunito
    # - nessun prompt + nessuna immagine -> 400
    # - prompt presente -> usa prompt con vincoli (monocromatico + maglina incassata + no traforo)
    if not user_prompt_input:
        if first_user_inline is None:
            raise HTTPException(status_code=400, detail="Scrivi un motivo oppure allega un’immagine.")
        final_prompt = _build_copy_prompt(size_val)
    else:
        final_prompt = _build_pendant_prompt(user_prompt_input, style, size_val)

    # Prepara contents per il modello immagine
    contents: List[object] = [final_prompt]

    # 1) Contromaglia fissa SEMPRE
    if _CONTROMAGLIA_INLINE:
        contents.append({"inline_data": _CONTROMAGLIA_INLINE})

    # 2) Riferimento utente (se presente)
    if first_user_inline:
        contents.append({"inline_data": first_user_inline})

    # Chiamata al modello immagine
    try:
        response = client.models.generate_content(model=IMAGE_MODEL_ID, contents=contents)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Errore chiamata modello: {e}")

    # Estrai la prima immagine utile dal primo candidato
    results: List[str] = []
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        raise HTTPException(status_code=500, detail="Risposta vuota.")

    for cand in candidates[:1]:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) if content else None
        if not parts:
            continue
        for part in parts:
            inline_data = getattr(part, "inline_data", None)
            if inline_data and getattr(inline_data, "data", None):
                data_field = inline_data.data
                raw_bytes = data_field if isinstance(data_field, bytes) else base64.b64decode(data_field)

                # === Post-filtro: argento brunito + auto-framing + nitidezza adattiva ===
                fixed = _silver_brunito_bytes(raw_bytes)
                fixed = _autoframe_on_black(fixed, target_px=target_px, fill_margin=0.86)
                fixed = _apply_detail_enhance(fixed, size_val)

                results.append(_to_data_url(fixed, "image/png"))
                break

    if not results:
        raise HTTPException(status_code=422, detail="Nessuna immagine generata.")

    return GenerateResponse(images=results)


# =============================================================================
# Static & Index
# =============================================================================
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
