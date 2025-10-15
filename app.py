import os
import glob
import base64
import mimetypes
from typing import Optional, List

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from google import genai

# Post-processing immagini (monocromatico argento brunito)
try:
    from PIL import Image, ImageEnhance
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
CAPTION_MODEL_ID = os.getenv("GEMINI_CAPTION_MODEL", "gemini-1.5-flash")  # fallback raro
ROOT_PATH = os.getenv("ROOT_PATH", "")

client = genai.Client(api_key=API_KEY)
app = FastAPI(title="NOVE25 Pendant Generator", root_path=ROOT_PATH)

# CORS
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
    format: Optional[str] = "png"
    width: Optional[int] = 1024
    height: Optional[int] = 1024


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
        im = Image.open(BytesIO(raw)).convert("RGB")

        # 1) Monocromatico (grayscale)
        gray = im.convert("L").convert("RGB")

        # 2) Leggero boost di contrasto e piccola riduzione luminosità per 'brunito'
        contrast = ImageEnhance.Contrast(gray).enhance(1.18)   # +18% contrasto
        bright = ImageEnhance.Brightness(contrast).enhance(0.94)  # -6% luce

        out = BytesIO()
        bright.save(out, format="PNG", optimize=True)
        return out.getvalue()
    except Exception:
        return raw


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
# Prompt builder
# =============================================================================
def _build_pendant_prompt(user_prompt: str, style: str, size_mm: int) -> str:
    """
    Vincoli generali quando c'è un prompt testo:
    - SOLO argento BRUNITO LUCIDO (monocromatico). Ignora colori.
    - Contromaglia OBBLIGATORIA identica al riferimento, ovale 8x4mm,
      integrata e LEGGERMENTE INCASSATA nel corpo (affogata con raccordo morbido).
    - Vietato: traforato, outline-only, wireframe, ritaglio piatto; corpo SOLIDO con smussi/bevel.
    - Nessun testo/numeri/smalti.
    """
    s = (style or "3d").lower()
    size = size_mm if size_mm in (20, 30, 40, 60) else 30
    style_line = (
        "BAS-RELIEF frontale, basso rilievo (ma con volume reale, non piatto)."
        if s == "basrelief"
        else "FULL 3D quasi frontale (leggero tre-quarti)."
    )

    base = (
        "Fotografia prodotto di un CIONDOLO in ARGENTO BRUNITO LUCIDO (monocromatico). "
        "Ignora qualunque colore: usa un unico metallo argento brunito. "
        f"Stile: {style_line} "
        f"Dimensioni corpo ciondolo: ~{size}mm (ESCLUSO anellino). "
        "Usa ESATTAMENTE la contromaglia del riferimento allegato: ovale 8x4mm, "
        "integrata e LEGGERMENTE INCASSATA nel corpo (affogata), con raccordo morbido. "
        "Vietato: traforo/outline/wireframe/ritaglio piatto; il corpo deve essere SOLIDO con smussi. "
        "Sfondo NERO pieno, luce da studio coerente. "
        "Nessuna incisione/lettering/numeri/smalti. "
        "Soggetto richiesto (senza lettere): "
    )
    return base + (user_prompt or "").strip()


def _build_copy_prompt(size_mm: int) -> str:
    """
    Prompt usato quando c'è immagine utente e prompt vuoto:
    - Copia fedele del soggetto
    - FULL 3D
    - Monocromatico argento brunito (ignora i colori del riferimento)
    - Niente traforo/outline
    - Contromaglia leggermente incassata
    """
    size = size_mm if size_mm in (20, 30, 40, 60) else 30
    return (
        "Fotografia prodotto di un CIONDOLO in ARGENTO BRUNITO LUCIDO (monocromatico). "
        "IGNORA COMPLETAMENTE i colori presenti nell'immagine utente: "
        "rendi tutto in un unico metallo ARGENTO BRUNITO LUCIDO. "
        "Stile: FULL 3D quasi frontale (leggero tre-quarti). "
        f"Dimensioni corpo ciondolo: ~{size}mm (ESCLUSO anellino). "
        "RIPRODUCI FEDELMENTE il soggetto dell'immagine utente senza interpretazioni né varianti. "
        "Mantieni contorni e proporzioni del riferimento. "
        "Usa ESATTAMENTE la contromaglia allegata (ovale 8x4mm), "
        "INTEGRATA e LEGGERMENTE INCASSATA nel corpo (affogata, raccordo morbido). "
        "Vietato: traforo/outline/wireframe/ritaglio piatto; corpo SOLIDO con smussi. "
        "Sfondo NERO pieno, luce da studio coerente. "
        "Nessuna scritta, lettering, numeri o smalti."
    )


# =============================================================================
# Captioning (fallback rarissimo: quando mancano sia prompt sia immagine)
# =============================================================================
def _infer_prompt_from_image(inline_image: dict) -> str:
    instruction = (
        "Analizza l'immagine e descrivi SOLO il soggetto/motivo in 3-8 parole, "
        "senza lettere, numeri o testi. Evita colori; non menzionare scritte. "
        "Esempi: 'cane stilizzato', 'rosa stilizzata', 'croce latina pulita'."
    )
    contents = [instruction, {"inline_data": inline_image}]
    try:
        resp = client.models.generate_content(model=CAPTION_MODEL_ID, contents=contents)
        text = (getattr(resp, "text", "") or "").strip()
        return text or "motivo astratto organico"
    except Exception:
        return "motivo astratto organico"


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
            return JSONResponse(status_code=400, content={"detail": "Scrivi un motivo oppure allega un’immagine."})
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
        return JSONResponse(status_code=500, content={"detail": f"Errore chiamata API: {e}"})

    # Estrai la prima immagine utile dal primo candidato
    results: List[str] = []
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return JSONResponse(status_code=500, content={"detail": "Risposta vuota."})

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

                # === post-filtro: forza argento brunito monocromatico ===
                fixed = _silver_brunito_bytes(raw_bytes)
                results.append(_to_data_url(fixed, "image/png"))
                break

    if not results:
        return JSONResponse(status_code=422, content={"detail": "Nessuna immagine generata."})

    return GenerateResponse(images=results)


# =============================================================================
# Static & Index
# =============================================================================
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
