import os
import base64
import logging
from io import BytesIO
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from google import genai

# ------------------------------
# Config & client
# ------------------------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("Imposta la variabile d'ambiente GOOGLE_API_KEY")

MODEL_ID = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")

client = genai.Client(api_key=API_KEY)

# Logger basico
logger = logging.getLogger("aipendant")
logging.basicConfig(level=logging.INFO)

# ------------------------------
# FastAPI app
# ------------------------------
app = FastAPI(title="NOVE25 Pendant Generator", root_path="/aipendant")

# CORS: di default senza credenziali per poter usare "*".
# Se servono cookie/Authorization, imposta allow_origins con domini espliciti e metti allow_credentials=True.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # metti True e specifica origini se necessario
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# Schemi richiesta/risposta
# ------------------------------
class GenerateRequest(BaseModel):
    prompt: str
    style: Optional[str] = "3d"           # accetta "3d" | "basrelief" (normalizzato più sotto)
    size_mm: Optional[int] = 30            # accetta 20 | 30 | 40 (normalizzato più sotto)
    images: Optional[List[str]] = None     # data URL base64 (max 5)
    format: Optional[str] = "png"         # "png" | "jpeg" | "jpg"

class GenerateResponse(BaseModel):
    images: List[str]  # data URL

# ------------------------------
# Utilità
# ------------------------------
SUPPORTED_OUT = {"png": "image/png", "jpeg": "image/jpeg", "jpg": "image/jpeg"}
MAX_TOTAL_IMAGE_BYTES = 10 * 1024 * 1024  # 10MB


def _to_data_url(raw_bytes: bytes, mime: str) -> str:
    b64 = base64.b64encode(raw_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _normalize_format(fmt: Optional[str]) -> str:
    f = (fmt or "png").lower()
    if f == "jpg":
        return "jpeg"
    return f if f in ("png", "jpeg") else "png"


def _sanitize_user_prompt(s: str, max_len: int = 2000) -> str:
    s = " ".join((s or "").split())
    return s[:max_len]


def _wants_text(user_prompt: str) -> bool:
    up = (user_prompt or "").lower()
    return any(k in up for k in ["scritta", "testo", "text", "wording", "lettering"])  # IT/EN key hints


def _style_clause(style: str) -> str:
    s = (style or "3d").lower()
    if s == "basrelief":
        return (
            "Design it as a BAS-RELIEF (bassorilievo): shallow relief, minimal depth, "
            "FRONT-FACING (frontal view). "
        )
    return (
        "Design it as a FULL 3D pendant: volumetric, realistic occlusions/reflections, "
        "shown almost frontal with a very slight three-quarter tilt (subtle) only to suggest depth. "
    )


def _size_clause(size_mm: int) -> str:
    size = size_mm if size_mm in (20, 30, 40) else 30
    ratio = round(6 / size, 2)
    return (
        f"The PENDANT BODY HEIGHT (excluding the bail) must be ~{size} mm. "
        "Keep the bail (\"maglina\") ABSOLUTELY CONSTANT across images: external height ~6 mm, "
        "internal opening ~4 mm, same exact design and scale as previous images. "
        f"Ensure the visual proportion reflects this: at {size} mm body height the 6 mm bail should appear "
        f"about {int(100 * 6 / size)}% of the body's height (approx. ratio {ratio}:1). "
        "Do NOT scale the bail to match the body; only scale the pendant body to reach the requested size. "
    )


def _finish_clause():
    return (
        "MATERIAL: single polished darkened silver only (argento brunito lucido). "
        "Metalness 1.0, roughness ~0.15 (non matte), clearcoat 0, no brushed finish, "
        "no patina beyond subtle darkening in recesses. "
        "Consistent micro-scratches barely visible only in grazing light. "
        "NO variation of material parameters between images. "
    )


def _lighting_clause():
    return (
        "LIGHTING: identical studio setup for all images: two large softboxes at 45° left/right, "
        "one weaker rim light top-back. No color gels (white light ~5600K). "
        "Keep exposure/contrast identical across images. Use identical tone mapping (gamma 2.2). "
        "Background: pure black (#000000), no gradient, no vignette. "
    )


def _camera_clause():
    return (
        "CAMERA: same focal length (~80mm equivalent), same distance and framing, "
        "front-facing with a very slight three-quarter tilt (<10°), horizon level, "
        "no perspective exaggeration, no DOF blur. "
    )


def _build_pendant_prompt(user_prompt: str, style: str, size_mm: int) -> str:
    BASE = (
        "You are a jewelry designer. Create a photorealistic PRODUCT PHOTO of a "
        'NECKLACE PENDANT made entirely of polished darkened silver ("argento brunito lucido"). '
        + _style_clause(style)
        + _size_clause(size_mm)
        + _finish_clause()
        + _lighting_clause()
        + _camera_clause()
        + 'Include a bail ("maglina") at the top that is ALWAYS the same design across images: '
        + "an oval polished silver bail, external height ~6mm, internal opening ~4mm tall, seamlessly connected to the pendant. "
        + "Do NOT include counter-bail or extra ring. "
        + "Use ONLY silver (argento brunito lucido). NEVER use gold, rose gold, bronze, copper, or other materials/colors. "
        + "NO watermarks/logos/text on the image. "
    )
    if not _wants_text(user_prompt):
        BASE += "Do NOT depict letters. Interpret words/names abstractly as motifs. "
    BASE += "User design motif: "
    return BASE + user_prompt


def _data_url_to_inline(durl: str):
    if not durl or not durl.startswith("data:"):
        return None
    try:
        header, b64 = durl.split(",", 1)
    except ValueError:
        return None
    b64 = b64.strip()
    try:
        mime = header.split("data:", 1)[1].split(";", 1)[0] or "image/png"
    except Exception:
        mime = "image/png"
    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception:
        return None
    if mime not in ("image/png", "image/jpeg"):
        return None
    return {"mime_type": mime, "data": raw}

# ------------------------------
# Endpoint principale
# ------------------------------
@app.post("/api/generate", response_model=GenerateResponse)
def generate_images(req: GenerateRequest):
    prompt = _sanitize_user_prompt(req.prompt)
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt mancante.")

    style = (req.style or "3d").lower()
    if style not in ("3d", "basrelief"):
        style = "3d"
    try:
        size_val = int(req.size_mm or 30)
    except Exception:
        size_val = 30
    if size_val not in (20, 30, 40):
        size_val = 30

    out_fmt = _normalize_format(req.format)
    out_mime = SUPPORTED_OUT[out_fmt]

    final_prompt = _build_pendant_prompt(prompt, style, size_val)

    # Costruzione contents: prompt + (eventuali) immagini inline
    contents = [final_prompt]

    total_bytes = 0
    if req.images:
        for durl in req.images[:5]:
            inline = _data_url_to_inline(durl)
            if inline:
                total_bytes += len(inline["data"])
                if total_bytes > MAX_TOTAL_IMAGE_BYTES:
                    raise HTTPException(status_code=413, detail="Immagini troppo grandi (max 10MB).")
                contents.append({"inline_data": inline})

    # Chiamata al modello
    try:
        response = client.models.generate_content(model=MODEL_ID, contents=contents)
    except Exception:
        logger.exception("Errore chiamata API")
        raise HTTPException(status_code=502, detail="Servizio di generazione non disponibile al momento.")

    # Parsing risposta
    if not getattr(response, "candidates", None):
        raise HTTPException(status_code=502, detail="Risposta vuota dal servizio di generazione.")

    results: List[str] = []
    seen = set()

    # Per massima coerenza della finitura, consideriamo solo il primo candidato
    for cand in response.candidates[:1]:
        content = getattr(cand, "content", None)
        if not content or not getattr(content, "parts", None):
            continue
        for part in content.parts:
            inline_data = getattr(part, "inline_data", None)
            if inline_data and getattr(inline_data, "data", None):
                data_field = inline_data.data
                raw_bytes = data_field if isinstance(data_field, bytes) else base64.b64decode(data_field)
                key = hash(raw_bytes)
                if key in seen:
                    continue
                seen.add(key)
                results.append(_to_data_url(raw_bytes, out_mime))

    if not results:
        raise HTTPException(status_code=422, detail="Nessuna immagine generata.")

    return GenerateResponse(images=results)

# ------------------------------
# Static & index
# ------------------------------
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
