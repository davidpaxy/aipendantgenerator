import os, base64, asyncio, logging, hashlib, threading
from collections import OrderedDict
from typing import Optional, List, Final, Tuple

import anyio
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai

# ============================== Config ======================================
load_dotenv()
log = logging.getLogger("toygen")
logging.basicConfig(level=logging.INFO)

API_KEY = os.getenv("GOOGLE_API_KEY")
IMAGE_MODEL_ID = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
ROOT_PATH = os.getenv("ROOT_PATH", "")
GENAI_TIMEOUT_S: float = float(os.getenv("GENAI_TIMEOUT_S", "45"))
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS","").split(",") if o.strip()] or ["http://localhost:3000","http://127.0.0.1:3000"]

PIPELINE_VERSION: Final = "pendant-i2i-silver-gold-tolerant-v1"

SUPPORTED_MIME = {"image/png","image/jpeg","image/webp"}
MAX_IMAGE_BYTES = 18 * 1024 * 1024
CANVAS_PX: Final = 1024
FILL_RATIO_TARGET: Final = 0.72
FILL_TOLERANCE: Final = 0.03
FRAME_MARGIN_FRAC: Final = 0.10

_MAX_CACHE_ITEMS = int(os.getenv("MAX_CACHE_ITEMS", "128"))
_CACHE: "OrderedDict[str, bytes]" = OrderedDict()
_CACHE_LOCK = threading.Lock()

# ============================== App =========================================
client: Optional[genai.Client] = None
app = FastAPI(title="Pendant Generator (tolerant)", root_path=ROOT_PATH)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS, allow_credentials=False,
    allow_methods=["POST","GET","OPTIONS"], allow_headers=["Authorization","Content-Type"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# ============================== Schemi ======================================
class GenerateRequest(BaseModel):
    prompt: Optional[str] = None
    style: Optional[str] = "3d"          # tollerante
    size_mm: Optional[int] = 30          # tollerante
    material: Optional[str] = None       # "silver" | "gold" | "argento" | "oro" | None
    images: List[str] = Field(default_factory=list)
    format: Optional[str] = "png"
    width: Optional[int] = 1024
    height: Optional[int] = 1024
    lighting: Optional[str] = "soft"     # "soft" | "hard"

class GenerateResponse(BaseModel):
    images: List[str]

# ============================== Prompt ======================================
NEGATIVE_INSTRUCTIONS = (
    "Vietato: gancio, anellino, occhiello, bail, asola, foro, catena, laccetto. "
    "Vietato: supporti, piedistalli, pavimenti, ombre a terra. "
    "Vietato: testo, loghi, watermark, etichette, appunti, frecce, quote, righelli, numeri, simboli tecnici, targhette o marchi."
)

_ALLOWED_SIZES = [10,20,30,40,60]

def _norm_style(s: Optional[str]) -> str:
    if not s: return "3d"
    s = s.strip().lower()
    return "basrelief" if s.startswith("bas") else "3d"

def _norm_material(m: Optional[str]) -> str:
    if not m: return "silver"
    m = m.strip().lower()
    if m in ("gold","oro"): return "gold"
    return "silver"

def _norm_size(mm: Optional[int]) -> int:
    try:
        mm = int(mm or 30)
    except Exception:
        return 30
    # mappa al più vicino consentito
    return min(_ALLOWED_SIZES, key=lambda v: abs(v - mm))

def _norm_lighting(l: Optional[str]) -> str:
    if not l: return "soft"
    l = l.strip().lower()
    return "hard" if l.startswith("hard") else "soft"

def _constraints_block(size_mm: int, lighting: str, style: str, material: str) -> str:
    target_px = int(round(CANVAS_PX * FILL_RATIO_TARGET))
    tol_px = int(round(CANVAS_PX * FILL_TOLERANCE))
    margin_pct = int(round(FRAME_MARGIN_FRAC * 100))
    return (
        "CONSTRAINTS OBBLIGATORIE (seguire ALLA LETTERA):\n"
        f"- LOOK & MATERIAL: Metallo reale — "
        f"{('ARGENTO BRUNITO LUCIDO' if material=='silver' else 'ORO LUCIDO SATINATO (toni giallo oro realistici)')}.\n"
        + (
            "Saturazione colori = 0. Nessun colore residuo. I colori dell'immagine di riferimento, "
            "se presenti, vanno tradotti in luminanza/texture/roughness del metallo (mai tinta).\n"
            if material == "silver" else
            "Colore metallo ORO realistico (18–22k). Tono fra #C9A227 e #E0C15A; riflessi caldi, mezzitoni dorati. "
            "Vietato: tonalità argento/acciaio/grigio, verde o arancione intenso; niente patine non metalliche.\n"
        )
        + ("ILLUMINAZIONE: Softbox diffuso (riflessi ampi, zero hotspot).\n" if lighting=="soft"
           else "ILLUMINAZIONE: Studio direzionale controllato (riflessi definiti, zero blowout).\n")
        + "- SFONDO: Grigio chiarissimo #F2F2F2, uniforme, senza gradiente né ombre.\n"
        + f"- INQUADRATURA/SCALA: Altezza oggetto ≈ {FILL_RATIO_TARGET*100:.0f}% del frame {CANVAS_PX}px "
          f"(~{target_px}px, tolleranza ±{tol_px}px). Margini regolari ≈ {margin_pct}% ai bordi. "
          "Niente crop aggressivo, niente zoom variabile tra generazioni.\n"
        + f"- DIMENSIONE FISICA: Considera l’oggetto alto ~{size_mm} mm; spessori e rilievi plausibili per questa scala.\n"
        + ("STILE: TRIDIMENSIONALE a tutto volume, leggera prospettiva 3/4.\n" if style=="3d"
           else "STILE: BASSORILIEVO con percezione volumetrica credibile.\n")
        + f"- DIVIETI: {NEGATIVE_INSTRUCTIONS}\n"
    )

def _detail_instruction(size_mm: int) -> str:
    if size_mm <= 20:  return "Dettagli semplificati; silhouette leggibile; microincisioni poco profonde."
    if size_mm == 30:  return "Dettagli moderati e puliti; evita micro-texture fitte; rilievi coerenti."
    if size_mm == 40:  return "Buon dettaglio ma pulito; rilievi netti non fragili."
    return "Dettaglio pieno ma pulito; rilievi e spessori ben definiti."

def _header_line() -> str:
    return "Fotografia prodotto 3D di un piccolo oggetto/pendente in metallo."

def _build_text_only_prompt(user_prompt: str, style: str, size_mm: int, lighting: str, material: str) -> str:
    return (
        f"{_header_line()}\n"
        f"{_constraints_block(size_mm, lighting, style, material)}"
        f"{_detail_instruction(size_mm)}\n"
        f"Soggetto: {user_prompt.strip()}"
    )

def _build_image_repro_prompt(style: str, size_mm: int, lighting: str, material: str, user_prompt: Optional[str]) -> str:
    addon = f"\nIstruzioni aggiuntive: {user_prompt.strip()}" if user_prompt else ""
    return (
        f"{_header_line()}\n"
        "TRASFORMAZIONE: Riproduci fedelmente l’IMMAGINE DI RIFERIMENTO nei minimi particolari, "
        "ma come oggetto 3D reale (volume e rilievi fisicamente plausibili). Mantieni silhouette, proporzioni, posa, "
        "posizione degli elementi, incisioni e micro-dettagli.\n"
        f"CONVERSIONE MATERIALE: Mantieni forma, volume, dettagli, inquadratura e luce IDENTICI; "
        f"cambia SOLO la finitura del metallo in {material}. \n"
        f"{_constraints_block(size_mm, lighting, style, material)}"
        f"{_detail_instruction(size_mm)}"
        f"{addon}"
    )

# ============================== Utils =======================================
def _cache_get(k: str) -> Optional[bytes]:
    with _CACHE_LOCK:
        v = _CACHE.get(k)
        if v is not None: _CACHE.move_to_end(k)
        return v

def _cache_set(k: str, v: bytes) -> None:
    with _CACHE_LOCK:
        _CACHE[k] = v; _CACHE.move_to_end(k)
        while len(_CACHE) > _MAX_CACHE_ITEMS: _CACHE.popitem(last=False)

def _extract_first_image_bytes(response) -> Optional[bytes]:
    candidates = getattr(response, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) if content else None
        if not parts: continue
        for part in parts:
            inline_data = getattr(part, "inline_data", None)
            if inline_data and getattr(inline_data, "data", None):
                data_field = inline_data.data
                return data_field if isinstance(data_field, bytes) else base64.b64decode(data_field)
            if getattr(part, "mime_type", "") == "image/png" and getattr(part, "data", None):
                data_field = part.data
                return data_field if isinstance(data_field, bytes) else base64.b64decode(data_field)
    return None

def _to_data_url(raw: bytes) -> str:
    return f"data:image/png;base64,{base64.b64encode(raw).decode()}"

def _decode_image_data_url(data_url: str) -> Tuple[Optional[bytes], Optional[str]]:
    if not data_url or not isinstance(data_url, str): return None, None
    s = data_url.strip().replace("\n","").replace("\r","")
    if not s.startswith("data:"): return None, None
    try: head, payload = s.split(",", 1)
    except ValueError: return None, None
    mime = "image/png"
    if ";base64" in head:
        m = head[5: head.index(";base64")].strip().lower()
        if m: mime = m
    try: raw = base64.b64decode(payload)
    except Exception: return None, None
    return raw, mime

def _detect_true_mime(raw: bytes) -> Optional[str]:
    if not raw or len(raw) < 12: return None
    sig4 = raw[:4]; sig3 = raw[:3]
    if sig4 == b"\x89PNG": return "image/png"
    if sig3 == b"\xff\xd8\xff": return "image/jpeg"
    if sig4 == b"RIFF" and raw[8:12] == b"WEBP": return "image/webp"
    head = raw[:256].lstrip()
    if head.startswith(b"<svg") or head.startswith(b"<?xml"): return "image/svg+xml"
    if sig4 in (b"ftyp", b"\x00\x00\x01\x00", b"BM", b"GIF8"): return "unsupported"
    return None

def _build_contents(final_prompt: str, ref_bytes: Optional[bytes], ref_mime: Optional[str]):
    if ref_bytes:
        return [{
            "role":"user",
            "parts":[
                {"inline_data":{"mime_type":(ref_mime or "image/png"), "data": ref_bytes}},
                {"text": final_prompt},
            ],
        }]
    else:
        return [{"role":"user","parts":[{"text":final_prompt}]}]

# ============================== Startup =====================================
@app.on_event("startup")
def _startup():
    global client
    if not API_KEY: raise RuntimeError("Imposta GOOGLE_API_KEY")
    client = genai.Client(api_key=API_KEY)

# ============================== Core ========================================
async def _run_inference(user_prompt: str,
                         ref_bytes: Optional[bytes],
                         ref_mime: Optional[str],
                         style: str, size_mm: int, lighting: str, material: str) -> GenerateResponse:
    if not user_prompt and not ref_bytes:
        raise HTTPException(422, "Prompt mancante e nessuna immagine di riferimento")

    final_prompt = (
        _build_image_repro_prompt(style, size_mm, lighting, material, user_prompt or None)
        if ref_bytes else
        _build_text_only_prompt(user_prompt, style, size_mm, lighting, material)
    )

    if ref_bytes:
        true_mime = _detect_true_mime(ref_bytes)
        if true_mime in (None,"unsupported","image/svg+xml"):
            raise HTTPException(422, "Formato immagine non supportato. Usa PNG/JPEG/WEBP.")
        if true_mime not in SUPPORTED_MIME:
            raise HTTPException(422, f"Formato non accettato: {true_mime}.")
        if len(ref_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(413, "Immagine troppo grande (>18MB).")
        ref_mime = true_mime

    parts = [PIPELINE_VERSION, IMAGE_MODEL_ID, ("i2i" if ref_bytes else "text"),
             style, str(size_mm), lighting, material, final_prompt]
    if ref_bytes: parts.append(hashlib.sha256(ref_bytes).hexdigest())
    cache_key = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()

    cached = _cache_get(cache_key)
    if cached is not None: return GenerateResponse(images=[_to_data_url(cached)])

    try:
        response = await asyncio.wait_for(
            anyio.to_thread.run_sync(lambda: client.models.generate_content(
                model=IMAGE_MODEL_ID, contents=_build_contents(final_prompt, ref_bytes, ref_mime)
            ), cancellable=True),
            timeout=GENAI_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        raise HTTPException(504, "Timeout generazione")
    except Exception:
        log.exception("Errore modello")
        raise HTTPException(502, "Servizio non disponibile")

    raw = _extract_first_image_bytes(response)
    if not raw: raise HTTPException(422, "Nessuna immagine generata")

    _cache_set(cache_key, raw)
    return GenerateResponse(images=[_to_data_url(raw)])

# ============================== Endpoint ====================================
@app.post("/api/generate", response_model=GenerateResponse)
async def generate_images(req: GenerateRequest):
    # Normalizzazione robusta (evita 422 per input “strani”)
    style = _norm_style(req.style)
    size_mm = _norm_size(req.size_mm)
    lighting = _norm_lighting(req.lighting)
    material = _norm_material(req.material)

    user_prompt = (req.prompt or "").strip()

    ref_bytes, ref_mime = (None, None)
    if isinstance(req.images, list) and req.images and isinstance(req.images[0], str):
        ref_bytes, ref_mime = _decode_image_data_url(req.images[0])

    log.info("req: has_prompt=%s, images_len=%d, decoded_img=%s, style=%s, size=%s, lighting=%s, material=%s",
             bool(user_prompt), len(req.images or []), bool(ref_bytes), style, size_mm, lighting, material)

    return await _run_inference(user_prompt, ref_bytes, ref_mime, style, size_mm, lighting, material)

# ============================== Cache & Health ===============================
def _require_admin(token: Optional[str]):
    if ADMIN_TOKEN and token != ADMIN_TOKEN: raise HTTPException(403, "Forbidden")

@app.get("/cache/stats")
def cache_stats():
    with _CACHE_LOCK: return {"items": len(_CACHE), "max_items": _MAX_CACHE_ITEMS}

@app.post("/cache/clear")
def cache_clear(authorization: Optional[str] = Header(None)):
    _require_admin(authorization); _CACHE.clear(); return {"ok": True}

@app.get("/health")
def health():
    return {"ok": True, "pipeline": PIPELINE_VERSION}

@app.get("/")
def index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        return {"message": "Pendant Generator (tolerant) - ready"}
    return FileResponse(index_path)
