import os, base64, asyncio, logging, hashlib, threading, time
from collections import OrderedDict
from typing import Optional, List, Final, Tuple

import anyio
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai

from io import BytesIO
try:
    from PIL import Image  # pip install pillow
except Exception:
    Image = None  # se Pillow manca, la normalizzazione verrà saltata

# ============================== Config ======================================
load_dotenv()
log = logging.getLogger("toygen")
logging.basicConfig(level=logging.INFO)

API_KEY = os.getenv("GOOGLE_API_KEY")
IMAGE_MODEL_ID = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
ROOT_PATH = os.getenv("ROOT_PATH", "")

# ⏱️ Timeout più aggressivo per ridurre la latenza
GENAI_TIMEOUT_S: float = float(os.getenv("GENAI_TIMEOUT_S", "20"))

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS","").split(",") if o.strip()] or [
    "http://localhost:3000","http://127.0.0.1:3000"
]

PIPELINE_VERSION: Final = "pendant-i2i-silver-gold-v5.3-fast72-lod10-extra"

SUPPORTED_MIME = {"image/png","image/jpeg","image/webp"}
MAX_IMAGE_BYTES = 18 * 1024 * 1024
CANVAS_PX: Final = 1024
FILL_RATIO_TARGET: Final = 0.72
FILL_TOLERANCE: Final = 0.02
FRAME_MARGIN_FRAC: Final = 0.10

# ✅ Spessore minimo per elementi strutturali
MIN_STRUCTURAL_MM: Final = 1.0

_BG_RGB = (242, 242, 242)  # #F2F2F2
DECOLORIZE_REF: Final = True  # desatura sempre la reference in i2i

_MAX_CACHE_ITEMS = int(os.getenv("MAX_CACHE_ITEMS", "128"))
_CACHE: "OrderedDict[str, bytes]" = OrderedDict()
_CACHE_LOCK = threading.Lock()

# Sampling config
GENERATION_CONFIG = {
    "temperature": float(os.getenv("GEN_TEMPERATURE", "0.5")),
    "top_p": float(os.getenv("GEN_TOP_P", "0.9")),
    "top_k": int(os.getenv("GEN_TOP_K", "40")),
}

def _build_gen_config():
    """Config compatibile SDK; filtra chiavi non ammesse."""
    cfg = {k: v for k, v in GENERATION_CONFIG.items() if k != "response_mime_type"}
    try:
        from google.genai import types as genai_types
        return genai_types.GenerateContentConfig(**cfg)
    except Exception:
        return cfg

def _finish_info(resp):
    try:
        c = (getattr(resp, "candidates", None) or [])[0]
        return f"finish_reason={getattr(c, 'finish_reason', None)}, safety={getattr(c, 'safety_ratings', None)}"
    except Exception:
        return "finish_info=unavailable"

# ============================== App =========================================
client: Optional[genai.Client] = None
app = FastAPI(title="Pendant Generator (fast)", root_path=ROOT_PATH)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# ============================== Schemi ======================================
class GenerateRequest(BaseModel):
    prompt: Optional[str] = None
    style: Optional[str] = "3d"
    size_mm: Optional[int] = 30
    material: Optional[str] = None       # "silver" | "gold" | "argento" | "oro" | None
    images: List[str] = Field(default_factory=list)
    format: Optional[str] = "png"
    width: Optional[int] = 1024
    height: Optional[int] = 1024
    lighting: Optional[str] = "soft"     # "soft" | "hard"
    mode: Optional[str] = Field("fast", pattern="^(fast|balanced|polish)$")

class GenerateResponse(BaseModel):
    images: List[str]

# ============================== Sanitizer ====================================
# Semplificato: rimuoviamo solo i termini di fissaggio
_BAN_PENDANT_TERMS = {
    # IT
    "pendente","ciondolo","anello","anellino","gancio","bail","asola","occhiello","foro","foratura","catena","collana","laccetto",
    # EN
    "pendant","loop","ring","bail","hook","eyelet","hole","drill","chain","necklace","cord"
}

def _sanitize_prompt(text: str) -> str:
    """Rimuove solo termini di fissaggio; le parole di colore restano."""
    if not text:
        return ""
    import re
    t = text
    ban_pat = r"\b(" + "|".join(map(re.escape, sorted(_BAN_PENDANT_TERMS, key=len, reverse=True))) + r")\b"
    t = re.sub(ban_pat, "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t

def _primary_concept(text: str) -> str:
    if not text:
        return ""
    import re
    parts = re.split(r"\s*(?:,|;|/|\+|\band\b|\be\b)\s*", text, maxsplit=1, flags=re.IGNORECASE)
    return parts[0].strip() if parts else text.strip()

# ============================== Prompt ======================================
NEGATIVE_INSTRUCTIONS = (
    "Vietato: gancio, anellino, occhiello, bail, asola, foro, catena, laccetto. "
    "Vietato: supporti, piedistalli, pavimenti, ombre a terra. "
    "Vietato: testo, loghi, watermark, etichette, appunti, frecce, quote, righelli, numeri, simboli tecnici, targhette o marchi. "
    "Vietato: qualunque vernice/smalto/patina non metallica."
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
    if m in ("silver","argento"): return "silver"
    return "silver"

def _norm_size(mm: Optional[int]) -> int:
    try:
        mm = int(mm or 30)
    except Exception:
        return 30
    return min(_ALLOWED_SIZES, key=lambda v: abs(v - mm))

def _norm_lighting(l: Optional[str]) -> str:
    if not l: return "soft"
    l = l.strip().lower()
    return "hard" if l.startswith("hard") else "soft"

def _extra_rules_10mm() -> str:
    return (
        "REGOLE SPECIALI 10mm (SEMPLIFICAZIONE FORTE):\n"
        "- Silhouette liscia e leggibile da lontano; nessuna micro-frastagliatura.\n"
        "- Bevel/smussi principali equivalenti a 0.30–0.40 mm; spigoli non taglienti.\n"
        "- Vietate micro-incisioni, micro-pori e texture fini: **rimuoverle**.\n"

  
    
        "- Nessun elemento < 1.0 mm; parti sottili → ispessire e raccordare.\n"
    )

def _constraints_block(size_mm: int, lighting: str, style: str, material: str) -> str:
    target_px = int(round(CANVAS_PX * FILL_RATIO_TARGET))
    tol_px = int(round(CANVAS_PX * FILL_TOLERANCE))
    margin_pct = int(round(FRAME_MARGIN_FRAC * 100))

    # --- Colore super-compatto con caso Silver brunito robusto ---
    if material == "silver":
        color_line = (
            "COLORE: metallo ARGENTO realistico (achromatico). "
            "Brunitura **solo** nelle cavità/incisioni come occlusion/roughness; "
            "vietati pannelli scuri piatti o smalti. "
            "Niente nero pieno: recessi 20–35% più scuri della base.\n"
        )
    else:
        color_line = (
            "COLORE: metallo ORO realistico (18–22k). "
            "Vietate vernici/smalti; resa fisica del metallo.\n"
        )

    txt = (
        "CONSTRAINTS OBBLIGATORIE (seguire ALLA LETTERA):\n"
        f"- LOOK & MATERIAL: {('ARGENTO chiaro satinato, cavità brunite' if material=='silver' else 'ORO lucido satinato')}.\n"
        + color_line
        + ("ILLUMINAZIONE: Softbox diffuso (riflessi ampi, zero hotspot).\n" if lighting=="soft"
           else "ILLUMINAZIONE: Studio direzionale controllato (riflessi definiti, zero blowout).\n")
        + "- SFONDO: Grigio chiarissimo #F2F2F2, uniforme, senza gradiente né ombre.\n"
        + f"- INQUADRATURA/SCALA: Altezza oggetto ≈ {FILL_RATIO_TARGET*100:.0f}% del frame {CANVAS_PX}px "
          f"(~{target_px}px, tolleranza ±{tol_px}px). Margini regolari ≈ {margin_pct}% ai bordi. "
          "Niente crop aggressivo, niente zoom variabile tra generazioni.\n"
        + f"- DIMENSIONE FISICA (per LOD, non per scala): considera l’oggetto alto ~{size_mm} mm.\n"
        + f"- ROBUSTEZZA STRUTTURALE: **nessun elemento strutturale < {MIN_STRUCTURAL_MM:.1f} mm** "
          "(es. aste sottili, punte, anelli interni, setti, ponticelli). "
          "Se necessario, ispessisci e raccorda i tratti critici.\n"
        + ("STILE: TRIDIMENSIONALE a tutto volume, leggera prospettiva 3/4.\n" if style=="3d"
           else "STILE: BASSORILIEVO con percezione volumetrica credibile.\n")
        + "- COERENZA SEMANTICA: un SOLO soggetto principale, chiaro e leggibile. Niente ibridi o oggetti estranei.\n"
        + f"- DIVIETI: {NEGATIVE_INSTRUCTIONS}\n"
        + "- SENZA FORI/AGGANCI: non creare gancio, anellino, bail, asole o fori; se presenti in reference, rimuovili ricostruendo il bordo.\n"
        + "- BACKGROUND SANITY: sfondo #F2F2F2 uniforme, nessun oggetto separato o appoggio sul piano.\n"
    )
    if size_mm == 10:
        txt += _extra_rules_10mm()
    return txt

# --------------------------- LOD policy --------------------------------------
def _lod_key(size_mm: int) -> int:
    return min([10, 20, 30, 40, 60], key=lambda v: abs(v - size_mm))

def _lod_policy_text(size_mm: int) -> str:
    key = _lod_key(size_mm)
    if key == 10:
        return (
            "LOD 10mm — RIELABORAZIONE SEMPLIFICATA:\n"
            "- Priorità a silhouette e volumi primari; CONSENTITA semplificazione/aggregazione.\n"
            "- Bevel bordi principali: 0.30–0.40 mm (spigoli morbidi e continui).\n"
            "- **Rimuovi** micro-incisioni (<0.25 mm), pori e trame fini; NO rumore.\n"
        
            "- Nessun elemento strutturale < 1.0 mm: ispessire e raccordare.\n"
            "- Rispetta sempre gli spessori strutturali minimi (≥ 1.0 mm) per parti portanti.\n"
        )
    if key == 20:
        return (
            "LOD 20mm — SEMPLIFICAZIONE MODERATA:\n"
            "- Mantieni volumi primari; dettagli secondari solo se migliorano la leggibilità.\n"
            "- Bevel bordi: 0.20–0.30 mm.\n"
            "- Incisioni principali sì; evita micro-incisioni fitte e grana granulare.\n"
            "- Rispetta sempre gli spessori strutturali minimi (≥ 1.0 mm) per parti portanti.\n"
        )
    if key == 30:
        return (
            "LOD 30mm — DETTAGLIO A BUONA DEFINIZIONE:\n"
            "- Maggiore precisione di bordi e interruzioni; micro-dettaglio ammesso se migliora la lettura.\n"
            "- Bevel bordi: 0.16–0.24 mm (spigoli più nitidi ma non taglienti).\n"
            "- Rispetta sempre gli spessori strutturali minimi (≥ 1.0 mm) per parti portanti.\n"


        )
    if key == 40:
        return (
            "LOD 40mm — DETTAGLIO ALTO PULITO:\n"
            "- Bevel bordi: 0.14–0.20 mm.\n"
            "- Ammesse incisioni fini coerenti; suture e fessure nitide ma non fragili.\n"
            "- Pori/segni superficiali solo accennati e distribuiti con criterio (no pattern rumoroso).\n"
            "- Rispetta sempre gli spessori strutturali minimi (≥ 1.0 mm) per parti portanti.\n"
        )
    return (
        "LOD 60mm — MICRO-DETTAGLIO CONTROLLATO:\n"
        "- Bevel bordi: 0.12–0.18 mm.\n"
        "- **Consenti micro-incisioni** sottili (suture, fessure denti, orbite) se migliorano la lettura.\n"
        "- Pori/texture finissime appena percepibili; evita artefatti casuali.\n"
        "- Rispetta sempre gli spessori strutturali minimi (≥ 1.0 mm) per parti portanti.\n"
    )

def _detail_instruction(size_mm: int) -> str:
    key = _lod_key(size_mm)
    return {
        10: "Rielabora e semplifica: pochi segni grossi e puliti; elimina micro-texture e dentini sottili.",
        20: "Semplifica i pattern minuti; privilegia silhouette e tratti iconici, dettaglio solo dove serve.",
        30: "Buona definizione: dettagliato;  dettagli fini controllati (no grana).",
        40: "Dettaglio alto ma ordinato; incidi fine dove migliora la lettura; evita trame ‘a rumore’.",
        60: "Introduci micro-dettaglio coerente (micro-incisioni/pori lievi); mantieni robustezza e pulizia generale."
    }[key]

def _header_line() -> str:
    return "Fotografia prodotto 3D di un piccolo oggetto in metallo (non montato)."

# ===== Prompts ===============================================================
def _build_text_only_prompt(user_prompt: str, style: str, size_mm: int, lighting: str, material: str) -> str:
    target_px = int(round(CANVAS_PX * FILL_RATIO_TARGET))
    tol_px = max(1, int(round(CANVAS_PX * 0.015)))
    margin_px = int(round(FRAME_MARGIN_FRAC * CANVAS_PX))
    return (
        f"{_header_line()}\n"
        f"{_constraints_block(size_mm, lighting, style, material)}"
        "OUTPUT: Genera **UNA** sola immagine PNG, 1024×1024, sfondo uniforme #F2F2F2.\n"
        f"FRAMING: ALTEZZA soggetto = {target_px}px (±{tol_px}px); centra; margini ≈ {margin_px}px.\n"
        "CAMERA: distanza/focale costanti; prospettiva minima; niente crop aggressivo.\n"
        "COERENZA: attieniti al soggetto richiesto; non introdurre altri soggetti/scene/attributi.\n"
        f"{_lod_policy_text(size_mm)}"
        "⚠️ In caso di conflitto tra le istruzioni dell’utente e le regole LOD per la dimensione scelta, "
        "prevalgono le regole LOD (livello di dettaglio coerente con la scala fisica dell’oggetto).\n"
        "Evita campiture scure uniformi: traduci eventuali aree scure in **brunitura metallica** (non smalto/vernice).\n"
        f"{_detail_instruction(size_mm)}\n"
        f"Soggetto: {user_prompt.strip()}"
    )

def _build_image_repro_prompt(style: str, size_mm: int, lighting: str, material: str, user_prompt: Optional[str]) -> str:
    addon = f"\nIstruzioni aggiuntive (no colore): {user_prompt.strip()}" if user_prompt else ""
    target_px = int(round(CANVAS_PX * FILL_RATIO_TARGET))
    tol_px = max(1, int(round(CANVAS_PX * 0.015)))
    margin_px = int(round(FRAME_MARGIN_FRAC * CANVAS_PX))
    return (
        f"{_header_line()}\n"
        "TRASFORMAZIONE: Riproduci fedelmente l’immagine di riferimento (silhouette, proporzioni, posa, incisioni). "
        "Converti il materiale nel metallo richiesto.\n"
        f"{_constraints_block(size_mm, lighting, style, material)}"
        "OUTPUT: **UNA** PNG 1024×1024, sfondo #F2F2F2.\n"
        f"FRAMING: ALTEZZA soggetto = {target_px}px (±{tol_px}px); centra; margini ≈ {margin_px}px.\n"
        "CAMERA: distanza/focale fisse; prospettiva bassa; nessun crop aggressivo.\n"
        "COERENZA: attieniti strettamente alla reference e alle istruzioni; non aggiungere soggetti o dettagli non presenti.\n"
        f"{_lod_policy_text(size_mm)}"
        "⚠️ In caso di conflitto tra reference/istruzioni e le regole LOD per la dimensione scelta, "
        "prevalgono le regole LOD (livello di dettaglio coerente con la scala fisica dell’oggetto).\n"
        "Evita campiture scure uniformi: traduci eventuali aree scure della reference in **brunitura metallica** (non smalto/vernice).\n"
        f"{_detail_instruction(size_mm)}"
        f"{addon}"
    )

def _build_text_only_prompt_lite(user_prompt, style, size_mm, lighting, material):
    return (
        "Product photo, single small metal object on uniform light gray background.\n"
        "ONE 1024x1024 PNG only. Center the subject. No extra props.\n"
        f"Material: {'silver (semi-matte, darker only in cavities)' if material=='silver' else 'gold (18–22k)'}.\n"
        "No hooks/holes/text/logos. No painted/enamel colors; reinterpret any colored areas as metal-only.\n"
        "If the user's request conflicts with LOD rules for the chosen size, LOD rules take precedence.\n"
        "Avoid flat dark paint-like fills; use **metal burnishing** in recesses instead.\n"
        f"Subject: {user_prompt.strip()}"
    )

def _build_image_repro_prompt_lite(style, size_mm, lighting, material, user_prompt):
    extra = f"\nNotes: {user_prompt.strip()}" if user_prompt else ""
    return (
        "Transform the reference image into a realistic metal object.\n"
        "Return ONE 1024x1024 PNG on uniform light gray, centered. No extra items.\n"
        f"Material: {'silver (semi-matte, darker only in cavities)' if material=='silver' else 'gold (18–22k)'}.\n"
        "No painted/enamel colors; if the reference has colors, reinterpret them as metal tonality only.\n"
        "If instructions or reference conflict with LOD rules for the size, LOD rules prevail.\n"
        "Avoid flat dark paint-like fills; render **burnished metal** only in recesses."
        + extra
    )

# ============================== Utils =======================================
def _cache_get(k: str) -> Optional[bytes]:
    with _CACHE_LOCK:
        v = _CACHE.get(k)
        if v is not None:
            _CACHE.move_to_end(k)
        return v

def _cache_set(k: str, v: bytes) -> None:
    with _CACHE_LOCK:
        _CACHE[k] = v
        _CACHE.move_to_end(k)
        while len(_CACHE) > _MAX_CACHE_ITEMS:
            _CACHE.popitem(last=False)

def _extract_first_image_bytes(response) -> Optional[bytes]:
    candidates = getattr(response, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) if content else None
        if not parts:
            continue
        for part in parts:
            inline_data = getattr(part, "inline_data", None)
            if inline_data and getattr(inline_data, "data", None):
                data_field = inline_data.data
                return data_field if isinstance(data_field, bytes) else base64.b64decode(data_field)
            mt = getattr(part, "mime_type", "")
            if mt in {"image/png", "image/jpeg", "image/webp"} and getattr(part, "data", None):
                data_field = part.data
                return data_field if isinstance(data_field, bytes) else base64.b64decode(data_field)
    return None

def _to_data_url(raw: bytes) -> str:
    return f"data:image/png;base64,{base64.b64encode(raw).decode()}"

def _decode_image_data_url(data_url: str) -> Tuple[Optional[bytes], Optional[str]]:
    if not data_url or not isinstance(data_url, str):
        return None, None
    s = data_url.strip().replace("\n","").replace("\r","")
    if not s.startswith("data:"):
        return None, None
    try:
        head, payload = s.split(",", 1)
    except ValueError:
        return None, None
    mime = "image/png"
    if ";base64" in head:
        m = head[5: head.index(";base64")].strip().lower()
        if m:
            mime = m
    try:
        raw = base64.b64decode(payload)
    except Exception:
        return None, None
    return raw, mime

def _detect_true_mime(raw: bytes) -> Optional[str]:
    if not raw or len(raw) < 12:
        return None
    sig4 = raw[:4]; sig3 = raw[:3]
    if sig4 == b"\x89PNG": return "image/png"
    if sig3 == b"\xff\xd8\xff": return "image/jpeg"
    if sig4 == b"RIFF" and raw[8:12] == b"WEBP": return "image/webp"
    head = raw[:256].lstrip()
    if head.startswith(b"<svg") or head.startswith(b"<?xml"): return "image/svg+xml"
    if len(raw) >= 12 and raw[4:8] == b"ftyp": return "unsupported"  # HEIC/AVIF
    if sig4 in (b"\x00\x00\x01\x00", b"BM", b"GIF8"): return "unsupported"
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

# -------- Normalizzazione a 1024×1024 (solo se non in fast) ------------------
def _looks_already_normalized(raw: bytes) -> bool:
    if not Image:
        return False
    try:
        with Image.open(BytesIO(raw)) as im:
            if im.size != (CANVAS_PX, CANVAS_PX):
                return False
            px = im.convert("RGB").load()
            pts = [(4,4), (CANVAS_PX-5, 4), (4, CANVAS_PX-5), (CANVAS_PX-5, CANVAS_PX-5), (CANVAS_PX//2, CANVAS_PX//2)]
            def d(a,b): return sum(abs(a[i]-b[i]) for i in range(3))
            return all(d(px[x,y], _BG_RGB) <= 9 for x,y in pts)
    except Exception:
        return False

def _normalize_to_square_canvas(ref_bytes: bytes, size: int = CANVAS_PX, bg_rgb: Tuple[int,int,int] = _BG_RGB) -> Optional[bytes]:
    if not Image:
        return None
    try:
        with Image.open(BytesIO(ref_bytes)) as im:
            im = im.convert("RGBA")
            w, h = im.size
            if w == 0 or h == 0:
                return None
            if (w, h) == (size, size) and _looks_already_normalized(ref_bytes):
                return ref_bytes
            scale = min(size / w, size / h)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
            im_resized = im.resize((new_w, new_h), resample)

            canvas = Image.new("RGBA", (size, size), (bg_rgb[0], bg_rgb[1], bg_rgb[2], 255))
            off_x = (size - new_w) // 2
            off_y = (size - new_h) // 2
            canvas.alpha_composite(im_resized, (off_x, off_y))

            out = BytesIO()
            canvas.save(out, format="PNG", optimize=True)
            return out.getvalue()
    except Exception:
        log.exception("Normalize to 1024 square failed")
        return None

# ---------- Utility: desaturazione reference i2i ----------
def _desaturate_ref(ref_bytes: bytes) -> Optional[bytes]:
    if not Image:
        return None
    try:
        with Image.open(BytesIO(ref_bytes)) as im:
            im = im.convert("RGBA")
            gray = im.convert("L").convert("RGBA")  # scala di grigi semplice
            out = BytesIO()
            gray.save(out, format="PNG", optimize=True)
            return out.getvalue()
    except Exception:
        return None

# ============== Rimozione oggetti estranei (solo non fast) ===================
def _cleanup_foreign_blobs(png_bytes: bytes, bg=_BG_RGB, tol=10) -> bytes:
    """Rimuove pixel non di sfondo che non appartengono al componente principale."""
    if not Image:
        return png_bytes
    try:
        im = Image.open(BytesIO(png_bytes)).convert("RGBA")
        w, h = im.size
        pix = im.load()

        def is_fg(p):
            return sum(abs(p[i]-bg[i]) for i in range(3)) > tol and p[3] >= 8

        visited = [[False]*w for _ in range(h)]
        areas, comps = [], []
        from collections import deque

        for y in range(h):
            for x in range(w):
                if visited[y][x]: continue
                r,g,b,a = pix[x,y]
                if not is_fg((r,g,b,a)):
                    visited[y][x] = True
                    continue
                q = deque([(x,y)])
                visited[y][x] = True
                comp = [(x,y)]
                while q:
                    cx,cy = q.popleft()
                    for nx,ny in ((cx-1,cy),(cx+1,cy),(cx,cy-1),(cx,cy+1)):
                        if 0 <= nx < w and 0 <= ny < h and not visited[ny][nx]:
                            r2,g2,b2,a2 = pix[nx,ny]
                            if is_fg((r2,g2,b2,a2)):
                                visited[ny][nx] = True
                                q.append((nx,ny))
                                comp.append((nx,ny))
                            else:
                                visited[ny][nx] = True
                comps.append(comp); areas.append(len(comp))

        if not areas:
            return png_bytes

        keep_idx = max(range(len(areas)), key=lambda i: areas[i])
        bg_rgba = (*bg, 255)
        for i, comp in enumerate(comps):
            if i == keep_idx: continue
            for (x,y) in comp:
                pix[x,y] = bg_rgba

        out = BytesIO(); im.save(out, format="PNG", optimize=True)
        return out.getvalue()
    except Exception:
        return png_bytes

# ============================== Modalità & Budget ============================
def _mode_settings(mode: str) -> Tuple[int, int]:
    if mode == "fast":
        return (0, 0)
    if mode == "polish":
        return (2, 3)
    return (1, 1)

def _is_fast(mode: str) -> bool:
    return (mode or "fast").lower() == "fast"

class Budget:
    def __init__(self, seconds: float):
        self.remaining = seconds
    async def run(self, coro_factory, min_needed: float = 0.5):
        if self.remaining < min_needed:
            raise asyncio.TimeoutError("Budget esaurito")
        t0 = time.perf_counter()
        res = await asyncio.wait_for(coro_factory(), timeout=self.remaining)
        spent = time.perf_counter() - t0
        self.remaining = max(0.0, self.remaining - spent)
        return res

# ============================== Startup ======================================
@app.on_event("startup")
def _startup():
    global client
    if not API_KEY:
        raise RuntimeError("Imposta GOOGLE_API_KEY")
    client = genai.Client(api_key=API_KEY)
    try:
        anyio.from_thread.run(
            lambda: client.models.generate_content(
                model=IMAGE_MODEL_ID,
                contents=[{"role":"user","parts":[{"text":"ping"}]}],
                config=_build_gen_config(),
            )
        )
    except Exception:
        pass

# ============================== Core =========================================
async def _run_inference(user_prompt: str,
                         ref_bytes: Optional[bytes],
                         ref_mime: Optional[str],
                         style: str, size_mm: int, lighting: str, material: str,
                         mode: str = "fast") -> GenerateResponse:
    if not user_prompt and not ref_bytes:
        raise HTTPException(422, "Prompt mancante e nessuna immagine di riferimento")

    retries_full, postproc_passes = _mode_settings(mode)
    budget = Budget(14 if _is_fast(mode) else GENAI_TIMEOUT_S)

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

        if not _is_fast(mode):
            if not _looks_already_normalized(ref_bytes):
                norm = _normalize_to_square_canvas(ref_bytes, size=CANVAS_PX, bg_rgb=_BG_RGB)
                if norm:
                    ref_bytes = norm
                    ref_mime = "image/png"

        # Desatura sempre la reference per evitare trasferimento di cromie
        if DECOLORIZE_REF:
            decol = _desaturate_ref(ref_bytes)
            if decol:
                ref_bytes = decol
                ref_mime = "image/png"

    parts = [PIPELINE_VERSION, IMAGE_MODEL_ID, ("i2i" if ref_bytes else "text"),
             f"mode={mode}", f"retries={retries_full}", f"pp={postproc_passes}",
             "postproc-via-model-lod-nohooks-framingV2",
             style, str(size_mm), lighting, material, final_prompt]
    if ref_bytes:
        parts.append(hashlib.sha256(ref_bytes).hexdigest())
    cache_key = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()

    cached = _cache_get(cache_key)
    if cached is not None:
        return GenerateResponse(images=[_to_data_url(cached)])

    # -------- Tentativo iniziale ---------------------------------------------
    try:
        response = await budget.run(
            lambda: anyio.to_thread.run_sync(
                lambda: client.models.generate_content(
                    model=IMAGE_MODEL_ID,
                    contents=_build_contents(final_prompt, ref_bytes, ref_mime),
                    config=_build_gen_config(),
                ),
                cancellable=True
            ),
            min_needed=2.0
        )
        log.info("gen: %s", _finish_info(response))
    except asyncio.TimeoutError:
        raise HTTPException(504, "Timeout generazione")
    except Exception as e:
        log.exception("Errore modello")
        if "safety" in str(e).lower() or "blocked" in str(e).lower():
            raise HTTPException(422, "Richiesta bloccata dai filtri del modello.")
        raise HTTPException(502, "Servizio non disponibile")

    raw = _extract_first_image_bytes(response)

    # -------- Retry compatti (solo non-fast) ---------------------------------
    did_retry = False
    if not raw and not _is_fast(mode) and retries_full >= 1 and budget.remaining > 3.0:
        did_retry = True
        log.warning("no image → retry #1 (lite)")
        final_prompt_lite = (
            _build_image_repro_prompt_lite(style, size_mm, lighting, material, user_prompt or None)
            if ref_bytes else
            _build_text_only_prompt_lite(user_prompt, style, size_mm, lighting, material)
        )
        try:
            response = await budget.run(
                lambda: anyio.to_thread.run_sync(
                    lambda: client.models.generate_content(
                        model=IMAGE_MODEL_ID,
                        contents=_build_contents(final_prompt_lite, ref_bytes, ref_mime),
                        config=_build_gen_config(),
                    ),
                    cancellable=True
                ),
                min_needed=1.5
            )
            log.info("gen(retry1): %s", _finish_info(response))
            raw = _extract_first_image_bytes(response)
        except Exception:
            log.exception("retry #1 failed")

    if not raw and not _is_fast(mode) and retries_full >= 2 and budget.remaining > 3.0:
        log.warning("no image → retry #2 (ultra-lite)")
        ultra = "Generate exactly ONE 1024x1024 PNG image only. Center the object on light gray background. Do not return text."
        try:
            response = await budget.run(
                lambda: anyio.to_thread.run_sync(
                    lambda: client.models.generate_content(
                        model=IMAGE_MODEL_ID,
                        contents=_build_contents(ultra, ref_bytes, ref_mime),
                        config=_build_gen_config(),
                    ),
                    cancellable=True
                ),
                min_needed=1.5
            )
            log.info("gen(retry2): %s", _finish_info(response))
            raw = _extract_first_image_bytes(response)
        except Exception:
            log.exception("retry #2 failed")

    if not raw:
        raise HTTPException(422, "Nessuna immagine generata")

    # -------- Pulizia deterministica (solo non fast) -------------------------
    if not _is_fast(mode):
        raw = _cleanup_foreign_blobs(raw)

    # -------- Post-process (solo non fast) -----------------------------------
    if not _is_fast(mode) and postproc_passes > 0 and not did_retry and budget.remaining > 3.0:
        current = raw
        for i in range(1, postproc_passes + 1):
            if budget.remaining <= 2.0:
                break
            try:
                pprompt = _postprocess_prompt(size_mm=size_mm, material=material, pass_idx=i)
                response = await budget.run(
                    lambda: anyio.to_thread.run_sync(
                        lambda: client.models.generate_content(
                            model=IMAGE_MODEL_ID,
                            contents=[{
                                "role":"user",
                                "parts":[
                                    {"inline_data":{"mime_type":"image/png","data": current}},
                                    {"text": pprompt},
                                ],
                            }],
                            config=_build_gen_config(),
                        ),
                        cancellable=True
                    ),
                    min_needed=1.5
                )
                new_img = _extract_first_image_bytes(response)
                if new_img:
                    current = _cleanup_foreign_blobs(new_img)
                else:
                    break
            except Exception:
                log.exception("postprocess pass %d failed", i)
                break
        raw = current

    _cache_set(cache_key, raw)
    return GenerateResponse(images=[_to_data_url(raw)])

# ============================== Endpoint =====================================
@app.post("/api/generate", response_model=GenerateResponse)
async def generate_images(req: GenerateRequest):
    style = _norm_style(req.style)
    size_mm = _norm_size(req.size_mm)
    lighting = _norm_lighting(req.lighting)
    material = _norm_material(req.material)
    mode = (req.mode or "fast").lower()

    user_prompt = (req.prompt or "").strip()
    user_prompt = _sanitize_prompt(user_prompt)
    user_prompt = _primary_concept(user_prompt)

    ref_bytes, ref_mime = (None, None)
    if isinstance(req.images, list) and req.images and isinstance(req.images[0], str):
        ref_bytes, ref_mime = _decode_image_data_url(req.images[0])

    _log_prompt = (user_prompt[:300] + "…") if len(user_prompt) > 300 else user_prompt
    log.info(
        "req: has_prompt=%s, images_len=%d, decoded_img=%s, style=%s, size=%s, lighting=%s, material=%s, mode=%s, prompt='%s'",
        bool(_log_prompt), len(req.images or []), bool(ref_bytes), style, size_mm, lighting, material, mode, _log_prompt
    )

    return await _run_inference(user_prompt, ref_bytes, ref_mime, style, size_mm, lighting, material, mode=mode)

# 🔌 Endpoint turbo che forza sempre fast (opzionale)
@app.post("/api/generate_fast", response_model=GenerateResponse)
async def generate_images_fast(req: GenerateRequest):
    req.mode = "fast"
    return await generate_images(req)

@app.get("/cache/stats")
def cache_stats():
    with _CACHE_LOCK:
        return {"items": len(_CACHE), "max_items": _MAX_CACHE_ITEMS}

def _require_admin(token: Optional[str]):
    if ADMIN_TOKEN and token != ADMIN_TOKEN:
        raise HTTPException(403, "Forbidden")

@app.post("/cache/clear")
def cache_clear(authorization: Optional[str] = Header(None)):
    _require_admin(authorization)
    _CACHE.clear()
    return {"ok": True}

@app.get("/health")
def health():
    with _CACHE_LOCK:
        cache_items = len(_CACHE)
    return {"ok": True, "pipeline": PIPELINE_VERSION, "model": IMAGE_MODEL_ID, "cache_items": cache_items}

@app.get("/ping")
def ping():
    return {"ok": True}

@app.get("/")
def index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        return {"message": "Pendant Generator (fast) - ready"}
    return FileResponse(index_path)
