import os
import base64
from io import BytesIO
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from google import genai

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("Imposta la variabile d'ambiente GOOGLE_API_KEY")

MODEL_ID = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")

client = genai.Client(api_key=API_KEY)

app = FastAPI(title="NOVE25 Pendant Generator", root_path="/aipendant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    prompt: str
    style: Optional[str] = "3d"
    size_mm: Optional[int] = 30
    images: Optional[List[str]] = None
    format: Optional[str] = "png"

class GenerateResponse(BaseModel):
    images: List[str]

def _to_data_url(raw_bytes: bytes, mime: str = "image/png") -> str:
    import base64
    b64 = base64.b64encode(raw_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def _wants_text(user_prompt: str) -> bool:
    up = (user_prompt or "").lower()
    return any(k in up for k in ["scritta", "testo", "text", "wording", "lettering"])

def _style_clause(style: str) -> str:
    s = (style or "3d").lower()
    if s == "basrelief":
        return "Design it as a BAS-RELIEF (bassorilievo): shallow relief, minimal depth, FRONT-FACING (frontal view). "
    return "Design it as a FULL 3D pendant: volumetric, realistic occlusions/reflections, shown almost frontal with a very slight three-quarter tilt (subtle) only to suggest depth. "

def _size_clause(size_mm: int) -> str:
    size = size_mm if size_mm in (20, 30, 40) else 30
    ratio = round(6/size, 2)
    return (
        f"The PENDANT BODY HEIGHT (excluding the bail) must be ~{size} mm. "
        "Keep the bail (\"maglina\") ABSOLUTELY CONSTANT across images: external height ~6 mm, internal opening ~4 mm, same exact design and scale as previous images. "
        f"Ensure the visual proportion reflects this: at {size} mm body height the 6 mm bail should appear about {int(100*6/size)}% of the body's height (approx. ratio {ratio}:1). "
        + "Do NOT scale the bail to match the body; only scale the pendant body to reach the requested size. "
    )

def _build_pendant_prompt(user_prompt: str, style: str, size_mm: int) -> str:
    BASE = (
        "You are a jewelry designer. Create a photorealistic PRODUCT PHOTO of a "
        'NECKLACE PENDANT made entirely of polished darkened silver ("argento brunito lucido"). '
        + _style_clause(style) +
        _size_clause(size_mm) +
        'Include a bail ("maglina") at the top that is ALWAYS the same design across images: '
        "an oval polished metal bail, external height ~6mm, internal opening ~4mm tall, seamlessly connected to the pendant. "
        "Do NOT include counter-bail or extra ring. "
        "Background: solid black. Lighting: consistent studio-style. "
        "Use ONLY silver (argento brunito lucido). NEVER use gold, rose gold, bronze, copper, or other materials/colors. "
    )
    if not _wants_text(user_prompt):
        BASE += "Do NOT depict letters. Interpret words/names abstractly as motifs. "
    BASE += "User design motif: "
    return BASE + user_prompt

def _data_url_to_inline(durl: str):
    if not durl or not durl.startswith("data:"):
        return None
    header, b64 = durl.split(",", 1)
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

@app.post("/api/generate", response_model=GenerateResponse)
def generate_images(req: GenerateRequest):
    prompt = (req.prompt or "").strip()
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

    final_prompt = _build_pendant_prompt(prompt, style, size_val)

    contents = [final_prompt]
    if req.images:
        for durl in req.images[:5]:
            inline = _data_url_to_inline(durl)
            if inline:
                contents.append({"inline_data": inline})

    try:
        response = client.models.generate_content(model=MODEL_ID, contents=contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore chiamata API: {e}")

    results: List[str] = []
    if not getattr(response, "candidates", None):
        raise HTTPException(status_code=500, detail="Risposta vuota.")

    for cand in response.candidates:
        content = getattr(cand, "content", None)
        if not content or not getattr(content, "parts", None):
            continue
        for part in content.parts:
            inline_data = getattr(part, "inline_data", None)
            if inline_data and getattr(inline_data, "data", None):
                data_field = inline_data.data
                raw_bytes = data_field if isinstance(data_field, bytes) else base64.b64decode(data_field)
                results.append(_to_data_url(raw_bytes, "image/png"))

    if not results:
        raise HTTPException(status_code=422, detail="Nessuna immagine generata.")
    return GenerateResponse(images=results)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
