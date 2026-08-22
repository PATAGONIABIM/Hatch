"""HATCH.it API — expone core_logic.py (DXF/Imagen → PAT Revit) vía JSON."""

import base64
import hashlib
import re
import sys
import tempfile
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core_logic import DXFtoPatConverter, ImageToPatConverter, render_pat_preview

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_PAT_BYTES = 2 * 1024 * 1024

EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0000FE00-\U0000FE0F"
    "\U00002B00-\U00002BFF"
    "]+"
)


def clean_text(text):
    """Elimina emojis de los strings generados por core_logic (frontend sin emojis)."""
    if not isinstance(text, str):
        return text
    cleaned = EMOJI_RE.sub("", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def img_to_data_url(img):
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


class LRUCache:
    """Cache en memoria replicando st.cache_data (hash de bytes + params)."""

    def __init__(self, maxsize=32):
        self.maxsize = maxsize
        self.store = OrderedDict()

    def get(self, key):
        if key in self.store:
            self.store.move_to_end(key)
            return self.store[key]
        return None

    def set(self, key, value):
        if key in self.store:
            self.store.move_to_end(key)
        self.store[key] = value
        while len(self.store) > self.maxsize:
            self.store.popitem(last=False)


image_cache = LRUCache()
dxf_cache = LRUCache()

app = FastAPI(title="HATCH.it API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

IMAGE_METHODS = ("hough", "lsd", "fld", "contour")
FILTER_MODES = ("gaussian", "bilateral", "none")


def _validate_image_params(method, filter_mode):
    if method not in IMAGE_METHODS:
        raise HTTPException(400, detail={"error": f"Método inválido: {method}"})
    if filter_mode not in FILTER_MODES:
        raise HTTPException(400, detail={"error": f"Filtro inválido: {filter_mode}"})


@app.get("/api/health")
def health():
    return {"status": "ok", "name": "HATCH.it", "version": "2.0.0"}


def _build_image_result(result, elapsed_ms):
    if "error" in result:
        raise HTTPException(status_code=400, detail={"error": clean_text(result["error"])})
    warnings = [clean_text(w) for w in (result.get("warnings") or [])]
    pat_content = result["pat_content"].replace(
        "HatchCraft precision pattern", "HATCH.it precision pattern"
    )
    return {
        "pat_content": pat_content,
        "preview": img_to_data_url(result["pat_preview"]),
        "debug": img_to_data_url(result["debug_img"]),
        "stats": clean_text(result.get("stats", "")),
        "warnings": warnings,
        "num_entries": result.get("num_entries"),
        "tile_w": result.get("tile_w"),
        "tile_h": result.get("tile_h"),
        "segments": result.get("segments"),
        "period_x_px": result.get("period_x_px"),
        "period_y_px": result.get("period_y_px"),
        "stroke_width_px": result.get("stroke_width_px"),
        "dedup_px": result.get("dedup_px"),
        "elapsed_ms": elapsed_ms,
    }


@app.post("/api/convert/image")
def convert_image(
    file: UploadFile = File(...),
    method: str = Form("hough"),
    canny_low: int = Form(50),
    canny_high: int = Form(150),
    blur_size: int = Form(3),
    param1: float = Form(20),
    param2: float = Form(5),
    use_clahe: bool = Form(False),
    clahe_clip: float = Form(2.0),
    use_adaptive: bool = Form(False),
    adaptive_block: int = Form(11),
    adaptive_c: int = Form(2),
    use_auto_canny: bool = Form(False),
    filter_mode: str = Form("gaussian"),
    use_blackhat: bool = Form(False),
    blackhat_ksize: int = Form(15),
    use_skeleton: bool = Form(False),
    merge_segments: bool = Form(False),
    merge_angle_tol: float = Form(5.0),
    merge_gap_tol: float = Form(10.0),
    dedup_auto: bool = Form(True),
    dedup_k: float = Form(2.5),
    dedup_threshold: float = Form(8.0),
    offset_x: float = Form(0.0),
    offset_y: float = Form(0.0),
    max_resolution: int = Form(400),
    tile_mm: float = Form(100.0),
    min_dash_mm: float = Form(0.3),
    min_gap_mm: float = Form(0.3),
):
    """Convierte una imagen a PAT. Los parámetros replican el contrato de app.py."""
    _validate_image_params(method, filter_mode)

    data = file.file.read()
    if not data:
        raise HTTPException(400, detail={"error": "Archivo vacío"})
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, detail={"error": "El archivo supera 12 MB"})

    params = {
        "method": method, "canny_low": canny_low, "canny_high": canny_high,
        "blur_size": blur_size, "param1": param1, "param2": param2,
        "use_clahe": use_clahe, "clahe_clip": clahe_clip,
        "use_adaptive": use_adaptive, "adaptive_block": adaptive_block,
        "adaptive_c": adaptive_c, "use_auto_canny": use_auto_canny,
        "filter_mode": filter_mode, "use_blackhat": use_blackhat,
        "blackhat_ksize": blackhat_ksize, "use_skeleton": use_skeleton,
        "merge_segments": merge_segments, "merge_angle_tol": merge_angle_tol,
        "merge_gap_tol": merge_gap_tol, "dedup_auto": dedup_auto,
        "dedup_k": dedup_k, "dedup_threshold": dedup_threshold,
        "offset_x": offset_x, "offset_y": offset_y,
        "max_resolution": max_resolution, "tile_mm": tile_mm,
        "min_dash_mm": min_dash_mm, "min_gap_mm": min_gap_mm,
    }
    key = hashlib.md5(data + repr(sorted(params.items())).encode()).hexdigest()

    cached = image_cache.get(key)
    if cached is not None:
        return _build_image_result(cached, 0)

    t0 = time.perf_counter()
    result = ImageToPatConverter().convert(image_bytes=data, **params)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    image_cache.set(key, result)
    return _build_image_result(result, elapsed_ms)


@app.post("/api/convert/dxf")
def convert_dxf(
    file: UploadFile = File(...),
    pattern_name: str = Form("DXF_Pattern"),
):
    """Convierte un DXF a PAT (explosión completa de entidades)."""
    data = file.file.read()
    if not data:
        raise HTTPException(400, detail={"error": "Archivo vacío"})
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, detail={"error": "El archivo supera 12 MB"})

    key = hashlib.md5(data + pattern_name.encode()).hexdigest()
    cached = dxf_cache.get(key)
    if cached is not None:
        return _build_dxf_result(cached, 0)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf", mode="wb") as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        t0 = time.perf_counter()
        result = DXFtoPatConverter().convert(tmp_path, pattern_name=pattern_name)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

    dxf_cache.set(key, result)
    return _build_dxf_result(result, elapsed_ms)


def _build_dxf_result(result, elapsed_ms):
    if "error" in result:
        raise HTTPException(status_code=400, detail={"error": clean_text(result["error"])})
    warnings = [clean_text(w) for w in (result.get("warnings") or [])]
    return {
        "pat_content": result["pat_content"],
        "preview": img_to_data_url(result["pat_preview"]),
        "debug": img_to_data_url(result["debug_img"]),
        "stats": clean_text(result.get("stats", "")),
        "warnings": warnings,
        "num_entries": result.get("num_entries"),
        "tile_w": result.get("tile_w"),
        "tile_h": result.get("tile_h"),
        "segments": result.get("segments"),
        "ignored_entities": result.get("ignored_entities"),
        "elapsed_ms": elapsed_ms,
    }


class PreviewRequest(BaseModel):
    pat_content: str
    tile_count: int = Field(3, ge=1, le=9)
    preview_size: int = Field(600, ge=100, le=1200)
    manual_scale: float = Field(1.0, ge=0.05, le=20.0)


@app.post("/api/preview")
def preview(req: PreviewRequest):
    """Renderiza un PAT con la simulación fiel de Revit (para slider de escala)."""
    if len(req.pat_content.encode()) > MAX_PAT_BYTES:
        raise HTTPException(400, detail={"error": "PAT demasiado grande"})
    try:
        img = render_pat_preview(
            req.pat_content,
            tile_count=req.tile_count,
            preview_size=req.preview_size,
            manual_scale=req.manual_scale,
        )
    except Exception as e:
        raise HTTPException(400, detail={"error": clean_text(str(e))})
    return {"preview": img_to_data_url(img)}


WEB_DIR = ROOT / "web"
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")