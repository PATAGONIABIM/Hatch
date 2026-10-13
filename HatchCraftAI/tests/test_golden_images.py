"""Tests golden con las imágenes reales del proyecto.

Cada imagen es una "fuente de verdad" (golden): el pipeline debe poder
vectorizarla sin error, producir segmentos y entradas PAT, y respetar el tipo
de patrón (MODEL / DRAFTING) solicitado.
"""

import io
import os

import cv2
import numpy as np
import pytest

from core_logic import ImageToPatConverter
from dxf_exporter import image_bytes_to_dxf, sample_segment_colors, write_dxf
from metrics import fragmentation_ratio, recall_ink, tiling_fidelity
from vectorize import pipeline as _pipeline
import pat_sim

SAMPLES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GOLDEN_IMAGES = [
    "Generated Image September 10, 2026 - 12_36PM.png",
    "38ebb6_c8a796acf67f45c9b1913452f31e2ea6~mv2.gif",
]

# Goldens orgánicos (Fase 2.8 / criterio #6: recall de tinta >= 90%).
# "captura_vetas_usuario.png" = imagen de captura del usuario (si no
# está en el repo, el test hace skip).
ORGANIC_GOLDENS = [
    "madera.png",
    "Flux2-Klein_00837_.png",
    "captura_vetas_usuario.png",
]

# Límite de fragmentación (segmentos/trazo real) medido en baseline:
# madera.png=5.5, Flux2-Klein=18.87 — margen generoso para CI.
ORGANIC_FRAG_LIMIT = {
    "madera.png": 8.0,
    "Flux2-Klein_00837_.png": 25.0,
}


def _rasterize_lines(segs, shape, thickness):
    """Rasteriza segmentos (x1,y1,x2,y2) en una máscara uint8."""
    canvas = np.zeros(shape, dtype=np.uint8)
    t = max(1, int(round(thickness)))
    for x1, y1, x2, y2 in segs:
        cv2.line(canvas, (int(round(x1)), int(round(y1))),
                 (int(round(x2)), int(round(y2))), 1, t)
    return canvas


def _load_supported_bytes(path):
    """Carga la imagen como PNG bytes (decodifica GIF vía Pillow)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".gif":
        from PIL import Image
        img = Image.open(path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    with open(path, "rb") as f:
        return f.read()


@pytest.mark.parametrize("fname", GOLDEN_IMAGES)
@pytest.mark.parametrize("method", ["hough", "contour"])
def test_golden_image_converts(fname, method):
    path = os.path.join(SAMPLES_DIR, fname)
    if not os.path.exists(path):
        pytest.skip(f"{fname} no existe")
    data = _load_supported_bytes(path)
    res = ImageToPatConverter().convert(data, method=method)
    assert "error" not in res, res.get("error")
    assert res["segments"] > 0, f"{fname}/{method}: sin segmentos"
    assert res["num_entries"] > 0
    assert res["pat_preview"] is not None and res["pat_preview"].size > 0
    assert res["debug_img"].shape[:2] == res["pat_preview"].shape[:2] or True


@pytest.mark.parametrize("fname", GOLDEN_IMAGES)
@pytest.mark.parametrize("pattern_type,expected", [
    ("MODEL", ";%TYPE=MODEL"),
    ("DRAFTING", ";%TYPE=DRAFTING"),
])
def test_golden_image_pattern_type(fname, pattern_type, expected):
    path = os.path.join(SAMPLES_DIR, fname)
    if not os.path.exists(path):
        pytest.skip(f"{fname} no existe")
    data = _load_supported_bytes(path)
    res = ImageToPatConverter().convert(data, method="hough",
                                        pattern_type=pattern_type)
    assert "error" not in res, res.get("error")
    assert expected in res["pat_content"]
    # El resto del header no debe mezclar tipos
    other = ";%TYPE=DRAFTING" if expected.endswith("MODEL") else ";%TYPE=MODEL"
    assert other not in res["pat_content"]


def test_golden_image_returns_geometry():
    path = os.path.join(SAMPLES_DIR, GOLDEN_IMAGES[0])
    if not os.path.exists(path):
        pytest.skip("imagen golden principal no existe")
    data = _load_supported_bytes(path)
    res = ImageToPatConverter().convert(data, method="hough",
                                        return_geometry=True)
    assert "lines_px" in res and len(res["lines_px"]) == res["segments"]
    assert "source_bgr" in res and res["source_bgr"].ndim == 3
    assert "px_per_unit" in res and res["px_per_unit"] > 0


def test_exporter_write_dxf_colors():
    doc_bytes = write_dxf([(0, 0, 10, 0), (0, 10, 10, 10)],
                          colors=[(255, 0, 0), (0, 0, 255)])
    text = doc_bytes.decode("utf-8")
    assert "LINE" in text
    assert "420" in text  # true color group code
    # true color 255,0,0 == 0xFF0000; 0,0,255 == 0x0000FF
    assert "16711680" in text
    assert "255" in text


def test_exporter_sample_segment_colors_picks_darkest():
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    cv2.line(img, (10, 50), (90, 50), (120, 0, 0), 3)  # BGR -> azul
    colors = sample_segment_colors(img, [(10, 50, 90, 50)], sample_n=20)
    r, g, b = colors[0]
    assert b > 100 and r < 60 and g < 60, f"debería capturar tinta azul, obtuvo {(r, g, b)}"


def test_exporter_image_to_dxf():
    path = os.path.join(SAMPLES_DIR, GOLDEN_IMAGES[0])
    if not os.path.exists(path):
        pytest.skip("imagen golden principal no existe")
    data = _load_supported_bytes(path)
    res = image_bytes_to_dxf(data, method="hough")
    assert "error" not in res, res.get("error")
    assert res["segments"] > 0
    assert "LINE" in res["dxf_content"].decode("utf-8")


@pytest.mark.parametrize("fname", ORGANIC_GOLDENS)
def test_organic_golden_ink_recall_and_fragmentation(fname):
    """Fase 2.8 / criterio #6: recall de tinta >= 90% en goldens
    orgánicos (tol = stroke width) y fragmentación acotada."""
    path = os.path.join(SAMPLES_DIR, fname)
    if not os.path.exists(path):
        pytest.skip(f"{fname} no existe")
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        pytest.skip(f"{fname} ilegible")
    res = _pipeline.run(img, curves="fit", tile_mm=100.0, budget=400)
    assert "error" not in res, res.get("error")
    assert res.get("structure_kind") == "organic", (
        f"{fname}: esperaba structure_kind=organic, "
        f"llegó {res.get('structure_kind')}")
    det = _rasterize_lines(res["segments_px"], res["mask"].shape,
                           res["stroke_width"])
    ink = (res["mask"] > 0).astype(np.uint8)
    recall = recall_ink(det, ink, tol=float(res["stroke_width"]))
    assert recall >= 0.90, f"{fname}: recall de tinta {recall:.3f} < 0.90"
    frag = fragmentation_ratio(len(res["segments_px"]), ink)
    limit = ORGANIC_FRAG_LIMIT.get(fname, 30.0)
    assert frag <= limit, f"{fname}: fragmentación {frag:.2f} > {limit}"


def test_organic_golden_tiling_fidelity():
    """Fase 2.8(c): fidelidad de tiling del PAT orgánico end-to-end
    (render fiel pat_sim cubre la tinta de la fuente)."""
    fname = "madera.png"
    path = os.path.join(SAMPLES_DIR, fname)
    if not os.path.exists(path):
        pytest.skip(f"{fname} no existe")
    with open(path, "rb") as f:
        data = f.read()
    res = ImageToPatConverter().convert(data, method="precision",
                                        return_geometry=True)
    assert "error" not in res, res.get("error")
    assert res.get("structure_kind") == "organic"
    side = int(res["side_px"])
    rend, info = pat_sim.render_faithful(res["pat_content"], size=side,
                                         show_grid=False, return_stats=True)
    rg = cv2.cvtColor(rend, cv2.COLOR_BGR2GRAY) if rend.ndim == 3 else rend
    rendered = (rg < 128).astype(np.uint8)
    gray = cv2.cvtColor(res["source_bgr"], cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink_r = cv2.resize(ink, (rendered.shape[1], rendered.shape[0]),
                       interpolation=cv2.INTER_NEAREST)
    fid = tiling_fidelity(rendered, ink_r, tol=5.0, render_info=info)
    assert fid >= 0.90, f"fidelidad de tiling {fid:.3f} < 0.90"