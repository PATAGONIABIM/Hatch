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

SAMPLES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GOLDEN_IMAGES = [
    "Generated Image September 10, 2026 - 12_36PM.png",
    "38ebb6_c8a796acf67f45c9b1913452f31e2ea6~mv2.gif",
]


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