"""Tests del API HATCH.it (FastAPI). Requiere: fastapi, uvicorn, python-multipart, httpx."""

import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app, clean_text

client = TestClient(app)


def make_synthetic_image(size=300, period=60):
    """Imagen sintética: grilla de líneas oscuras sobre fondo blanco."""
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    for x in range(0, size, period):
        cv2.line(img, (x, 0), (x, size), (20, 20, 20), 2)
    for y in range(0, size, period):
        cv2.line(img, (0, y), (size, y), (20, 20, 20), 2)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def make_synthetic_dxf():
    """DXF mínimo: un rectángulo de líneas."""
    import ezdxf

    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    msp.add_line((0, 0), (100, 0))
    msp.add_line((100, 0), (100, 100))
    msp.add_line((100, 100), (0, 100))
    msp.add_line((0, 100), (0, 0))
    bio = io.StringIO()
    doc.write(bio)
    return bio.getvalue().encode("utf-8")


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["name"] == "HATCH.it"


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "HATCH" in r.text


def test_convert_image_hough():
    data = make_synthetic_image()
    r = client.post(
        "/api/convert/image",
        files={"file": ("grid.png", data, "image/png")},
        data={"method": "hough", "merge_segments": "true"},
    )
    assert r.status_code == 200, r.text
    j = r.json()
    assert "*PAT" in j["pat_content"] or "PAT" in j["pat_content"].upper()
    assert j["preview"].startswith("data:image/png;base64,")
    assert j["debug"].startswith("data:image/png;base64,")
    assert j["num_entries"] > 0
    assert j["segments"] > 0
    assert "⚠️" not in j["stats"]
    assert "✅" not in j["stats"]
    assert "HatchCraft precision pattern" not in j["pat_content"]


def test_convert_image_contour():
    data = make_synthetic_image()
    r = client.post(
        "/api/convert/image",
        files={"file": ("grid.png", data, "image/png")},
        data={"method": "contour", "param1": "20", "param2": "0.005"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["num_entries"] > 0


def test_convert_image_invalid_method():
    data = make_synthetic_image()
    r = client.post(
        "/api/convert/image",
        files={"file": ("grid.png", data, "image/png")},
        data={"method": "bogus"},
    )
    assert r.status_code == 400
    assert "error" in r.json()["detail"]


def test_convert_image_empty_file():
    r = client.post("/api/convert/image", files={"file": ("x.png", b"", "image/png")})
    assert r.status_code == 400


def test_convert_dxf():
    data = make_synthetic_dxf()
    r = client.post(
        "/api/convert/dxf",
        files={"file": ("rect.dxf", data, "application/dxf")},
        data={"pattern_name": "TestRect"},
    )
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["pat_content"]
    assert j["num_entries"] > 0
    assert j["tile_w"] == 100.0
    assert j["preview"].startswith("data:image/png;base64,")


def test_convert_dxf_empty_file():
    r = client.post("/api/convert/dxf", files={"file": ("x.dxf", b"", "application/dxf")})
    assert r.status_code == 400


def test_preview_endpoint():
    pat = ";TEST\n*T,test\n0, 0,0, 0,10, 10,-10\n"
    r = client.post("/api/preview", json={"pat_content": pat, "tile_count": 3, "preview_size": 300, "manual_scale": 1.0})
    assert r.status_code == 200, r.text
    assert r.json()["preview"].startswith("data:image/png;base64,")


def test_preview_invalid_pat():
    r = client.post("/api/preview", json={"pat_content": "no es un pat"})
    assert r.status_code in (200, 400)


def test_clean_text_strips_emojis():
    assert clean_text("a ✅ b ⚠️ c") == "a b c"
    assert clean_text("sin emojis") == "sin emojis"