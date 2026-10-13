"""S2/S4: regresión de exportación DXF con primitivas reales (ARC/SPLINE).

image_bytes_to_dxf(method="precision", curves="fit") exporta las primitivas
del pipeline (líneas TLS, arcos, círculos, Béziers como SPLINE) sin
discretizar. Aquí se valida cada tipo de entidad contra la geometría esperada.
"""

import io

import cv2
import ezdxf
import numpy as np
import pytest

from dxf_exporter import image_bytes_to_dxf
from test_precision_s1 import (
    REPO_ROOT,
    dxf_segments,
    encode_png,
    rasterize_segments,
    transform_segments,
)


def _decode_dxf(res):
    return ezdxf.read(io.StringIO(res["dxf_content"].decode("utf-8")))


def _make_arc_image(size=512, radius=140, start=30, end=120, thickness=3):
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    c = size // 2
    cv2.ellipse(img, (c, c), (radius, radius), 0, start, end,
                (0, 0, 0), thickness, cv2.LINE_AA)
    return img


def _make_sine_image(size=512, amp=60, freq=40, thickness=3):
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    pts = [(x, int(round(size / 2 + amp * np.sin(x / freq))))
           for x in range(0, size + 1, 2)]
    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i + 1], (0, 0, 0), thickness, cv2.LINE_AA)
    return img


def _make_circle_image(size=512, radius=140, thickness=3):
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), radius, (0, 0, 0),
               thickness, cv2.LINE_AA)
    return img


def _ink_mask(img_bgr):
    return (cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) < 128).astype(np.uint8)


def _render_polyline_mask(points, shape, thickness=3):
    mask = np.zeros(shape, dtype=np.uint8)
    for i in range(len(points) - 1):
        cv2.line(mask, tuple(np.round(points[i]).astype(int)),
                 tuple(np.round(points[i + 1]).astype(int)), 1, thickness)
    return mask


def test_export_arc_precision():
    img = _make_arc_image()
    res = image_bytes_to_dxf(encode_png(img), method="precision", tile_mm=100.0)
    assert "error" not in res
    counts = res["entities"]
    assert counts["arcs"] == 1
    assert counts["circles"] == 0 and counts["splines"] == 0

    doc = _decode_dxf(res)
    arcs = list(doc.modelspace().query("ARC"))
    assert len(arcs) == 1
    a = arcs[0]
    # 512px -> 1024 work px, ppu = 1024/100 = 10.24 px/mm
    assert a.dxf.radius == pytest.approx(140 * 2 / 10.24, abs=0.6)
    assert a.dxf.center[0] == pytest.approx(50.0, abs=0.6)
    assert a.dxf.center[1] == pytest.approx(50.0, abs=0.6)
    # arco img 30..120 grados (y-down) -> DXF y-up CCW 240..330
    assert a.dxf.start_angle == pytest.approx(240.0, abs=2.5)
    assert a.dxf.end_angle == pytest.approx(330.0, abs=2.5)


def test_export_spline_precision():
    img = _make_sine_image()
    res = image_bytes_to_dxf(encode_png(img), method="precision", tile_mm=100.0)
    assert "error" not in res
    counts = res["entities"]
    assert counts["splines"] >= 1
    assert counts["arcs"] == 0 and counts["circles"] == 0

    doc = _decode_dxf(res)
    splines = list(doc.modelspace().query("SPLINE"))
    assert len(splines) == counts["splines"]
    for s in splines:
        ctrl = np.asarray(s.control_points)
        assert len(ctrl) % 4 == 0
        assert s.dxf.degree == 3

    # fidelidad geométrica: aplanar todos los splines y comparar con la tinta
    ink = _ink_mask(img)
    all_pts = []
    for s in splines:
        path = ezdxf.path.make_path(s)
        pts_mm = np.asarray(list(path.flattening(0.05)))[:, :2]
        pts_img = np.empty_like(pts_mm)
        pts_img[:, 0] = pts_mm[:, 0] * 10.24 / 2.0
        pts_img[:, 1] = 512.0 - pts_mm[:, 1] * 10.24 / 2.0
        all_pts.append(pts_img)
    from metrics import chamfer_px
    mask = np.zeros(ink.shape, dtype=np.uint8)
    for pts in all_pts:
        mask = np.maximum(mask, _render_polyline_mask(pts, ink.shape, thickness=3))
    assert np.logical_and(mask, ink).any()
    assert chamfer_px(mask, ink) < 1.5


def test_export_lines_herribone_precision():
    truth = transform_segments(dxf_segments(REPO_ROOT / "HERRIBONE.dxf"))
    img = rasterize_segments(truth)
    res = image_bytes_to_dxf(encode_png(img), method="precision", tile_mm=100.0)
    assert "error" not in res
    counts = res["entities"]
    assert counts["lines"] >= 4
    assert counts["arcs"] == 0
    assert counts["circles"] == 0
    assert counts["splines"] == 0

    doc = _decode_dxf(res)
    msp = doc.modelspace()
    assert len(list(msp.query("LINE"))) == counts["lines"]
    assert not list(msp.query("ARC")) and not list(msp.query("CIRCLE"))
    assert not list(msp.query("SPLINE"))


def test_export_rdp_fallback_all_lines():
    img = _make_circle_image()
    res = image_bytes_to_dxf(encode_png(img), method="precision",
                             tile_mm=100.0, curves="rdp")
    assert "error" not in res
    # sin primitives_px el exportador cae al camino legado: todo LINE
    assert "entities" not in res
    assert res["segments"] > 10
    doc = _decode_dxf(res)
    msp = doc.modelspace()
    assert len(list(msp.query("LINE"))) == res["segments"]
    assert not list(msp.query("ARC"))
    assert not list(msp.query("CIRCLE"))
    assert not list(msp.query("SPLINE"))


def test_api_precision_params_flow():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    img = _make_circle_image()
    data = encode_png(img)

    # valores inválidos -> 400
    r = client.post(
        "/api/convert/image",
        files={"file": ("c.png", data, "image/png")},
        data={"method": "precision", "curves": "bogus"},
    )
    assert r.status_code == 400
    r = client.post(
        "/api/convert/image",
        files={"file": ("c.png", data, "image/png")},
        data={"method": "precision", "periodic": "sometimes"},
    )
    assert r.status_code == 400

    # precision con defaults -> 200 con entradas
    r = client.post(
        "/api/convert/image",
        files={"file": ("c.png", data, "image/png")},
        data={"method": "precision", "curves": "fit", "snap_tol": "2.0",
              "sagitta_mm": "0.1", "budget": "400", "corner_angle": "135.0",
              "periodic": "auto"},
    )
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["num_entries"] > 0
    assert j["num_entries"] <= 400

    # export DXF por API con params precision
    r = client.post(
        "/api/export/dxf",
        files={"file": ("c.png", data, "image/png")},
        data={"method": "precision", "curves": "fit", "tile_mm": "100.0"},
    )
    assert r.status_code == 200, r.text
    import base64
    dxf = base64.b64decode(r.json()["dxf_base64"]).decode("utf-8")
    doc = ezdxf.read(io.StringIO(dxf))
    assert list(doc.modelspace().query("CIRCLE"))
