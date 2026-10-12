"""S1 — Aceptación del método "precision" (paquete vectorize).

Criterios del plan (plan_vectorizacion_precision.md, Sprint 1):

- Error angular de rectas < 0.3° (media) sobre el set de regresión
  rasterizado desde los DXF dorados.
- Cero líneas dobles.
- Cobertura ≥ 90% del largo de cada recta verdadera larga.

El set de regresión se genera al vuelo (determinista, sin binarios en el
repo): se explota el DXF a segmentos, se rasteriza con supersampling ×4 +
INTER_AREA (anti-aliasing real, como un escaneo) y los segmentos
transformados a px son la verdad vectorial.
"""

import math
from pathlib import Path

import cv2
import ezdxf
import numpy as np
import pytest

from core_logic import ImageToPatConverter
from geometry import fit_line_tls, max_perp_deviation
from metrics import chamfer_px, evaluate_lines
from vectorize import pipeline as precision_pipeline
from vectorize.centerline import merge_collinear_segments

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DXF = [
    "HERRIBONE.dxf",
    "HILADA.dxf",
    "TABLEADO HORIZONTAL.dxf",
    "ADOCRETO_MODEL.dxf",
]

CANVAS = 512
MARGIN = 16
STROKE_PX = 3.0
SUPERSAMPLE = 4


# --------------------------------------------------------------------------
# Utilidades del set de regresión (raster + verdad vectorial)
# --------------------------------------------------------------------------

def seg_len(s):
    return float(math.hypot(s[2] - s[0], s[3] - s[1]))


def dxf_segments(path):
    """Explota un DXF a segmentos 2D en unidades originales del dibujo."""
    doc = ezdxf.readfile(str(path))
    segs = []

    def handle(e):
        kind = e.dxftype()
        if kind == "INSERT":
            for sub in e.virtual_entities():
                handle(sub)
        elif kind == "LINE":
            segs.append((e.dxf.start.x, e.dxf.start.y,
                         e.dxf.end.x, e.dxf.end.y))
        elif kind in ("LWPOLYLINE", "POLYLINE"):
            if kind == "LWPOLYLINE":
                pts = [(p[0], p[1]) for p in e.get_points("xy")]
            else:
                pts = [(v.dxf.location.x, v.dxf.location.y)
                       for v in e.vertices]
            if e.closed and pts and pts[0] != pts[-1]:
                pts = pts + [pts[0]]
            for a, b in zip(pts, pts[1:]):
                if a != b:
                    segs.append((a[0], a[1], b[0], b[1]))
        else:
            try:
                from ezdxf import path as ezpath
                verts = [tuple(v[:2])
                         for v in ezpath.make_path(e).flattening(0.05)]
                for a, b in zip(verts, verts[1:]):
                    if a != b:
                        segs.append((a[0], a[1], b[0], b[1]))
            except Exception:
                pass  # entidades no geométricas (TEXT, HATCH…) se omiten

    for e in doc.modelspace():
        handle(e)
    return segs


def transform_segments(segs, canvas=CANVAS, margin=MARGIN):
    """Escala los segmentos al lienzo px (mismo marco que el raster)."""
    xs = [v for s in segs for v in (s[0], s[2])]
    ys = [v for s in segs for v in (s[1], s[3])]
    min_x, min_y = min(xs), min(ys)
    extent = max(max(xs) - min_x, max(ys) - min_y, 1e-9)
    scale = (canvas - 2 * margin) / extent
    return [(margin + (x1 - min_x) * scale, margin + (y1 - min_y) * scale,
             margin + (x2 - min_x) * scale, margin + (y2 - min_y) * scale)
            for x1, y1, x2, y2 in segs]


def rasterize_segments(segs_px, canvas=CANVAS, stroke_px=STROKE_PX,
                       ss=SUPERSAMPLE):
    """Raster anti-aliased: dibuja a ×4 y baja con INTER_AREA."""
    big = np.full((canvas * ss, canvas * ss), 255, dtype=np.uint8)
    thickness = max(1, int(round(stroke_px * ss)))
    for x1, y1, x2, y2 in segs_px:
        cv2.line(big,
                 (int(round(x1 * ss)), int(round(y1 * ss))),
                 (int(round(x2 * ss)), int(round(y2 * ss))),
                 0, thickness)
    small = cv2.resize(big, (canvas, canvas),
                       interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)


def encode_png(img):
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


# --------------------------------------------------------------------------
# Regresión: aceptación S1 sobre los DXF dorados rasterizados
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dxf_name", GOLDEN_DXF)
def test_regression_precision_on_golden_dxf(dxf_name):
    truth = transform_segments(dxf_segments(REPO_ROOT / dxf_name))
    assert truth, "el DXF debe contener geometría"
    img = rasterize_segments(truth)

    res = ImageToPatConverter().convert(
        encode_png(img), method="precision", tile_mm=100.0,
        return_geometry=True)
    assert "error" not in res, res.get("error")

    k = CANVAS / res["side_px"]
    detected = [(x1 * k, y1 * k, x2 * k, y2 * k)
                for x1, y1, x2, y2 in res["lines_px"]]
    stroke = res["stroke_width_px"] * k

    min_len = max(15.0, 10.0 * stroke)
    long_truth = [s for s in truth if seg_len(s) >= min_len]
    assert long_truth, "debe haber verdades largas que evaluar"

    ev = evaluate_lines(long_truth, detected, stroke_px=stroke)
    assert ev["n_matched"] == ev["n_truth"], (
        f"verdades sin emparejar: {ev['unmatched']}")
    assert ev["double_lines"] == 0, (
        f"líneas dobles: {ev['doubled_truth']}")
    assert ev["mean_angular_error"] < 0.3, ev
    assert ev["max_angular_error"] < 0.75, ev
    assert ev["mean_coverage"] >= 0.9, ev


# --------------------------------------------------------------------------
# Tests unitarios: geometría, fusión colineal, métricas
# --------------------------------------------------------------------------

def test_fit_line_tls_perfect_line():
    pts = [(0.0, 0.0), (10.0, 5.0), (20.0, 10.0)]
    fit = fit_line_tls(pts)
    assert fit is not None
    expected = np.array([2.0, 1.0]) / math.sqrt(5.0)
    assert np.allclose(fit["dir"], expected, atol=1e-9)
    assert fit["rms"] < 1e-9


def test_fit_line_tls_noisy_line():
    pts = [(float(x), 0.5 * x + 0.3 * math.sin(x)) for x in range(0, 101, 5)]
    fit = fit_line_tls(pts)
    assert fit is not None
    expected = np.array([2.0, 1.0]) / math.sqrt(5.0)
    angle = math.degrees(math.acos(
        min(1.0, abs(float(np.dot(fit["dir"], expected))))))
    assert angle < 0.5
    assert fit["rms"] < 0.5


def test_fit_line_tls_degenerate():
    assert fit_line_tls([(1.0, 1.0)]) is None
    assert fit_line_tls([(1.0, 1.0), (1.0, 1.0)]) is None


def test_max_perp_deviation():
    fit = fit_line_tls([(0.0, 0.0), (10.0, 5.0)])
    dev = max_perp_deviation([(0.0, 0.0), (10.0, 5.0), (15.0, 5.0)], fit)
    # distancia de (15,5) a la recta y = x/2: |15·1 − 5·2| / √5
    assert dev == pytest.approx(5.0 / math.sqrt(5.0), abs=1e-6)


def test_merge_collinear_segments_gap():
    segs = [(0.0, 0.0, 50.0, 0.0), (52.0, 0.0, 100.0, 0.0)]
    merged = merge_collinear_segments(segs, sw=4.0)
    assert len(merged) == 1
    assert seg_len(merged[0]) > 99.0


def test_merge_collinear_segments_angle_wrap():
    # 0.4° y 179.78° son la misma orientación (módulo 180): deben fusionarse.
    segs = [(0.0, 0.0, 50.0, 0.35), (52.5, 0.36, 100.0, 0.18)]
    merged = merge_collinear_segments(segs, sw=4.0)
    assert len(merged) == 1


def test_merge_collinear_segments_parallel_offset_kept():
    segs = [(0.0, 0.0, 50.0, 0.0), (52.0, 6.0, 100.0, 6.0)]
    merged = merge_collinear_segments(segs, sw=4.0)
    assert len(merged) == 2


def test_evaluate_lines_match():
    truth = [(10.0, 10.0, 110.0, 10.0)]
    det = [(9.0, 10.2, 111.0, 10.1)]
    ev = evaluate_lines(truth, det, stroke_px=2.0)
    assert ev["n_truth"] == 1
    assert ev["n_matched"] == 1
    assert ev["double_lines"] == 0
    assert ev["mean_angular_error"] < 0.2
    assert ev["mean_coverage"] >= 0.99


def test_evaluate_lines_detects_double():
    # Dos portadores en rectas físicas DISTINTAS (offsets separados más
    # allá del umbral de agrupación) que ambos emparejan la misma verdad.
    truth = [(10.0, 10.0, 110.0, 10.0)]
    det = [(9.0, 10.2, 111.0, 10.1), (11.0, 12.0, 109.0, 12.0)]
    ev = evaluate_lines(truth, det, stroke_px=2.0)
    assert ev["n_matched"] == 1
    assert ev["double_lines"] == 1


def test_evaluate_lines_unmatched():
    truth = [(10.0, 10.0, 110.0, 10.0), (0.0, 0.0, 0.0, 120.0)]
    det = [(9.0, 10.2, 111.0, 10.1)]
    ev = evaluate_lines(truth, det, stroke_px=2.0)
    assert ev["n_matched"] == 1
    assert len(ev["unmatched"]) == 1


def test_chamfer_px():
    # Chamfer simétrico medio: crece con el desplazamiento y es 0 si igual.
    a = np.zeros((40, 40), np.uint8)
    a[10:20, 10:20] = 255
    assert chamfer_px(a, a) == 0.0
    b5 = np.zeros((40, 40), np.uint8)
    b5[10:20, 15:25] = 255
    c5 = chamfer_px(a, b5)
    b10 = np.zeros((40, 40), np.uint8)
    b10[10:20, 20:30] = 255
    c10 = chamfer_px(a, b10)
    assert c5 > 0.3
    assert c10 > c5
    assert math.isinf(chamfer_px(a, np.zeros((40, 40), np.uint8)))


# --------------------------------------------------------------------------
# Integración: convert() y pipeline
# --------------------------------------------------------------------------

def test_convert_precision_pat_content():
    img = rasterize_segments(
        [(16.0, 16.0, 496.0, 16.0), (256.0, 16.0, 256.0, 496.0)])
    res = ImageToPatConverter().convert(encode_png(img), method="precision")
    assert "error" not in res, res.get("error")
    assert "*Image_Pattern" in res["pat_content"]
    assert "lines_px" not in res
    assert res["num_entries"] > 0


def test_convert_precision_blank_image_errors_cleanly():
    img = np.full((256, 256, 3), 255, dtype=np.uint8)
    res = ImageToPatConverter().convert(encode_png(img), method="precision")
    assert "error" in res


def test_region_mode_forced():
    img = np.full((300, 300, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (200, 200), (0, 0, 0), -1)
    res = precision_pipeline.run(img, force_mode="region")
    assert "error" not in res, res.get("error")
    assert res["mode"] == "region"
    assert len(res["segments_px"]) > 0


def test_api_exposes_precision_method():
    pytest.importorskip("fastapi")
    from api.main import IMAGE_METHODS
    assert "precision" in IMAGE_METHODS
