# -*- coding: utf-8 -*-
"""Tests S2 del método "precision": esquinas (F4), ajuste de primitivas
recta/arco/círculo/Bézier (F5) y discretización por sagita con budget (F8).

Aceptación S2: Chamfer < 0.5 px en set orgánico y −40 % de entradas PAT
con la misma fidelidad (n_fit ≤ 0.6·n_rdp y chamfer_fit ≤ chamfer_rdp + 0.05).
"""

import io
import math

import cv2
import ezdxf
import numpy as np
import pytest

from core_logic import ImageToPatConverter
from dxf_exporter import image_bytes_to_dxf
from geometry import classify_segment, fit_bezier_schneider, fit_circle_hyper
from metrics import chamfer_px
from vectorize import pipeline
from vectorize.corners import detect_corners
from vectorize.discretize import (
    arc_chord_count,
    arc_points,
    bezier_chords,
    primitives_to_segments,
)
from vectorize.fitting import chain_primitives


# ---------------------------------------------------------------------------
# Helpers geométricos sintéticos
# ---------------------------------------------------------------------------

def _square_points(side=100.0):
    """Cuadrado cerrado muestreado a ~1 px por punto (400 puntos)."""
    corners = [((0, 0), (side, 0)), ((side, 0), (side, side)),
               ((side, side), (0, side)), ((0, side), (0, 0))]
    parts = []
    for a, b in corners:
        n = int(round(math.hypot(b[0] - a[0], b[1] - a[1])))
        t = np.linspace(0.0, 1.0, n, endpoint=False)
        parts.append(np.stack([a[0] + (b[0] - a[0]) * t,
                               a[1] + (b[1] - a[1]) * t], axis=1))
    return np.vstack(parts)


def _circle_points(radius=50.0, n=200, center=(0.0, 0.0)):
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([center[0] + radius * np.cos(ang),
                     center[1] + radius * np.sin(ang)], axis=1)


def _bezier_eval(ctrl, u):
    """Bézier cúbica por Bernstein (independiente de geometry.py)."""
    b0 = (1 - u) ** 3
    b1 = 3 * u * (1 - u) ** 2
    b2 = 3 * u * u * (1 - u)
    b3 = u ** 3
    return (np.outer(b0, ctrl[0]) + np.outer(b1, ctrl[1]) +
            np.outer(b2, ctrl[2]) + np.outer(b3, ctrl[3]))


def _pt_seg_dist(p, a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
    return math.hypot(*(p - (a + t * ab)))


# ---------------------------------------------------------------------------
# Helpers de imagen (integración)
# ---------------------------------------------------------------------------

def _circle_image(size=512, radius=140, thickness=3):
    img = np.full((size, size, 3), 255, np.uint8)
    cv2.circle(img, (size // 2, size // 2), radius, (0, 0, 0),
               thickness, cv2.LINE_AA)
    return img


def _encode_png(bgr):
    ok, buf = cv2.imencode(".png", bgr)
    assert ok
    return buf.tobytes()


def _raster_mask(segments, shape, thickness):
    """Máscara de tinta 0/1 con los segmentos rasterizados."""
    mask = np.zeros(shape, np.uint8)
    t = max(1, int(round(thickness)))
    for x1, y1, x2, y2 in segments:
        cv2.line(mask, (int(round(x1)), int(round(y1))),
                 (int(round(x2)), int(round(y2))), 1, t)
    return mask


# ---------------------------------------------------------------------------
# F5: ajuste geométrico (geometry.py)
# ---------------------------------------------------------------------------

class TestFitCircleHyper:
    def test_short_arc_90deg(self):
        rng = np.random.default_rng(42)
        ang = np.linspace(0.0, np.pi / 2, 80)
        cx, cy, r = 20.0, -10.0, 100.0
        pts = np.stack([cx + r * np.cos(ang), cy + r * np.sin(ang)], axis=1)
        pts = pts + rng.normal(0.0, 0.2, pts.shape)
        res = fit_circle_hyper(pts)
        assert res is not None
        fcx, fcy, fr, rms = res
        assert abs(fr - r) < 1.0
        assert math.hypot(fcx - cx, fcy - cy) < 1.0
        assert rms < 0.35

    def test_full_circle(self):
        rng = np.random.default_rng(7)
        ang = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
        cx, cy, r = 5.0, -3.0, 80.0
        pts = np.stack([cx + r * np.cos(ang), cy + r * np.sin(ang)], axis=1)
        pts = pts + rng.normal(0.0, 0.15, pts.shape)
        res = fit_circle_hyper(pts)
        assert res is not None
        fcx, fcy, fr, rms = res
        assert abs(fr - r) < 0.5
        assert math.hypot(fcx - cx, fcy - cy) < 0.5
        assert rms < 0.35


class TestFitBezier:
    def test_exact_bezier_single_segment(self):
        ctrl = np.array([[0, 0], [10, 40], [30, 40], [40, 0]], float)
        pts = _bezier_eval(ctrl, np.linspace(0.0, 1.0, 50))
        fit = fit_bezier_schneider(pts, tol=0.1)
        assert fit is not None
        assert len(fit["beziers"]) == 1
        assert fit["max_error"] <= 0.1
        recovered = fit["beziers"][0]
        assert np.allclose(recovered, ctrl, atol=0.6)

    def test_sine_reconstructed_within_tol(self):
        x = np.linspace(0.0, 20.0, 80)
        pts = np.stack([x, 20.0 * np.sin(x)], axis=1)
        fit = fit_bezier_schneider(pts, tol=0.25)
        assert fit is not None
        assert fit["max_error"] <= 0.25
        assert 2 <= len(fit["beziers"]) <= 12


class TestClassifySegment:
    def test_line(self):
        x = np.arange(100, dtype=float)
        pts = np.stack([x, 0.3 * x + 5.0], axis=1)
        assert classify_segment(pts)["type"] == "line"

    def test_circle_is_arc(self):
        assert classify_segment(_circle_points(50.0, 200))["type"] == "arc"

    def test_short_arc(self):
        ang = np.linspace(0.0, np.pi / 6, 60)
        pts = np.stack([100.0 * np.cos(ang), 100.0 * np.sin(ang)], axis=1)
        assert classify_segment(pts)["type"] == "arc"

    def test_sine_is_bezier(self):
        x = np.linspace(0.0, 100.0, 100)
        pts = np.stack([x, 20.0 * np.sin(x / 10.0)], axis=1)
        assert classify_segment(pts)["type"] == "bezier"


# ---------------------------------------------------------------------------
# F4: detección de esquinas
# ---------------------------------------------------------------------------

class TestDetectCorners:
    def test_square_closed_four_corners(self):
        corners = detect_corners(_square_points(), 6.0, closed=True)
        assert len(corners) == 4

    def test_circle_closed_none(self):
        assert detect_corners(_circle_points(50.0, 200), 6.0, closed=True) == []

    def test_l_shape_open_one_corner(self):
        l_poly = np.vstack([
            np.stack([np.linspace(0, 100, 101)[:-1], np.zeros(100)], axis=1),
            np.stack([np.full(81, 100.0), np.linspace(0, 80, 81)], axis=1),
        ])
        corners = detect_corners(l_poly, 6.0, closed=False)
        assert len(corners) == 1

    def test_smooth_sine_open_none(self):
        x = np.linspace(0, 120, 121)
        pts = np.stack([x, 8.0 * np.sin(x / 15.0)], axis=1)
        assert detect_corners(pts, 6.0, closed=False) == []


# ---------------------------------------------------------------------------
# F5b: cadenas -> primitivas
# ---------------------------------------------------------------------------

class TestChainPrimitives:
    def test_dense_circle_becomes_circle(self):
        pts = _circle_points(50.0, 300, center=(10.0, -5.0))
        prims = chain_primitives(pts, 6.0, closed=True)
        assert len(prims) == 1
        prim = prims[0]
        assert prim["type"] == "circle"
        assert abs(prim["radius"] - 50.0) < 0.5
        assert math.hypot(prim["center"][0] - 10.0,
                          prim["center"][1] + 5.0) < 0.5

    def test_square_becomes_four_lines(self):
        prims = chain_primitives(_square_points(), 6.0, closed=True)
        assert len(prims) == 4
        assert all(p["type"] == "line" for p in prims)


# ---------------------------------------------------------------------------
# F8: discretización por sagita + budget
# ---------------------------------------------------------------------------

class TestDiscretize:
    def test_arc_chord_count_formula(self):
        assert arc_chord_count(100.0, math.pi / 2, 1.0) == 6

    def test_arc_chords_within_sagitta(self):
        prim = {"type": "arc", "center": (0.0, 0.0), "radius": 100.0,
                "a0": 0.0, "sweep": math.pi / 2, "rms": 0.0}
        pts = arc_points(prim, sagitta=1.0)
        assert pts.shape == (7, 2)
        worst = 0.0
        for i in range(len(pts) - 1):
            mid_angle = prim["sweep"] * (i + 0.5) / (len(pts) - 1)
            arc_mid = np.array([100.0 * math.cos(mid_angle),
                                100.0 * math.sin(mid_angle)])
            chord_mid = (pts[i] + pts[i + 1]) / 2.0
            worst = max(worst, math.hypot(*(arc_mid - chord_mid)))
        assert worst <= 1.0

    def test_bezier_chords_within_sagitta(self):
        ctrl = np.array([[0, 0], [10, 40], [30, 40], [40, 0]], float)
        out = []
        bezier_chords(ctrl, 1.0, out)
        assert len(out) >= 2
        samples = _bezier_eval(ctrl, np.linspace(0.0, 1.0, 1000))
        worst = max(min(_pt_seg_dist(p, s[0], s[1]) for s in out)
                    for p in samples)
        assert worst <= 1.0

    def test_budget_degrades_curves_not_lines(self):
        prims = ([{"type": "circle", "center": (float(i * 50), 0.0),
                   "radius": 100.0, "rms": 0.0} for i in range(5)] +
                 [{"type": "line", "p0": (0.0, 300.0), "p1": (200.0, 300.0),
                   "rms": 0.0},
                  {"type": "line", "p0": (0.0, 320.0), "p1": (200.0, 320.0),
                   "rms": 0.0}])
        segs_free, _, counts_free = primitives_to_segments(prims, 0.5)
        assert counts_free["chords"] == 160  # 5 × 32 cuerdas sin budget
        segs, flags, counts = primitives_to_segments(prims, 0.5, budget=100)
        assert len(segs) <= 100
        assert counts["line"] == 2          # las rectas nunca se degradan
        assert counts["chords"] < counts_free["chords"]
        assert sum(1 for f in flags if f) == counts["chords"]


# ---------------------------------------------------------------------------
# Integración: pipeline end-to-end con círculo orgánico (AA)
# ---------------------------------------------------------------------------

class TestCircleIntegration:
    def test_pipeline_fit_yields_circle_primitive(self):
        res = pipeline.run(_circle_image(), curves="fit", tile_mm=100.0)
        assert res["mode"] == "centerline"
        circles = [p for p in res["primitives"] if p["type"] == "circle"]
        assert len(circles) == 1
        expected = 140.0 * res["scale"]
        assert abs(circles[0]["radius"] - expected) <= 0.01 * expected

    def test_fit_vs_rdp_entries_and_chamfer(self):
        img = _circle_image()
        fit = pipeline.run(img, curves="fit", tile_mm=100.0)
        rdp = pipeline.run(img, curves="rdp")
        n_fit = len(fit["segments_px"])
        n_rdp = len(rdp["segments_px"])
        assert n_fit > 0 and n_rdp > 0
        assert n_fit <= 0.6 * n_rdp  # −40 % entradas (objetivo del plan)

        mask_fit = _raster_mask(fit["segments_px"], fit["mask"].shape,
                                fit["stroke_width"])
        mask_rdp = _raster_mask(rdp["segments_px"], rdp["mask"].shape,
                                rdp["stroke_width"])
        chamfer_fit = chamfer_px(mask_fit, fit["mask"])
        chamfer_rdp = chamfer_px(mask_rdp, rdp["mask"])
        assert chamfer_fit < 0.5                    # fidelidad orgánica
        assert chamfer_fit <= chamfer_rdp + 0.05    # misma fidelidad que RDP

    def test_convert_precision_circle_entries(self):
        png = _encode_png(_circle_image())
        conv = ImageToPatConverter()
        res = conv.convert(png, method="precision", tile_mm=100.0,
                           return_geometry=True)
        assert "primitives_px" in res
        assert 0 < res["num_entries"] <= 400

    def test_dxf_precision_exports_real_circle(self):
        png = _encode_png(_circle_image())
        out = image_bytes_to_dxf(png, method="precision", tile_mm=100.0)
        doc = ezdxf.read(io.StringIO(out["dxf_content"].decode("utf-8")))
        circles = list(doc.modelspace().query("CIRCLE"))
        assert len(circles) == 1
        # r_px ≈ 140·scale (scale=2) → mm: 280 / (side/tile) = 27.34
        assert abs(circles[0].dxf.radius - 27.33) < 0.7
