"""Tests de geometry.py — simplificación adaptativa y ajuste de arcos."""

import math

import numpy as np
import pytest

from geometry import fit_arc, fit_circle_taubin, simplify_polyline_adaptive
from metrics import chord_error


def _circle_points(cx, cy, r, n=360, a0=0.0, a1=2 * math.pi):
    ts = np.linspace(a0, a1, n)
    return np.stack([cx + r * np.cos(ts), cy + r * np.sin(ts)], axis=1)


def test_simplify_preserves_endpoints():
    pts = _circle_points(50.0, 60.0, 30.0, n=200)
    out = simplify_polyline_adaptive(pts, tol=1.0)
    assert out.shape[1] == 2
    assert np.allclose(out[0], pts[0])
    assert np.allclose(out[-1], pts[-1])


def test_simplify_bounds_chord_error():
    # Un círculo denso simplificado con tol: el error de cuerda debe ser <= tol.
    pts = _circle_points(0.0, 0.0, 100.0, n=720)
    tol = 2.0
    out = simplify_polyline_adaptive(pts, tol=tol)
    assert len(out) < len(pts)
    assert chord_error(pts, out) <= tol + 1e-6


def test_simplify_edge_cases():
    pts = np.array([[0.0, 0.0], [10.0, 0.0]])
    out = simplify_polyline_adaptive(pts, tol=5.0)
    assert len(out) == 2
    # tol <= 0 devuelve copia sin simplificar
    many = _circle_points(0.0, 0.0, 10.0, n=50)
    out0 = simplify_polyline_adaptive(many, tol=0.0)
    assert len(out0) == len(many)


def test_fit_circle_recovers_params():
    pts = _circle_points(20.0, -15.0, 42.0, n=360)
    res = fit_circle_taubin(pts)
    assert res is not None
    cx, cy, r, rms = res
    assert abs(cx - 20.0) < 1e-3
    assert abs(cy + 15.0) < 1e-3
    assert abs(r - 42.0) < 1e-3
    assert rms < 1e-3


def test_fit_circle_degenerate_returns_none():
    assert fit_circle_taubin(np.array([[0.0, 0.0], [1.0, 1.0]])) is None


def test_fit_circle_collinear_gives_bad_fit():
    # Kasa siempre ajusta "algo" a puntos colineales, pero con rms grande;
    # ese rms es lo que hace que fit_arc rechace líneas rectas.
    line = np.array([[float(i), 0.0] for i in range(20)])
    res = fit_circle_taubin(line)
    assert res is None or res[3] > 1.0


def test_fit_arc_accepts_good_arc():
    pts = _circle_points(0.0, 0.0, 50.0, n=60, a0=0.0, a1=math.pi / 2)
    res = fit_arc(pts, tol=1.0)
    assert res is not None
    cx, cy = res["center"]
    assert abs(cx) < 1e-2 and abs(cy) < 1e-2
    assert abs(res["radius"] - 50.0) < 1e-2
    assert res["rms"] <= 1.0
    assert set(res.keys()) == {"center", "radius", "start_angle", "end_angle", "rms"}


def test_fit_arc_rejects_straight_line_or_zero_tol():
    line = np.array([[float(i), 0.01 * math.sin(i)] for i in range(50)])
    # una línea casi recta no debe ajustarse con tol pequeña
    assert fit_arc(line, tol=1e-6) is None
    arc = _circle_points(0.0, 0.0, 50.0, n=60, a0=0.0, a1=math.pi / 2)
    assert fit_arc(arc, tol=0.0) is None
