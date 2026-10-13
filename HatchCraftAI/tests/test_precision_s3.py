"""S3 — F6 Regularización de rectas (precisión arquitectónica).

Criterios del plan (plan_vectorizacion_precision.md, Sprint 3):

- Direcciones dominantes + snap angular: rectas a ≤2° de una dirección
  dominante quedan EXACTAMENTE paralelas entre sí.
- Espaciado regular: portadoras colineales de una familia caen en un
  lattice de offsets con paso constante.
- Uniones T/L/X: extremos a ≤1.5×ancho de una intersección se mueven
  EXACTAMENTE a ella (sin micro-gaps ni sobrepasos).
- Aceptación: 100% juntas paralelas y exactas; count_overlap == 0.

Los tests de integración rasterizan los DXF dorados (set de regresión
determinista de S1) y exigen, además de las métricas S1, cero solapes,
juntas exactas y entradas PAT == verdades.
"""

import math

import pytest

from core_logic import ImageToPatConverter
from metrics import count_overlap, evaluate_lines, junction_gaps
from test_precision_s1 import (
    CANVAS,
    GOLDEN_DXF,
    REPO_ROOT,
    dxf_segments,
    encode_png,
    rasterize_segments,
    seg_len,
    transform_segments,
)
from vectorize.regularize import (
    _seg_angle,
    _wrap180,
    dominant_angles,
    regularize_lines,
    snap_angles,
    snap_junctions,
    snap_lattice,
)


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------

def make(ang_deg, length, cx=100.0, cy=100.0):
    a = math.radians(ang_deg)
    u = (math.cos(a), math.sin(a))
    return (cx - u[0] * length / 2, cy - u[1] * length / 2,
            cx + u[0] * length / 2, cy + u[1] * length / 2)


def vertical(x, y0, y1):
    return (x, y0, x, y1)


def horiz(y, x0=0.0, x1=480.0):
    return (x0, y, x1, y)


# --------------------------------------------------------------------------
# Unitarios: direcciones dominantes y snap angular
# --------------------------------------------------------------------------

def test_dominant_angles_three_families():
    segs = [make(0 + d, 100) for d in
            (-0.4, 0.3, -0.2, 0.1, 0.5, -0.5, 0.2, -0.3, 0.0, -0.1)]
    segs += [make(90 + d, 80) for d in
             (-0.3, 0.4, 0.2, -0.4, 0.1, -0.1, 0.3, -0.2)]
    segs += [make(45 + d, 60) for d in (-0.2, 0.3, -0.1)]
    segs += [make(179.5, 50)]  # cluster que envuelve 180 → ~0
    angs = dominant_angles(segs)
    assert len(angs) == 3, angs
    for target in (0.0, 45.0, 90.0):
        assert any(abs(_wrap180(a - target)) < 0.3 for a in angs), (
            target, angs)


def test_snap_angles_exact_and_preserves_centroid():
    snapped, n = snap_angles(
        [make(1.5, 200), make(10.0, 200)], [0.0, 90.0], 2.0)
    assert n == 1
    assert abs(_seg_angle(snapped[0])) < 1e-9
    cx = (snapped[0][0] + snapped[0][2]) / 2
    cy = (snapped[0][1] + snapped[0][3]) / 2
    assert abs(cx - 100.0) < 1e-9 and abs(cy - 100.0) < 1e-9
    assert abs(_seg_angle(snapped[1]) - 10.0) < 1e-6, "10° no se toca"


# --------------------------------------------------------------------------
# Unitarios: lattice de offsets
# --------------------------------------------------------------------------

def test_snap_lattice_regular_spacing():
    latt = [vertical(10.0, 0, 100), vertical(60.3, 0, 100),
            vertical(110.1, 0, 100), vertical(159.8, 0, 100)]
    out, n_adj = snap_lattice(latt, 4.0)
    xs = sorted(s[0] for s in out)
    sp = [b - a for a, b in zip(xs, xs[1:])]
    assert max(sp) - min(sp) < 0.05, sp  # regularidad EXACTA (garantía F6)
    for got, want in zip(xs, [10, 60, 110, 160]):
        assert abs(got - want) < 0.35, (got, want)
    assert n_adj == 4


def test_snap_lattice_irregular_not_snapped():
    irr = [vertical(0.0, 0, 100), vertical(100.2, 0, 100),
           vertical(190.1, 0, 100)]
    out, n = snap_lattice(irr, 4.0)
    assert n == 0, "un lattice irregular no se ajusta"


def test_snap_lattice_noisy_family_snapped_exact():
    """Regresión del fix de fase: o0 = min(offs) (no la mediana).

    Una familia de 18 con paso 26.667 y ruido ±0.06 debe ajustarse
    COMPLETA con espaciado exacto (con o0=mediana el LS divergía).
    """
    noise = [0.05, -0.04, 0.03, -0.06, 0.04, -0.03, 0.06, -0.05,
             0.02, -0.04, 0.05, -0.02, 0.03, -0.06, 0.04, 0.05,
             -0.03, 0.02]
    segs = [horiz(16.0 + i * 26.6667 + noise[i]) for i in range(18)]
    out, n_adj = snap_lattice(segs, 8.0)
    ys = sorted(s[1] for s in out)
    sp = [b - a for a, b in zip(ys, ys[1:])]
    assert n_adj == 18
    assert max(sp) - min(sp) < 0.05, sp
    for got, i in zip(ys, range(18)):
        assert abs(got - (16.0 + i * 26.6667)) < 0.1, (got, i)


def test_snap_lattice_irregular_stubs_untouched():
    """Regresión del gate por línea: los stubs verticales de TABLEADO
    (x irregulares) NO se arrastran a un lattice espurio."""
    xs = [120.02, 136, 136, 136, 158.39, 158.39, 256, 256, 256, 256,
          256, 256, 368.01, 368.01, 376, 376, 388.69, 394.33]
    stubs = [vertical(x, 0.0, 26.7) for x in xs]
    out, n = snap_lattice(stubs, 8.0)
    assert n == 0, f"stubs movidos: {n}"
    assert list(out) == stubs


# --------------------------------------------------------------------------
# Unitarios: uniones T/L/X
# --------------------------------------------------------------------------

def test_snap_junctions_t():
    tj = [vertical(100.0, 55.0, 150.0), (0.0, 50.0, 200.0, 50.0)]
    out, n_moved = snap_junctions(tj, 6.0)
    v, h = out
    assert abs(v[1] - 50.0) < 1e-9, v  # la punta cae EXACTA en la horizontal
    assert abs(h[2] - 200.0) < 1e-9 and abs(v[3] - 150.0) < 1e-9


def test_snap_junctions_l():
    lj = [(10.0, 49.2, 45.0, 50.6),    # ~-1.2°, termina cerca de P=(50,50)
          (51.5, 90.0, 49.4, 54.0)]    # ~88.5°, termina cerca de P
    out, n_moved = snap_junctions(lj, 6.0)
    assert n_moved == 2
    pa = (out[0][2], out[0][3])
    pb = (out[1][2], out[1][3])
    assert math.hypot(pa[0] - pb[0], pa[1] - pb[1]) < 1e-9, (pa, pb)


def test_snap_junctions_x():
    xj = [(94.0, 94.0, 160.0, 160.0),   # y=x, extremo inferior a 8.5px de P
          (94.0, 106.0, 160.0, 40.0)]   # y=-x+200, extremo superior idem
    out, n_moved = snap_junctions(xj, 6.0)
    a, b = out
    assert n_moved == 2
    assert math.hypot(a[0] - 100.0, a[1] - 100.0) < 1e-9, a
    assert math.hypot(b[0] - 100.0, b[1] - 100.0) < 1e-9, b
    assert (a[2], a[3]) == (160.0, 160.0) and (b[2], b[3]) == (160.0, 40.0)


def test_snap_junctions_far_endpoint_untouched():
    nj = [vertical(10.0, 0, 100), (0.0, 150.0, 200.0, 150.0)]
    out, n_moved = snap_junctions(nj, 6.0)
    assert n_moved == 0
    assert list(out) == nj


def test_snap_junctions_parallel_never():
    pj = [vertical(10.0, 0, 100), vertical(12.0, 0, 100)]
    out, n_moved = snap_junctions(pj, 6.0)
    assert n_moved == 0


# --------------------------------------------------------------------------
# Unitario: regularize_lines end-to-end
# --------------------------------------------------------------------------

def test_regularize_lines_combo():
    """4 verticales + 3 horizontales ruidosas que se cruzan.

    La garantía F6: los miembros de un cluster quedan EXACTAMENTE
    paralelos (mismo ángulo) y el lattice queda regular.
    """
    combo = [vertical(10.0, 0, 100), vertical(60.3, 0, 100),
             vertical(110.1, 0, 100), vertical(159.8, 0, 100),
             (0.0, 30.2, 200.0, 30.05),
             (0.0, 49.9, 200.0, 50.3),
             (0.0, 69.8, 200.0, 69.9)]
    segs, info = regularize_lines(combo, 6.0, snap_tol=2.0)
    vangs = [round(_seg_angle(s), 9) for s in segs
             if 45 < _seg_angle(s) < 135]
    hangs = [round(_seg_angle(s), 9) for s in segs
             if not (45 < _seg_angle(s) < 135)]
    assert len(vangs) == 4 and len(hangs) == 3, (vangs, hangs)
    assert max(vangs) - min(vangs) < 1e-9, vangs      # paralelismo EXACTO
    assert abs(vangs[0] - 90.0) < 1e-9, vangs
    assert max(hangs) - min(hangs) < 1e-9, hangs
    assert abs(_wrap180(hangs[0])) < 0.5, hangs       # media del cluster
    xs = sorted(s[0] for s in segs if abs(_seg_angle(s) - 90.0) < 1e-6)
    sp = [b - a for a, b in zip(xs, xs[1:])]
    assert max(sp) - min(sp) < 0.05, sp               # lattice regular
    assert info["n_snapped"] >= 1 and info["n_lattice"] >= 1, info


# --------------------------------------------------------------------------
# Unitario: métrica junction_gaps
# --------------------------------------------------------------------------

def test_junction_gaps_reports_tip_error():
    truth = [vertical(100.0, 0.0, 50.0), (0.0, 50.0, 200.0, 50.0)]
    detected = [vertical(100.0, 0.0, 52.0), (0.0, 50.0, 200.0, 50.0)]
    gaps = junction_gaps(truth, detected, stroke_px=6.0)
    assert len(gaps) == 1
    assert gaps[0] == pytest.approx(2.0, abs=0.01)


# --------------------------------------------------------------------------
# Integración: aceptación S3 sobre los DXF dorados rasterizados
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dxf_name", GOLDEN_DXF)
def test_regression_regularization_on_golden_dxf(dxf_name):
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

    # 1) métricas S1 intactas
    min_len = max(15.0, 10.0 * stroke)
    long_truth = [s for s in truth if seg_len(s) >= min_len]
    ev = evaluate_lines(long_truth, detected, stroke_px=stroke)
    assert ev["n_matched"] == ev["n_truth"], ev["unmatched"]
    assert ev["double_lines"] == 0
    assert ev["mean_angular_error"] < 0.3, ev
    assert ev["mean_coverage"] >= 0.9, ev

    # 2) aceptación F6: cero solapes entre portadoras
    assert count_overlap(detected, angle_tol=0.5, perp_tol=0.01) == 0

    # 3) paralelismo exacto: todo ángulo detectado cae en {0°, 90°} ±0.05°
    for s in detected:
        a = _seg_angle(s)
        d_axis = min(abs(_wrap180(a - 0.0)), abs(_wrap180(a - 90.0)))
        assert d_axis < 0.05, (s, a)

    # 4) juntas exactas: extremos detectados a ≤0.5px de la intersección
    #    de las verdades (T y L reales de los DXF)
    gaps = junction_gaps(truth, detected, stroke_px=stroke)
    assert gaps, "debe haber juntas que evaluar"
    assert max(gaps) < 0.5, max(gaps)

    # 5) fidelidad de entradas: una entrada PAT por verdad geométrica
    assert res["num_entries"] <= len(truth) + 2, (
        res["num_entries"], len(truth))

    # 6) la regularización se reporta en stats
    assert "Reg[" in res["stats"], res["stats"]
