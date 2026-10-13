"""S4 — F7 Periodicidad: detección de lattice 2D, promediado de celdas y wrap de costuras.

Cubre:
- detect_lattice: rectangular, oblicuo (running bond), aperiódico, reducción de
  múltiplos (_reduce_cell) — regresión del bug A(múltiplo) > A(fundamental).
- wrap_segments: división en bordes, piezas que suman la longitud original.
- average_cell: promediado reduce ruido; nx·ny==1 = recorte simple.
- Integración end-to-end: patrón bond cruzando bordes → tile = UNA celda del
  período (25.0×12.5 mm), 3×3 sin costuras (chamfer), sin líneas dobles.
- Activación conservadora: sin tinta en el borde → F7 desactivado.
"""
import cv2
import numpy as np
import pytest

from metrics import chamfer_px, evaluate_lines
from pat_compiler import compile_pat_perline
from pat_sim import render_faithful
from core_logic import ImageToPatConverter
from vectorize.periodic import (
    _reduce_cell,
    average_cell,
    detect_lattice,
    wrap_segments,
)

from test_precision_s1 import encode_png


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grid_mask(h, w, row_step, col_step, thickness=1, seed=0):
    """Rectangular grid: horizontales cada row_step, verticales cada col_step."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), dtype=np.float32)
    for y in range(row_step // 2, h, row_step):
        cv2.line(img, (0, y), (w - 1, y), 1.0, thickness)
    for x in range(col_step // 2, w, col_step):
        cv2.line(img, (x, 0), (x, h - 1), 1.0, thickness)
    del rng
    return (img > 0).astype(np.uint8)


def _bond_mask(h, w, bed_step=32, head_step=128, thickness=1):
    """Running bond: beds cada bed_step; heads con offset alterno por fila."""
    img = np.zeros((h, w), dtype=np.float32)
    for y in range(bed_step // 2, h, bed_step):
        cv2.line(img, (0, y), (w - 1, y), 1.0, thickness)
    band = 0
    for y in range(bed_step // 2, h - bed_step // 2, bed_step):
        x0 = (head_step // 4) if band % 2 == 0 else (3 * head_step // 4)
        for x in range(x0, w, head_step):
            cv2.line(img, (x, y), (x, y + bed_step), 1.0, thickness)
        band += 1
    return (img > 0).astype(np.uint8)


def _make_bond_image(size=512, bed_step=32, head_step=128, noise_sigma=None,
                     seed=1234, stroke_px=3):
    """Imagen bond anti-aliased (supersampling ×4). Beds cruzan bordes."""
    ss = 4
    big = np.full((size * ss, size * ss, 3), 255.0, dtype=np.float32)
    th = stroke_px * ss

    def _px(v):
        return int(round(v * ss))

    for y in range(bed_step // 2, size, bed_step):
        cv2.line(big, (-8 * ss, _px(y)), (size * ss + 8 * ss, _px(y)),
                  (0, 0, 0), th, cv2.LINE_AA)
    band = 0
    for y in range(bed_step // 2, size - bed_step // 2, bed_step):
        x0 = (head_step // 4) if band % 2 == 0 else (3 * head_step // 4)
        for x in range(x0 - head_step, size + head_step, head_step):
            cv2.line(big, (_px(x), _px(y)), (_px(x), _px(y + bed_step)),
                     (0, 0, 0), th, cv2.LINE_AA)
        band += 1

    img = cv2.resize(big, (size, size), interpolation=cv2.INTER_AREA)
    if noise_sigma:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, noise_sigma, (size, size, 1)).astype(np.float32)
        img = np.clip(img + noise, 0, 255)
    return np.clip(img, 0, 255).astype(np.uint8)


def _bond_cell_truth_work(scale=2, bed_step=32, head_step=128):
    """Verdad de UNA celda (work px): 2 beds + 2 stubs (uno partido por wrap)."""
    cw, ch = head_step * scale, bed_step * 2 * scale
    y1, y2 = bed_step // 2 * scale, (bed_step + bed_step // 2) * scale
    xs = head_step // 4 * scale
    xo = 3 * head_step // 4 * scale
    return [
        (0.0, float(y1), float(cw), float(y1)),
        (0.0, float(y2), float(cw), float(y2)),
        (float(xs), float(y1), float(xs), float(y2)),
        (float(xo), float(y2), float(xo), float(ch)),
        (float(xo), 0.0, float(xo), float(y1)),
    ]


def _render_mask(pat_content, tile_w, tile_h):
    img = render_faithful(pat_content, tile_w, tile_h, tiles=3, size=600)
    return (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < 128).astype(np.uint8)


# ---------------------------------------------------------------------------
# detect_lattice
# ---------------------------------------------------------------------------

def test_lattice_rectangular():
    mask = _grid_mask(256, 256, row_step=32, col_step=64)
    lat = detect_lattice(mask, stroke_width=2.0)
    assert lat is not None
    assert abs(lat["w"] - 64) <= 1.0
    assert abs(lat["h"] - 32) <= 1.0


def test_lattice_oblicuo_running_bond():
    mask = _bond_mask(256, 256, bed_step=32, head_step=128)
    lat = detect_lattice(mask, stroke_width=2.0)
    assert lat is not None
    assert abs(lat["w"] - 128) <= 1.0
    assert abs(lat["h"] - 64) <= 1.0


def test_lattice_aperiodico_none():
    rng = np.random.default_rng(7)
    img = np.zeros((256, 256), dtype=np.float32)
    for _ in range(30):
        x1, y1 = int(rng.integers(0, 256)), int(rng.integers(0, 256))
        x2, y2 = int(rng.integers(0, 256)), int(rng.integers(0, 256))
        cv2.line(img, (x1, y1), (x2, y2), 1.0, 1)
    mask = (img > 0).astype(np.uint8)
    assert detect_lattice(mask, stroke_width=2.0) is None


def test_reduce_cell_fundamental_desde_multiplo():
    """Un pico en el MÚLTIPLO con A≥0.85·peak1 se reduce a la fundamental."""
    w = h = 128
    A = np.zeros((h, w), dtype=np.float32)
    A[0, 64] = 0.95          # múltiplo 2× de la fundamental 32
    A[0, 32] = 0.92          # ≥ 0.85·0.95 = 0.8075 → debe reducir
    cw, ch = _reduce_cell(A, 64, 128, peak1=0.95, min_size=8.0)
    assert abs(cw - 32) < 1.0
    assert ch == 128


def test_reduce_cell_medio_genuino_no_reduce():
    """Un medio-período GENUINO (A≈0.78 < 0.85·peak1) NO se reduce."""
    A = np.zeros((128, 128), dtype=np.float32)
    A[0, 64] = 0.95
    A[0, 32] = 0.78
    cw, ch = _reduce_cell(A, 64, 128, peak1=0.95, min_size=8.0)
    assert cw == 64
    assert ch == 128


def test_lattice_jitter_fundamental():
    """Jitter ±1px degrada A(fundamental) pero el lattice se recupera
    (en limpio los múltiplos empatan a 1.0 — ver test_reduce_cell_*)."""
    rng = np.random.default_rng(42)
    img = np.zeros((512, 512), dtype=np.float32)
    for y in range(32, 512, 64):
        cv2.line(img, (0, y + int(rng.integers(-1, 2))),
                 (511, y + int(rng.integers(-1, 2))), 1.0, 1)
    for x in range(32, 512, 64):
        cv2.line(img, (x + int(rng.integers(-1, 2)), 0),
                 (x + int(rng.integers(-1, 2)), 511), 1.0, 1)
    mask = (img > 0).astype(np.uint8)
    lat = detect_lattice(mask, stroke_width=2.0)
    assert lat is not None
    assert abs(lat["w"] - 64) <= 1.5
    assert abs(lat["h"] - 64) <= 1.5


# ---------------------------------------------------------------------------
# wrap_segments
# ---------------------------------------------------------------------------

def test_wrap_cruce_vertical_dos_piezas():
    segs = [(90.0, 5.0, 110.0, 5.0)]
    out, flags = wrap_segments(segs, [True], 100.0, 50.0)
    assert len(out) == 2
    total = sum(np.hypot(s[2] - s[0], s[3] - s[1]) for s in out)
    assert abs(total - 20.0) < 1e-6
    assert all(max(s[0], s[2]) <= 100.0 + 1e-9 for s in out)
    assert flags == [True, True]


def test_wrap_interior_intacto():
    segs = [(30.0, 10.0, 70.0, 10.0), (5.0, 5.0, 5.0, 45.0)]
    out, _ = wrap_segments(segs, [False, False], 100.0, 50.0)
    assert len(out) == 2
    assert out[0] == segs[0]
    assert out[1] == segs[1]


def test_wrap_sobre_borde_canonico():
    """Segmento EXACTAMENTE sobre el borde: una sola pieza, envuelta a la
    posición canónica x≡0 (equivalente mod tile)."""
    segs = [(100.0, 5.0, 100.0, 45.0)]
    out, flags = wrap_segments(segs, [False], 100.0, 50.0)
    assert len(out) == 1
    assert out[0] == (0.0, 5.0, 0.0, 45.0)
    assert flags == [False]


# ---------------------------------------------------------------------------
# average_cell
# ---------------------------------------------------------------------------

def test_average_cell_reduce_ruido():
    rng = np.random.default_rng(5)
    cell = np.zeros((32, 64), dtype=np.float32)
    cv2.line(cell, (0, 16), (63, 16), 1.0, 1)
    ink = np.tile(cell, (2, 2))  # 64×128, celda 32×64
    noisy = np.clip(ink + rng.normal(0, 0.25, ink.shape), 0, 1)

    avg = average_cell(noisy, 0, 0, 64.0, 32.0, 2, 2)
    assert avg.shape == (32, 64)
    single = noisy[0:32, 0:64]
    clean_m = (cell > 0).astype(np.float32)
    mse_avg = float(np.mean((avg - clean_m) ** 2))
    mse_single = float(np.mean((single - clean_m) ** 2))
    assert mse_avg < mse_single
    assert float(np.mean(np.abs(avg - clean_m))) < 0.25


def test_average_cell_unica_es_recorte():
    rng = np.random.default_rng(6)
    ink = rng.random((40, 80)).astype(np.float32)
    avg = average_cell(ink, 8, 4, 64.0, 32.0, 1, 1)
    assert avg.shape == (32, 64)
    assert np.allclose(avg, ink[4:36, 8:72])


# ---------------------------------------------------------------------------
# Integración: bond cruzando bordes
# ---------------------------------------------------------------------------

def _assert_bond(res, truth_work, mean_tol=0.3, cov_tol=0.9):
    assert "period" in res
    period = res["period"]
    assert abs(period["cell_w_img"] - 128.0) / 128.0 <= 0.005
    assert abs(res["tile_w"] - 25.0) <= 0.05
    assert abs(res["tile_h"] - 12.5) <= 0.05
    assert period["nx"] == 4 and period["ny"] == 8
    assert period["averaged"]
    assert res["num_entries"] <= 12
    assert "Periodo[" in res["stats"]
    assert res["num_entries"] <= 400

    detected = res["lines_px"]
    stroke = res["stroke_width_px"]
    ev = evaluate_lines(truth_work, detected, stroke)
    assert ev["n_matched"] == ev["n_truth"] == len(truth_work)
    assert ev["double_lines"] == 0
    assert ev["mean_angular_error"] < mean_tol
    assert ev["mean_coverage"] >= cov_tol
    return ev


def test_bond_end_to_end_periodo_y_tile():
    img = _make_bond_image()
    res = ImageToPatConverter().convert(
        encode_png(img), method="precision", tile_mm=100.0, return_geometry=True)
    assert "error" not in res
    truth = _bond_cell_truth_work()
    _assert_bond(res, truth)


def test_bond_end_to_end_sin_costuras_3x3():
    img = _make_bond_image()
    conv = ImageToPatConverter()
    res = conv.convert(encode_png(img), method="precision", tile_mm=100.0,
                       return_geometry=True)
    assert "error" not in res

    truth = _bond_cell_truth_work()
    ppu = res["px_per_unit"]
    work_h = res["side_px"]
    segs_mm = []
    for x1, y1, x2, y2 in truth:
        segs_mm.append((x1 / ppu, (work_h - y1) / ppu,
                        x2 / ppu, (work_h - y2) / ppu))
    truth_pat = compile_pat_perline(
        segs_mm, name="TRUTH", desc="truth", tile_size=25.0,
        tile_w=25.0, tile_h=12.5, quantize_angles=None, coord_nd=5, dash_nd=5)

    det_mask = _render_mask(res["pat_content"], res["tile_w"], res["tile_h"])
    tru_mask = _render_mask(truth_pat["pat_content"], 25.0, 12.5)
    ch = chamfer_px(det_mask, tru_mask)
    assert ch <= 0.5


def test_bond_con_ruido_promediado():
    img = _make_bond_image(noise_sigma=22.0)
    res = ImageToPatConverter().convert(
        encode_png(img), method="precision", tile_mm=100.0, return_geometry=True)
    assert "error" not in res
    truth = _bond_cell_truth_work()
    _assert_bond(res, truth, mean_tol=0.3, cov_tol=0.8)


def test_bond_sin_borde_no_activa_periodicidad():
    """Margen 16 sin tinta en bordes → F7 desactivado (regresión S1-S3)."""
    size = 512
    ss = 4
    big = np.full((size * ss, size * ss, 3), 255.0, dtype=np.float32)
    th = 3 * ss
    for y in range(64, size - 32, 64):
        cv2.line(big, (16 * ss, y * ss), ((size - 16) * ss, y * ss),
                 (0, 0, 0), th, cv2.LINE_AA)
    for x in range(64, size - 32, 64):
        cv2.line(big, (x * ss, 16 * ss), (x * ss, (size - 16) * ss),
                 (0, 0, 0), th, cv2.LINE_AA)
    img = np.clip(cv2.resize(big, (size, size), interpolation=cv2.INTER_AREA),
                  0, 255).astype(np.uint8)
    res = ImageToPatConverter().convert(
        encode_png(img), method="precision", tile_mm=100.0, return_geometry=True)
    assert "error" not in res
    assert "period" not in res
    assert res["num_entries"] > 0
    assert "Periodo[" not in res["stats"]


def test_bond_periodic_off_fuerza_camino_legacy():
    img = _make_bond_image()
    res = ImageToPatConverter().convert(
        encode_png(img), method="precision", tile_mm=100.0,
        return_geometry=True, periodic="off")
    assert "error" not in res
    assert "period" not in res
    assert res["num_entries"] > 0
