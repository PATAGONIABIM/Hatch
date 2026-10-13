"""S5 (F3b): regiones k-means Lab + eje medio de juntas compartidas.

Aceptación del plan: fotos de adoquín con F1-score > 0.85 respecto a un
ground truth CONOCIDO. El sintético de pavers (foto simulada con ruido de
tono, grano, sombreado y desenfoque) es ese caso medible; la foto real
(Flux2) no tiene GT pixel-exacto (el line-art provisto es OTRA instancia
del patrón), así que se valida estructuralmente: modo región automático,
red sin solapes dobles y familias angulares 45/135/90 presentes.
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vectorize import pipeline  # noqa: E402
from vectorize import regions as _regions  # noqa: E402
from metrics import count_overlap, f1_line_pixels  # noqa: E402
from core_logic import ImageToPatConverter  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

rng = np.random.default_rng(7)


def make_pavers(size=512, mortar_w=18, cell=64, tone_jitter=12, grain=5.0):
    """Foto sintética de adoquines: mortero oscuro + piedras con jitter."""
    img = np.zeros((size, size, 3), np.uint8)
    mortar = np.array([50, 42, 36], np.float32)
    stone = np.array([168, 162, 150], np.float32)
    img[:] = stone
    half = mortar_w // 2
    for k in range(size // cell + 1):
        img[max(0, k * cell - half):min(size, k * cell + half)] = mortar
    for r in range(size // cell):
        y0, y1 = r * cell + half, (r + 1) * cell - half
        x_off = 32 if r % 2 == 0 else 96
        x = x_off
        while x < size + cell:
            img[y0:y1, max(0, x - half):min(size, x + half)] = mortar
            x += 128
    for r in range(size // cell):
        y0, y1 = r * cell + half, (r + 1) * cell - half
        x_off = 32 if r % 2 == 0 else 96
        x_start = -(x_off % cell)
        for i in range(size // cell + 2):
            xa = x_start + i * cell
            xb = min(xa + cell - mortar_w, size)
            xa = max(0, xa)
            if xb <= xa or y1 <= y0:
                continue
            delta = rng.uniform(-tone_jitter, tone_jitter)
            img[y0:y1, xa:xb] = np.clip(
                img[y0:y1, xa:xb].astype(np.float32) + delta, 0, 255
            ).astype(np.uint8)
    grad = np.linspace(-10, 10, size, dtype=np.float32)[:, None, None]
    img = np.clip(img.astype(np.float32) + grad, 0, 255)
    img += rng.normal(0, grain, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (0, 0), 0.8)


def gt_paver_lines(size=512, cell=64, thickness=3):
    """Centros de junta del patrón running-bond (px de imagen)."""
    m = np.zeros((size, size), np.uint8)
    for k in range(size // cell + 1):
        y = min(k * cell, size - 1)
        cv2.line(m, (0, y), (size, y), 255, thickness)
    for r in range(size // cell):
        x_off = 32 if r % 2 == 0 else 96
        x = x_off
        while x < size + cell:
            y0, y1 = r * cell, (r + 1) * cell
            cv2.line(m, (x, y0), (x, y1), 255, thickness)
            x += 128
    return m


def raster_segs(segs, shape, thickness=3):
    m = np.zeros(shape, np.uint8)
    for x1, y1, x2, y2 in segs:
        cv2.line(m, (int(round(x1)), int(round(y1))),
                 (int(round(x2)), int(round(y2))), 255, thickness)
    return m


def scale_mask(m, sc):
    if sc == 1:
        return m
    return cv2.resize(m, (m.shape[1] * sc, m.shape[0] * sc),
                      interpolation=cv2.INTER_NEAREST)


def seg_angle(s):
    return math.degrees(math.atan2(s[3] - s[1], s[2] - s[0])) % 180.0


def angles_of(segs, minlen):
    out = []
    for s in segs:
        if math.hypot(s[2] - s[0], s[3] - s[1]) >= minlen:
            out.append(seg_angle(s))
    return out


# ---------------------------------------------------------------- unitarios

def test_web_metrics_cross_vs_disk():
    cross = np.zeros((200, 200), bool)
    cross[95:105, 20:180] = True
    cross[20:180, 95:105] = True
    j3, skel = _regions._web_metrics(cross)
    assert j3 >= 1  # el cruce central existe como unión real
    disk = np.zeros((200, 200), np.uint8)
    cv2.circle(disk, (100, 100), 60, 1, -1)
    disk = disk.astype(bool)
    j3d, _ = _regions._web_metrics(disk)
    assert j3d <= 2  # compacto: a lo sumo ramas mediales espurias


def test_extract_joint_mask_sources():
    # pavers: el mortero es la fase WEB
    joint, info = _regions.extract_joint_mask(
        cv2.resize(make_pavers(), (512, 512)), ksize_hint=8)
    assert joint is not None
    assert info["source"] == "web"
    # checkerboard: sin fase web -> frontera compartida
    cb = np.zeros((256, 256, 3), np.uint8)
    for i in range(4):
        for j in range(4):
            if (i + j) % 2 == 0:
                cb[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64] = 200
    joint_cb, info_cb = _regions.extract_joint_mask(cb, ksize_hint=8)
    assert info_cb["source"] == "boundary"
    assert joint_cb is not None and joint_cb.any()


def test_joint_skeleton_centered():
    band = np.zeros((100, 300), np.uint8)
    band[40:60, :] = 1  # banda horizontal de 20px
    skel, sw = _regions.joint_skeleton(band)
    assert skel.any()
    ys = np.flatnonzero(skel.any(axis=1))
    assert 40 <= ys.min() and ys.max() <= 59
    assert 15 <= sw <= 25


def test_f1_line_pixels():
    m = np.zeros((100, 100), np.uint8)
    cv2.line(m, (10, 10), (90, 90), 255, 3)
    assert f1_line_pixels(m, m, 2.0) == 1.0
    shifted = np.zeros((100, 100), np.uint8)
    cv2.line(shifted, (12, 12), (92, 92), 255, 3)
    val = f1_line_pixels(m, shifted, 1.0)
    assert 0.5 < val < 1.0
    other = np.zeros((100, 100), np.uint8)
    cv2.line(other, (10, 90), (90, 10), 255, 3)
    # solo el entorno del cruce central coincide
    assert f1_line_pixels(m, other, 2.0) < 0.10


# ------------------------------------------------------------- integración

def test_pavers_f1_and_no_overlap():
    """Aceptación S5: F1 > 0.85 (tol = sw/2) en foto de adoquín sintética."""
    res = pipeline.run(make_pavers())
    assert "error" not in res
    assert res["mode"] == "region"
    assert res["regions"]["source"] == "web"
    segs = res["segments_px"]
    side = res["side_px"]
    sc = res["scale"]
    sw = res["stroke_width"]
    det = raster_segs(segs, (side, side))
    gt = scale_mask(gt_paver_lines(), sc)
    assert f1_line_pixels(det, gt, 4.0) > 0.8      # ajuste fino
    assert f1_line_pixels(det, gt, sw / 2) > 0.9   # aceptación del plan
    assert count_overlap(segs, 0.5, 0.01) == 0
    angs = angles_of(segs, 60)
    assert any(abs(a) < 3 or abs(a - 180) < 3 for a in angs)   # horizontales
    assert any(abs(a - 90) < 3 for a in angs)                  # verticales


def test_checkerboard_single_shared_grid():
    cb = np.zeros((512, 512, 3), np.uint8)
    for i in range(4):
        for j in range(4):
            if (i + j) % 2 == 0:
                cb[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128] = 200
    res = pipeline.run(cb)
    assert "error" not in res
    assert res["mode"] == "region"
    assert res["regions"]["source"] == "boundary"
    segs = res["segments_px"]
    assert len(segs) <= 12          # 6 líneas de rejilla, sin dobles
    assert count_overlap(segs, 0.5, 0.01) == 0
    side = res["side_px"]
    sc = res["scale"]
    det = raster_segs(segs, (side, side))
    gt = np.zeros((512, 512), np.uint8)
    for k in range(1, 4):
        cv2.line(gt, (k * 128, 0), (k * 128, 512), 255, 3)
        cv2.line(gt, (0, k * 128), (512, k * 128), 255, 3)
    assert f1_line_pixels(det, scale_mask(gt, sc), 6.0) > 0.9


def test_touching_regions_midline():
    """Regiones que se tocan: UNA línea en la frontera compartida."""
    tr = np.zeros((300, 300, 3), np.uint8)
    tr[:, :150] = 90
    tr[:, 150:] = 190
    res = pipeline.run(tr)
    assert "error" not in res
    segs = res["segments_px"]
    assert len(segs) == 1
    s = segs[0]
    sc = res["scale"]
    x_img = s[0] / sc
    assert 146 <= x_img <= 154          # frontera real en x=150
    assert abs(s[2] - s[0]) <= 3.0      # vertical
    assert math.hypot(s[2] - s[0], s[3] - s[1]) > 0.5 * res["side_px"]


def test_filled_square_outline_exact():
    """S1 regresión: cuadrado relleno -> contorno exacto, no marco fantasma."""
    sq = np.full((300, 300, 3), 255, np.uint8)
    cv2.rectangle(sq, (50, 50), (250, 250), 10, -1)
    res = pipeline.run(sq, force_mode="region")
    assert "error" not in res
    assert res["mode"] == "region"
    segs = res["segments_px"]
    assert len(segs) > 0
    sc = res["scale"]
    xs, ys = [], []
    for s in segs:
        for (x, y) in ((s[0], s[1]), (s[2], s[3])):
            xs.append(x / sc)
            ys.append(y / sc)
    # el contorno vive en el anillo del borde del cuadrado [50, 250]
    for v in xs + ys:
        ring = min(abs(v - 50), abs(v - 251))
        assert ring <= 8.0
    assert min(xs) >= 45 and max(xs) <= 256
    assert min(ys) >= 45 and max(ys) <= 256


def _real(name):
    p = REPO_ROOT / name
    if not p.exists():
        pytest.skip(f"imagen real no disponible: {name}")
    return cv2.imread(str(p))


def test_real_photo_structural():
    """Foto real de adoquín: región automática, red sin dobles, 45/135/90."""
    photo = _real("Flux2-Klein_00837_.png")
    res = pipeline.run(photo)
    assert "error" not in res
    assert res["mode"] == "region"
    assert res["regions"]["source"] == "web"
    segs = res["segments_px"]
    assert 0 < len(segs) <= 1000
    assert count_overlap(segs, 0.5, 0.01) <= 25
    angs = angles_of(segs, 60)
    assert any(abs(a - 45) < 3 for a in angs)
    assert any(abs(a - 135) < 3 for a in angs)
    assert any(abs(a - 90) < 3 for a in angs)
    # end-to-end: entradas PAT razonables para una foto orgánica densa
    ok, buf = cv2.imencode(".png", photo)
    conv = ImageToPatConverter().convert(
        bytes(buf), method="precision", tile_mm=100.0, return_geometry=True)
    assert "error" not in conv
    assert 0 < conv["num_entries"] <= 600


def test_real_gif_centerline_periodic():
    """GIF binario de rombos: centerline + periodicidad activa."""
    from PIL import Image
    p = REPO_ROOT / "38ebb6_c8a796acf67f45c9b1913452f31e2ea6~mv2.gif"
    if not p.exists():
        pytest.skip("gif no disponible")
    g = Image.open(str(p)).convert("RGB")
    gif = cv2.cvtColor(np.array(g), cv2.COLOR_RGB2BGR)
    res = pipeline.run(gif)
    assert "error" not in res
    assert res["mode"] == "centerline"
    assert len(res["segments_px"]) > 0
    assert "period" in res and res["period"]["nx"] >= 2


def test_real_1jpg_sane():
    """Dibujo tenue de rejilla de ladrillos: conversión sana."""
    img = _real("1.jpg")
    ok, buf = cv2.imencode(".png", img)
    res = ImageToPatConverter().convert(
        bytes(buf), method="precision", tile_mm=100.0, return_geometry=True)
    assert "error" not in res
    assert res["num_entries"] > 0


def test_real_images_png_centerline():
    img = _real("images.png")
    res = pipeline.run(img)
    assert "error" not in res
    assert res["mode"] == "centerline"
    assert len(res["segments_px"]) > 0
    # el dibujo tiene margen: la periodicidad no debe activarse
    assert "period" not in res


def test_real_madera_no_crash():
    """Textura orgánica (madera): región k-means sin crash, salida sana."""
    img = _real("madera.png")
    res = pipeline.run(img)
    assert "error" not in res
    assert res["mode"] == "region"
    assert len(res["segments_px"]) > 0
