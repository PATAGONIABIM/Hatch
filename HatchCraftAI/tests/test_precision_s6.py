"""S6 — F9 auto-tuning (botón Auto) + UI sección 4 wiring."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_precision_s1 import (CANVAS, REPO_ROOT, dxf_segments, encode_png,
                               rasterize_segments, seg_len,
                               transform_segments)
from test_precision_s4 import _make_bond_image
from test_precision_s5 import make_pavers
from core_logic import ImageToPatConverter
from metrics import count_overlap, evaluate_lines
from vectorize import tuning

WEB = REPO_ROOT / "web"
SHORT_COMBOS = [
    dict(snap_tol=2.0, corner_angle=135.0, sagitta_mm=0.1),
    dict(snap_tol=0.0, corner_angle=135.0, sagitta_mm=0.1),
    dict(snap_tol=4.0, corner_angle=135.0, sagitta_mm=0.1),
]


def _golden_png(name):
    segs = transform_segments(dxf_segments(REPO_ROOT / name))
    return encode_png(rasterize_segments(segs)), segs


def _circle_png(size=512, r=140):
    img = np.full((size, size, 3), 255, np.uint8)
    cv2.circle(img, (size // 2, size // 2), r, (0, 0, 0), 3, cv2.LINE_AA)
    return encode_png(img)


def _real_png(name):
    path = REPO_ROOT / name
    if not path.exists():
        pytest.skip(f"missing real image: {name}")
    from PIL import Image
    im = Image.open(path).convert("RGB")
    return encode_png(cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR))


def _decode(png):
    return cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)


def _long_truth(segs, stroke):
    return [s for s in segs if seg_len(s) >= max(15.0, 10.0 * stroke)]


def _det_canvas(res):
    k = CANVAS / float(res["side_px"])
    det = [(x1 * k, y1 * k, x2 * k, y2 * k)
           for x1, y1, x2, y2 in res["lines_px"]]
    return det, float(res["stroke_width_px"]) * k


def _convert_auto(png, **kw):
    return ImageToPatConverter().convert(
        png, method="precision", tile_mm=100.0, return_geometry=True,
        auto=True, **kw)


# ---------------------------------------------------------------- unitarios

def test_auto_tune_tableado_elige_rigidez():
    png, _ = _golden_png("TABLEADO HORIZONTAL.dxf")
    best_params, best_res, scores = tuning.auto_tune(
        _decode(png), tile_mm=100.0, budget=400)
    assert best_params is not None and best_res is not None
    assert scores and len(scores) == len(tuning.DEFAULT_COMBOS)
    # imagen arquitectónica → debe preferir snap (no orgánico)
    assert best_params["snap_tol"] >= 2.0
    # scores ascendentes y el mejor no peor que el combo default
    assert scores[0][1] <= scores[-1][1]
    default_score = next(
        s for p, s in scores
        if p["snap_tol"] == 2.0 and p["corner_angle"] == 135.0
        and p["sagitta_mm"] == 0.1)
    assert scores[0][1] <= default_score
    assert best_res["segments_px"]
    assert np.isfinite(best_res["auto_score"])


def test_auto_tune_circulo_sagita_del_grid():
    best_params, best_res, scores = tuning.auto_tune(
        _decode(_circle_png()), tile_mm=100.0, budget=400)
    assert best_params is not None and best_res is not None
    grid_sag = {c["sagitta_mm"] for c in tuning.DEFAULT_COMBOS}
    assert best_params["sagitta_mm"] in grid_sag
    assert len(best_res["segments_px"]) <= 400


def test_auto_tune_combos_cortos():
    best_params, best_res, scores = tuning.auto_tune(
        _decode(_circle_png()), tile_mm=100.0, budget=400,
        combos=SHORT_COMBOS[:2])
    assert best_params is not None
    assert best_res["auto_n"] == 2
    assert len(scores) == 2


# ------------------------------------------------------------ end-to-end

def test_convert_auto_tableado_calidad_intacta():
    png, segs = _golden_png("TABLEADO HORIZONTAL.dxf")
    res = _convert_auto(png)
    assert "error" not in res
    assert res["analysis"]["auto_params"]
    assert "Auto[" in res["stats"]
    assert res["analysis"]["dominant_deg"]
    assert "debug_types" in res and "debug_heat" in res
    det, stroke = _det_canvas(res)
    truth = _long_truth(segs, stroke)
    ev = evaluate_lines(truth, det, stroke)
    assert ev["n_matched"] == len(truth)
    assert ev["double_lines"] == 0
    assert ev["mean_angular_error"] < 0.3
    assert count_overlap(det) == 0
    assert res["num_entries"] > 0


def test_convert_sin_auto_no_lleva_auto_params():
    png, _ = _golden_png("TABLEADO HORIZONTAL.dxf")
    res = ImageToPatConverter().convert(
        png, method="precision", tile_mm=100.0, return_geometry=True)
    assert "error" not in res
    assert not (res.get("analysis") or {}).get("auto_params")


# -------------------------------------------------------------------- API

def _client():
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)


def test_api_auto_true_devuelve_auto_params():
    client = _client()
    r = client.post(
        "/api/convert/image",
        files={"file": ("t.png", _circle_png(), "image/png")},
        data={"method": "precision", "auto": "true", "tile_mm": "100"})
    assert r.status_code == 200
    j = r.json()
    assert j["num_entries"] > 0
    assert j.get("analysis", {}).get("auto_params")


def test_api_force_mode_invalido_400():
    client = _client()
    r = client.post(
        "/api/convert/image",
        files={"file": ("t.png", _circle_png(), "image/png")},
        data={"method": "precision", "force_mode": "bogus"})
    assert r.status_code == 400


def test_api_force_mode_region_200():
    client = _client()
    r = client.post(
        "/api/convert/image",
        files={"file": ("t.png", _circle_png(), "image/png")},
        data={"method": "precision", "force_mode": "region"})
    assert r.status_code == 200
    assert r.json()["num_entries"] > 0


# ------------------------------------------------- aceptación ≥80% del set

def _golden_ok(res, truth):
    if "error" in res or res["num_entries"] <= 0:
        return False
    det, stroke = _det_canvas(res)
    tl = _long_truth(truth, stroke)
    if not tl:
        return False
    ev = evaluate_lines(tl, det, stroke)
    return (ev["n_matched"] == len(tl) and ev["double_lines"] == 0
            and ev["mean_angular_error"] < 0.3 and count_overlap(det) == 0)


def _sane_ok(res):
    return "error" not in res and 0 < res.get("num_entries", 0) <= 600


def test_auto_aceptacion_primer_intento():
    """Aceptación S6 (plan L172): Auto aceptable en ≥80% del set."""
    results = {}
    for name in ("HERRIBONE.dxf", "HILADA.dxf",
                 "TABLEADO HORIZONTAL.dxf", "ADOCRETO_MODEL.dxf"):
        png, segs = _golden_png(name)
        results[name] = _golden_ok(_convert_auto(png), segs)
    results["circle"] = _sane_ok(_convert_auto(_circle_png()))
    results["bond"] = _sane_ok(_convert_auto(encode_png(_make_bond_image())))
    results["pavers"] = _sane_ok(_convert_auto(encode_png(make_pavers())))
    for name in ("38ebb6_c8a796acf67f45c9b1913452f31e2ea6~mv2.gif",
                 "1.jpg", "images.png", "madera.png"):
        img = _decode(_real_png(name))
        best_params, best_res, _ = tuning.auto_tune(
            img, tile_mm=100.0, budget=400, combos=SHORT_COMBOS)
        ok = (best_res is not None and "error" not in best_res
              and 0 < len(best_res["segments_px"]) <= 6000)
        results[name] = bool(ok)
    n_ok = sum(1 for v in results.values() if v)
    print(f"\nAuto aceptable: {n_ok}/{len(results)} -> {results}")
    assert n_ok >= 9, f"Auto aceptable en {n_ok}/{len(results)}: {results}"


# ------------------------------------------------------------ UI estática

def test_ui_wiring_estatico():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert 'value="precision"' in html
    assert "workbench-grid" in html
    assert "timer-value" in html
    assert 'name="precmode"' in html
    assert "view-types" in html
    js = (WEB / "js" / "main.js").read_text(encoding="utf-8")
    assert "sagitta_mm" in js
    assert "append('auto'" in js
    assert "precmode" in js
    css = (WEB / "css" / "style.css").read_text(encoding="utf-8")
    assert ".analysis-strip" in css
    assert ".cad-view-btn" in css
