"""F9: auto-tuning de parámetros del método precision.

Grid corto de combinaciones {snap_tol, corner_angle, sagitta_mm} evaluado
con pipeline.run completo. Score = Chamfer(raster(segs), mask)
+ λ·(n_segmentos/budget): fidelidad primero, complejidad como desempate.
"""

import cv2
import numpy as np


def _raster_segments(segs, shape, thickness):
    """Rasteriza segmentos (4-tuplas px) en una máscara uint8."""
    out = np.zeros((shape[0], shape[1]), dtype=np.uint8)
    t = max(1, int(round(thickness)))
    for x1, y1, x2, y2 in segs:
        cv2.line(out, (int(round(x1)), int(round(y1))),
                 (int(round(x2)), int(round(y2))), 1, t, cv2.LINE_AA)
    return out


DEFAULT_COMBOS = (
    {"snap_tol": 2.0, "corner_angle": 135.0, "sagitta_mm": 0.1},
    {"snap_tol": 0.0, "corner_angle": 135.0, "sagitta_mm": 0.1},
    {"snap_tol": 4.0, "corner_angle": 135.0, "sagitta_mm": 0.1},
    {"snap_tol": 2.0, "corner_angle": 115.0, "sagitta_mm": 0.1},
    {"snap_tol": 2.0, "corner_angle": 155.0, "sagitta_mm": 0.1},
    {"snap_tol": 2.0, "corner_angle": 135.0, "sagitta_mm": 0.05},
    {"snap_tol": 2.0, "corner_angle": 135.0, "sagitta_mm": 0.3},
)


def auto_tune(img_bgr, tile_mm=100.0, budget=400, combos=None, lam=0.15):
    """Prueba el grid y devuelve (best_params, best_result, scores).

    best_result es el dict completo de pipeline.run con la mejor
    combinación (incluye best_params bajo la key "auto_params" y el nº de
    pruebas bajo "auto_n"). scores es una lista (params, score) ordenada
    por score ascendente; score = inf para combinaciones con error.
    """
    from . import pipeline as _pipeline
    from metrics import chamfer_px

    if combos is None:
        combos = DEFAULT_COMBOS
    scores = []
    best_params = None
    best_res = None
    best_score = float("inf")
    for combo in combos:
        params = {
            "snap_tol": float(combo.get("snap_tol", 2.0)),
            "corner_angle": float(combo.get("corner_angle", 135.0)),
            "sagitta_mm": float(combo.get("sagitta_mm", 0.1)),
        }
        res = _pipeline.run(
            img_bgr, force_mode=combo.get("force_mode"),
            curves="fit", corner_angle=params["corner_angle"],
            tile_mm=tile_mm, sagitta_mm=params["sagitta_mm"],
            budget=budget, snap_tol=params["snap_tol"], periodic="auto")
        if "error" in res:
            scores.append((params, float("inf")))
            continue
        shape = (res["side_px"], res.get("tile_w_px", res["side_px"]))
        raster = _raster_segments(res["segments_px"], shape,
                                   max(res["stroke_width"], 1.0))
        ink = (res["mask"] > 0).astype(np.uint8)
        chamfer = chamfer_px(raster, ink)
        if not np.isfinite(chamfer):
            scores.append((params, float("inf")))
            continue
        n_seg = len(res["segments_px"])
        score = float(chamfer) + lam * min(n_seg, budget) / float(budget)
        scores.append((params, score))
        if score < best_score:
            best_score = score
            best_params = dict(params)
            best_res = res
    scores.sort(key=lambda p: p[1])
    if best_res is not None:
        best_res = dict(best_res)
        best_res["auto_params"] = best_params
        best_res["auto_score"] = best_score
        best_res["auto_n"] = len(combos)
    return best_params, best_res, scores
