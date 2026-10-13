"""F9: auto-tuning de parámetros del método precision.

Grid corto de combinaciones {snap_tol, corner_angle, sagitta_mm} evaluado
con pipeline.run completo. Score = Chamfer(raster(segs), mask)
+ λ·(n_segmentos/budget) + λ_r·(1 − recall_tinta): fidelidad primero,
complejidad y cobertura de tinta como desempate (Fase 2.8: las métricas
nuevas de metrics.py son el objetivo del tuning). En estructuras
orgánicas los combos con snap_tol>0 se descartan (2.1/2.5: el snap se
desactiva en modo orgánico; recomendarlo sería un parámetro muerto).
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


def auto_tune(img_bgr, tile_mm=100.0, budget=400, combos=None, lam=0.15,
              lam_recall=0.5):
    """Prueba el grid y devuelve (best_params, best_result, scores).

    best_result es el dict completo de pipeline.run con la mejor
    combinación (incluye best_params bajo la key "auto_params" y el nº de
    pruebas bajo "auto_n"). scores es una lista (params, score) ordenada
    por score ascendente; score = inf para combinaciones con error y
    para combos con snap_tol>0 cuando la imagen resulta orgánica.
    lam_recall pondera (1 − recall_tinta con tol=stroke_width).
    """
    from . import pipeline as _pipeline
    from metrics import chamfer_px, recall_ink

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
        if res.get("structure_kind") == "organic" and params["snap_tol"] > 0:
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
        sw = max(float(res.get("stroke_width", 1.0)), 1.0)
        recall = recall_ink(raster, ink, tol=sw)
        score = (float(chamfer)
                 + lam * min(n_seg, budget) / float(budget)
                 + float(lam_recall) * (1.0 - recall))
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
