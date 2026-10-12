"""Orquestador del método de precisión (S1).

run(img_bgr) -> dict con:
- segments_px: lista de (x1, y1, x2, y2) float en la resolución de trabajo
- side_px: lado de trabajo (px) para la conversión px -> mm
- stroke_width: grosor de trazo estimado (px de trabajo)
- mode: "centerline" | "region"
- scale: factor de escalado aplicado por el preproceso
- gray / mask / debug_base: para integración con core_logic y depuración
- "error": mensaje si el pipeline falla
"""

import math

import cv2
import numpy as np

from . import preprocess as _preprocess
from . import classify as _classify
from . import centerline as _centerline
from geometry import simplify_polyline_adaptive


def _region_segments(mask, sw, rdp_tol):
    """F3b básico (semilla): contornos de zonas rellenas -> cuerdas.
    El eje medial dual de calidad llega en S5."""
    from skimage import measure
    segs = []
    min_len = max(3.0, 0.75 * max(sw, 1.0))
    contours = measure.find_contours(mask.astype(np.float32) * 0.5, 0.5 - 1e-6)
    for c in contours:
        pts = np.asarray(c, dtype=np.float64)[:, ::-1]  # (row,col) -> (x,y)
        if len(pts) < 3:
            continue
        simplified = simplify_polyline_adaptive(pts, max(rdp_tol, 0.75))
        n = len(simplified)
        for i in range(n):
            a = simplified[i]
            b = simplified[(i + 1) % n]  # contornos cerrados
            seg = (float(a[0]), float(a[1]), float(b[0]), float(b[1]))
            if np.hypot(seg[2] - seg[0], seg[3] - seg[1]) >= min_len:
                segs.append(seg)
    return segs


def run(img_bgr, force_mode=None, rdp_tol=0.5):
    prep = _preprocess.preprocess(img_bgr)
    info = _classify.analyze_strokes(prep["mask"], force_mode=force_mode)
    if "error" in info:
        return {"error": info["error"]}
    h, w = prep["size"]
    sw = info["stroke_width"]

    if info["mode"] == "centerline":
        chain_data, surgery = _centerline.extract_chains(
            info["skeleton"], prep["ink"], sw)
        if not chain_data:
            return {"error": "No se pudieron extraer cadenas del esqueleto."}
        segs = _centerline.chains_to_segments(chain_data, sw, rdp_tol=rdp_tol)
        segs = _centerline.merge_collinear_segments(segs, sw=sw)
        # filtro definitivo de trazos cortos, tras la curación de gaps
        min_len = max(3.0, 0.75 * max(sw, 1.0))
        segs = [s for s in segs
                if math.hypot(s[2] - s[0], s[3] - s[1]) >= min_len]
        n_chains = len(chain_data)
    else:
        segs = _region_segments(prep["mask"], sw, rdp_tol)
        n_chains = 0

    if not segs:
        return {"error": "No se detectaron trazos válidos en la imagen."}

    debug_base = np.full((h, w, 3), 255, dtype=np.uint8)
    debug_base[prep["mask"] > 0] = (215, 215, 215)

    return {
        "segments_px": [(float(a), float(b), float(c), float(d))
                        for (a, b, c, d) in segs],
        "side_px": int(h),
        "stroke_width": float(sw),
        "mode": info["mode"],
        "scale": int(prep["scale"]),
        "ink_fraction": float(info["ink_fraction"]),
        "width_cv": float(info["width_cv"]),
        "num_chains": int(n_chains),
        "surgery": surgery if info["mode"] == "centerline" else {},
        "mask": prep["mask"],
        "gray": prep["gray"],
        "debug_base": debug_base,
    }
