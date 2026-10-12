"""F2 — Clasificador de trazos.

Decide cómo vectorizar el dibujo:

- modo "centerline": trazos finos tipo línea (la tinta es el trazo) →
  esqueleto + centro sub-píxel (F3a).
- modo "region": manchas / zonas rellenas → contornos (F3b básico; el
  eje medial dual llega en S5).

Estimación del grosor REAL: transformada de distancia sobre la MÁSCARA de
tinta binaria (no sobre bordes Canny) muestreada en el esqueleto: cada
píxel del esqueleto está en el centro del trazo, así que 2·dt(skel) es el
grosor local sin sesgo de borde.
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize


def analyze_strokes(mask, force_mode=None,
                    max_centerline_width=25.0,
                    max_ink_fraction=0.25,
                    max_width_cv=0.45):
    """Analiza la máscara binaria (uint8 0/1, 1=tinta).

    Devuelve dict con stroke_width (px), width_cv, ink_fraction, mode,
    distance (mapa dt) y skeleton (bool array). "error" si no hay tinta.
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if cv2.countNonZero(binary) == 0:
        return {"error": "No se detectó tinta en la imagen."}

    dt = cv2.distanceTransform(binary, cv2.DIST_L2, cv2.DIST_MASK_5)
    skel = skeletonize(binary > 0)

    vals = dt[skel]
    if vals.size == 0:
        vals = dt[binary > 0]
    if vals.size == 0:
        return {"error": "No se detectó tinta en la imagen."}

    stroke_width = float(2.0 * np.median(vals))
    mean_dt = float(np.mean(vals))
    std_dt = float(np.std(vals))
    width_cv = std_dt / mean_dt if mean_dt > 1e-9 else 1.0
    ink_fraction = float(np.mean(binary))

    if force_mode in ("centerline", "region"):
        mode = force_mode
    elif (stroke_width <= max_centerline_width
          and ink_fraction <= max_ink_fraction
          and width_cv <= max_width_cv):
        mode = "centerline"
    else:
        mode = "region"

    return {
        "stroke_width": stroke_width,
        "width_cv": width_cv,
        "ink_fraction": ink_fraction,
        "mode": mode,
        "distance": dt,
        "skeleton": skel,
    }
