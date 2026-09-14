"""Primitivas geométricas para vectorización fiel de trazos (Fase 2).

- simplify_polyline_adaptive: RDP con tolerancia de cuerda ABSOLUTA (predecible
  independiente del tamaño de la forma). .PAT solo soporta líneas rectas, así
  que la fidelidad de curvas = aproximar con cuerdas acotadas por `tol`.
- fit_circle_taubin / fit_arc: ajuste algebraico de círculo/arco. Se usa en el
  exportador DXF (Fase 5) para emitir ARC reales en vez de polilíneas.
"""

import math

import cv2
import numpy as np


def simplify_polyline_adaptive(points, tol):
    """Ramer-Douglas-Peucker con tolerancia de cuerda absoluta.

    points: (N,2) array-like de puntos ordenados (px o unidades reales).
    tol: distancia perpendicular máxima permitida entre la polilínea original
         y la simplificada (error de cuerda). tol<=0 devuelve los puntos tal cual.
    Devuelve (M,2) float array con los vértices conservados (incluye extremos).
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] <= 2 or tol is None or tol <= 0:
        return pts.copy()
    cnt = pts.reshape(-1, 1, 2).astype(np.float32)
    approx = cv2.approxPolyDP(cnt, float(tol), False)
    out = approx.reshape(-1, 2).astype(np.float64)
    if out.shape[0] < 2:  # seguridad: nunca colapsar a un punto
        return pts[:2].copy()
    # Garantizar extremos exactos: approxPolyDP puede fundir el último punto
    # cuando coincide con el primero (trazos cerrados).
    if not np.allclose(out[0], pts[0]):
        out = np.vstack([pts[0], out])
    if not np.allclose(out[-1], pts[-1]):
        out = np.vstack([out, pts[-1]])
    return out


def fit_circle_taubin(points):
    """Ajuste algebraico de círculo (Kasa). Devuelve (cx, cy, r, rms_residual).

    rms_residual = desviación RMS de la distancia radial respecto a r.
    Devuelve None si hay <3 puntos o el ajuste es degenerado.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 3:
        return None
    x = pts[:, 0]
    y = pts[:, 1]
    z = x * x + y * y
    A = np.column_stack([x, y, np.ones_like(x)])
    # x²+y² = D·x + E·y + F  →  centro=(D/2, E/2), r²=F+(D²+E²)/4
    sol, *_ = np.linalg.lstsq(A, z, rcond=None)
    d, e, f = sol
    cx, cy = d / 2.0, e / 2.0
    radicand = f + (d * d + e * e) / 4.0
    if radicand <= 0:
        return None
    r = math.sqrt(radicand)
    if r <= 1e-9:
        return None
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    rms = float(np.sqrt(np.mean((dist - r) ** 2)))
    return cx, cy, r, rms


def fit_arc(points, tol, min_radius=1e-6, max_radius=1e9):
    """Intenta ajustar un arco a la polilínea. Devuelve dict o None.

    Acepta si el ajuste de círculo tiene rms_residual <= tol y el radio está
    en rango. Devuelve {center, radius, start_angle, end_angle, rms} con
    ángulos en grados (convención DXF, CCW desde +X).
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    fit = fit_circle_taubin(pts)
    if fit is None:
        return None
    cx, cy, r, rms = fit
    if rms > tol or not (min_radius <= r <= max_radius):
        return None
    ang = np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)) % 360.0
    return {
        "center": (float(cx), float(cy)),
        "radius": float(r),
        "start_angle": float(ang[0]),
        "end_angle": float(ang[-1]),
        "rms": rms,
    }
