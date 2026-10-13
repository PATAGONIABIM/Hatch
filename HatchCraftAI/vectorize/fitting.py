"""F5: descomposición de cadenas en primitivas (recta / arco / círculo / Bézier).

Cada cadena del grafo de esqueleto se divide en tramos libres de esquinas
(F4) y cada tramo se ajusta a la primitiva de menor costo MDL
(classify_segment). Las primitivas usan coordenadas de imagen (y hacia
abajo); los arcos se guardan con sweep CCW en radianes.
"""

import numpy as np

from geometry import classify_segment, fit_line_tls
from vectorize.corners import detect_corners


def _wrap_slice(pts, i0, i1):
    """Slice cíclico: _wrap_slice(p, 8, 3) con len(p)==10 → p[8:]+p[:3]."""
    n = len(pts)
    i0 %= n
    i1 %= n
    if i1 > i0:
        return pts[i0:i1]
    return np.vstack([pts[i0:], pts[:i1]])


def _split_spans(pts, corners, closed):
    """Divide la polilínea en tramos entre esquinas.

    Devuelve lista de arrays (cada uno incluye ambos extremos compartidos
    con el vecino). Para cadenas cerradas el primer punto == último.
    """
    n = len(pts)
    if not closed:
        if not corners:
            return [pts]
        spans = []
        prev = 0
        for c in sorted(corners):
            spans.append(pts[prev:c + 1])
            prev = c
        spans.append(pts[prev:])
        return [s for s in spans if len(s) >= 1]
    # Cerrada: rodar para que la primera esquina (o el propio 0) inicie.
    if not corners:
        return [np.vstack([pts, pts[:1]])]
    start = corners[0]
    rolled = np.vstack([pts[start:], pts[:start]])
    rest = sorted(c for c in corners if c != start)
    rolled_corners = [(c - start) % n for c in rest]
    spans = []
    prev = 0
    for c in rolled_corners:
        spans.append(rolled[prev:c + 1])
        prev = c
    tail = np.vstack([rolled[prev:], rolled[:1]])
    spans.append(tail)
    return spans


def _line_prim(span):
    fit = fit_line_tls(span)
    if fit is None:
        p0, p1 = span[0], span[-1]
        return {"type": "line", "p0": (p0[0], p0[1]), "p1": (p1[0], p1[1]),
                "rms": 0.0}
    d = np.array(fit["dir"], dtype=np.float64)
    p0 = np.array(fit["point"], dtype=np.float64)
    t0 = float(np.dot(span[0] - p0, d))
    t1 = float(np.dot(span[-1] - p0, d))
    a = p0 + t0 * d
    b = p0 + t1 * d
    return {"type": "line", "p0": (a[0], a[1]), "p1": (b[0], b[1]),
            "rms": float(fit["rms"])}


def _arc_prim(span, cx, cy, r, rms, loop=False):
    ang = np.arctan2(span[:, 1] - cy, span[:, 0] - cx)
    a0 = float(ang[0])
    if len(ang) > 1:
        d = np.diff(ang)
        d = (d + np.pi) % (2 * np.pi) - np.pi
        sweep = float(np.sum(d))
    else:
        sweep = 0.0
    # Lazo cerrado (extremos coincidentes) con vuelta casi completa ->
    # círculo. Umbral de seguridad: más de ~2 vueltas también es círculo.
    if (loop and abs(sweep) > np.pi) or abs(sweep) >= 1.95 * 2 * np.pi:
        return {"type": "circle", "center": (cx, cy), "radius": float(r),
                "rms": float(rms)}
    if sweep < 0:
        # Normalizar a CCW (y-down: misma convención geométrica local)
        a0 = float(ang[-1])
        sweep = -sweep
    return {"type": "arc", "center": (cx, cy), "radius": float(r),
            "a0": a0, "sweep": sweep, "rms": float(rms)}


def _cls_to_prim(span, cls, loop=False):
    kind = cls["type"]
    if kind == "line":
        return _line_prim(span)
    if kind == "arc":
        cx, cy, r, rms = cls["arc"]
        return _arc_prim(span, cx, cy, r, rms, loop=loop)
    ctrl = np.vstack(cls["beziers"])
    return {"type": "bezier", "ctrl": ctrl, "max_error": float(cls["max_error"])}


def chain_primitives(coords, stroke_width, closed=False, corner_angle=135.0,
                     sigma=None, rdp_fallback_tol=0.5):
    """Convierte una cadena en lista de primitivas.

    coords: array (n, 2) de la polilínea (ya refinada sub-píxel).
    Devuelve lista de dicts primitiva (ver _cls_to_prim) — vacía si no hay
    nada utilizable.
    """
    pts = np.asarray(coords, dtype=np.float64)
    if pts.ndim != 2 or len(pts) < 2:
        return []
    sw = max(stroke_width, 1.0)
    if sigma is None:
        sigma = min(0.75, max(0.2, 0.05 * sw))
    corners = detect_corners(pts, sw, corner_angle=corner_angle,
                             closed=closed)
    spans = _split_spans(pts, corners, closed)
    prims = []
    for span in spans:
        m = len(span)
        if m < 2:
            continue
        if m == 2:
            prims.append(_line_prim(span))
            continue
        # lazo: extremos del tramo coincidentes (cadena cerrada sin esquinas)
        loop = float(np.hypot(span[0][0] - span[-1][0],
                              span[0][1] - span[-1][1])) <= max(1.5, 0.3 * sw)
        cls = classify_segment(span, sigma=sigma)
        if cls is None:
            prims.append(_line_prim(span))
            continue
        prim = _cls_to_prim(span, cls, loop=loop)
        if prim is not None:
            prims.append(prim)
    return prims
