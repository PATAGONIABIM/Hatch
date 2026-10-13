"""F4 — Detección de esquinas sobre cadenas refinadas sub-píxel.

El esqueleto ya corta las cadenas en uniones (grado >= 3) y extremos; aquí
se detectan las esquinas RESTANTES dentro de cadenas suaves (p. ej. el
contorno de un rectángulo es una cadena cerrada con 4 esquinas).

Método (IPAN99 simplificado):
1. Perfil de giro en ventana adaptativa w ~ 0.75 x grosor de trazo:
   giro(i) = ángulo entre las cuerdas (P[i-w] -> P[i]) y (P[i] -> P[i+w]).
2. Esquina donde giro > 180° - corner_angle (interior < corner_angle),
   máximo local y validado con DOS rectas TLS (una por lado):
   - el giro TLS confirma el de las cuerdas, y
   - la intersección de ambas rectas cae cerca del vértice (aunque fuera,
     una curva suave tiene rectas casi paralelas => intersección lejana).
3. Supresión de no-máximos con radio ~w y margen de w respecto a los
   extremos (los extremos ya son cortes naturales de la cadena).

Los falsos positivos por escalera de raster quedan descartados por la
validación TLS (los perfiles sub-píxel refinados apenas giran).
"""

import math

import numpy as np

from geometry import fit_line_tls


def detect_corners(points, stroke_width, corner_angle=135.0, closed=False):
    """Devuelve índices ordenados de esquinas en `points` ((N,2) float)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = len(pts)
    if n < 6:
        return []
    sw = max(float(stroke_width), 1.0)
    w = int(round(max(3.0, 0.75 * sw)))
    if closed:
        w = min(w, max(2, n // 4))
    else:
        w = min(w, (n - 2) // 2)
    if w < 2:
        return []
    turn_thresh = max(5.0, 180.0 - float(corner_angle))

    def at(i):
        return pts[i % n]

    turn = np.zeros(n)
    valid = np.zeros(n, dtype=bool)
    for i in range(n):
        if not closed and (i - w < 0 or i + w >= n):
            continue
        a = at(i - w)
        b = at(i)
        c = at(i + w)
        v1 = b - a
        v2 = c - b
        l1 = float(np.linalg.norm(v1))
        l2 = float(np.linalg.norm(v2))
        if l1 < 1e-9 or l2 < 1e-9:
            continue
        cosang = float(np.clip(np.dot(v1, v2) / (l1 * l2), -1.0, 1.0))
        turn[i] = math.degrees(math.acos(cosang))
        valid[i] = True

    if not np.any(valid):
        return []

    # candidatos: giro sobre umbral y máximo local en radio w//2
    nms = max(1, w // 2)
    cands = []
    for i in range(n):
        if not valid[i] or turn[i] < turn_thresh:
            continue
        local_max = True
        for k in range(-nms, nms + 1):
            j = (i + k) % n
            if j == i:
                continue
            if valid[j] and turn[j] > turn[i] + 1e-12:
                local_max = False
                break
        if local_max:
            cands.append((turn[i], i))

    if not cands:
        return []

    def wrap_slice(i0, i1):
        """pts[i0:i1] con wrap para cadenas cerradas."""
        idx = [(k % n) for k in range(i0, i1)]
        return pts[idx]

    # validación TLS (dos rectas) + intersección cercana al vértice
    int_tol = max(2.0, 0.9 * sw)
    confirmed = []
    for tval, i in cands:
        if closed:
            left = wrap_slice(i - w, i + 1)
            right = wrap_slice(i, i + w + 1)
        else:
            left = pts[max(0, i - w):i + 1]
            right = pts[i:min(n, i + w + 1)]
        if len(left) < 2 or len(right) < 2:
            continue
        fl = fit_line_tls(left)
        fr = fit_line_tls(right)
        if fl is None or fr is None:
            continue
        dl = np.asarray(fl["dir"])
        dr = np.asarray(fr["dir"])
        dot = abs(float(np.dot(dl, dr)))
        tls_turn = math.degrees(math.acos(min(1.0, dot)))
        if tls_turn < turn_thresh * 0.75:
            continue
        # intersección de las rectas TLS
        pl = np.asarray(fl["point"])
        pr = np.asarray(fr["point"])
        A = np.stack([dl, -dr], axis=1)
        if abs(np.linalg.det(A)) < 1e-9:
            continue
        s, _ = np.linalg.solve(A, pr - pl)
        q = pl + s * dl
        if float(np.linalg.norm(q - pts[i])) > int_tol:
            continue
        confirmed.append((tval, i))

    if not confirmed:
        return []

    # NMS global por magnitud de giro; separación mínima ~w índices
    confirmed.sort(reverse=True)
    kept = []
    sep = max(2, w)
    for tval, i in confirmed:
        ok = True
        for _, j in kept:
            d = abs(i - j)
            d = min(d, n - d) if closed else d
            if d < sep:
                ok = False
                break
        if ok:
            kept.append((tval, i))

    idx = sorted(i for (_, i) in kept)
    if not closed:
        idx = [i for i in idx if w <= i <= n - 1 - w]
    return idx
