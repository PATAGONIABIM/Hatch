"""Primitivas geométricas para vectorización fiel de trazos (Fase 2 + F5).

- simplify_polyline_adaptive: RDP con tolerancia de cuerda ABSOLUTA (predecible
  independiente del tamaño de la forma). .PAT solo soporta líneas rectas, así
  que la fidelidad de curvas = aproximar con cuerdas acotadas por `tol`.
- fit_line_tls / max_perp_deviation: recta por Total Least Squares (S1).
- fit_circle_taubin / fit_arc: ajuste algebraico Kasa. Se usa en el exportador
  DXF (Fase 5) para emitir ARC reales en vez de polilíneas.
- fit_circle_hyper: algebraico por autovectores (Taubin/Hyper) + refinado
  geométrico Levenberg-Marquardt — sin el sesgo de Kasa en arcos cortos (S2).
- fit_bezier_schneider: Béziers cúbicas por mínimos cuadrados con
  reparametrización de Newton y subdivisión en el punto de error máximo
  (Graphics Gems, S2).
- classify_segment: elección recta/arco/Bézier por costo MDL
  (error_rms²/σ² + n_parámetros).
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


def fit_line_tls(points):
    """Ajuste de recta por Total Least Squares (PCA).

    Minimiza la suma de distancias perpendiculares de todos los puntos a la
    recta (a diferencia deOLS sobre y). Devuelve
    {"point": (px, py), "dir": (ux, uy) unitario, "rms": float} o None si los
    puntos son degenerados (<2 puntos o todos idénticos).
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 2:
        return None
    centroid = pts.mean(axis=0)
    d = pts - centroid
    cov = d.T @ d
    if cov[0, 0] + cov[1, 1] <= 1e-12:
        return None
    evals, evecs = np.linalg.eigh(cov)
    if evals[-1] <= 1e-12:
        return None
    direction = evecs[:, -1]
    if direction[0] < 0 or (direction[0] == 0 and direction[1] < 0):
        direction = -direction
    proj = d @ direction
    resid = d - np.outer(proj, direction)
    rms = float(np.sqrt(np.mean(np.sum(resid * resid, axis=1))))
    return {
        "point": (float(centroid[0]), float(centroid[1])),
        "dir": (float(direction[0]), float(direction[1])),
        "rms": rms,
    }


def max_perp_deviation(points, fit):
    """Desviación perpendicular máxima de `points` a la recta TLS `fit`."""
    if fit is None:
        return float("inf")
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] == 0:
        return 0.0
    p0 = np.asarray(fit["point"], dtype=np.float64)
    d = np.asarray(fit["dir"], dtype=np.float64)
    rel = pts - p0
    perp = rel - np.outer(rel @ d, d)
    return float(np.max(np.linalg.norm(perp, axis=1)))


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


def _fit_circle_algebraic(pts):
    """Ajuste algebraico por autovectores (Taubin/Hyper, S2 del plan).

    Minimiza Σ(A·zc + B·u + C·v)² con (A,B,C) unitario, donde u,v son
    coordenadas centradas y z = u²+v² centrada por su media. El centrado
    elimina el sesgo de Kasa en arcos cortos; el autovector menor de la
    matriz de dispersión 3x3 da la solución cerrada. Sirve de inicializador
    para el refinado geométrico LM.
    """
    x = pts[:, 0]
    y = pts[:, 1]
    mx = float(x.mean())
    my = float(y.mean())
    u = x - mx
    v = y - my
    z = u * u + v * v
    zc = z - z.mean()
    M = np.array([
        [np.sum(zc * zc), np.sum(zc * u), np.sum(zc * v)],
        [np.sum(zc * u), np.sum(u * u), np.sum(u * v)],
        [np.sum(zc * v), np.sum(u * v), np.sum(v * v)],
    ])
    evals, evecs = np.linalg.eigh(M)
    A, B, C = evecs[:, 0]
    if abs(A) < 1e-12:
        return None
    D = -A * z.mean()
    uc = -B / (2.0 * A)
    vc = -C / (2.0 * A)
    radicand = (B * B + C * C) / (4.0 * A * A) - D / A
    if radicand <= 0:
        return None
    r = math.sqrt(radicand)
    if r < 1e-9:
        return None
    return mx + uc, my + vc, r


def _fit_circle_lm(pts, init, max_iter=30):
    """Refinado geométrico Levenberg-Marquardt: min Σ(dist_i − r)².

    Elimina el sesgo residual del ajuste algebraico (importante en arcos
    cortos). Devuelve (cx, cy, r, rms_geométrico).
    """
    x = pts[:, 0]
    y = pts[:, 1]
    cx, cy, r = (float(init[0]), float(init[1]), float(init[2]))
    dx = x - cx
    dy = y - cy
    d = np.sqrt(dx * dx + dy * dy)
    r = max(float(d.mean()), 1e-9)
    sse = float(np.sum((d - r) ** 2))
    lam = 1e-3
    for _ in range(max_iter):
        dx = x - cx
        dy = y - cy
        d = np.sqrt(dx * dx + dy * dy)
        d_safe = np.maximum(d, 1e-12)
        e = d - r
        J = np.stack([-dx / d_safe, -dy / d_safe, -np.ones_like(d)], axis=1)
        JtJ = J.T @ J
        g = J.T @ e
        try:
            diag = np.diag(np.diag(JtJ)) + 1e-12
            delta = np.linalg.solve(JtJ + lam * diag, -g)
        except np.linalg.LinAlgError:
            break
        ncx, ncy, nr = cx + delta[0], cy + delta[1], max(r + delta[2], 1e-9)
        nd = np.sqrt((x - ncx) ** 2 + (y - ncy) ** 2)
        nsse = float(np.sum((nd - nr) ** 2))
        if nsse < sse:
            move = float(np.linalg.norm(delta))
            cx, cy, r, sse = ncx, ncy, nr, nsse
            lam = max(lam / 3.0, 1e-9)
            if move < 1e-10:
                break
        else:
            lam *= 5.0
            if lam > 1e9:
                break
    n = pts.shape[0]
    return cx, cy, r, math.sqrt(sse / n)


def fit_circle_hyper(points):
    """Círculo por algebraico (autovectores) + LM geométrico (plan F5).

    Devuelve (cx, cy, r, rms) con rms GEOMÉTRICO (Σ(dist−r)²/n), mucho menos
    sesgado que Kasa (`fit_circle_taubin`) en arcos cortos. None si degenera.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 3:
        return None
    init = _fit_circle_algebraic(pts)
    if init is None:
        init = fit_circle_taubin(pts)
    if init is None:
        return None
    return _fit_circle_lm(pts, init)


# ------------------------------------------------------- Bézier (Schneider)

def _bezier_eval(ctrl, u):
    """Punto de la Bézier cúbica ctrl(4,2) en parámetros u (array)."""
    u = np.asarray(u, dtype=np.float64)
    b = 1.0 - u
    w = np.stack([b ** 3, 3.0 * b * b * u, 3.0 * b * u * u, u ** 3], axis=1)
    return w @ ctrl


def _bezier_deriv(ctrl, u):
    """Derivada (×1/3) de la Bézier cúbica en u (array)."""
    u = np.asarray(u, dtype=np.float64)
    b = 1.0 - u
    w = np.stack([-b * b, b * b - 2.0 * b * u,
                  2.0 * b * u - u * u, u * u], axis=1)
    return 3.0 * (w @ ctrl)


def _solve_bezier_free(pts, u):
    """Bézier cúbica con extremos fijos y C1/C2 LIBRES por LS lineal.

    Con p0/p3 fijos, B(u) es lineal en (C1, C2) y x/y desacoplan en un
    sistema 2x2 compartido. Evita las tangentes estimadas del método
    clásico de Schneider (una tangente 3° desviada hace diverger el
    ajuste con tangentes fijas). Los controles se limitan a un sobre
    |C−anclaje| <= 3·cuerda + 12 por robustez ante parametrizaciones
    malas (Newton las corrige en las siguientes iteraciones).
    """
    p0 = pts[0]
    p3 = pts[-1]
    bu = 1.0 - u
    b0 = bu ** 3
    b1 = 3.0 * bu * bu * u
    b2 = 3.0 * bu * u * u
    b3 = u ** 3
    t = pts - np.outer(b0, p0) - np.outer(b3, p3)
    m11 = float(np.sum(b1 * b1))
    m12 = float(np.sum(b1 * b2))
    m22 = float(np.sum(b2 * b2))
    chord = float(np.linalg.norm(p3 - p0))
    dh = (p3 - p0) / max(chord, 1e-12)
    det = m11 * m22 - m12 * m12
    c1 = p0 + dh * (chord / 3.0)
    c2 = p3 - dh * (chord / 3.0)
    if abs(det) > 1e-12 and m11 > 1e-12 and m22 > 1e-12:
        m = np.array([[m11, m12], [m12, m22]])
        rx = np.array([float(np.sum(b1 * t[:, 0])),
                       float(np.sum(b2 * t[:, 0]))])
        ry = np.array([float(np.sum(b1 * t[:, 1])),
                       float(np.sum(b2 * t[:, 1]))])
        c1x, c2x = np.linalg.solve(m, rx)
        c1y, c2y = np.linalg.solve(m, ry)
        cand1 = np.array([c1x, c1y])
        cand2 = np.array([c2x, c2y])
        if np.all(np.isfinite(cand1)) and np.all(np.isfinite(cand2)):
            c1, c2 = cand1, cand2
    bound = 3.0 * chord + 12.0
    v1 = c1 - p0
    l1 = float(np.linalg.norm(v1))
    if l1 > bound:
        c1 = p0 + v1 * (bound / l1)
    v2 = c2 - p3
    l2 = float(np.linalg.norm(v2))
    if l2 > bound:
        c2 = p3 + v2 * (bound / l2)
    return np.stack([p0, c1, c2, p3])


def _newton_reparam(ctrl, pts, u):
    """Reparametrización de Schneider: u ← u − (Q−P)·Q'/|Q'|², clamp [0,1]."""
    q = _bezier_eval(ctrl, u)
    dq = _bezier_deriv(ctrl, u)
    diff = q - pts
    denom = np.sum(dq * dq, axis=1)
    denom = np.maximum(denom, 1e-12)
    du = np.sum(diff * dq, axis=1) / denom
    u2 = u - du
    u2 = np.clip(u2, 0.0, 1.0)
    u2.sort()
    u2[0] = 0.0
    u2[-1] = 1.0
    # forzar monotonía estricta
    eps = 1e-9
    for i in range(1, len(u2)):
        if u2[i] <= u2[i - 1]:
            u2[i] = u2[i - 1] + eps
    u2[-1] = 1.0
    return u2


def _chord_params(pts):
    """Parametrización por longitud de cuerda, normalizada a [0,1]."""
    d = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
    u = np.concatenate([[0.0], np.cumsum(d)])
    total = u[-1]
    if total < 1e-12:
        return np.linspace(0.0, 1.0, len(pts))
    return u / total


def _fit_bezier_range(pts, tol, out, err_box, w_hint, depth=0, max_depth=9):
    """FitCurves de Schneider sobre pts (array contiguo)."""
    n = len(pts)
    if n < 2:
        return
    if n == 2:
        chord = pts[1] - pts[0]
        ctrl = np.stack([pts[0], pts[0] + chord / 3.0,
                         pts[1] - chord / 3.0, pts[1]])
        out.append(ctrl)
        return
    u = _chord_params(pts)
    best_err = math.inf
    best_split = None
    prev_err = math.inf
    for _ in range(12):
        ctrl = _solve_bezier_free(pts, u)
        q = _bezier_eval(ctrl, u)
        d2 = np.sum((q - pts) ** 2, axis=1)
        split_i = int(np.argmax(d2))
        max_err = float(math.sqrt(d2[split_i]))
        if max_err <= tol:
            out.append(ctrl)
            err_box[0] = max(err_box[0], max_err)
            return
        if max_err < best_err:
            best_err = max_err
            best_split = split_i
        # estancamiento: el ajuste alternante ya no mejora → subdividir
        if max_err > prev_err * 0.98:
            break
        prev_err = max_err
        if depth >= max_depth:
            break
        u = _newton_reparam(ctrl, pts, u)
    # no converge: subdividir en el punto de error máximo de la MEJOR
    # iteración (la última puede ser una curva divergente sin sentido).
    if best_split is None or best_split <= 0 or best_split >= n - 1:
        best_split = n // 2
    out_left = []
    _fit_bezier_range(pts[:best_split + 1], tol, out_left, err_box,
                      w_hint, depth + 1, max_depth)
    out_right = []
    _fit_bezier_range(pts[best_split:], tol, out_right, err_box,
                      w_hint, depth + 1, max_depth)
    out.extend(out_left)
    out.extend(out_right)


def fit_bezier_schneider(points, tol, closed=False):
    """Ajuste de curva por Béziers cúbicas (Schneider, Graphics Gems).

    Devuelve {"beziers": [(4,2) arrays de control], "max_error": px} o
    None si <2 puntos. `closed` une tangentes a través del inicio/fin.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 2:
        return None
    if closed and not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[:1]])
    if tol is None or tol <= 0:
        tol = 0.25
    w_hint = max(3, min(len(pts) // 4, 12))
    out = []
    err_box = [0.0]
    _fit_bezier_range(pts, tol, out, err_box, w_hint)
    if not out:
        return None
    return {"beziers": out, "max_error": err_box[0]}


# ------------------------------------------------------- clasificación MDL


def classify_segment(points, sigma=0.5, bezier_tol=None, max_arc_radius=None):
    """Elige primitiva (recta/arco/Bézier) por costo MDL (plan F5).

    costo = (rms/σ)² + n_parámetros. σ = ruido sub-píxel estimado.
    Devuelve {"type", ..., "rms", "cost", "n_params"}:
      line   → {"line": fit_line_tls, 4 parámetros}
      arc    → {"arc": (cx, cy, r, rms), 5}
      bezier → {"beziers": [...], "max_error", 8 por tramo}
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = pts.shape[0]
    if n < 2:
        return None
    sigma = max(float(sigma), 1e-6)

    # --- recta (TLS, 4 parámetros)
    best = None
    line_fit = fit_line_tls(pts)
    if line_fit is not None:
        cost = (line_fit["rms"] / sigma) ** 2 + 4.0
        best = {"type": "line", "line": line_fit,
                "rms": line_fit["rms"], "cost": cost, "n_params": 4}

    # --- arco (algebraico+LM, 5 parámetros)
    arc_fit = fit_circle_hyper(pts)
    if arc_fit is not None:
        cx, cy, r, rms = arc_fit
        if r >= 1.0 and (max_arc_radius is None or r <= max_arc_radius):
            cost = (rms / sigma) ** 2 + 5.0
            if best is None or cost < best["cost"]:
                best = {"type": "arc", "arc": arc_fit,
                        "rms": rms, "cost": cost, "n_params": 5}

    # --- Bézier (Schneider, 8 parámetros por tramo)
    btol = bezier_tol if bezier_tol is not None else 2.0 * sigma
    bez = fit_bezier_schneider(pts, tol=btol)
    if bez is not None and bez["beziers"]:
        n_seg = len(bez["beziers"])
        rms_approx = 0.5 * bez["max_error"]
        cost = (rms_approx / sigma) ** 2 + 8.0 * n_seg
        if best is None or cost < best["cost"]:
            best = {"type": "bezier", "beziers": bez["beziers"],
                    "max_error": bez["max_error"], "rms": rms_approx,
                    "cost": cost, "n_params": 8 * n_seg}

    if best is None:
        # n==2 u otros degenerados: cuerda directa
        best = {"type": "line",
                "line": {"point": tuple(pts[0]),
                         "dir": _safe_dir(pts[-1] - pts[0]),
                         "rms": 0.0},
                "rms": 0.0, "cost": 4.0, "n_params": 4}
    return best


def _safe_dir(v):
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return (1.0, 0.0)
    return (float(v[0] / n), float(v[1] / n))


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
