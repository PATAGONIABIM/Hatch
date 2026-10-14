"""F7 — Periodicidad 2D y tile exacto.

- detect_lattice: autocorrelación 2D por FFT sobre la máscara de tinta;
  devuelve los vectores de lattice (v1, v2) (soporta retículos oblicuos,
  p.ej. trabas al 50 %) y la supercelda RECTANGULAR mínima (W, H), que es
  la única que un .PAT puede expresar.
- plan_crop: recorte exacto a un número entero de celdas (n_x·W × n_y·H),
  centrado, con regla de sobrante para rechazar periodicidades falsas.
- average_cell: promediado de los n_x·n_y tiles alineados sobre el campo
  de tinta (reduce el ruido en √N antes de vectorizar).
- wrap_segments: recorte de los segmentos que cruzan el borde del tile y
  traslado mod tile de la parte sobrante, para que la repetición en Revit
  no tenga costuras.
"""

import math

import numpy as np


def _autocorr_norm(mask):
    """Autocorrelación normalizada (A[0,0] = 1) de la máscara binaria."""
    f = mask.astype(np.float32)
    f = f - float(f.mean())
    a0 = float(np.sum(f * f))
    if a0 <= 1e-12:
        return None
    F = np.fft.rfft2(f)
    A = np.fft.irfft2(F * np.conj(F), s=f.shape)
    return A / a0


def _refine_peak(A, px, py):
    """Refinamiento sub-píxel de un pico entero por parábola por eje."""
    h, w = A.shape
    dx = dy = 0.0
    if 0 < px < w - 1:
        den = float(A[py, px - 1] - 2.0 * A[py, px] + A[py, px + 1])
        if den < -1e-12:
            dx = 0.5 * (float(A[py, px - 1]) - float(A[py, px + 1])) / den
            dx = max(-0.5, min(0.5, dx))
    if 0 < py < h - 1:
        den = float(A[py - 1, px] - 2.0 * A[py, px] + A[py + 1, px])
        if den < -1e-12:
            dy = 0.5 * (float(A[py - 1, px]) - float(A[py + 1, px])) / den
            dy = max(-0.5, min(0.5, dy))
    return (px + dx, py + dy)


def _rect_cell(v1, v2, tol=0.5, rng=4):
    """Supercelda rectangular mínima del lattice {a·v1 + b·v2}.

    Devuelve (W, H): mínimos vectores puros (W,0) y (0,H) del lattice,
    o None si no existen combinaciones pequeñas (lattice no rectangular
    en rango útil).
    """
    best_w = None
    best_h = None
    for a in range(-rng, rng + 1):
        for b in range(-rng, rng + 1):
            if a == 0 and b == 0:
                continue
            x = a * v1[0] + b * v2[0]
            y = a * v1[1] + b * v2[1]
            if abs(y) <= tol and x > 1.0:
                if best_w is None or x < best_w:
                    best_w = x
            if abs(x) <= tol and y > 1.0:
                if best_h is None or y < best_h:
                    best_h = y
    if best_w is None or best_h is None:
        return None
    return (float(best_w), float(best_h))


def _reduce_cell(A, cw, ch, peak1, min_size):
    """Reduce la supercelda a la fundamental dividiendo ejes por 2/3.

    Un múltiplo del período tiene autocorrelación ~igual a la fundamental
    (el argmax puede caer en el múltiplo por ruido). Si A en (cw/d, 0) o
    (0, ch/d) >= 0.85·peak1, el patrón es invariante bajo ese vector y la
    celda se reduce. Un semi-período GENUINO da A mucho menor (p.ej. 0.78
    vs 0.97) y es rechazado.
    """
    h, w = A.shape
    changed = True
    while changed:
        changed = False
        for d in (2, 3):
            if cw / d >= min_size:
                ix = int(round(cw / d)) % w
                if float(A[0, ix]) >= 0.85 * peak1:
                    cw = _refine_peak(A, ix, 0)[0]
                    changed = True
            if ch / d >= min_size:
                iy = int(round(ch / d)) % h
                if float(A[iy, 0]) >= 0.85 * peak1:
                    ch = _refine_peak(A, 0, iy)[1]
                    changed = True
    return float(cw), float(ch)


def detect_lattice(mask, stroke_width=1.0, min_peak1=0.45, min_peak2=0.30,
                   max_frac=0.62):
    """Detecta el lattice 2D de periodicidad de la máscara.

    Devuelve {"v1", "v2", "w", "h", "peak1", "peak2"} en px de la máscara,
    o None si no hay periodicidad confiable. (w, h) es la supercelda
    RECTANGULAR mínima (la que puede expresar un .PAT).
    """
    A = _autocorr_norm(mask)
    if A is None:
        return None
    h, w = mask.shape[:2]
    r0 = max(8.0, 3.0 * max(float(stroke_width), 1.0))
    rmax = max_frac * float(min(h, w))
    yy, xx = np.mgrid[0:h, 0:w]
    # distancia toroidal: el desplazamiento k y -k son equivalentes
    dxa = np.minimum(xx, w - xx).astype(np.float32)
    dya = np.minimum(yy, h - yy).astype(np.float32)
    r = np.hypot(dxa, dya)
    valid = (r >= r0) & (r <= rmax)
    if not valid.any():
        return None

    Av = np.where(valid, A, -np.inf)
    py, px = np.unravel_index(int(np.argmax(Av)), A.shape)
    peak1 = float(A[py, px])
    if peak1 < min_peak1:
        return None
    v1 = _refine_peak(A, int(px), int(py))

    # v2: mejor pico no colineal con v1 (perp dist suficiente)
    cross = np.abs(v1[0] * yy - v1[1] * xx).astype(np.float32)
    perp = cross / max(math.hypot(*v1), 1e-9)
    cand = valid & (perp >= max(3.0, 0.3 * r0))
    if not cand.any():
        return None
    Av2 = np.where(cand, A, -np.inf)
    py2, px2 = np.unravel_index(int(np.argmax(Av2)), A.shape)
    peak2 = float(A[py2, px2])
    if peak2 < min_peak2:
        return None
    v2 = _refine_peak(A, int(px2), int(py2))

    cell = _rect_cell(v1, v2)
    if cell is None:
        return None
    cw, ch = cell
    cw, ch = _reduce_cell(A, cw, ch, peak1, max(8.0, 2.0 * max(float(stroke_width), 1.0)))
    if cw < r0 or ch < r0 or cw > rmax + 1.0 or ch > rmax + 1.0:
        return None
    return {
        "v1": (float(v1[0]), float(v1[1])),
        "v2": (float(v2[0]), float(v2[1])),
        "w": float(cw),
        "h": float(ch),
        "peak1": peak1,
        "peak2": peak2,
    }


def plan_crop(size_w, size_h, cell_w, cell_h, max_leftover_frac=0.45):
    """Recorte exacto a n_x·cell_w × n_y·cell_h, centrado.

    Rechaza (None) si el sobrante sugiere que el contenido no es
    realmente periódico (p.ej. 1.4 períodos en la imagen).
    """
    if cell_w <= 0 or cell_h <= 0:
        return None
    nx = int(round(size_w / cell_w))
    nx = max(1, nx)
    while nx * cell_w > size_w + 0.5 and nx > 1:
        nx -= 1
    if nx * cell_w > size_w + 0.5:
        return None
    if size_w - nx * cell_w > max_leftover_frac * cell_w:
        return None
    ny = int(round(size_h / cell_h))
    ny = max(1, ny)
    while ny * cell_h > size_h + 0.5 and ny > 1:
        ny -= 1
    if ny * cell_h > size_h + 0.5:
        return None
    if size_h - ny * cell_h > max_leftover_frac * cell_h:
        return None
    cw = int(round(nx * cell_w))
    ch = int(round(ny * cell_h))
    if cw < 8 or ch < 8:
        return None
    x0 = int(round((size_w - cw) / 2.0))
    y0 = int(round((size_h - ch) / 2.0))
    return (nx, ny, x0, y0, cw, ch)


def average_cell(ink, x0, y0, cell_w, cell_h, nx, ny):
    """Promedia los nx·ny tiles alineados; devuelve UNA celda promediada.

    La celda (i, j) se toma en (x0 + i·cell_w, y0 + j·cell_h). El
    resultado (round(cell_h), round(cell_w)) float32 es la media de
    todas las celdas: reduce el ruido en √(nx·ny) antes de vectorizar.
    Con nx·ny == 1 equivale al recorte simple de la primera celda.
    Devuelve None si la celda resulta demasiado pequeña.
    """
    cw = int(round(cell_w))
    ch = int(round(cell_h))
    if cw < 8 or ch < 8:
        return None
    acc = np.zeros((ch, cw), dtype=np.float64)
    H, W = ink.shape[:2]
    n = 0
    for i in range(nx):
        for j in range(ny):
            xs = int(round(x0 + i * cell_w))
            ys = int(round(y0 + j * cell_h))
            xs = max(0, min(xs, W - cw))
            ys = max(0, min(ys, H - ch))
            acc += ink[ys:ys + ch, xs:xs + cw]
            n += 1
    if n <= 0:
        return None
    return (acc / float(n)).astype(np.float32)


def wrap_segments(segments, flags, tile_w, tile_h, min_piece=0.05, seg_types=None):
    """Envuelve segmentos mod (tile_w, tile_h) para un PAT sin costuras.

    Los segmentos que cruzan el borde se cortan en los cruces y cada
    trozo se traslada a su posición dentro del tile. Las piezas mínimas
    (slivers) se descartan. Devuelve (segmentos, flags) con los flags
    paralelos conservados (None si la entrada era None), o
    (segmentos, flags, seg_types) si seg_types no es None.
    """
    if not segments:
        res_flags = list(flags) if flags is not None else None
        if seg_types is not None:
            return list(segments), res_flags, list(seg_types)
        return list(segments), res_flags
    W = float(tile_w)
    H = float(tile_h)
    if W <= 0 or H <= 0:
        res_flags = list(flags) if flags is not None else None
        if seg_types is not None:
            return list(segments), res_flags, list(seg_types)
        return list(segments), res_flags
    has_flags = flags is not None and len(flags) == len(segments)
    has_types = seg_types is not None and len(seg_types) == len(segments)
    out = []
    out_flags = []
    out_types = []
    for idx, (x1, y1, x2, y2) in enumerate(segments):
        fl = bool(flags[idx]) if has_flags else None
        st = seg_types[idx] if has_types else None
        dx = float(x2) - float(x1)
        dy = float(y2) - float(y1)
        ts = set()
        if abs(dx) > 1e-12:
            kmin = math.floor(min(x1, x2) / W)
            kmax = math.ceil(max(x1, x2) / W)
            for k in range(int(kmin), int(kmax) + 1):
                t = (k * W - x1) / dx
                if 1e-9 < t < 1.0 - 1e-9:
                    ts.add(round(t, 12))
        if abs(dy) > 1e-12:
            kmin = math.floor(min(y1, y2) / H)
            kmax = math.ceil(max(y1, y2) / H)
            for k in range(int(kmin), int(kmax) + 1):
                t = (k * H - y1) / dy
                if 1e-9 < t < 1.0 - 1e-9:
                    ts.add(round(t, 12))
        params = [0.0] + sorted(ts) + [1.0]
        for a, b in zip(params[:-1], params[1:]):
            if b - a <= 1e-12:
                continue
            pax = x1 + dx * a
            pay = y1 + dy * a
            pbx = x1 + dx * b
            pby = y1 + dy * b
            if math.hypot(pbx - pax, pby - pay) < min_piece:
                continue
            mx = (pax + pbx) / 2.0
            my = (pay + pby) / 2.0
            kx = math.floor(mx / W)
            ky = math.floor(my / H)
            out.append((pax - kx * W, pay - ky * H,
                        pbx - kx * W, pby - ky * H))
            out_flags.append(fl)
            out_types.append(st)
    if not has_flags:
        out_flags = None
    if seg_types is not None:
        return out, out_flags, (out_types if has_types else None)
    return out, out_flags
