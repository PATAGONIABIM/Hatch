import math

import cv2
import numpy as np

EPS = 1e-9


def parse_pat(content):
    """Parsea contenido PAT devolviendo (nombre, lista de entradas).

    Cada entrada: {angle, ox, oy, dx, dy, dashes[]} con la semántica oficial
    de AutoCAD/Revit: familia infinita de líneas paralelas separadas por el
    vector delta, con ciclo de dashes repetido a lo largo de cada miembro.
    """
    name = "PATTERN"
    entries = []
    for raw in content.strip().replace("\r\n", "\n").split("\n"):
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("*"):
            name = line[1:].split(",")[0].strip()
            continue
        if ";" in line:
            line = line.split(";")[0].strip()
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            angle, ox, oy, dx, dy = (float(p) for p in parts[:5])
            dashes = [float(p) for p in parts[5:] if p.strip()]
        except ValueError:
            continue
        entries.append({
            "angle": angle, "ox": ox, "oy": oy,
            "dx": dx, "dy": dy, "dashes": dashes,
        })
    return name, entries


def _clip_segment(p1, p2, rect):
    x0, y0, w, h = rect
    xr, yr = x0 + w, y0 + h
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, p1[0] - x0), (dx, xr - p1[0]),
                 (-dy, p1[1] - y0), (dy, yr - p1[1])):
        if abs(p) < EPS:
            if q < 0:
                return None
        else:
            r = q / p
            if p < 0:
                if r > t1:
                    return None
                if r > t0:
                    t0 = r
            else:
                if r < t0:
                    return None
                if r < t1:
                    t1 = r
    return ((p1[0] + t0 * dx, p1[1] + t0 * dy),
            (p1[0] + t1 * dx, p1[1] + t1 * dy))


def iter_entry_segments(entry, rect):
    """Genera segmentos 2D de una entrada PAT recortados al rectángulo.

    Implementa la semántica completa: réplicas de familia en k·(dx,dy) y
    ciclado bidireccional de dashes a lo largo de cada miembro.
    """
    rad = math.radians(entry["angle"])
    ux, uy = math.cos(rad), math.sin(rad)
    ox, oy = entry["ox"], entry["oy"]
    dxv, dyv = entry["dx"], entry["dy"]
    dashes = entry["dashes"]

    corners = [(rect[0], rect[1]),
               (rect[0] + rect[2], rect[1]),
               (rect[0], rect[1] + rect[3]),
               (rect[0] + rect[2], rect[1] + rect[3])]

    dlen = math.hypot(dxv, dyv)
    cycle = sum(abs(d) for d in dashes)

    if dlen > EPS:
        dux, duy = dxv / dlen, dyv / dlen
        dprojs = [px * dux + py * duy for px, py in corners]
        kmin = int(math.floor(min(dprojs) / dlen)) - 1
        kmax = int(math.ceil(max(dprojs) / dlen)) + 1
    else:
        kmin, kmax = 0, 0

    uprojs = [(px - ox) * ux + (py - oy) * uy for px, py in corners]
    t_lo, t_hi = min(uprojs), max(uprojs)

    for k in range(kmin, kmax + 1):
        bx = ox + k * dxv
        by = oy + k * dyv
        shift_t = k * (dxv * ux + dyv * uy)
        rmin = t_lo - shift_t
        rmax = t_hi - shift_t

        if not dashes or cycle < EPS:
            seg = _clip_segment((bx + rmin * ux, by + rmin * uy),
                                (bx + rmax * ux, by + rmax * uy), rect)
            if seg:
                yield seg
            continue

        m_start = int(math.floor(rmin / cycle)) - 1
        m_end = int(math.ceil(rmax / cycle)) + 1
        for m in range(m_start, m_end + 1):
            base_s = m * cycle
            pos = base_s
            for dv in dashes:
                length = abs(dv)
                a = max(pos, rmin)
                b = min(pos + length, rmax)
                if dv > 0 and b - a > EPS:
                    seg = _clip_segment((bx + a * ux, by + a * uy),
                                        (bx + b * ux, by + b * uy), rect)
                    if seg:
                        yield seg
                pos += length


def segments_in_rect(pat_content, rect):
    """Devuelve todos los segmentos del patrón dentro del rectángulo dado."""
    _, entries = parse_pat(pat_content)
    out = []
    for e in entries:
        for (p1, p2) in iter_entry_segments(e, rect):
            out.append((p1[0], p1[1], p2[0], p2[1]))
    return out


def infer_tile_size(entries):
    t = 0.0
    for e in entries:
        t = max(t, abs(e["ox"]), abs(e["oy"]), abs(e["dx"]), abs(e["dy"]))
    return t if t > EPS else 1.0


def render_faithful(pat_content, tile_w=None, tile_h=None,
                    tiles=3, size=600, manual_scale=1.0, show_grid=True,
                    bg=(255, 255, 255)):
    """Renderiza el PAT exactamente como lo interpretaría Revit."""
    name, entries = parse_pat(pat_content)
    img = np.full((size, size, 3), bg, dtype=np.uint8)
    if not entries:
        return img

    if tile_w is None or tile_h is None:
        t_inf = infer_tile_size(entries)
        if tile_w is None:
            tile_w = t_inf
        if tile_h is None:
            tile_h = t_inf

    span_x = max(tile_w * tiles, EPS)
    span_y = max(tile_h * tiles, EPS)
    pad_frac = 0.04
    rect = (-span_x * pad_frac, -span_y * pad_frac,
            span_x * (1 + 2 * pad_frac), span_y * (1 + 2 * pad_frac))
    spp = size / max(rect[2], EPS) * manual_scale

    def to_screen(x, y):
        sx = (x - rect[0]) * spp
        sy = size - (y - rect[1]) * spp
        return int(round(sx)), int(round(sy))

    if show_grid:
        gx = 0.0
        while gx <= span_x + EPS:
            p1 = to_screen(gx, rect[1])
            p2 = to_screen(gx, rect[1] + rect[3])
            cv2.line(img, p1, p2, (215, 215, 215), 1, cv2.LINE_AA)
            gx += tile_w
        gy = 0.0
        while gy <= span_y + EPS:
            p1 = to_screen(rect[0], gy)
            p2 = to_screen(rect[0] + rect[2], gy)
            cv2.line(img, p1, p2, (215, 215, 215), 1, cv2.LINE_AA)
            gy += tile_h

    for e in entries:
        for (p1, p2) in iter_entry_segments(e, rect):
            cv2.line(img, to_screen(*p1), to_screen(*p2),
                     (0, 0, 0), 1, cv2.LINE_AA)

    return img


render_pat_preview = render_faithful
