"""Métricas de regresión para la fidelidad de patrones (Fase 0).

- count_overlap / count_overlap_entries: detectan líneas duplicadas o
  superpuestas (el síntoma reportado en la simulación Revit).
- chord_error: error máximo de cuerda al aproximar curvas con polilíneas
  (fidelidad de trazos orgánicos, Fase 2).
"""

import math


def _angle_mod180(x1, y1, x2, y2):
    ang = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
    return ang


def _perp_offset(angle_deg, px, py):
    """Offset perpendicular de la recta (angle, punto) al origen."""
    rad = math.radians(angle_deg)
    return -px * math.sin(rad) + py * math.cos(rad)


def _ang_diff(a, b):
    """Diferencia angular mínima mod 180."""
    d = abs((a - b) % 180.0)
    return min(d, 180.0 - d)


def count_overlap(segments, angle_tol=0.5, perp_tol=0.01):
    """Cuenta pares de segmentos (x1,y1,x2,y2) que se superponen.

    Dos segmentos se superponen si son colineales (mismo ángulo mod 180 y
    mismo offset perpendicular, dentro de tolerancias) y sus intervalos a lo
    largo de la dirección se solapan.
    """
    n = len(segments)
    lines = []
    for (x1, y1, x2, y2) in segments:
        ang = _angle_mod180(x1, y1, x2, y2)
        off = _perp_offset(ang, x1, y1)
        rad = math.radians(ang)
        ux, uy = math.cos(rad), math.sin(rad)
        t1, t2 = x1 * ux + y1 * uy, x2 * ux + y2 * uy
        lines.append((ang, off, min(t1, t2), max(t1, t2)))

    count = 0
    for i in range(n):
        ai, oi, ta, tb = lines[i]
        for j in range(i + 1, n):
            aj, oj, tc, td = lines[j]
            if _ang_diff(ai, aj) > angle_tol:
                continue
            if abs(oi - oj) > perp_tol:
                continue
            if min(tb, td) - max(ta, tc) > 1e-9:  # solape de intervalos
                count += 1
    return count


def _dash_intervals(entry):
    """Intervalos (a, b) con tinta a lo largo de la dirección de la entrada,
    en coordenada absoluta (el dash arranca en el origen (ox, oy))."""
    rad = math.radians(entry["angle"])
    t0 = entry["ox"] * math.cos(rad) + entry["oy"] * math.sin(rad)
    ivs = []
    pos = 0.0
    for d in entry["dashes"]:
        if d > 0:
            ivs.append((t0 + pos, t0 + pos + d))
        pos += abs(d)
    return ivs


def _overlap_mod(a, b, period, tol=1e-9):
    """¿Los intervalos a y b se solapan módulo `period`?"""
    if period <= 0:
        return min(a[1], b[1]) - max(a[0], b[0]) > tol
    mid = ((a[0] + a[1]) - (b[0] + b[1])) / 2.0
    k0 = int(round(mid / period))
    for k in (k0 - 1, k0, k0 + 1):
        if min(a[1], b[1] + k * period) - max(a[0], b[0] + k * period) > tol:
            return True
    return False


def count_overlap_entries(entries, tile, angle_tol=0.5, offset_tol=1e-3):
    """Cuenta pares de entradas PAT (parse_pat) que se superponen VISUALMENTE.

    Con delta=(tile,tile), la familia se repite cada `tile` en offset
    perpendicular: dos entradas se superponen si tienen el mismo ángulo
    (mod 180), su offset coincide módulo `tile` Y sus intervalos de dash
    se solapan módulo `tile`. Las familias en la misma portadora pero con
    dashes en distinta fase (p.ej. juntas trabadas de un tableado) NO
    cuentan: no se dibujan una encima de otra.
    """
    n = len(entries)
    count = 0
    for i in range(n):
        ei = entries[i]
        oi = _perp_offset(ei["angle"], ei["ox"], ei["oy"]) % tile
        ivs_i = None
        for j in range(i + 1, n):
            ej = entries[j]
            if _ang_diff(ei["angle"], ej["angle"]) > angle_tol:
                continue
            oj = _perp_offset(ej["angle"], ej["ox"], ej["oy"]) % tile
            d = abs(oi - oj)
            if min(d, tile - d) > offset_tol:
                continue
            if ivs_i is None:
                ivs_i = _dash_intervals(ei)
            ivs_j = _dash_intervals(ej)
            if any(_overlap_mod(a, b, tile) for a in ivs_i for b in ivs_j):
                count += 1
    return count


def chord_error(source_pts, simplified_pts):
    """Error máximo de cuerda (unilateral): distancia máxima de los vértices
    de la polilínea fuente a los segmentos de la polilínea simplificada.

    source_pts / simplified_pts: listas de (x, y).
    """
    def point_seg_dist(px, py, ax, ay, bx, by):
        dx, dy = bx - ax, by - ay
        if dx == 0 and dy == 0:
            return math.hypot(px - ax, py - ay)
        t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        cx, cy = ax + t * dx, ay + t * dy
        return math.hypot(px - cx, py - cy)

    worst = 0.0
    for (px, py) in source_pts:
        best = float("inf")
        for k in range(len(simplified_pts) - 1):
            ax, ay = simplified_pts[k]
            bx, by = simplified_pts[k + 1]
            best = min(best, point_seg_dist(px, py, ax, ay, bx, by))
        worst = max(worst, best)
    return worst
