"""Limpieza de segmentos compartida (Fase 3).

Con la emisión per-line (delta=(tile,tile)), cada segmento genera una FAMILIA
de líneas que se repite cada `tile` unidades en perpendicular y a lo largo.
Dos segmentos distintos en el dibujo fuente pueden generar la MISMA familia
(p.ej. una línea en el borde mínimo y otra en el máximo del tile, o trazos
duplicados). Este módulo elimina esos duplicados visuales reales SIN tocar
trazos distintos a distinto offset.
"""

import math

EPS = 1e-9


def _angle_mod180(x1, y1, x2, y2):
    a = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
    return a


def _ang_diff(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _perp_offset(x, y, rad):
    return -x * math.sin(rad) + y * math.cos(rad)


def _mod_dist(a, b, period):
    """Distancia entre a y b módulo `period` (en [0, period/2])."""
    if period <= EPS:
        return abs(a - b)
    d = abs(a - b) % period
    return min(d, period - d)


def _intervals_overlap_mod(a0, a1, b0, b1, period, tol):
    """True si [a0,a1] y [b0,b1] se solapan módulo `period`."""
    if period <= EPS:
        return min(a1, b1) - max(a0, b0) > tol
    mid = ((a0 + a1) - (b0 + b1)) / 2.0
    k0 = int(round(mid / period))
    for k in (k0, k0 - 1, k0 + 1):
        shift = k * period
        if min(a1, b1 + shift) - max(a0, b0 + shift) > tol:
            return True
    return False


def _segment_info(seg):
    x1, y1, x2, y2 = seg
    ang = _angle_mod180(x1, y1, x2, y2)
    rad = math.radians(ang)
    ux, uy = math.cos(rad), math.sin(rad)
    # offset perpendicular (usa el punto medio, robusto ante extremos)
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    off = _perp_offset(mx, my, rad)
    t0, t1 = x1 * ux + y1 * uy, x2 * ux + y2 * uy
    return ang, rad, off, min(t0, t1), max(t0, t1)


def dedup_duplicate_segments(segments, tile, angle_tol=0.5, offset_tol=None,
                             overlap_tol=1e-9):
    """Elimina segmentos cuya FAMILIA PAT coincide con otra ya conservada.

    Dos segmentos son duplicados visuales si:
      - ángulos mod 180 difieren <= angle_tol, Y
      - offsets perpendiculares coinciden MÓDULO tile (<= offset_tol), Y
      - sus intervalos a lo largo de la dirección se solapan MÓDULO tile.

    Conserva el PRIMER segmento de cada grupo (orden estable).
    Devuelve (segmentos_conservados, num_eliminados).
    """
    if offset_tol is None:
        offset_tol = max(tile * 1e-6, 1e-9)

    kept = []
    infos = []
    removed = 0
    for seg in segments:
        x1, y1, x2, y2 = seg
        if math.hypot(x2 - x1, y2 - y1) <= EPS:
            removed += 1
            continue
        ang, rad, off, t0, t1 = _segment_info(seg)
        duplicate = False
        for (k_ang, k_rad, k_off, k_t0, k_t1) in infos:
            if _ang_diff(ang, k_ang) > angle_tol:
                continue
            if _mod_dist(off, k_off, tile) > offset_tol:
                continue
            # proyectar t del nuevo segmento al frame del conservado es
            # suficientemente equivalente dentro de angle_tol
            if _intervals_overlap_mod(t0, t1, k_t0, k_t1, tile, overlap_tol):
                duplicate = True
                break
        if duplicate:
            removed += 1
            continue
        kept.append(seg)
        infos.append((ang, rad, off, t0, t1))
    return kept, removed
