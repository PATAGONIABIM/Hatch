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


def _seg_line_info(x1, y1, x2, y2):
    """(ang mod 180, ux, uy, t1, t2) de un segmento ordenado por t."""
    ang = _angle_mod180(x1, y1, x2, y2)
    rad = math.radians(ang)
    ux, uy = math.cos(rad), math.sin(rad)
    t1, t2 = x1 * ux + y1 * uy, x2 * ux + y2 * uy
    return ang, ux, uy, min(t1, t2), max(t1, t2)


def _point_line_dist(px, py, x1, y1, ux, uy):
    """Distancia perpendicular de (px, py) a la recta infinita (x1,y1)+t·(ux,uy)."""
    return abs(-(px - x1) * uy + (py - y1) * ux)


def evaluate_lines(truth_segs, detected_segs, stroke_px=1.0,
                   angle_tol=10.0, perp_factor=1.5,
                   distinct_angle_tol=1.0):
    """Métricas S1 de precisión de rectas (coordenadas en px, mismo marco).

    Empareja cada recta verdadera con los segmentos detectados candidatos
    (ángulo < angle_tol, offsets perpendiculares mutuos < perp_factor·grosor
    y solape de extensión). El solape y los offsets de agrupación se
    calculan SIEMPRE en la base direccional de la recta verdadera (evita el
    salto de signo en el límite 0°/180°). Devuelve dict con:

    - n_truth / n_matched / unmatched: conteos de cobertura
    - angular_errors: {i: |Δángulo|°} con el mejor match (mayor solape)
    - mean_angular_error / max_angular_error
    - double_lines / doubled_truth: verdades emparejadas por >= 2 rectas
      físicas distintas (ángulo y offset perpendicular separados)
    - coverages: {i: fracción de longitud cubierta}, mean_coverage
    """
    stroke_px = max(float(stroke_px), 1.0)
    perp_tol = perp_factor * stroke_px

    det = []
    for s in detected_segs:
        x1, y1, x2, y2 = map(float, s)
        ang = _angle_mod180(x1, y1, x2, y2)
        det.append({
            "ang": ang,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "mx": (x1 + x2) / 2.0, "my": (y1 + y2) / 2.0,
        })
    buckets = {}
    for j, d in enumerate(det):
        buckets.setdefault(int(d["ang"]) % 180, []).append(j)

    angular_errors = {}
    coverages = {}
    unmatched = []
    doubled_truth = []
    for i, s in enumerate(truth_segs):
        x1, y1, x2, y2 = map(float, s)
        tang, ux, uy, ta, tb = _seg_line_info(x1, y1, x2, y2)
        tlen = tb - ta
        if tlen < 1e-9:
            continue
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        trad = math.radians(tang)
        tsin, tcos = math.sin(trad), math.cos(trad)
        cands = []
        for b in (int(tang) - 1, int(tang), int(tang) + 1):
            cands.extend(buckets.get(b % 180, ()))
        matches = []
        for j in cands:
            d = det[j]
            if _ang_diff(tang, d["ang"]) > angle_tol:
                continue
            perp1 = _point_line_dist(d["mx"], d["my"], x1, y1, ux, uy)
            perp2 = _point_line_dist(mx, my, d["x1"], d["y1"],
                                     math.cos(math.radians(d["ang"])),
                                     math.sin(math.radians(d["ang"])))
            if max(perp1, perp2) > perp_tol:
                continue
            # solape a lo largo de la DIRECCIÓN de la verdad
            dta = d["x1"] * ux + d["y1"] * uy
            dtb = d["x2"] * ux + d["y2"] * uy
            dlo, dhi = min(dta, dtb), max(dta, dtb)
            ov = min(dhi, tb) - max(dlo, ta)
            if ov <= 1e-9:
                continue
            matches.append((ov, j, dlo, dhi))
        if not matches:
            unmatched.append(i)
            continue
        best = max(matches, key=lambda m: m[0])
        angular_errors[i] = _ang_diff(tang, det[best[1]]["ang"])
        # cobertura: unión de intervalos solapados (base de la verdad)
        ivs = sorted((max(dlo, ta), min(dhi, tb))
                     for (_, _, dlo, dhi) in matches)
        merged = []
        for a, b in ivs:
            if merged and a <= merged[-1][1] + 1e-9:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        covered = sum(b - a for (a, b) in merged)
        coverages[i] = min(1.0, covered / tlen)
        # líneas dobles: matches en >= 2 rectas físicas distintas, con el
        # offset perpendicular medido en la base angular de la verdad
        offs = {}
        for (_, j, _, _) in matches:
            d = det[j]
            offs[j] = -d["x1"] * tsin + d["y1"] * tcos
        clusters = []
        for j, off in offs.items():
            placed = False
            for c in clusters:
                j0 = c[0]
                if (_ang_diff(det[j]["ang"], det[j0]["ang"]) <= distinct_angle_tol
                        and abs(off - offs[j0]) <= max(0.5, 0.75 * stroke_px)):
                    c.append(j)
                    placed = True
                    break
            if not placed:
                clusters.append([j])
        if len(clusters) >= 2:
            doubled_truth.append(i)

    n_err = len(angular_errors)
    return {
        "n_truth": len(truth_segs),
        "n_matched": n_err,
        "unmatched": unmatched,
        "angular_errors": angular_errors,
        "mean_angular_error": (sum(angular_errors.values()) / n_err
                              if n_err else float("nan")),
        "max_angular_error": (max(angular_errors.values())
                             if angular_errors else float("nan")),
        "double_lines": len(doubled_truth),
        "doubled_truth": doubled_truth,
        "coverages": coverages,
        "mean_coverage": (sum(coverages.values()) / len(coverages)
                          if coverages else 0.0),
    }


def chamfer_px(mask_a, mask_b):
    """Chamfer medio simétrico (px) entre dos máscaras binarias."""
    import cv2
    import numpy as np
    a = (np.asarray(mask_a) > 0).astype(np.uint8)
    b = (np.asarray(mask_b) > 0).astype(np.uint8)
    na, nb = int(a.sum()), int(b.sum())
    if na == 0 or nb == 0:
        return float("inf")
    dt_b = cv2.distanceTransform(1 - b, cv2.DIST_L2, 3)
    dt_a = cv2.distanceTransform(1 - a, cv2.DIST_L2, 3)
    return float((dt_b[a > 0].sum() + dt_a[b > 0].sum()) / (na + nb))
