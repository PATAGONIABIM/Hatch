import math

EPS = 1e-9


def fmt_num(v, nd=6):
    s = f"{float(v):.{nd}f}".rstrip("0").rstrip(".")
    if s in ("", "-", "-0"):
        return "0"
    return s


def norm_angle_180(ang):
    a = math.fmod(ang, 180.0)
    if a < 0:
        a += 180.0
    return a


def angle_dist_180(a, b):
    d = abs(a - b)
    return min(d, 180.0 - d)


def _circular_mean_180(pairs):
    sx = sy = 0.0
    for ang, w in pairs:
        r = math.radians(2.0 * ang)
        sx += math.cos(r) * w
        sy += math.sin(r) * w
    if abs(sx) < EPS and abs(sy) < EPS:
        return pairs[0][0]
    m = math.degrees(math.atan2(sy, sx)) / 2.0
    return norm_angle_180(m)


def group_collinear(segments, angle_tol=0.5, perp_tol=0.05):
    """Agrupa segmentos en líneas portadoras por ángulo y offset perpendicular."""
    step = max(angle_tol, 1e-6)
    n_buckets = max(1, int(round(180.0 / step)))
    items = []
    for (x1, y1, x2, y2) in segments:
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < EPS:
            continue
        ang = norm_angle_180(math.degrees(math.atan2(dy, dx)))
        rad = math.radians(ang)
        nx, ny = -math.sin(rad), math.cos(rad)
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        items.append({
            "ang": ang,
            "key": int(round(ang / step)) % n_buckets,
            "c": mx * nx + my * ny,
            "len": length,
            "p": (x1, y1, x2, y2),
        })

    used = [False] * len(items)
    result = []
    for i, it in enumerate(items):
        if used[i]:
            continue
        used[i] = True
        members = [it]
        weights = [it["len"]]
        for j in range(i + 1, len(items)):
            if used[j]:
                continue
            jt = items[j]
            kd = abs(it["key"] - jt["key"])
            if min(kd, n_buckets - kd) > 1:
                continue
            if angle_dist_180(it["ang"], jt["ang"]) > angle_tol:
                continue
            if abs(it["c"] - jt["c"]) > perp_tol:
                continue
            members.append(jt)
            weights.append(jt["len"])
            used[j] = True

        ang = _circular_mean_180([(m["ang"], w)
                                  for m, w in zip(members, weights)])
        rad = math.radians(ang)
        ux, uy = math.cos(rad), math.sin(rad)
        nx, ny = -math.sin(rad), math.cos(rad)
        c_sum = sum(w * ((m["p"][0] + m["p"][2]) / 2.0 * nx
                         + (m["p"][1] + m["p"][3]) / 2.0 * ny)
                    for m, w in zip(members, weights))
        c = c_sum / sum(weights)
        raw = [m["p"] for m in members]
        intervals = []
        for m in members:
            t1 = m["p"][0] * ux + m["p"][1] * uy
            t2 = m["p"][2] * ux + m["p"][3] * uy
            intervals.append((min(t1, t2), max(t1, t2)))
        result.append({"angle": ang, "c": c, "segments": raw,
                       "intervals": intervals})
    return result


def _merge_intervals(intervals, join_gap):
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for a, b in intervals[1:]:
        if a <= merged[-1][1] + join_gap:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def _rect_line_interval(c, ux, uy, nx, ny, w, h):
    """Intervalo paramétrico de rectángulo∩línea para una línea paralela."""
    lo, hi = -1e18, 1e18
    if abs(ux) > EPS:
        a = (0.0 - c * nx) / ux
        b = (w - c * nx) / ux
        lo = max(lo, min(a, b))
        hi = min(hi, max(a, b))
    elif not (-EPS <= c * nx <= w + EPS):
        return None
    if abs(uy) > EPS:
        a = (0.0 - c * ny) / uy
        b = (h - c * ny) / uy
        lo = max(lo, min(a, b))
        hi = min(hi, max(a, b))
    elif not (-EPS <= c * ny <= h + EPS):
        return None
    if hi - lo <= EPS:
        return None
    return (lo, hi)


def compile_carrier_entries(angle, member_segments, tile_w, tile_h,
                            min_dash, min_gap, line_tol=1e-4):
    """Genera entradas PAT para un grupo colineal.

    Enumera las copias de cada segmento bajo el retículo del tile y las
    agrupa por la línea paralela real en la que caen (offset perpendicular
    propio). Cada línea distinta produce una entrada con delta paralelo a la
    dirección y ventana igual a la intersección exacta línea∩tile,
    garantizando continuidad entre tiles sin líneas fantasma.
    """
    rad = math.radians(angle)
    ux, uy = math.cos(rad), math.sin(rad)
    nx, ny = -math.sin(rad), math.cos(rad)

    lines = []
    ext = tile_w * abs(nx) + tile_h * abs(ny)
    # The member segments are inside or immediately adjacent to the tile [0, tile_w] x [0, tile_h].
    # Periodic tiling shifts (da, db) only intersect the tile window for immediate neighbors (da, db in [-2, 2]).
    # When nx or ny is near zero (horizontal or vertical lines), ext / abs(tile * n) blows up to tens of thousands,
    # causing catastrophic O(N^2) lockups. Clamping da_span and db_span to 2 guarantees correctness in O(1) time.
    if abs(tile_w * nx) > EPS:
        da_span = min(2, max(1, int(math.ceil(ext / abs(tile_w * nx)))))
    else:
        da_span = 1
    if abs(tile_h * ny) > EPS:
        db_span = min(2, max(1, int(math.ceil(ext / abs(tile_h * ny)))))
    else:
        db_span = 1
    for (x1, y1, x2, y2) in member_segments:
        t1 = x1 * ux + y1 * uy
        t2 = x2 * ux + y2 * uy
        if t1 > t2:
            t1, t2 = t2, t1
        base_c = ((x1 + x2) / 2.0) * nx + ((y1 + y2) / 2.0) * ny
        for da in range(-da_span, da_span + 1):
            for db in range(-db_span, db_span + 1):
                sc = base_c + da * tile_w * nx + db * tile_h * ny
                target = None
                for ln in lines:
                    if abs(ln["c"] - sc) <= line_tol:
                        target = ln
                        break
                if target is None:
                    target = {"c": sc, "intervals": []}
                    lines.append(target)
                st1 = t1 + da * tile_w * ux + db * tile_h * uy
                st2 = t2 + da * tile_w * ux + db * tile_h * uy
                target["intervals"].append((min(st1, st2), max(st1, st2)))

    entries = []
    for ln in lines:
        win = _rect_line_interval(ln["c"], ux, uy, nx, ny, tile_w, tile_h)
        if win is None:
            continue
        s_lo, s_hi = win
        clipped = []
        for (a, b) in ln["intervals"]:
            lo = max(a, s_lo)
            hi = min(b, s_hi)
            if hi - lo > EPS:
                clipped.append((lo, hi))
        if not clipped:
            continue

        kept = [(a, b) for (a, b) in clipped if (b - a) >= min_dash]
        if not kept:
            kept = [max(clipped, key=lambda ab: ab[1] - ab[0])]
        merged = _merge_intervals(kept, min_gap)

        dashes = []
        cursor = s_lo
        for (a, b) in merged:
            gap = a - cursor
            if gap > EPS:
                dashes.append(-gap)
            dashes.append(b - a)
            cursor = b
        tail = s_hi - cursor
        if tail > EPS:
            dashes.append(-tail)
        if not any(d > 0 for d in dashes):
            continue

        window = s_hi - s_lo
        ox = s_lo * ux + ln["c"] * nx
        oy = s_lo * uy + ln["c"] * ny
        if abs(tile_w * nx) >= abs(tile_h * ny):
            fdx, fdy = tile_w, 0.0
        else:
            fdx, fdy = 0.0, tile_h
        entries.append({
            "angle": angle,
            "ox": ox, "oy": oy,
            "dx": fdx, "dy": fdy,
            "dashes": dashes,
            "cycle": window,
        })
    return entries


def compile_pat(segments, name="HATCHCRAFT", desc="",
                tile_w=None, tile_h=None,
                min_dash=0.3, min_gap=0.3,
                angle_tol=0.5, perp_tol=0.05,
                pattern_type="MODEL"):
    """Compila segmentos (x1,y1,x2,y2) en unidades Y-up a contenido PAT fiel.

    Cada línea portadora produce una entrada cuyo delta es paralelo a la
    dirección de la línea con magnitud igual al ciclo de dashes, de modo que
    la familia generada colapsa sobre una sola línea sin fantasmas ni
    desfases, y las copias envueltas garantizan continuidad entre tiles.

    pattern_type: "MODEL" (patrón de relleno de modelo, unidades reales con
    escala) o "DRAFTING" (patrón de relleno de diseño/detalle, escala fija de
    símbolo en Revit). Se emite como `;%TYPE=` en el header.
    """
    pattern_type = str(pattern_type).upper()
    if pattern_type not in ("MODEL", "DRAFTING"):
        raise ValueError(f"pattern_type inválido: {pattern_type!r} (usa MODEL o DRAFTING)")
    segs = [tuple(map(float, s)) for s in segments]
    if not segs:
        raise ValueError("No hay segmentos para compilar")

    if tile_w is None or tile_h is None:
        xs = [v for s in segs for v in (s[0], s[2])]
        ys = [v for s in segs for v in (s[1], s[3])]
        if tile_w is None:
            tile_w = max(xs) - min(xs)
        if tile_h is None:
            tile_h = max(ys) - min(ys)
    tile_w = float(tile_w)
    tile_h = float(tile_h)
    if tile_w < EPS or tile_h < EPS:
        raise ValueError("Tamaño de tile inválido")

    carriers = group_collinear(segs, angle_tol=angle_tol, perp_tol=perp_tol)
    entries = []
    for car in carriers:
        entries.extend(compile_carrier_entries(
            car["angle"], car["segments"],
            tile_w, tile_h, min_dash, min_gap
        ))

    if not entries:
        raise ValueError("Ninguna portadora produjo geometría válida")

    lines = [f"*{name}, {desc}", f";%TYPE={pattern_type}"]
    for e in entries:
        parts = [
            fmt_num(e["angle"]),
            fmt_num(e["ox"]), fmt_num(e["oy"]),
            fmt_num(e["dx"]), fmt_num(e["dy"]),
        ]
        parts.extend(fmt_num(d) for d in e["dashes"])
        lines.append(", ".join(parts))
    content = "\r\n".join(lines) + "\r\n"
    return {
        "pat_content": content,
        "entries": entries,
        "tile_w": tile_w,
        "tile_h": tile_h,
        "num_entries": len(entries),
    }


# Cuantización estándar de ángulos (DXF, pre-v2): múltiplos de 15°.
VALID_ANGLES_15 = list(range(0, 360, 15))


def _quantize_angle_15(deg):
    """Devuelve el múltiplo de 15° más cercano (con wrap 0..360)."""
    return min(VALID_ANGLES_15, key=lambda a: min(abs(deg - a), 360 - abs(deg - a)))


def get_pat_shift(ang, tile_size):
    """Calcula el vector de desplazamiento (dx, dy) según la periodicidad angular del tile.

    Garantiza que las líneas inclinadas (Chevron, diagonales, etc.) se repitan de forma
    uniforme y continua entre tiles contiguos sin dejar huecos vacíos en tablero de ajedrez.
    Para conservar simetría y compatibilidad con Revit, dx y dy se mantienen simétricos:
    - Ortogonales (cercanas a 0° o 90°): s = tile_size
    - Inclinadas / Diagonales (Chevron a 45°/135°): s = 0.7071067812 * tile_size
    """
    ang_norm = ang % 180.0
    dist_ortho = min(ang_norm, abs(ang_norm - 90.0), 180.0 - ang_norm)
    if dist_ortho < 15.0:
        return tile_size, tile_size
    s = 0.7071067812 * tile_size
    return s, s


def compile_pat_perline(segments, name, desc, tile_size, pattern_type="MODEL",
                        quantize_angles=None, coord_nd=6, dash_nd=6):
    """Emite un .pat en el formato "una entrada por línea" (el que Revit acepta).

    Reproduce fielmente la emisión pre-v2:
      - Una entrada PAT por cada segmento dibujado (NO agrupa en portadoras).
      - delta = (tile_size, tile_size)  → cada entrada dibuja una línea por tile
        (siguiente copia a un tile en ambas direcciones) ⇒ sin solapado.
      - dash = length; gap = -(tile_size - length), con mínimo -0.001
        (trazo casi continuo dentro del tile).
      - Ángulo en 0..180; si >= 180 se resta 180 y se intercambian puntos.
      - quantize_angles: callable opcional (p.ej. _quantize_angle_15) para el DXF.

    `segments` debe venir YA normalizado al origen (min_x,min_y restados) e Y-up.
    Devuelve el mismo dict que compile_pat (pat_content/entries/tile_w/tile_h/num_entries)
    para que render_faithful y parse_pat funcionen sin cambios.
    """
    if not segments:
        raise ValueError("no hay segmentos para compilar")

    pt = str(pattern_type).upper()
    if pt not in ("MODEL", "DRAFTING"):
        raise ValueError(f"pattern_type inválido: {pattern_type} (usa MODEL o DRAFTING)")

    tile = float(tile_size)
    if tile <= 0:
        raise ValueError("tile_size debe ser > 0")
    tile_r = round(tile, coord_nd)

    pat_lines = []
    entries = []
    skipped = 0
    for (x1, y1, x2, y2) in segments:
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 360.0
        if quantize_angles is not None:
            ang = float(quantize_angles(ang))
        if ang >= 180.0:
            ang -= 180.0
            x1, y1, x2, y2 = x2, y2, x1, y1
        ox = round(x1, coord_nd)
        oy = round(y1, coord_nd)
        length = math.hypot(x2 - x1, y2 - y1)
        dash = round(length, dash_nd)
        if dash <= 0:
            # Segmento degenerado (longitud ~0): un dash 0.0 es inválido en Revit.
            skipped += 1
            continue
        ang_q = round(ang, coord_nd) if quantize_angles is None else ang
        shift_x, shift_y = get_pat_shift(ang, tile)
        shift_xr = round(shift_x, coord_nd)
        shift_yr = round(shift_y, coord_nd)
        gap = round(-(shift_x - length), dash_nd)
        if gap >= 0:
            gap = -0.001
        pat_lines.append(f"{ang_q}, {ox},{oy}, {shift_xr},{shift_yr}, {dash},{gap}")
        entries.append({
            "angle": float(ang_q),
            "ox": float(ox),
            "oy": float(oy),
            "dx": float(shift_xr),
            "dy": float(shift_yr),
            "dashes": [float(dash), float(gap)],
        })

    if not pat_lines:
        raise ValueError("todos los segmentos son degenerados (longitud ~0)")

    lines = [f"*{name}, {desc}", f";%TYPE={pt}"] + pat_lines
    pat_content = "\r\n".join(lines) + "\r\n"
    return {
        "pat_content": pat_content,
        "entries": entries,
        "tile_w": tile,
        "tile_h": tile,
        "num_entries": len(entries),
        "skipped_degenerate": skipped,
    }
