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


# ---------------------------------------------------------------------------
# 1.1  Entrada PAT "lattice-rational": familia exacta sobre el retículo
# ---------------------------------------------------------------------------
# Una familia PAT solo se repite EXACTAMENTE con el tile si su dirección es un
# vector del retículo (n·Tw, m·Th) con n, m coprimos. Se cuantiza el ángulo al
# racional más simple que respete una tolerancia de error en mm en los extremos
# y se calculan dx (paralelo) / dy (perpendicular) con Bézout para que cada
# tile reciba EXACTAMENTE una copia. La versión anterior (delta = tile·(1,1) o
# 0.7071·tile) solo era correcta en 0°/90° y producía ruido de 3×3 tiles en el
# resto de ángulos.

_LATTICE_CACHE = {}


def lattice_candidates(tw, th, n_max=64, ratio_max=60.0):
    """Vectores del retículo (n·Tw, m·Th) coprimos, ordenados por |L|."""
    key = (float(tw), float(th), int(n_max), float(ratio_max))
    cands = _LATTICE_CACHE.get(key)
    if cands is None:
        cands = []
        span = max(tw, th)
        for n in range(-int(n_max), int(n_max) + 1):
            for m in range(0, int(n_max) + 1):
                if (n == 0 and m == 0) or math.gcd(abs(n), m) != 1:
                    continue
                if m == 0 and n != 1:
                    continue
                Lx, Ly = n * tw, m * th
                L = math.hypot(Lx, Ly)
                if L / span > ratio_max:
                    continue
                cands.append((L, n, m, math.atan2(Ly, Lx)))
        cands.sort()
        _LATTICE_CACHE[key] = cands
    return cands


def lattice_direction(theta, length, tw, th, tol_mm=0.05,
                      n_max=64, ratio_max=60.0):
    """(n, m, L, ang) coprimos con error de extremo <= tol_mm y |L| mínimo."""
    for L, n, m, a in lattice_candidates(tw, th, n_max, ratio_max):
        if length * abs(math.sin(theta - a)) <= tol_mm:
            return n, m, L, a
    return None


def lattice_direction_best(theta, length, tw, th, n_max=64, ratio_max=60.0):
    """Dirección de retículo con menor error (sin imponer tolerancia)."""
    best = None
    for L, n, m, a in lattice_candidates(tw, th, n_max, ratio_max):
        err = length * abs(math.sin(theta - a))
        if best is None or err < best[0]:
            best = (err, n, m, L, a)
            if err <= 1e-12:
                break
    return best[1], best[2], best[3], best[4]


def lattice_candidates_within(theta, length, tw, th, tol_mm,
                              n_max=64, ratio_max=60.0, limit=8):
    """Primeras `limit` direcciones (por menor |L|) dentro de la tolerancia."""
    out = []
    for L, n, m, a in lattice_candidates(tw, th, n_max, ratio_max):
        if length * abs(math.sin(theta - a)) <= tol_mm:
            out.append((n, m, L, a))
            if len(out) >= limit:
                break
    return out


def lattice_entry(seg, tw, th, tol_mm=0.05, n_max=64, ratio_max=60.0,
                  start=None):
    """Entrada PAT cuantizada al retículo para un segmento.

    Rota el segmento alrededor de su punto medio (o lo ancla en `start`) al
    ángulo cuantizado y calcula dx/dy con Bézout. Devuelve None si ningún
    (n, m) cumple la tolerancia (el llamador debe partir el segmento).
    """
    x1, y1, x2, y2 = seg
    if y2 < y1 or (y2 == y1 and x2 < x1):
        x1, y1, x2, y2 = x2, y2, x1, y1
    length = math.hypot(x2 - x1, y2 - y1)
    theta = math.atan2(y2 - y1, x2 - x1)  # [0, pi)
    d = lattice_direction(theta, length, tw, th, tol_mm, n_max, ratio_max)
    if d is None:
        return None
    res = _lattice_place(x1, y1, x2, y2, length, d, tw, th, start=start)
    res["n"], res["m"] = d[0], d[1]
    res["err_mm"] = length * abs(math.sin(theta - d[3]))
    return res


def _lattice_place(x1, y1, x2, y2, length, d, tw, th, start=None):
    """Coloca un segmento en la dirección cuantizada `d`=(n,m,L,a)."""
    n, m, L, a = d
    ux, uy = math.cos(a), math.sin(a)
    if start is None:
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        ox, oy = mx - ux * length / 2.0, my - uy * length / 2.0
    else:
        ox, oy = start
    # Bézout: n·q − m·p = 1
    g, s, t = _egcd(n, m)          # n·s + m·t = g = ±1
    q, p = s * g, -t * g           # n·q − m·p = 1
    dy = tw * th / L
    dx = (p * n * tw * tw + q * m * th * th) / L
    dx = math.fmod(dx, L)
    dash = min(length, L)
    gap = -(L - dash)
    return {
        "angle": math.degrees(a),
        "ox": ox, "oy": oy,
        "dx": dx, "dy": dy,
        "dashes": [dash, gap] if gap < -1e-9 else [],
        "L": L,
        "end": (ox + ux * length, oy + uy * length),
    }


def _egcd(a, b):
    """MCD extendido: devuelve (g, x, y) con a·x + b·y = g."""
    if b == 0:
        return a, 1, 0
    g, x, y = _egcd(b, a % b)
    return g, y, x - (a // b) * y


def is_lattice_entry(entry, tw, th, tol=0.01, cycle_tol=2e-3):
    """True si la entrada PAT es una familia exacta sobre el retículo (Tw, Th).

    Comprueba que existe (n, m) coprimos con dy = Tw·Th/|L| y ángulo =
    atan2(m·Th, n·Tw) (hasta la redondeo del ángulo emitido), y que el ciclo
    de dashes es |L| (o la línea es continua). Lo usan los tests para validar
    el formato per-line.
    """
    ang = math.radians(entry["angle"] % 180.0)
    dy = abs(entry["dy"])
    if dy < 1e-9:
        return False
    L = tw * th / dy
    n, m = round(L * math.cos(ang) / tw), round(L * math.sin(ang) / th)
    if n == 0 and m == 0:
        return False
    if math.gcd(abs(n), m) != 1:
        return False
    # (n·Tw, m·Th) debe ser paralelo al ángulo emitido: error medido en mm
    # sobre |L| (tolera el redondeo del ángulo en el .pat).
    nx, ny = n * tw, m * th
    nl = math.hypot(nx, ny)
    if nl < 1e-12:
        return False
    if abs(math.cos(ang) * ny - math.sin(ang) * nx) / nl * L > tol:
        return False
    if abs(entry["dx"]) > L + 1e-6:
        return False
    dashes = entry["dashes"]
    if dashes:
        cycle = sum(abs(d) for d in dashes)
        # Tol. relativa: el .pat redondea dy/dashes (5-6 decimales) y con
        # |L| grandes eso mueve el ciclo unas pocas milésimas de mm.
        if abs(cycle - L) > max(cycle_tol, 1e-5 * L):
            return False
    return True


def _pick_for_anchor(anchor, x1, y1, x2, y2, length, cands):
    """Candidato cuya pieza colocada minimiza la deriva del punto final."""
    best = None
    for cand in cands:
        a = cand[3]
        nx = anchor[0] + math.cos(a) * length
        ny = anchor[1] + math.sin(a) * length
        err = math.hypot(nx - x2, ny - y2)
        if best is None or err < best[0]:
            best = (err, cand)
    return best[1]


def compile_pat_perline(segments, name, desc, tile_size, pattern_type="MODEL",
                        quantize_angles=None, coord_nd=6, dash_nd=6,
                        tile_w=None, tile_h=None, angle_tol_mm=None,
                        chain_ids=None, n_max=64, ratio_max=60.0,
                        max_split_depth=20):
    """Emite un .pat en el formato "una entrada por línea" (el que Revit acepta).

    Reglas de emisión:
      - Una entrada PAT por cada segmento dibujado (NO agrupa en portadoras).
      - Ángulo en 0..180; si >= 180 se resta 180 y se intercambian puntos.
      - quantize_angles: callable opcional (p.ej. _quantize_angle_15) para el DXF.
      - Direcciones EXACTAMENTE ortogonales (0°/90° tras el retículo): se
        conserva la emisión histórica (delta = _shift_for(ángulo)).
      - Cualquier otra dirección se cuantiza a un vector del retículo
        (n·Tw, m·Th) coprimo cuyo error en los extremos no supere
        `angle_tol_mm` (default 0.05 mm); dx/dy salen de Bézout. Si ningún
        (n, m) cumple la tolerancia, el segmento se parte por la mitad y se
        reintenta recursivamente (`max_split_depth`).
      - len >= |L| → trazo continuo (dashes = []).
      - chain_ids: opcional, lista paralela a `segments`. Los segmentos
        consecutivos con el mismo id forman una cadena: el inicio de cada
        pieza se ancla al final de la anterior (sin huecos en los vértices)
        y se elige la dirección de retículo que minimiza la deriva.

    `segments` debe venir YA normalizado al origen (min_x,min_y restados) e Y-up.
    Devuelve el mismo dict que compile_pat (pat_content/entries/tile_w/tile_h/
    num_entries) para que render_faithful y parse_pat funcionen sin cambios.
    """
    if not segments:
        raise ValueError("no hay segmentos para compilar")

    pt = str(pattern_type).upper()
    if pt not in ("MODEL", "DRAFTING"):
        raise ValueError(f"pattern_type inválido: {pattern_type} (usa MODEL o DRAFTING)")

    tile = float(tile_size)
    if tile <= 0:
        raise ValueError("tile_size debe ser > 0")

    tw = float(tile_w) if tile_w is not None else tile
    th = float(tile_h) if tile_h is not None else tile
    non_square = abs(tw - th) > 1e-9
    tol = 0.05 if angle_tol_mm is None else float(angle_tol_mm)

    if chain_ids is not None and len(chain_ids) != len(segments):
        raise ValueError("chain_ids debe tener la misma longitud que segments")

    def _shift_for(ang_deg):
        """Delta de la familia para direcciones ortogonales exactas.

        Conserva el comportamiento histórico: con tile cuadrado siempre es
        (tile, tile); con tile rectangular usa el período/espaciado del eje
        correspondiente.
        """
        ang_n = ang_deg % 180.0
        if ang_n < 45.0 or ang_n > 135.0:
            # horizontal-ish: período paralelo = tile_w, espaciado perp = tile_h
            return tw, th
        if not non_square:
            return tile, tile
        # vertical-ish: período paralelo = tile_h, espaciado perp = -tile_w
        return th, -tw

    pat_lines = []
    entries = []
    skipped = 0

    def _emit(angle_deg, ox, oy, dash, gap, shift_x, shift_y, extra=None):
        ang_q = round(angle_deg, coord_nd)
        oxr = round(ox, coord_nd)
        oyr = round(oy, coord_nd)
        sxr = round(shift_x, coord_nd)
        syr = round(shift_y, coord_nd)
        dash_r = round(dash, dash_nd)
        gap_r = round(gap, dash_nd)
        if gap_r >= 0:
            gap_r = -0.001
        pat_lines.append(f"{ang_q}, {oxr},{oyr}, {sxr},{syr}, {dash_r},{gap_r}")
        entry = {
            "angle": float(ang_q),
            "ox": float(oxr),
            "oy": float(oyr),
            "dx": float(sxr),
            "dy": float(syr),
            "dashes": [float(dash_r), float(gap_r)],
        }
        if extra:
            entry.update(extra)
        entries.append(entry)

    # --- normalización y agrupación en cadenas (Fase 1.2) ------------------
    norm = []
    for idx, (x1, y1, x2, y2) in enumerate(segments):
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 360.0
        if quantize_angles is not None:
            ang = float(quantize_angles(ang))
        if ang >= 180.0:
            ang -= 180.0
            x1, y1, x2, y2 = x2, y2, x1, y1
        length = math.hypot(x2 - x1, y2 - y1)
        if round(length, dash_nd) <= 0:
            # Segmento degenerado (longitud ~0): un dash 0.0 es inválido en Revit.
            skipped += 1
            continue
        cid = chain_ids[idx] if chain_ids is not None else idx
        norm.append((x1, y1, x2, y2, ang, length, cid))

    chains = []
    for item in norm:
        if chains and chains[-1][0] == item[6]:
            chains[-1][1].append(item)
        else:
            chains.append((item[6], [item]))

    for _cid, pieces in chains:
        anchor = None
        for (x1, y1, x2, y2, ang, length, _c) in pieces:
            theta = math.radians(ang)
            pend = [(x1, y1, x2, y2, theta, length, 0)]
            while pend:
                px1, py1, px2, py2, ptheta, plen, pdepth = pend.pop()
                cands = lattice_candidates_within(
                    ptheta, plen, tw, th, tol, n_max, ratio_max)
                if not cands:
                    if pdepth < max_split_depth:
                        mx, my = (px1 + px2) / 2.0, (py1 + py2) / 2.0
                        pend.append((mx, my, px2, py2, ptheta,
                                     plen / 2.0, pdepth + 1))
                        pend.append((px1, py1, mx, my, ptheta,
                                     plen / 2.0, pdepth + 1))
                        continue
                    cands = [lattice_direction_best(ptheta, plen, tw, th,
                                                    n_max, ratio_max)]
                if anchor is None:
                    sel = cands[0]
                else:
                    sel = _pick_for_anchor(anchor, px1, py1, px2, py2,
                                           plen, cands)
                place = _lattice_place(px1, py1, px2, py2, plen, sel,
                                       tw, th, start=anchor)
                anchor = place["end"]
                n, m, L = sel[0], sel[1], sel[2]
                if (n, m) in ((1, 0), (0, 1)):
                    # Ortodo exacta: emisión histórica (byte a byte).
                    ang_deg = 0.0 if (n, m) == (1, 0) else 90.0
                    sx, sy = _shift_for(ang_deg)
                    dash = min(plen, sx)
                    gap = -(sx - dash)
                    _emit(ang_deg, place["ox"], place["oy"], dash, gap,
                          sx, sy, extra={"n": n, "m": m,
                                         "err_mm": plen * abs(math.sin(ptheta - sel[3]))})
                    continue
                dash = min(plen, L)
                gap = -(L - dash)
                extra = {"n": n, "m": m,
                         "err_mm": plen * abs(math.sin(ptheta - sel[3]))}
                if gap < -1e-9:
                    _emit(place["angle"], place["ox"], place["oy"], dash, gap,
                          place["dx"], place["dy"], extra=extra)
                else:
                    # len >= |L|: la familia de copias cubre el portador
                    # entero → línea continua, sin dashes.
                    ang_q = round(place["angle"], coord_nd)
                    oxr = round(place["ox"], coord_nd)
                    oyr = round(place["oy"], coord_nd)
                    dxr = round(place["dx"], coord_nd)
                    dyr = round(place["dy"], coord_nd)
                    pat_lines.append(f"{ang_q}, {oxr},{oyr}, {dxr},{dyr}")
                    entry = {
                        "angle": float(ang_q),
                        "ox": float(oxr),
                        "oy": float(oyr),
                        "dx": float(dxr),
                        "dy": float(dyr),
                        "dashes": [],
                    }
                    entry.update(extra)
                    entries.append(entry)

    if not pat_lines:
        raise ValueError("todos los segmentos son degenerados (longitud ~0)")

    lines = [f"*{name}, {desc}", f";%TYPE={pt}"] + pat_lines
    pat_content = "\r\n".join(lines) + "\r\n"
    return {
        "pat_content": pat_content,
        "entries": entries,
        "tile_w": tw,
        "tile_h": th,
        "num_entries": len(entries),
        "skipped_degenerate": skipped,
    }
