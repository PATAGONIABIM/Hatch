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
    for (x1, y1, x2, y2) in member_segments:
        t1 = x1 * ux + y1 * uy
        t2 = x2 * ux + y2 * uy
        if t1 > t2:
            t1, t2 = t2, t1
        base_c = ((x1 + x2) / 2.0) * nx + ((y1 + y2) / 2.0) * ny
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
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
        entries.append({
            "angle": angle,
            "ox": ox, "oy": oy,
            "dx": window * ux, "dy": window * uy,
            "dashes": dashes,
            "cycle": window,
        })
    return entries


def compile_pat(segments, name="HATCHCRAFT", desc="",
                tile_w=None, tile_h=None,
                min_dash=0.3, min_gap=0.3,
                angle_tol=0.5, perp_tol=0.05):
    """Compila segmentos (x1,y1,x2,y2) en unidades Y-up a contenido PAT fiel.

    Cada línea portadora produce una entrada cuyo delta es paralelo a la
    dirección de la línea con magnitud igual al ciclo de dashes, de modo que
    la familia generada colapsa sobre una sola línea sin fantasmas ni
    desfases, y las copias envueltas garantizan continuidad entre tiles.
    """
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

    lines = [f"*{name}, {desc}", ";%TYPE=MODEL"]
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
