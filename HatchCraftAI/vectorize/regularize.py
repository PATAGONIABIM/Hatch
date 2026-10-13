"""F6 — Regularización de rectas (precisión arquitectónica).

Secuencia:
1. Direcciones dominantes: histograma de ángulos ponderado por longitud +
   mean-shift circular (mod 180°).
2. Snap angular: recta a <= snap_tol de una dirección dominante se rota sobre
   su centroide al ángulo exacto. Las curvas nunca se tocan (se filtran antes).
3. Colinealidad: merge_collinear sobre las líneas snapped.
4. Espaciado regular: offsets por dirección -> lattice offset = o0 + k*p (LS).
5. Uniones T/L/X: extremos a <= tol de la intersección exacta se mueven a ella.
"""
import math

import numpy as np

from .centerline import merge_collinear_segments


def _wrap180(a):
    """Diferencia de ángulos al intervalo [-90, 90)."""
    return (a + 90.0) % 180.0 - 90.0


def _seg_angle(seg):
    """Ángulo de un segmento 4-tupla en grados, mod 180."""
    return math.degrees(math.atan2(seg[3] - seg[1], seg[2] - seg[0])) % 180.0


def _seg_len(seg):
    return math.hypot(seg[2] - seg[0], seg[3] - seg[1])


def _circular_mean_deg(angles, weights):
    """Media circular mod 180 (truco del ángulo doblado)."""
    a2 = np.radians(np.asarray(angles, dtype=np.float64) * 2.0)
    w = np.asarray(weights, dtype=np.float64)
    s = float(np.sum(w * np.sin(a2)))
    c = float(np.sum(w * np.cos(a2)))
    if abs(s) < 1e-12 and abs(c) < 1e-12:
        return 0.0
    return (math.degrees(math.atan2(s, c)) / 2.0) % 180.0


def dominant_angles(segments, window_deg=3.0, min_weight_frac=0.05,
                    sigma_deg=2.5):
    """Direcciones dominantes de un conjunto de segmentos (grados, mod 180).

    Histograma de 180 bins de 1° ponderado por longitud, suavizado circular
    gaussiano, picos con separación >= 3 bins y peso >= min_weight_frac del
    total, refinados por mean-shift circular.
    """
    segs = [s for s in segments if _seg_len(s) >= 2.0]
    if not segs:
        return []
    angs = np.array([_seg_angle(s) for s in segs], dtype=np.float64)
    lens = np.array([_seg_len(s) for s in segs], dtype=np.float64)

    hist = np.zeros(180, dtype=np.float64)
    np.add.at(hist, np.floor(angs).astype(np.int64) % 180, lens)

    # Suavizado circular gaussiano.
    radius = max(1, int(round(3.0 * sigma_deg)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma_deg) ** 2)
    kernel /= kernel.sum()
    padded = np.concatenate([hist[-radius:], hist, hist[:radius]])
    smooth = np.convolve(padded, kernel, mode="valid")
    if len(smooth) != 180:
        smooth = smooth[:180]

    total_w = float(lens.sum())
    if total_w <= 0:
        return []
    # El suavizado diluye un pico puntual en kernel[radius] su fracción;
    # el umbral de candidatos se escala por eso y la validación final se
    # hace sobre el peso real (longitudes) del clúster refinado.
    kernel_center = float(kernel[len(kernel) // 2])
    min_peak = min_weight_frac * total_w * kernel_center

    order = np.argsort(smooth)[::-1]
    peaks = []
    for idx in order:
        if smooth[idx] < min_peak:
            break
        if all(abs(_wrap180(float(idx) - float(p))) >= 3.0 for p in peaks):
            peaks.append(float(idx))
    if not peaks:
        return []

    # Mean-shift circular por pico (mod 180) + validación de peso real.
    refined = []
    for peak in peaks:
        center = peak
        for _ in range(60):
            mask = np.abs(np.array([_wrap180(a - center) for a in angs])) <= window_deg
            if not mask.any():
                break
            new_c = _circular_mean_deg(angs[mask], lens[mask])
            if abs(_wrap180(new_c - center)) < 1e-9:
                center = new_c
                break
            center = new_c
        mask = np.abs(np.array([_wrap180(a - center) for a in angs])) <= window_deg
        cluster_w = float(lens[mask].sum())
        if cluster_w >= min_weight_frac * total_w:
            refined.append((center, cluster_w))

    # Fusionar refinados a < 2° (se queda el de mayor peso).
    refined.sort(key=lambda t: -t[1])
    merged = []
    for ang, w in refined:
        if all(abs(_wrap180(ang - m[0])) >= 2.0 for m in merged):
            merged.append((ang, w))
    return sorted(ang for ang, _ in merged)


def snap_angles(segments, angles, snap_tol):
    """Rota cada segmento a <= snap_tol de una dirección dominante.

    La rotación es sobre el centroide: el punto medio de la recta se
    preserva y los extremos se deslizan por la nueva línea infinita.
    """
    out = []
    n_snapped = 0
    if not angles:
        return [tuple(map(float, s)) for s in segments], 0
    for s in segments:
        x1, y1, x2, y2 = map(float, s)
        L = math.hypot(x2 - x1, y2 - y1)
        if L < 1e-9:
            out.append((x1, y1, x2, y2))
            continue
        a = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
        best_d, best_t = None, None
        for t in angles:
            d = abs(_wrap180(a - t))
            if best_d is None or d < best_d:
                best_d, best_t = d, t
        if best_d is not None and best_d <= snap_tol and best_d > 1e-9:
            t = best_t
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            u = (math.cos(math.radians(t)), math.sin(math.radians(t)))
            t0 = (x1 - cx) * u[0] + (y1 - cy) * u[1]
            t1 = (x2 - cx) * u[0] + (y2 - cy) * u[1]
            if t0 > t1:
                t0, t1 = t1, t0
            out.append((cx + u[0] * t0, cy + u[1] * t0,
                        cx + u[0] * t1, cy + u[1] * t1))
            n_snapped += 1
        else:
            out.append((x1, y1, x2, y2))
    return out, n_snapped


def _angle_groups(segments, tol_deg=0.01):
    """Agrupa segmentos por ángulo exacto (con tolerancia y wrap 179<->0)."""
    items = sorted(((_seg_angle(s), i) for i, s in enumerate(segments)),
                   key=lambda p: p[0])
    groups = []
    cur_rep, cur_idx = None, []
    for ang, i in items:
        if cur_rep is None or abs(_wrap180(ang - cur_rep)) <= tol_deg:
            if cur_rep is None:
                cur_rep = ang
            cur_idx.append(i)
        else:
            groups.append((cur_rep, cur_idx))
            cur_rep, cur_idx = ang, [i]
    if cur_rep is not None:
        groups.append((cur_rep, cur_idx))
    # Wrap: fusionar primer y último grupo si están a < tol por 179.999<->0.
    if len(groups) >= 2:
        rep0, idx0 = groups[0]
        repN, idxN = groups[-1]
        if abs(_wrap180(repN - rep0)) <= tol_deg:
            merged = _circular_mean_deg([rep0, repN],
                                        [len(idx0), len(idxN)])
            groups[0] = (merged, idx0 + idxN)
            groups.pop()
    return groups


def snap_lattice(segments, stroke_width, min_step=None, resid_tol=None):
    """Ajusta offsets perpendiculares a un lattice o0 + k*p por dirección.

    Solo grupos con >= 3 segmentos paralelos; el paso debe ser >= min_step
    y los residuos <= resid_tol (nunca colapsa juntas distintas).
    """
    sw = max(float(stroke_width), 1.0)
    if min_step is None:
        min_step = max(4.0, 0.75 * sw)
    if resid_tol is None:
        resid_tol = max(0.25, 0.3 * sw)
    out = [tuple(map(float, s)) for s in segments]
    n_adj = 0
    if len(out) < 3:
        return out, 0
    for rep, idxs in _angle_groups(out):
        if len(idxs) < 3:
            continue
        rep_rad = math.radians(rep)
        n = (-math.sin(rep_rad), math.cos(rep_rad))
        offs = np.array([
            (out[i][0] + out[i][2]) / 2.0 * n[0] +
            (out[i][1] + out[i][3]) / 2.0 * n[1]
            for i in idxs
        ], dtype=np.float64)
        diffs = np.diff(np.sort(offs))
        diffs = diffs[diffs > 1e-9]
        if diffs.size == 0:
            continue
        cands = [float(np.median(diffs))] + [float(d) for d in diffs]
        offs_sorted = np.sort(offs)
        best = None  # (score, p, o0, k_arr, resid_arr)
        for p_cand in cands:
            if p_cand < min_step:
                continue
            p = p_cand
            # o0 = min(offs): ancla k=0 en la primera línea. Con recuento
            # par y offsets simétricos, la mediana cae en fase SEMIENTERA y
            # el redondeo half-up asigna k inconsistentes con cualquier
            # ruido -> el LS diverge y el drift guard rechaza lattices
            # perfectamente válidos.
            o0 = float(offs_sorted[0])
            for _ in range(3):
                # floor(x+0.5+eps): redondeo half-up con epsilon (np.round
                # es bancario y el FP puede dejar el empate en 0.999...9).
                k = np.floor((offs - o0) / p + 0.5 + 1e-9)
                if np.ptp(k) < 1:
                    break
                A = np.column_stack([np.ones_like(k), k])
                sol, *_ = np.linalg.lstsq(A, offs, rcond=None)
                o0, p = float(sol[0]), abs(float(sol[1]))
            k = np.floor((offs - o0) / p + 0.5 + 1e-9)
            # El LS refina el paso; si deriva > 20% el candidato divergió.
            if np.ptp(k) < 1 or p < min_step or abs(p - p_cand) > 0.2 * p_cand:
                continue
            resid = offs - (o0 + k * p)
            # Gate por LÍNEA: TODOS los miembros deben caer a <= resid_tol de
            # su posición del lattice. Un gate solo por suma (RMS) tolera
            # outliers individuales y arrastra líneas reales varios px a un
            # lattice espurio de paso corto (p ~ sw).
            if np.abs(resid).max() > resid_tol:
                continue
            score = float(np.sum(resid ** 2))
            if best is None or score < best[0] - 1e-12 or (
                    abs(score - best[0]) <= 1e-12 and p > best[1]):
                best = (score, p, o0, k, resid)
        if best is None:
            continue
        score, p, o0, k, resid = best
        if score > resid_tol ** 2 * len(offs):
            continue
        for j, i in enumerate(idxs):
            target = o0 + k[j] * p
            dr = float(target - offs[j])
            if abs(dr) <= 1e-9:
                continue
            x1, y1, x2, y2 = out[i]
            out[i] = (x1 + dr * n[0], y1 + dr * n[1],
                      x2 + dr * n[0], y2 + dr * n[1])
            n_adj += 1
    return out, n_adj


def snap_junctions(segments, stroke_width, tol=None, min_cross_deg=20.0):
    """Mueve extremos cercanos a la intersección exacta de dos rectas.

    Unión T/L/X: para cada extremo de una línea i se busca la intersección
    con una recta j no paralela (>= min_cross_deg); si el punto cae dentro
    del extent de j (+/- tol) y a <= tol del extremo, este se mueve a la
    intersección. La intersección depende solo de las líneas infinitas
    (direcciones ya regularizadas), no de los extremos de i: los
    movimientos se computan en paralelo y son deterministas.
    """
    sw = max(float(stroke_width), 1.0)
    if tol is None:
        tol = 1.5 * sw
    n = len(segments)
    if n == 0:
        return [], 0
    info = []
    for s in segments:
        x1, y1, x2, y2 = map(float, s)
        L = math.hypot(x2 - x1, y2 - y1)
        if L < 1e-9:
            info.append(None)
            continue
        u = ((x2 - x1) / L, (y2 - y1) / L)
        ang = math.degrees(math.atan2(u[1], u[0]))
        info.append((x1, y1, u[0], u[1], L, ang))

    out = [tuple(map(float, s)) for s in segments]
    n_moved = 0
    for i in range(n):
        if info[i] is None:
            continue
        xi, yi, ux, uy, Li, angi = info[i]
        for e_idx in (0, 2):
            ex, ey = out[i][e_idx], out[i][e_idx + 1]
            best = None  # (d, P)
            for j in range(n):
                if j == i or info[j] is None:
                    continue
                xj, yj, vx, vy, Lj, angj = info[j]
                if abs(_wrap180(angi - angj)) < min_cross_deg:
                    continue
                # Intersección: xi + t*ui = pj + s*uj
                rx, ry = xj - xi, yj - yi
                denom = ux * vy - uy * vx
                if abs(denom) < 1e-12:
                    continue
                t = (rx * vy - ry * vx) / denom
                s = (rx * uy - ry * ux) / denom
                # s: parámetro sobre j desde su origen
                if s < -tol or s > Lj + tol:
                    continue
                P = (xi + t * ux, yi + t * uy)
                d = math.hypot(P[0] - ex, P[1] - ey)
                if d <= tol and d > 1e-9:
                    if best is None or d < best[0]:
                        best = (d, P)
            if best is not None:
                x1, y1, x2, y2 = out[i]
                if e_idx == 0:
                    out[i] = (best[1][0], best[1][1], x2, y2)
                else:
                    out[i] = (x1, y1, best[1][0], best[1][1])
                n_moved += 1
    return out, n_moved


def regularize_lines(segments, stroke_width, snap_tol=2.0, min_coverage=0.6):
    """Pipeline F6 completo sobre segmentos rectos (las curvas se excluyen).

    Devuelve (segments, info) donde info = {"dominant_deg", "n_snapped",
    "n_merged", "n_lattice", "n_junctions", "coverage"}.

    Gate de cobertura: el snap angular + lattice solo se aplican si las
    direcciones dominantes cubren >= min_coverage de la longitud total.
    En estructuras ORGÁNICAS (fotos irregulares, vetas) los "dominantes"
    son difusos y el snap solo añade error: orgánico sigue orgánico
    (el merge colineal y las uniones T/L/X sí se aplican siempre).
    """
    segs = [tuple(map(float, s)) for s in segments]
    info = {"dominant_deg": [], "n_snapped": 0, "n_merged": 0,
            "n_lattice": 0, "n_junctions": 0, "coverage": 0.0}
    if snap_tol is not None and snap_tol > 0 and segs:
        angles = dominant_angles(segs)
        info["dominant_deg"] = [round(a, 3) for a in angles]
        total = sum(_seg_len(s) for s in segs)
        cov = 0.0
        if total > 0 and angles:
            for s in segs:
                a = _seg_angle(s)
                d = min(abs(_wrap180(a - t)) for t in angles)
                if d <= snap_tol:
                    cov += _seg_len(s)
            cov /= total
        info["coverage"] = round(cov, 3)
        snapped = cov >= min_coverage
        if snapped:
            segs, info["n_snapped"] = snap_angles(segs, angles, snap_tol)
        sw = max(float(stroke_width), 1.0)
        merged = merge_collinear_segments(
            segs, sw=sw,
            perp_tol=max(1.5, 0.5 * sw), gap_tol=max(2.5, 0.75 * sw))
        info["n_merged"] = len(segs) - len(merged)
        segs = [tuple(map(float, s)) for s in merged]
        if snapped:
            segs, info["n_lattice"] = snap_lattice(segs, sw)
        segs, info["n_junctions"] = snap_junctions(segs, sw)
        merged = merge_collinear_segments(
            segs, sw=sw,
            perp_tol=max(1.5, 0.5 * sw), gap_tol=max(2.5, 0.75 * sw))
        info["n_merged"] += len(segs) - len(merged)
        segs = [tuple(map(float, s)) for s in merged]
    segs = [s for s in segs if _seg_len(s) >= 0.05]
    return segs, info
