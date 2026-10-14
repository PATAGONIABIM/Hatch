"""F8: discretización de primitivas en segmentos por sagita máxima.

Arcos → cuerdas con desviación ≤ sagita; Béziers → subdivisión adaptativa
por planitud. Presupuesto: si el total supera `budget` segmentos, se
degrada la sagita de las curvas por búsqueda binaria (las rectas nunca se
degradan).
"""

import math

import numpy as np


def arc_chord_count(radius, sweep, sagitta):
    """Nº de cuerdas para cubrir un arco con desviación ≤ sagitta."""
    s = min(max(sagitta, 1e-6), 0.75 * radius)
    ratio = 1.0 - s / radius
    theta_max = 2.0 * math.acos(max(-1.0, min(1.0, ratio)))
    if theta_max <= 1e-9:
        return max(2, int(math.ceil(abs(sweep) / 1e-9)))
    return max(2, int(math.ceil(abs(sweep) / theta_max)))


def arc_points(prim, sagitta):
    """Polilínea del arco/círculo con cuerdas por sagita."""
    cx, cy = prim["center"]
    r = prim["radius"]
    if prim["type"] == "circle":
        a0 = 0.0
        sweep = 2.0 * math.pi
    else:
        a0 = prim["a0"]
        sweep = prim["sweep"]
    n = arc_chord_count(r, sweep, sagitta)
    ang = a0 + np.linspace(0.0, sweep, n + 1)
    xs = cx + r * np.cos(ang)
    ys = cy + r * np.sin(ang)
    return np.stack([xs, ys], axis=1)


def _bezier_flat(ctrl, sagitta):
    """True si la Bézier es plana respecto a la cuerda C0→C3."""
    p0, p3 = ctrl[0], ctrl[3]
    d = p3 - p0
    L = float(np.hypot(*d))
    if L < 1e-9:
        return False
    n = d / L
    worst = 0.0
    for c in (ctrl[1], ctrl[2]):
        v = c - p0
        worst = max(worst, abs(float(n[0] * v[1] - n[1] * v[0])))
    return worst <= sagitta


def bezier_chords(ctrl, sagitta, out, depth=0):
    """Añade cuerdas de una Bézier cúbica con planitud ≤ sagitta."""
    ctrl = np.asarray(ctrl, dtype=np.float64)
    if depth >= 14 or _bezier_flat(ctrl, sagitta):
        out.append((ctrl[0], ctrl[-1]))
        return
    # de Casteljau en t=0.5
    p01 = (ctrl[0] + ctrl[1]) * 0.5
    p12 = (ctrl[1] + ctrl[2]) * 0.5
    p23 = (ctrl[2] + ctrl[3]) * 0.5
    p012 = (p01 + p12) * 0.5
    p123 = (p12 + p23) * 0.5
    mid = (p012 + p123) * 0.5
    left = np.vstack([ctrl[0], p01, p012, mid])
    right = np.vstack([mid, p123, p23, ctrl[3]])
    bezier_chords(left, sagitta, out, depth + 1)
    bezier_chords(right, sagitta, out, depth + 1)


def _prim_segments(prim, sagitta):
    """Segmentos (4-tuplas) e info de una primitiva con sagita dada."""
    kind = prim["type"]
    if kind == "line":
        return [(prim["p0"][0], prim["p0"][1], prim["p1"][0], prim["p1"][1])], 1
    if kind in ("arc", "circle"):
        pts = arc_points(prim, sagitta)
        segs = []
        for a, b in zip(pts[:-1], pts[1:]):
            segs.append((a[0], a[1], b[0], b[1]))
        return segs, len(segs)
    if kind == "bezier":
        ctrl = np.asarray(prim["ctrl"], dtype=np.float64).reshape(-1, 4, 2)
        chords = []
        for c in ctrl:
            bezier_chords(c, sagitta, chords)
        segs = []
        for a, b in chords:
            segs.append((a[0], a[1], b[0], b[1]))
        return segs, len(segs)
    return [], 0


def _count_all(prims, sagitta):
    total = 0
    for prim in prims:
        _, n = _prim_segments(prim, sagitta)
        total += n
    return total


def _is_curve(prim):
    return prim["type"] in ("arc", "circle", "bezier")


def _prim_length(prim):
    """Longitud (px) de la primitiva: líneas/arcos exactas, Bézier por
    polígono de control (cota superior suficiente para presupuesto)."""
    kind = prim["type"]
    if kind == "line":
        return math.hypot(prim["p1"][0] - prim["p0"][0],
                          prim["p1"][1] - prim["p0"][1])
    if kind == "arc":
        return abs(prim["radius"] * prim["sweep"])
    if kind == "circle":
        return 2.0 * math.pi * abs(prim["radius"])
    if kind == "bezier":
        ctrl = np.asarray(prim["ctrl"], dtype=np.float64).reshape(-1, 4, 2)
        return float(sum(math.hypot(c[0][0] - c[3][0], c[0][1] - c[3][1])
                         for c in ctrl))
    return 0.0


def primitives_to_segments(prims, sagitta, budget=None, with_types=False):
    """Discretiza primitivas en segmentos + flags is_chord.

    Devuelve (segments, is_chord, counts) o, con with_types=True,
    (segments, is_chord, counts, seg_types) donde seg_types es paralelo a
    segments con el tipo de la primitiva madre ("line" | "arc" |
    "bezier"). Si budget es dado y el total lo excede, la sagita de las
    CURVAS se degrada con búsqueda binaria del multiplicador ∈ [1, 64]
    (las rectas no se tocan).

    Fase 2.6: el presupuesto escala con la longitud total de las
    primitivas (400 es el suelo para imágenes típicas, no un techo
    global: una imagen orgánica larga no debe degradar la sagita ×64).
    counts lleva "sagitta_eff" (sagita realmente aplicada) y
    "budget_eff" para que la UI la muestre.
    """
    counts = {"line": 0, "arc": 0, "circle": 0, "bezier": 0}
    sagitta_eff = max(sagitta, 1e-6)
    has_curves = any(_is_curve(p) for p in prims)
    eff_budget = budget
    if budget is not None and has_curves:
        total_len = sum(_prim_length(p) for p in prims)
        # longitud media mínima forzada por el presupuesto: nunca peor
        # que max(25px, 100·sagita) de media por segmento
        per_seg = max(25.0, 100.0 * sagitta_eff)
        eff_budget = max(int(budget), int(math.ceil(total_len / per_seg)))
        base_count = _count_all(prims, sagitta_eff)
        if base_count > eff_budget:
            lo, hi = 1.0, 64.0
            for _ in range(12):
                mid = 0.5 * (lo + hi)
                if _count_all(prims, mid * sagitta_eff) <= eff_budget:
                    hi = mid
                else:
                    lo = mid
            sagitta_eff = hi * sagitta_eff
    counts["sagitta_eff"] = float(sagitta_eff)
    counts["budget_eff"] = int(eff_budget) if eff_budget is not None else None
    segments = []
    is_chord = []
    seg_types = []
    for prim in prims:
        segs, _ = _prim_segments(prim, sagitta_eff)
        curve = _is_curve(prim)
        kind = "bezier" if prim["type"] == "bezier" else (
            "line" if prim["type"] == "line" else "arc")
        counts[prim["type"]] = counts.get(prim["type"], 0) + 1
        for s in segs:
            length = math.hypot(s[2] - s[0], s[3] - s[1])
            if length < 0.05:
                continue
            segments.append(s)
            is_chord.append(curve)
            seg_types.append(kind)
    counts["chords"] = int(sum(is_chord))
    if with_types:
        return segments, is_chord, counts, seg_types
    return segments, is_chord, counts
