"""Orquestador del método de precisión (S1 + S2 + S3 + S4 + S5).

run(img_bgr) -> dict con:
- segments_px: lista de (x1, y1, x2, y2) float en la resolución de trabajo
  (envueltos mod tile si se activó la periodicidad F7)
- side_px: lado de trabajo (px) para la conversión px -> mm
- tile_w_px / tile_h_px: dimensiones del tile de trabajo (celda F7 o imagen)
- orig_side_px: lado original de la imagen (ancla física px -> mm)
- stroke_width: grosor de trazo estimado (px de trabajo)
- mode: "centerline" | "region"
- scale: factor de escalado aplicado por el preproceso
- primitives / is_chord / curve_counts / sagitta_px: solo curves="fit"
  (centerline y región k-means)
- regularization: info F6 (direcciones dominantes, snaps, lattice, uniones)
- period: info F7 (lattice, celda, promediado) si se activó la periodicidad
- regions: info F3b (k-means Lab) si la región se resolvió por eje medio
- gray / mask / debug_base: para integración con core_logic y depuración
- "error": mensaje si el pipeline falla
"""

import math

import cv2
import numpy as np

from . import preprocess as _preprocess
from . import classify as _classify
from . import centerline as _centerline
from . import fitting as _fitting
from . import discretize as _discretize
from . import regularize as _regularize
from . import periodic as _periodic
from . import regions as _regions
from geometry import simplify_polyline_adaptive


def _region_segments(mask, sw, rdp_tol):
    """F3b básico (semilla): contornos de zonas rellenas -> cuerdas.
    El eje medial dual de calidad llega en S5."""
    from skimage import measure
    segs = []
    min_len = max(3.0, 0.75 * max(sw, 1.0))
    contours = measure.find_contours(mask.astype(np.float32) * 0.5, 0.5 - 1e-6)
    for c in contours:
        pts = np.asarray(c, dtype=np.float64)[:, ::-1]  # (row,col) -> (x,y)
        if len(pts) < 3:
            continue
        simplified = simplify_polyline_adaptive(pts, max(rdp_tol, 0.75))
        n = len(simplified)
        for i in range(n):
            a = simplified[i]
            b = simplified[(i + 1) % n]  # contornos cerrados
            seg = (float(a[0]), float(a[1]), float(b[0]), float(b[1]))
            if np.hypot(seg[2] - seg[0], seg[3] - seg[1]) >= min_len:
                segs.append(seg)
    return segs


def _region_joint_segments(img_bgr, work_size, sw_hint, rdp_tol, curves,
                           corner_angle, tile_mm, sagitta_mm, sagitta_px,
                           budget, snap_tol):
    """F3b: k-means Lab -> eje medio de juntas -> maquinaria centerline.

    work_size = (h, w) de trabajo del preproceso: la imagen de color se
    sube a esa resolución ANTES del k-means para que los segmentos
    salgan en px de trabajo. Devuelve None si la máscara de juntas es
    inservible (el llamador cae al camino de contornos).
    """
    th, tw = work_size
    if img_bgr.shape[0] != th or img_bgr.shape[1] != tw:
        img_bgr = cv2.resize(img_bgr, (tw, th),
                             interpolation=cv2.INTER_CUBIC)
    joint, rinfo = _regions.extract_joint_mask(img_bgr, ksize_hint=sw_hint)
    if joint is None:
        return None
    skel, sw_joint = _regions.joint_skeleton(joint)
    if not skel.any():
        return None
    chain_data, surgery = _centerline.extract_chains(
        skel, joint.astype(np.float32), sw_joint)
    if not chain_data:
        return None
    # Tolerancias de LÍNEA con el grosor del EJE (no de la banda): la red
    # de juntas es fina aunque el mortero sea grueso; con sw_joint en
    # min_len/perp/gap se fusionarían piezas vecinas de ángulos distintos.
    sw_line = max(2.0, 0.1 * sw_joint)
    segs, is_chord, prims, counts, s_px, reg_info, seg_types = \
        _centerline_segments(
            chain_data, sw_line, rdp_tol, curves, corner_angle, tile_mm,
            sagitta_mm, sagitta_px, budget, joint.shape[0], snap_tol)
    if not segs:
        return None
    return {
        "segments": segs, "is_chord": is_chord, "primitives": prims,
        "counts": counts, "s_px": s_px, "reg_info": reg_info,
        "seg_types": seg_types, "sw": sw_joint, "surgery": surgery,
        "regions": rinfo, "mask": joint, "num_chains": len(chain_data),
    }


def _prim_span(prim):
    """Longitud geodésica aproximada de una primitiva (px)."""
    kind = prim["type"]
    if kind == "line":
        return math.hypot(prim["p1"][0] - prim["p0"][0],
                          prim["p1"][1] - prim["p0"][1])
    if kind == "arc":
        return abs(prim["radius"] * prim["sweep"])
    if kind == "circle":
        return 2.0 * math.pi * abs(prim["radius"])
    if kind == "bezier":
        ctrl = prim["ctrl"]
        return sum(math.hypot(c[0][0] - c[3][0], c[0][1] - c[3][1])
                   for c in np.asarray(ctrl, dtype=np.float64).reshape(-1, 4, 2))
    return 0.0


def _centerline_segments(chain_data, sw, rdp_tol, curves, corner_angle,
                         tile_mm, sagitta_mm, sagitta_px, budget, height_px,
                         snap_tol):
    """Trazos finos -> (segments, is_chord, prims, counts, sagitta, reg_info,
    seg_types). seg_types es paralelo a segments ("line"|"arc"|"bezier")
    o None en el camino rdp."""
    if curves == "fit":
        # F5: clasificar cada cadena en primitivas
        line_prims = []
        curve_prims = []
        for cd in chain_data:
            for prim in _fitting.chain_primitives(
                    cd["coords"], sw, closed=cd.get("closed", False),
                    corner_angle=corner_angle):
                if prim.get("type") == "line":
                    line_prims.append(prim)
                else:
                    curve_prims.append(prim)
        # Curvas tan cortas como el mínimo de recta son ruido de unión:
        # sus cuerdas diminutas generan solapes y entradas PAT basura.
        min_span = max(3.0, 0.75 * max(sw, 1.0))
        curve_prims = [p for p in curve_prims if _prim_span(p) >= min_span]
        reg_info = None
        if snap_tol > 0 and line_prims:
            # F6: regularizar rectas ANTES de discretizar
            line_segs = [(float(p["p0"][0]), float(p["p0"][1]),
                          float(p["p1"][0]), float(p["p1"][1]))
                         for p in line_prims]
            line_segs, reg_info = _regularize.regularize_lines(
                line_segs, sw, snap_tol=snap_tol)
            min_len = max(3.0, 0.75 * max(sw, 1.0))
            line_segs = [s for s in line_segs
                         if math.hypot(s[2] - s[0], s[3] - s[1]) >= min_len]
            prims = [{"type": "line", "p0": (s[0], s[1]),
                      "p1": (s[2], s[3]), "rms": 0.0} for s in line_segs]
            prims += curve_prims
        else:
            prims = line_prims + curve_prims
        if sagitta_px is not None:
            s_px = float(sagitta_px)
        elif tile_mm:
            s_px = float(sagitta_mm) * height_px / float(tile_mm)
        else:
            s_px = max(1.0, 0.1 * sw)
        segs, is_chord, counts, seg_types = _discretize.primitives_to_segments(
            prims, s_px, budget=budget, with_types=True)
        # Cuerdas degeneradas (< min_span) vienen de filas de Bézier
        # microscópicas en esquinas de unión: generan solapes dobles y
        # entradas PAT basura. Las cuerdas REALES miden >= 2·sqrt(2·r·s).
        min_span = max(3.0, 0.75 * max(sw, 1.0))
        kept, kept_flags, kept_types = [], [], []
        for s, ch, ty in zip(segs, is_chord, seg_types):
            if ch and math.hypot(s[2] - s[0], s[3] - s[1]) < min_span:
                continue
            kept.append(s)
            kept_flags.append(ch)
            kept_types.append(ty)
        segs, is_chord, seg_types = kept, kept_flags, kept_types
        counts["chords"] = int(sum(kept_flags))
        if snap_tol > 0:
            # las rectas ya vienen regularizadas/fusionadas; solo agrupar
            order = [i for i, ch in enumerate(is_chord) if not ch] + \
                    [i for i, ch in enumerate(is_chord) if ch]
            segs = [segs[i] for i in order]
            seg_types = [seg_types[i] for i in order]
            is_chord = [False] * sum(1 for ch in kept_flags if not ch) + \
                       [True] * sum(1 for ch in kept_flags if ch)
        else:
            # S2 legacy: post-proceso SOLO sobre rectas (cuerdas no se tocan)
            lines = [s for s, ch in zip(segs, is_chord) if not ch]
            line_types = [t for t, ch in zip(seg_types, is_chord) if not ch]
            chords = [s for s, ch in zip(segs, is_chord) if ch]
            chord_types = [t for t, ch in zip(seg_types, is_chord) if ch]
            merged = _centerline.merge_collinear_segments(lines, sw=sw)
            min_len = max(3.0, 0.75 * max(sw, 1.0))
            merged = [s for s in merged
                      if math.hypot(s[2] - s[0], s[3] - s[1]) >= min_len]
            segs = merged + chords
            seg_types = ["line"] * len(merged) + chord_types
            is_chord = [False] * len(merged) + [True] * len(chords)
        return segs, is_chord, prims, counts, s_px, reg_info, seg_types
    segs = _centerline.chains_to_segments(chain_data, sw, rdp_tol=rdp_tol)
    segs = _centerline.merge_collinear_segments(segs, sw=sw)
    # filtro definitivo de trazos cortos, tras la curación de gaps
    min_len = max(3.0, 0.75 * max(sw, 1.0))
    segs = [s for s in segs
            if math.hypot(s[2] - s[0], s[3] - s[1]) >= min_len]
    return segs, None, None, None, None, None, None


def run(img_bgr, force_mode=None, rdp_tol=0.5, curves="fit",
        corner_angle=135.0, tile_mm=None, sagitta_mm=0.1, sagitta_px=None,
        budget=400, snap_tol=2.0, periodic="auto"):
    prep = _preprocess.preprocess(img_bgr)
    info = _classify.analyze_strokes(prep["mask"], force_mode=force_mode)
    if "error" in info:
        return {"error": info["error"]}
    h, w = prep["size"]
    sw = info["stroke_width"]
    ink = prep["ink"]
    mask = prep["mask"]
    gray = prep["gray"]

    # --- F7: periodicidad (solo modo centerline) ---
    # Activación conservadora: tinta que toca el borde de la imagen (el
    # contenido probablemente sigue más allá del encuadre) + lattice 2D
    # confiable + recorte exacto a un número entero de celdas.
    period = None
    orig_h, orig_w = h, w
    if periodic != "off" and info["mode"] == "centerline":
        border_ink = bool(mask[0:2, :].any() or mask[-2:, :].any()
                          or mask[:, 0:2].any() or mask[:, -2:].any())
        if border_ink:
            lat = _periodic.detect_lattice(mask, sw)
            if lat:
                plan = _periodic.plan_crop(w, h, lat["w"], lat["h"])
                if plan:
                    nx, ny, x0, y0, _cw, _ch = plan
                    cell_ink = _periodic.average_cell(
                        ink, x0, y0, lat["w"], lat["h"], nx, ny)
                    if cell_ink is not None:
                        cell_mask = (cell_ink >= 0.5).astype(np.uint8)
                        if int(cell_mask.sum()) > 0:
                            info2 = _classify.analyze_strokes(
                                cell_mask, force_mode="centerline")
                            if "error" not in info2:
                                ink = cell_ink
                                mask = cell_mask
                                ch_c, cw_c = cell_ink.shape[:2]
                                gray = gray[y0:y0 + ch_c, x0:x0 + cw_c]
                                info = info2
                                sw = info["stroke_width"]
                                # el tile del PAT es UNA celda del lattice
                                h, w = ch_c, cw_c
                                period = {
                                    "v1": lat["v1"], "v2": lat["v2"],
                                    "cell_w": lat["w"], "cell_h": lat["h"],
                                    "nx": nx, "ny": ny,
                                    "peak1": lat["peak1"],
                                    "peak2": lat["peak2"],
                                    "averaged": bool(nx * ny > 1),
                                    "tile_w_px": int(w),
                                    "tile_h_px": int(h),
                                    "cell_w_img": float(w) / float(prep["scale"]),
                                    "cell_h_img": float(h) / float(prep["scale"]),
                                }

    regions_info = None
    seg_types = None
    if info["mode"] == "centerline":
        chain_data, surgery = _centerline.extract_chains(
            info["skeleton"], ink, sw)
        if not chain_data:
            return {"error": "No se pudieron extraer cadenas del esqueleto."}
        segs, is_chord, prims, counts, s_px, reg_info, seg_types = \
            _centerline_segments(
                chain_data, sw, rdp_tol, curves, corner_angle, tile_mm,
                sagitta_mm, sagitta_px, budget, h, snap_tol)
        n_chains = len(chain_data)
        if period is not None:
            # F7-4: wrap mod tile para un PAT sin costuras
            segs, is_chord = _periodic.wrap_segments(
                segs, is_chord, w, h,
                min_piece=max(0.05, 0.1 * max(sw, 1.0)))
    else:
        # F3b: intentar resolver la región por eje medio de juntas
        # (k-means Lab); si no hay estructura utilizable, contornos.
        alt = None
        if img_bgr is not None and getattr(img_bgr, "ndim", 2) == 3:
            alt = _region_joint_segments(
                img_bgr, (h, w), sw, rdp_tol, curves, corner_angle, tile_mm,
                sagitta_mm, sagitta_px, budget, snap_tol)
        if alt is not None:
            segs = alt["segments"]
            is_chord = alt["is_chord"]
            prims = alt["primitives"]
            counts = alt["counts"]
            s_px = alt["s_px"]
            reg_info = alt["reg_info"]
            seg_types = alt.get("seg_types")
            sw = alt["sw"]
            mask = alt["mask"]
            n_chains = alt["num_chains"]
            regions_info = alt["regions"]
        else:
            segs = _region_segments(mask, sw, rdp_tol)
            n_chains = 0
            is_chord = None
            prims = None
            counts = None
            s_px = None
            reg_info = None
            regions_info = None

    if not segs:
        return {"error": "No se detectaron trazos válidos en la imagen."}

    debug_base = np.full((h, w, 3), 255, dtype=np.uint8)
    debug_base[mask > 0] = (215, 215, 215)

    result = {
        "segments_px": [(float(a), float(b), float(c), float(d))
                        for (a, b, c, d) in segs],
        "side_px": int(h),
        "orig_side_px": int(orig_h),
        "tile_w_px": int(w),
        "tile_h_px": int(h),
        "stroke_width": float(sw),
        "mode": info["mode"],
        "scale": int(prep["scale"]),
        "ink_fraction": float(info["ink_fraction"]),
        "width_cv": float(info["width_cv"]),
        "num_chains": int(n_chains),
        "surgery": surgery if info["mode"] == "centerline" else {},
        "mask": mask,
        "gray": gray,
        "debug_base": debug_base,
    }
    if period is not None:
        result["period"] = period
    if curves == "fit" and prims is not None:
        result["primitives"] = prims
        result["is_chord"] = is_chord
        result["curve_counts"] = counts
        result["sagitta_px"] = s_px
        if reg_info is not None:
            result["regularization"] = reg_info
        if seg_types is not None:
            result["seg_types"] = seg_types
    if regions_info is not None:
        result["regions"] = regions_info
    return result
