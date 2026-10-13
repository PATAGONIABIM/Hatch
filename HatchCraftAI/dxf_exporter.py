"""Exportador DXF bidireccional con preservación de color fuente.

Permite:
- write_dxf: volcar segmentos (lineal) a un DXF con color True Color por entidad.
- sample_segment_colors: muestrear el color fuente (tinta) de cada segmento
  sobre la imagen original, para que el DXF respete el color real del dibujo.
- image_bytes_to_dxf: vectorizar una imagen y exportarla directamente a DXF
  conservando el color de cada trazo. Con method="precision" exporta las
  primitivas reales (LINE / ARC / CIRCLE / SPLINE) sin discretizar.
"""

import io
import math

import cv2
import ezdxf
import numpy as np

from core_logic import ImageToPatConverter
from geometry import fit_arc

DEFAULT_LAYER = "0"


def write_dxf(segments, colors=None, out_bytes=True, layer=DEFAULT_LAYER,
              doc=None):
    """Crea un DXF con entidades LINE.

    segments: iterable de (x1, y1, x2, y2) en unidades de dibujo.
    colors: opcional, lista del mismo largo con (r, g, b) int o None por
            segmento. Si es None, todos heredan el color del layer.
    out_bytes: si True devuelve bytes UTF-8; si False devuelve el doc ezdxf.
    """
    if doc is None:
        doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    n = len(segments)
    for i, seg in enumerate(segments):
        x1, y1, x2, y2 = (float(v) for v in seg)
        e = msp.add_line((x1, y1), (x2, y2))
        e.dxf.layer = layer
        if colors is not None and i < n and colors[i] is not None:
            e.rgb = tuple(int(c) for c in colors[i])

    if out_bytes:
        bio = io.StringIO()
        doc.write(bio)
        return bio.getvalue().encode("utf-8")
    return doc


def chain_segments(segments, tol=1.0):
    """Encadena segmentos en polilíneas uniendo extremos cercanos (distancia <= tol).

    Devuelve lista de {"points": [(x, y), ...], "indices": [i, ...]}.
    """
    tol2 = tol * tol
    chains = []
    used = [False] * len(segments)
    for i, (x1, y1, x2, y2) in enumerate(segments):
        if used[i]:
            continue
        used[i] = True
        points = [(float(x1), float(y1)), (float(x2), float(y2))]
        indices = [i]
        extended = True
        while extended:
            extended = False
            hx, hy = points[0]
            tx, ty = points[-1]
            for j, (a1, b1, a2, b2) in enumerate(segments):
                if used[j]:
                    continue
                if (a2 - hx) ** 2 + (b2 - hy) ** 2 <= tol2:
                    points.insert(0, (float(a1), float(b1)))
                elif (a1 - hx) ** 2 + (b1 - hy) ** 2 <= tol2:
                    points.insert(0, (float(a2), float(b2)))
                elif (a1 - tx) ** 2 + (b1 - ty) ** 2 <= tol2:
                    points.append((float(a2), float(b2)))
                elif (a2 - tx) ** 2 + (b2 - ty) ** 2 <= tol2:
                    points.append((float(a1), float(b1)))
                else:
                    continue
                used[j] = True
                indices.append(j)
                extended = True
                break
        chains.append({"points": points, "indices": indices})
    return chains


def write_dxf_entities(entities, colors=None, out_bytes=True, layer=DEFAULT_LAYER, doc=None):
    """Vuelca entidades variadas a DXF.

    entities: lista de dicts con type "line" | "polyline" | "arc" | "circle" | "spline":
      {"type": "line", "p1": (x, y), "p2": (x, y)}
      {"type": "polyline", "points": [(x, y), ...]}
      {"type": "arc", "center": (cx, cy), "radius": r, "start_angle": a0, "end_angle": a1}
      {"type": "circle", "center": (cx, cy), "radius": r}
      {"type": "spline", "control_points": [(x, y), ...]}  (Bézier cúbica: 4 puntos
               de control + knots [0,0,0,0,1,1,1,1], grado 3)
    """
    if doc is None:
        doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    for i, ent in enumerate(entities):
        t = ent["type"]
        if t == "line":
            e = msp.add_line(ent["p1"], ent["p2"])
        elif t == "polyline":
            e = msp.add_lwpolyline(ent["points"])
        elif t == "arc":
            e = msp.add_arc(ent["center"], ent["radius"], ent["start_angle"], ent["end_angle"])
        elif t == "circle":
            e = msp.add_circle(ent["center"], ent["radius"])
        elif t == "spline":
            ctrl = [tuple(float(c) for c in p) for p in ent["control_points"]]
            if len(ctrl) >= 4:
                e = msp.add_spline([], degree=3)
                e.control_points = ctrl
                e.knots = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
            else:
                e = msp.add_line(ctrl[0], ctrl[-1])
        else:
            raise ValueError(f"Tipo de entidad desconocido: {t}")
        e.dxf.layer = layer
        if colors is not None and i < len(colors) and colors[i] is not None:
            e.rgb = tuple(int(c) for c in colors[i])
    if out_bytes:
        stream = io.StringIO()
        doc.write(stream)
        return stream.getvalue().encode("utf-8")
    return doc


def sample_segment_colors(bgr, lines_px, sample_n=16, neighborhood=1):
    """Muestrea el color fuente (tinta) de cada segmento sobre la imagen BGR.

    bgr: imagen color (HxWx3) en el MISMO espacio de coordenadas que lines_px.
    lines_px: lista de (x1, y1, x2, y2) en píxeles.
    Devuelve lista de (r, g, b) — un color por segmento. Se elige el píxel más
    oscuro del entorno del trazo (asume tinta más oscura que el fondo).
    """
    h, w = bgr.shape[:2]
    colors = []
    sigma = max(1, int(neighborhood))
    for (x1, y1, x2, y2) in lines_px:
        best_luma = None
        best_rgb = None
        for k in range(sample_n):
            t = (k + 0.5) / sample_n
            cx = int(round(x1 + (x2 - x1) * t))
            cy = int(round(y1 + (y2 - y1) * t))
            for dx in range(-sigma, sigma + 1):
                for dy in range(-sigma, sigma + 1):
                    px = cx + dx
                    py = cy + dy
                    if 0 <= px < w and 0 <= py < h:
                        b, g, r = bgr[py, px]
                        luma = 0.299 * int(r) + 0.587 * int(g) + 0.114 * int(b)
                        if best_luma is None or luma < best_luma:
                            best_luma = luma
                            best_rgb = (int(r), int(g), int(b))
        colors.append(best_rgb)
    return colors


def _prim_entities_precision(res, colors_all, px_per_unit, side):
    """Primitivas S2 (px imagen y-down) → entidades DXF (mm, y-up)."""
    prims = res["primitives_px"]
    source = res["source_bgr"]
    rep_segs = []
    for p in prims:
        t = p["type"]
        if t == "line":
            rep_segs.append((p["p0"][0], p["p0"][1], p["p1"][0], p["p1"][1]))
        elif t == "arc":
            cx, cy = p["center"]
            r = p["radius"]
            a_mid = p["a0"] + 0.5 * p["sweep"]
            rep_segs.append((cx + r * math.cos(p["a0"]),
                             cy + r * math.sin(p["a0"]),
                             cx + r * math.cos(a_mid),
                             cy + r * math.sin(a_mid)))
        elif t == "circle":
            cx, cy = p["center"]
            r = p["radius"]
            rep_segs.append((cx, cy, cx + r, cy))
        elif t == "bezier":
            ctrl = np.asarray(p["ctrl"], dtype=float).reshape(-1, 4, 2)
            rep_segs.append((ctrl[0][0][0], ctrl[0][0][1],
                             ctrl[0][3][0], ctrl[0][3][1]))
    rep_colors = sample_segment_colors(source, rep_segs) if rep_segs else []

    def _mm(pt):
        return (pt[0] / px_per_unit, (side - pt[1]) / px_per_unit)

    entities = []
    entity_colors = []
    counts = {"lines": 0, "arcs": 0, "circles": 0, "splines": 0}
    for i, p in enumerate(prims):
        t = p["type"]
        color = rep_colors[i] if i < len(rep_colors) else None
        if t == "line":
            entities.append({"type": "line", "p1": _mm(p["p0"]),
                             "p2": _mm(p["p1"])})
            counts["lines"] += 1
        elif t == "circle":
            entities.append({"type": "circle",
                             "center": _mm(p["center"]),
                             "radius": p["radius"] / px_per_unit})
            counts["circles"] += 1
        elif t == "arc":
            # imagen y-down CCW -> DXF y-up: θ_dxf = -θ_img; el arco CCW
            # dxf de -(a0+sweep) a -a0 cubre el mismo conjunto de puntos.
            a0 = p["a0"] % (2 * math.pi)
            sweep = p["sweep"] % (2 * math.pi)
            start_dxf = math.degrees((-(a0 + sweep)) % (2 * math.pi))
            end_dxf = math.degrees((-a0) % (2 * math.pi))
            entities.append({"type": "arc", "center": _mm(p["center"]),
                             "radius": p["radius"] / px_per_unit,
                             "start_angle": start_dxf,
                             "end_angle": end_dxf})
            counts["arcs"] += 1
        elif t == "bezier":
            ctrl = np.asarray(p["ctrl"], dtype=float).reshape(-1, 4, 2)
            for c in ctrl:
                entities.append({"type": "spline",
                                  "control_points": [_mm(pt) for pt in c]})
                counts["splines"] += 1
        entity_colors.append(color)
    return entities, entity_colors, counts


def image_bytes_to_dxf(image_bytes, method="hough", canny_low=50,
                       canny_high=150, blur_size=3, param1=20, param2=5,
                       tile_mm=100.0, max_resolution=600,
                       filter_mode="gaussian", use_skeleton=False,
                       fit_arcs=None, **kwargs):
    """Vectoriza una imagen y la exporta a DXF conservando colores fuente.

    Reutiliza ImageToPatConverter con return_geometry=True para obtener los
    segmentos en píxeles y la imagen color fuente; muestrea el color de cada
    trazo y los reescala a unidades mm antes de volcar el DXF.

    method="precision" exporta las primitivas reales (LINE/ARC/CIRCLE/SPLINE)
    directamente desde el pipeline S2, sin discretizar curvas.

    fit_arcs=None → activo para method="contour": encadena segmentos y ajusta
    arcos/círculos reales (fit_arc de geometry); lo que no ajusta sale como
    LWPOLYLINE o LINE. fit_arcs=False → todo LINE (comportamiento clásico).
    """
    conv = ImageToPatConverter()
    res = conv.convert(
        image_bytes, method=method, canny_low=canny_low, canny_high=canny_high,
        blur_size=blur_size, param1=param1, param2=param2,
        tile_mm=tile_mm, max_resolution=max_resolution,
        filter_mode=filter_mode, use_skeleton=use_skeleton,
        return_geometry=True, **kwargs,
    )
    if "error" in res:
        return res

    lines_px = res["lines_px"]
    source = res["source_bgr"]
    side = res["side_px"]
    px_per_unit = res["px_per_unit"]

    colors = sample_segment_colors(source, lines_px)

    if method == "precision" and "primitives_px" in res:
        entities, entity_colors, counts = _prim_entities_precision(
            res, colors, px_per_unit, side)
        dxf_bytes = write_dxf_entities(entities, colors=entity_colors,
                                        out_bytes=True)
        ent_stats = " | ".join(f"{k}: {v}" for k, v in counts.items() if v)
        return {
            "dxf_content": dxf_bytes,
            "segments": len(lines_px),
            "entities": counts,
            "stats": f'{res["stats"]} | entidades DXF ({ent_stats})',
            "warnings": res.get("warnings", []),
        }

    segments = []
    for (x1, y1, x2, y2) in lines_px:
        ux1 = x1 / px_per_unit
        uy1 = (side - y1) / px_per_unit
        ux2 = x2 / px_per_unit
        uy2 = (side - y2) / px_per_unit
        segments.append((ux1, uy1, ux2, uy2))

    if fit_arcs is None:
        fit_arcs = (method == "contour")

    if not fit_arcs:
        dxf_bytes = write_dxf(segments, colors=colors, out_bytes=True)
        return {
            "dxf_content": dxf_bytes,
            "segments": len(segments),
            "stats": res["stats"],
            "warnings": res.get("warnings", []),
        }

    chain_tol = 2.0 / px_per_unit
    arc_tol = 3.0 / px_per_unit
    chains = chain_segments(segments, tol=chain_tol)
    entities = []
    entity_colors = []
    counts = {"lines": 0, "polylines": 0, "arcs": 0, "circles": 0}
    for chain in chains:
        points = chain["points"]
        idx = chain["indices"]
        color = colors[idx[0]] if colors else None
        closed = (len(points) >= 4
                  and (points[0][0] - points[-1][0]) ** 2
                  + (points[0][1] - points[-1][1]) ** 2 <= chain_tol ** 2)
        fit_pts = points[:-1] if closed else points
        arc = None
        if len(fit_pts) >= 3:
            arc = fit_arc(np.array(fit_pts, dtype=float), tol=arc_tol)
        if arc is not None:
            if closed:
                entities.append({"type": "circle", "center": arc["center"],
                                 "radius": arc["radius"]})
                counts["circles"] += 1
            else:
                entities.append({"type": "arc", "center": arc["center"],
                                 "radius": arc["radius"],
                                 "start_angle": arc["start_angle"],
                                 "end_angle": arc["end_angle"]})
                counts["arcs"] += 1
        elif len(points) > 2:
            entities.append({"type": "polyline", "points": points})
            counts["polylines"] += 1
        else:
            entities.append({"type": "line", "p1": points[0], "p2": points[-1]})
            counts["lines"] += 1
        entity_colors.append(color)

    dxf_bytes = write_dxf_entities(entities, colors=entity_colors, out_bytes=True)
    ent_stats = " | ".join(f"{k}: {v}" for k, v in counts.items())
    return {
        "dxf_content": dxf_bytes,
        "segments": len(segments),
        "entities": counts,
        "stats": f'{res["stats"]} | entidades DXF ({ent_stats})',
        "warnings": res.get("warnings", []),
    }