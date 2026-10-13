import math
from collections import defaultdict

import cv2
import numpy as np
import ezdxf

import pat_compiler
import pat_sim

from geometry import simplify_polyline_adaptive
from segment_cleaner import dedup_duplicate_segments


def render_pat_preview(pat_content, tile_count=3, preview_size=600, manual_scale=1.0):
    """Renderiza el PAT con el simulador fiel (semántica real de Revit)"""
    _, entries = pat_sim.parse_pat(pat_content)
    size = pat_sim.infer_tile_size(entries)
    return pat_sim.render_faithful(
        pat_content, size, size,
        tiles=tile_count, size=preview_size,
        manual_scale=manual_scale
    )


def render_dxf_debug(lines_data, min_x, min_y, tile_w, tile_h, preview_size=500):
    """Vista debug del DXF: segmentos detectados sobre el tile rectangular"""
    img = np.ones((preview_size, preview_size, 3), dtype=np.uint8) * 255
    tile_size = max(tile_w, tile_h)
    if not lines_data or tile_size == 0:
        return img

    # El tile debe tocar los bordes del canvas (sin margen perceptible).
    offset = 1
    scale = (preview_size - 2 * offset) / tile_size
    cx = offset + (preview_size - 2 * offset) / 2 - tile_w * scale / 2
    cy = offset + (preview_size - 2 * offset) / 2 + tile_h * scale / 2

    for x1, y1, x2, y2 in lines_data:
        nx1 = (x1 - min_x) * scale + cx
        ny1 = cy - (y1 - min_y) * scale
        nx2 = (x2 - min_x) * scale + cx
        ny2 = cy - (y2 - min_y) * scale
        cv2.line(img, (int(nx1), int(ny1)), (int(nx2), int(ny2)),
                 (60, 60, 60), 1, cv2.LINE_AA)

    x0 = int(cx)
    y0 = int(cy - tile_h * scale)
    x1r = int(cx + tile_w * scale)
    y1r = int(cy)
    cv2.rectangle(img, (x0, y0), (x1r, y1r), (150, 150, 150), 1)
    return img


class DXFtoPatConverter:
    """Convierte archivos DXF a PAT mediante explosión completa de entidades"""

    CURVE_TYPES = ('ARC', 'CIRCLE', 'ELLIPSE', 'SPLINE')

    def __init__(self):
        pass

    def _handle_simple(self, entity, segments, curves):
        t = entity.dxftype()
        if t == 'LINE':
            s, e = entity.dxf.start, entity.dxf.end
            segments.append((s.x, s.y, e.x, e.y))
        elif t in self.CURVE_TYPES:
            curves.append(entity)

    def _explode(self, entities, segments, curves, ignored):
        for ent in entities:
            t = ent.dxftype()
            try:
                if t == 'INSERT':
                    self._explode(ent.virtual_entities(), segments, curves, ignored)
                elif t in ('LWPOLYLINE', 'POLYLINE'):
                    subs = list(ent.virtual_entities())
                    for sub in subs:
                        st = sub.dxftype()
                        if st in ('LINE', 'ARC'):
                            self._handle_simple(sub, segments, curves)
                        elif st in self.CURVE_TYPES:
                            curves.append(sub)
                else:
                    self._handle_simple(ent, segments, curves)
            except Exception:
                ignored[t] += 1

    def _flatten_curves(self, curves, segments, tol):
        n = 0
        for ent in curves:
            try:
                pts = [p for p in ent.flattening(tol)]
                closed = ent.dxftype() in ('CIRCLE',)
                m = len(pts)
                for i in range(m - 1):
                    a, b = pts[i], pts[i + 1]
                    segments.append((a.x, a.y, b.x, b.y))
                    n += 1
                if closed and m > 2:
                    a, b = pts[-1], pts[0]
                    segments.append((a.x, a.y, b.x, b.y))
                    n += 1
            except Exception:
                pass
        return n

    def convert(self, dxf_file_path, pattern_name="DXF_Pattern", pattern_type="MODEL",
                unit="in"):
        """Lee un archivo DXF y genera un PAT fiel (sin cuantización de ángulo).

        unit: unidad de las coordenadas de origen del DXF ('cm' | 'in').
        Revit interpreta las coordenadas PAT en pulgadas, así que con 'cm'
        se escala todo por 1/2.54 para que 100 unidades = 100 cm reales.
        """
        try:
            doc = ezdxf.readfile(dxf_file_path)
            msp = doc.modelspace()

            segments = []
            curves = []
            ignored = defaultdict(int)
            self._explode(msp, segments, curves, ignored)

            if segments:
                xs = [v for s in segments for v in (s[0], s[2])]
                ys = [v for s in segments for v in (s[1], s[3])]
                diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
                curve_tol = max(diag * 0.001, 1e-4)
            else:
                curve_tol = 0.01

            num_curve_segs = self._flatten_curves(curves, segments, curve_tol)

            if not segments:
                msg = "No se encontraron entidades soportadas en el DXF"
                if ignored:
                    tipos = ", ".join(f"{k}×{v}" for k, v in sorted(ignored.items()))
                    msg += f" (ignoradas: {tipos})"
                return {"error": msg}

            min_x = min(v for s in segments for v in (s[0], s[2]))
            min_y = min(v for s in segments for v in (s[1], s[3]))
            norm = [(x1 - min_x, y1 - min_y, x2 - min_x, y2 - min_y)
                    for x1, y1, x2, y2 in segments]

            tile_w = max(v for s in norm for v in (s[0], s[2]))
            tile_h = max(v for s in norm for v in (s[1], s[3]))
            if tile_w <= 0 or tile_h <= 0:
                return {"error": "El dibujo tiene tamaño cero"}

            # Unidad del DXF de origen → pulgadas (Revit lee PAT en pulgadas).
            unit = "cm" if str(unit).lower() == "cm" else "in"
            src_w, src_h = tile_w, tile_h
            factor = (1.0 / 2.54) if unit == "cm" else 1.0
            if factor != 1.0:
                norm = [(x1 * factor, y1 * factor, x2 * factor, y2 * factor)
                        for x1, y1, x2, y2 in norm]
                tile_w *= factor
                tile_h *= factor

            # Emisión "una entrada por línea" (formato que Revit acepta),
            # con ángulos cuantizados a 15° y tile cuadrado = max(tile_w, tile_h).
            tile_size = max(tile_w, tile_h)
            # Fase 3: eliminar familias PAT duplicadas (offset ≡ mod tile),
            # que son las líneas superpuestas visibles en la simulación/Revit.
            norm, removed_dups = dedup_duplicate_segments(norm, tile_size)
            compiled = pat_compiler.compile_pat_perline(
                    norm, name=pattern_name, desc="Converted from AutoCAD DXF",
                    tile_size=tile_size,
                    quantize_angles=pat_compiler._quantize_angle_15,
                    coord_nd=6, dash_nd=6,
                    pattern_type=pattern_type
                )

            debug_img = render_dxf_debug(norm, 0.0, 0.0, tile_w, tile_h)
            preview = pat_sim.render_faithful(
                compiled['pat_content'], tile_size, tile_size, tiles=3, size=600
            )

            warnings = []
            if compiled['num_entries'] > 400:
                warnings.append(
                    f"⚠️ {compiled['num_entries']} entradas PAT: Revit puede ir lento (>400 recomendado)"
                )
            total_ignored = sum(ignored.values())
            if unit == "cm":
                tile_desc = (f"tile={src_w:.2f}×{src_h:.2f} cm "
                             f"= {tile_w:.2f}×{tile_h:.2f} in")
            else:
                tile_desc = f"tile={src_w:.2f}×{src_h:.2f} in"
            stats = (
                f"✅ DXF: {len(segments)} segmentos "
                f"({len(curves)} curvas → {num_curve_segs} tramos, tol={curve_tol:.4f}) → "
                f"{compiled['num_entries']} entradas PAT | {tile_desc}"
            )
            if ignored:
                tipos = ", ".join(f"{k}×{v}" for k, v in sorted(ignored.items()))
                stats += f" | ignoradas: {tipos}"
            if removed_dups:
                stats += f" | duplicados eliminados: {removed_dups}"

            return {
                "pat_content": compiled['pat_content'],
                "pat_preview": preview,
                "debug_img": debug_img,
                "stats": stats,
                "warnings": warnings,
                "tile_w": tile_w,
                "tile_h": tile_h,
                "num_entries": compiled['num_entries'],
                "ignored_entities": dict(ignored),
                "segments": len(segments),
                "segments_mm": [list(s) for s in norm],
            }

        except ezdxf.DXFError as e:
            return {"error": f"Error leyendo DXF: {str(e)}"}
        except Exception as e:
            return {"error": f"Error: {str(e)}"}


class ImageToPatConverter:
    """Convierte imágenes a PAT con pipeline CV avanzado.

    Novedades: auto-Canny (mediana±33%), filtro bilateral, black-hat,
    upscale automático para imágenes pequeñas,
    dedup por grosor de trazo vía transformada de distancia, detección de
    período por autocorrelación FFT y escala real en milímetros.
    Métodos de detección canónicos: hough, lsd, contour.
    """

    def __init__(self):
        pass

    @staticmethod
    def _merge_colinear(lines, angle_tol=5.0, dist_tol=3.0, gap_tol=10.0):
        """Une segmentos cercanos con ángulo similar para reducir fragmentación."""
        if not lines:
            return lines

        entries = []
        for (x1, y1, x2, y2) in lines:
            dx = x2 - x1
            dy = y2 - y1
            ang = math.degrees(math.atan2(dy, dx))
            if ang < 0:
                ang += 180
            if ang >= 180:
                ang -= 180
            ang_rad = math.radians(ang)
            perp = -x1 * math.sin(ang_rad) + y1 * math.cos(ang_rad)
            para1 = x1 * math.cos(ang_rad) + y1 * math.sin(ang_rad)
            para2 = x2 * math.cos(ang_rad) + y2 * math.sin(ang_rad)
            p_min, p_max = min(para1, para2), max(para1, para2)
            entries.append({
                'ang': ang, 'perp': perp,
                'p_min': p_min, 'p_max': p_max,
                'merged': False
            })

        merged_lines = []
        for i, a in enumerate(entries):
            if a['merged']:
                continue
            group_min = a['p_min']
            group_max = a['p_max']
            a['merged'] = True

            changed = True
            while changed:
                changed = False
                for j, b in enumerate(entries):
                    if b['merged']:
                        continue
                    ang_diff = abs(a['ang'] - b['ang'])
                    if ang_diff > angle_tol and (180 - ang_diff) > angle_tol:
                        continue
                    if abs(a['perp'] - b['perp']) > dist_tol:
                        continue
                    if b['p_min'] > group_max + gap_tol or b['p_max'] < group_min - gap_tol:
                        continue
                    group_min = min(group_min, b['p_min'])
                    group_max = max(group_max, b['p_max'])
                    b['merged'] = True
                    changed = True

            ang_rad = math.radians(a['ang'])
            cos_a, sin_a = math.cos(ang_rad), math.sin(ang_rad)
            mx1 = group_min * cos_a + a['perp'] * (-sin_a)
            my1 = group_min * sin_a + a['perp'] * cos_a
            mx2 = group_max * cos_a + a['perp'] * (-sin_a)
            my2 = group_max * sin_a + a['perp'] * cos_a
            merged_lines.append((mx1, my1, mx2, my2))

        return merged_lines

    @staticmethod
    def _point_to_line_dist(px, py, lx1, ly1, lx2, ly2):
        dx = lx2 - lx1
        dy = ly2 - ly1
        len_sq = dx * dx + dy * dy
        if len_sq < 1e-10:
            return math.hypot(px - lx1, py - ly1)
        return abs(dy * px - dx * py + lx2 * ly1 - ly2 * lx1) / math.sqrt(len_sq)

    @staticmethod
    def _segments_overlap(x1, y1, x2, y2, ux1, uy1, ux2, uy2):
        dx = x2 - x1
        dy = y2 - y1
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-10:
            return True
        ux, uy = dx / seg_len, dy / seg_len

        p1 = x1 * ux + y1 * uy
        p2 = x2 * ux + y2 * uy
        q1 = ux1 * ux + uy1 * uy
        q2 = ux2 * ux + uy2 * uy

        a_min, a_max = min(p1, p2), max(p1, p2)
        b_min, b_max = min(q1, q2), max(q1, q2)

        tol = seg_len * 0.2
        return a_min <= b_max + tol and b_min <= a_max + tol

    @staticmethod
    def _estimate_stroke_width(edges):
        """Grosor medio del trazo (px) vía transformada de distancia.

        La distancia de cada pixel de borde al fondo más cercano aproxima
        la mitad del grosor; la mediana × 2 da una estimación robusta.
        """
        inv = (edges == 0).astype(np.uint8)
        dt = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
        vals = dt[edges > 0]
        if vals.size == 0 or float(np.max(vals)) < 0.5:
            return 1.5
        return max(1.0, float(2.0 * np.median(vals)))

    @staticmethod
    def detect_period(gray, axis='x', lo_ratio=0.12, hi_ratio=0.95, min_peak=0.15):
        """Detecta el período del patrón (px) por autocorrelación FFT.

        axis='x' analiza columnas (variación horizontal), axis='y' filas.
        Devuelve None si no hay periodicidad clara.
        """
        prof = gray.mean(axis=(0 if axis == 'x' else 1)).astype(np.float64)
        prof -= prof.mean()
        n = len(prof)
        if n < 16:
            return None
        spec = np.fft.rfft(prof, n=2 * n)
        ac = np.fft.irfft(spec * np.conj(spec))[:n]
        if ac[0] <= 1e-12:
            return None
        ac /= ac[0]
        lag_lo = max(2, int(n * lo_ratio))
        lag_hi = min(n - 1, int(n * hi_ratio))
        if lag_hi <= lag_lo + 2:
            return None
        seg = ac[lag_lo:lag_hi]
        peak = int(np.argmax(seg)) + lag_lo
        if ac[peak] < min_peak:
            return None
        return float(peak)

    @staticmethod
    def _preprocess(gray, use_clahe=False, clahe_clip=2.0, use_blackhat=False,
                    blackhat_ksize=15, filter_mode="gaussian", blur_size=3,
                    use_adaptive=False, adaptive_block=11, adaptive_c=2,
                    use_auto_canny=False, canny_low=50, canny_high=150):
        """Preprocesado unificado: CLAHE → black-hat → blur → edges.

        Devuelve (blurred, edges). `blurred` alimenta LSD/Hough; `edges`
        alimenta Canny/contornos y el debug.
        """
        if use_clahe:
            clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

        if use_blackhat:
            k = max(3, int(blackhat_ksize))
            if k % 2 == 0:
                k += 1
            kernel_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_bh)
            gray = cv2.subtract(gray, bh)

        blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
        if filter_mode == "bilateral":
            blurred = cv2.bilateralFilter(gray, blur_size, 50, blur_size)
        elif filter_mode == "none":
            blurred = gray
        else:
            blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

        if use_adaptive:
            adaptive_block = adaptive_block if adaptive_block % 2 == 1 else adaptive_block + 1
            if adaptive_block < 3:
                adaptive_block = 3
            edges = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, adaptive_block, adaptive_c
            )
        elif use_auto_canny:
            med = float(np.median(blurred))
            low = max(0, int(0.66 * med))
            high = min(255, int(1.33 * med))
            edges = cv2.Canny(blurred, low, high)
        else:
            edges = cv2.Canny(blurred, canny_low, canny_high)

        return blurred, edges

    def convert(self, image_bytes, method="hough", canny_low=50, canny_high=150,
                blur_size=3, param1=20, param2=5,
                use_clahe=False, clahe_clip=2.0,
                use_adaptive=False, adaptive_block=11, adaptive_c=2,
                use_auto_canny=False,
                filter_mode="gaussian",
                use_blackhat=False, blackhat_ksize=15,
                use_skeleton=False,
                merge_segments=False, merge_angle_tol=5.0, merge_gap_tol=10.0,
                dedup_auto=True, dedup_k=2.5, dedup_threshold=8.0,
                offset_x=0.0, offset_y=0.0,
                max_resolution=600,
                tile_mm=100.0, min_dash_mm=0.3, min_gap_mm=0.3,
                pattern_type="MODEL", return_geometry=False,
                curves="fit", corner_angle=135.0, sagitta_mm=0.1,
                budget=400, snap_tol=2.0, periodic="auto",
                force_mode=None, auto=False):
        """Pipeline imagen → PAT con escala física en mm.

        tile_mm es el tamaño real del lado del tile cuadrado; los segmentos
        detectados se reescalan de px a mm antes de compilar el PAT.
        """
        try:
            import time
            timings = {}
            t0 = time.perf_counter()

            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return {"error": "Error al cargar la imagen"}

            h_orig, w_orig = img.shape[:2]
            side = min(h_orig, w_orig)
            start_x = (w_orig - side) // 2
            start_y = (h_orig - side) // 2
            img = img[start_y:start_y + side, start_x:start_x + side]

            # "precision" gestiona su propia resolución de trabajo en
            # vectorize.pipeline (escala sub-píxel); los métodos legacy
            # conservan su redimensionado histórico (baselines de oro).
            precision_mode = (method == "precision")

            upscaled = False
            if not precision_mode:
                if side < 200:
                    img = cv2.resize(img, (side * 2, side * 2),
                                     interpolation=cv2.INTER_CUBIC)
                    side *= 2
                    upscaled = True

                if side > max_resolution:
                    img = cv2.resize(img, (max_resolution, max_resolution),
                                     interpolation=cv2.INTER_AREA)
                    side = max_resolution

            if offset_x != 0.0 or offset_y != 0.0:
                shift_x = int(round(offset_x * side))
                shift_y = int(round(offset_y * side))
                img = np.roll(img, shift_x, axis=1)
                img = np.roll(img, shift_y, axis=0)

            timings['decode+resize'] = time.perf_counter() - t0
            t1 = time.perf_counter()

            raw_lines = []
            if precision_mode:
                from vectorize import pipeline as _precision
                t2 = time.perf_counter()
                if auto:
                    from vectorize import tuning as _tuning
                    _bp, prec, _scores = _tuning.auto_tune(
                        img, tile_mm=tile_mm, budget=budget)
                else:
                    prec = _precision.run(img, curves=curves,
                                          corner_angle=corner_angle,
                                          tile_mm=tile_mm,
                                          sagitta_mm=sagitta_mm, budget=budget,
                                          snap_tol=snap_tol,
                                          periodic=periodic,
                                          force_mode=force_mode)
                timings['detection'] = time.perf_counter() - t2
                t3 = time.perf_counter()
                if "error" in prec:
                    return {"error": prec["error"]}
                side = prec["side_px"]
                stroke_width = prec["stroke_width"]
                dedup_eff = dedup_threshold
                if dedup_auto:
                    dedup_eff = min(40.0, max(2.0, dedup_k * stroke_width))
                raw_lines = prec["segments_px"]
                edges = prec["mask"]
                gray = prec["gray"]
                debug_base = prec["debug_base"]
                debug_img = debug_base.copy()
                upscaled = prec["scale"] > 1
                precision_info = prec
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blurred, edges = self._preprocess(
                    gray, use_clahe=use_clahe, clahe_clip=clahe_clip,
                    use_blackhat=use_blackhat, blackhat_ksize=blackhat_ksize,
                    filter_mode=filter_mode, blur_size=blur_size,
                    use_adaptive=use_adaptive, adaptive_block=adaptive_block,
                    adaptive_c=adaptive_c, use_auto_canny=use_auto_canny,
                    canny_low=canny_low, canny_high=canny_high,
                )

                timings['preprocess'] = time.perf_counter() - t1
                t2 = time.perf_counter()

                stroke_width = self._estimate_stroke_width(edges)
                dedup_eff = dedup_threshold
                if dedup_auto:
                    dedup_eff = min(40.0, max(2.0, dedup_k * stroke_width))

                if use_skeleton:
                    try:
                        from skimage.morphology import skeletonize
                        skeleton_input = (edges > 0).astype(np.uint8)
                        skeleton = skeletonize(skeleton_input).astype(np.uint8) * 255
                        if cv2.countNonZero(skeleton) > 0:
                            edges = skeleton
                    except ImportError:
                        pass

                timings['skeleton'] = time.perf_counter() - t2
                t3 = time.perf_counter()

                debug_img = np.ones((side, side, 3), dtype=np.uint8) * 255
                edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                debug_edges = np.where(edges_color > 0,
                                       np.array([220, 220, 220], dtype=np.uint8),
                                       np.array([255, 255, 255], dtype=np.uint8))
                debug_img = debug_edges.astype(np.uint8)
                debug_base = debug_img.copy()

            if method == "hough":
                kernel = np.ones((2, 2), np.uint8)
                closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

                lines = cv2.HoughLinesP(closed_edges, 1, np.pi / 180, threshold=20,
                                        minLineLength=param1, maxLineGap=param2)
                if lines is not None:
                    for line in lines:
                        coords = np.asarray(line).ravel()
                        x1, y1, x2, y2 = coords[:4]
                        raw_lines.append((float(x1), float(y1),
                                          float(x2), float(y2)))

            elif method == "lsd":
                lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
                lines_lsd, widths, precs, nfas = lsd.detect(blurred)

                if lines_lsd is not None:
                    min_len = float(param1)
                    for line in lines_lsd:
                        coords = np.asarray(line).ravel()
                        x1, y1, x2, y2 = coords[:4]
                        seg_len = math.hypot(float(x2) - float(x1),
                                             float(y2) - float(y1))
                        if seg_len >= min_len:
                            raw_lines.append((float(x1), float(y1),
                                              float(x2), float(y2)))

            elif method == "precision":
                pass  # raw_lines ya viene del pipeline vectorize.precision
            else:
                kernel = np.ones((2, 2), np.uint8)
                dilated = cv2.dilate(edges, kernel, iterations=1)
                contours, _ = cv2.findContours(dilated, cv2.RETR_LIST,
                                               cv2.CHAIN_APPROX_SIMPLE)

                min_contour_len = param1
                epsilon_factor = float(param2)

                for cnt in contours:
                    arc_len = cv2.arcLength(cnt, False)
                    if arc_len < min_contour_len:
                        continue

                    # Simplificación adaptativa (RDP, tolerancia de cuerda
                    # absoluta): misma tolerancia que antes → mismo resultado,
                    # ahora enrutado por geometry.py (Fase 2, curvas orgánicas).
                    pts = simplify_polyline_adaptive(
                        cnt[:, 0, :], tol=epsilon_factor * arc_len)

                    for i in range(len(pts) - 1):
                        x1, y1 = pts[i]
                        x2, y2 = pts[i + 1]
                        raw_lines.append((float(x1), float(y1),
                                          float(x2), float(y2)))

            timings['detection'] = time.perf_counter() - t3
            t4 = time.perf_counter()

            if merge_segments and len(raw_lines) > 1:
                raw_lines = self._merge_colinear(
                    raw_lines,
                    angle_tol=merge_angle_tol,
                    gap_tol=merge_gap_tol
                )

            from collections import defaultdict as _dd
            angle_bucket_size = 5
            buckets = _dd(list)

            unique_lines = []
            kept_lines = []
            kept_types = []

            # Cuerdas de curvas (S2): nunca se eliminan por dedup (su perp
            # mutuo ≈ sagita < dedup_eff), pero sí entran a los buckets
            # como supresores. Solo válidos si no hubo merge intermedio.
            raw_flags = None
            if precision_mode and "is_chord" in precision_info:
                cand_flags = precision_info["is_chord"]
                if isinstance(cand_flags, (list, tuple)) and \
                        len(cand_flags) == len(raw_lines):
                    raw_flags = cand_flags

            # Tipos de primitiva (S6 UI): "line"|"arc"|"bezier" paralelo a
            # raw_lines; viaja intacto a través del dedup.
            raw_types = None
            if precision_mode and "seg_types" in precision_info:
                cand_types = precision_info["seg_types"]
                if isinstance(cand_types, (list, tuple)) and \
                        len(cand_types) == len(raw_lines):
                    raw_types = cand_types

            for line_idx, (x1, y1, x2, y2) in enumerate(raw_lines):
                ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
                if ang < 0:
                    ang += 180
                if ang >= 180:
                    ang -= 180
                bucket_id = int(ang / angle_bucket_size)
                mx = (x1 + x2) / 2.0
                my = (y1 + y2) / 2.0

                is_chord = bool(raw_flags is not None and raw_flags[line_idx])
                is_dup = False
                if not is_chord:
                    for bid in (bucket_id - 1, bucket_id, bucket_id + 1):
                        actual_bid = bid % (180 // angle_bucket_size)
                        for (ux1, uy1, ux2, uy2, umx, umy) in buckets.get(actual_bid, []):
                            if abs(mx - umx) > dedup_eff * 3 and abs(my - umy) > dedup_eff * 3:
                                continue
                            perp_dist = self._point_to_line_dist(mx, my, ux1, uy1, ux2, uy2)
                            if perp_dist < dedup_eff:
                                if self._segments_overlap(x1, y1, x2, y2, ux1, uy1, ux2, uy2):
                                    is_dup = True
                                    break
                        if is_dup:
                            break

                if is_dup:
                    continue

                buckets[bucket_id].append((x1, y1, x2, y2, mx, my))
                unique_lines.append((x1, y1, x2, y2))

                dx_raw = x2 - x1
                dy_raw = y2 - y1
                ang_raw = math.degrees(math.atan2(dy_raw, dx_raw))
                if ang_raw < 0:
                    ang_raw += 180

                if abs(ang_raw) < 15 or abs(ang_raw - 180) < 15:
                    line_color = (220, 50, 50)
                elif abs(ang_raw - 90) < 15:
                    line_color = (50, 50, 220)
                elif abs(ang_raw - 45) < 15:
                    line_color = (50, 180, 50)
                elif abs(ang_raw - 135) < 15:
                    line_color = (180, 50, 180)
                else:
                    line_color = (80, 80, 80)

                cv2.line(debug_img, (int(round(x1)), int(round(y1))),
                         (int(round(x2)), int(round(y2))),
                         line_color, 1, cv2.LINE_AA)

                kept_lines.append((x1, y1, x2, y2))
                if raw_types is not None:
                    kept_types.append(raw_types[line_idx])

            if not kept_lines:
                return {"error": "No se detectaron líneas válidas. Ajusta los parámetros."}

            # --- S6: renders para la vista de tipos y el heatmap de error ---
            debug_types = None
            debug_heat = None
            if raw_types is not None and len(kept_types) == len(kept_lines):
                type_colors = {
                    "line": (255, 70, 20),    # azul
                    "arc": (40, 180, 60),     # verde
                    "bezier": (40, 150, 255),  # naranja
                }
                debug_types = debug_base.copy()
                for (x1, y1, x2, y2), ty in zip(kept_lines, kept_types):
                    cv2.line(debug_types, (int(round(x1)), int(round(y1))),
                             (int(round(x2)), int(round(y2))),
                             type_colors.get(ty, (80, 80, 80)),
                             1, cv2.LINE_AA)
                # heatmap: desviación máxima de cada segmento respecto a la
                # tinta (distanceTransform sobre no-tinta)
                ink_mask = (edges > 0).astype(np.uint8) if edges is not None \
                    else None
                if ink_mask is not None and ink_mask.any():
                    dist = cv2.distanceTransform(
                        1 - ink_mask, cv2.DIST_L2, cv2.DIST_MASK_5)
                    h_img, w_img = dist.shape
                    sw_ref = max(stroke_width, 1.0)
                    heat = debug_base.copy()
                    for (x1, y1, x2, y2) in kept_lines:
                        n_samples = 16
                        ts = np.linspace(0.0, 1.0, n_samples)
                        xs = np.clip(x1 + (x2 - x1) * ts, 0, w_img - 1)
                        ys = np.clip(y1 + (y2 - y1) * ts, 0, h_img - 1)
                        ix = np.round(xs).astype(int)
                        iy = np.round(ys).astype(int)
                        err = float(dist[iy, ix].max())
                        t = min(1.0, err / (2.0 * sw_ref))
                        # verde (60,180,60) -> amarillo -> rojo (50,50,230)
                        if t < 0.5:
                            f = t / 0.5
                            color = (int(60 + (0 - 60) * f),
                                     int(180 + (20 * f)),
                                     int(60 + (170 * f)))
                        else:
                            f = (t - 0.5) / 0.5
                            color = (int(0 + 50 * f),
                                     int(200 - 150 * f),
                                     int(230 + 0 * f))
                        cv2.line(heat, (int(round(x1)), int(round(y1))),
                                 (int(round(x2)), int(round(y2))),
                                 color, 1, cv2.LINE_AA)
                    debug_heat = heat

            period_info = prec.get("period") if precision_mode else None

            if precision_mode:
                # F7: el tile puede ser el recorte de periodicidad (no
                # cuadrado); la escala física se ancla al lado ORIGINAL.
                work_w = prec.get("tile_w_px", side)
                work_h = prec.get("tile_h_px", side)
                orig_side = prec.get("orig_side_px", side)
                px_per_unit = orig_side / max(tile_mm, 1e-6)
                tile_w_mm = work_w / px_per_unit
                tile_h_mm = work_h / px_per_unit
            else:
                px_per_unit = side / max(tile_mm, 1e-6)
                tile_w_mm = tile_mm
                tile_h_mm = tile_mm
                work_h = side

            segs_units = []
            for x1, y1, x2, y2 in kept_lines:
                ux1 = x1 / px_per_unit
                uy1 = (work_h - y1) / px_per_unit
                ux2 = x2 / px_per_unit
                uy2 = (work_h - y2) / px_per_unit
                segs_units.append((ux1, uy1, ux2, uy2))

            # Normaliza al origen y emite "una entrada por línea" (formato que
            # Revit acepta): delta=(tile,tile), dash=length, gap=-(tile-length).
            # Para imagen, las coordenadas en mm ya están mapeadas dentro del tile [0, tile_mm]
            # No restar min_x/min_y para preservar la posición real y coincidir con el preview/lienzo
            segs_norm = list(segs_units)

            compiled = pat_compiler.compile_pat_perline(
                segs_norm, name="Image_Pattern",
                desc="HatchCraft precision pattern",
                tile_size=tile_w_mm,
                quantize_angles=None,
                coord_nd=5, dash_nd=5,
                pattern_type=pattern_type,
                tile_w=tile_w_mm, tile_h=tile_h_mm
            )

            timings['dedup+pat'] = time.perf_counter() - t4
            t5 = time.perf_counter()

            preview = pat_sim.render_faithful(
                compiled['pat_content'], tile_w_mm, tile_h_mm,
                tiles=3, size=600
            )

            timings['preview'] = time.perf_counter() - t5
            total_time = time.perf_counter() - t0

            if period_info is not None:
                period_x = period_info["cell_w"]
                period_y = period_info["cell_h"]
            else:
                period_x = self.detect_period(gray, axis='x')
                period_y = self.detect_period(gray, axis='y')

            extras = []
            if use_clahe:
                extras.append("CLAHE")
            if use_blackhat:
                extras.append("BlackHat")
            if filter_mode == "bilateral":
                extras.append("Bilateral")
            if use_auto_canny and not use_adaptive:
                extras.append("AutoCanny")
            if use_adaptive:
                extras.append("AdaptiveThresh")
            if use_skeleton:
                extras.append("Skeleton")
            if precision_mode:
                extras.append(
                    f"Precision({precision_info['mode']}, "
                    f"x{precision_info['scale']})"
                )
                cc = precision_info.get("curve_counts")
                if cc:
                    curve_parts = [f"{k}×{v}" for k, v in cc.items() if v]
                    if curve_parts:
                        extras.append("Curvas[" + ", ".join(curve_parts) + "]")
                ri = precision_info.get("regularization")
                if ri and (ri.get("n_snapped") or ri.get("n_lattice")
                           or ri.get("n_junctions")):
                    extras.append(
                        f"Reg[{ri.get('n_snapped', 0)}s, "
                        f"{ri.get('n_lattice', 0)}l, "
                        f"{ri.get('n_junctions', 0)}j]"
                    )
                if period_info is not None:
                    ptxt = f"Periodo[{period_info['nx']}×{period_info['ny']}"
                    if period_info.get("averaged"):
                        ptxt += ", avg"
                    if abs(tile_w_mm - tile_h_mm) > 1e-9:
                        ptxt += (f", {tile_w_mm:.1f}×{tile_h_mm:.1f}mm")
                    extras.append(ptxt + "]")
                if auto and precision_info.get("auto_n"):
                    extras.append(f"Auto[{precision_info['auto_n']} pruebas]")
            if merge_segments:
                extras.append(f"Merge({len(raw_lines)}→{len(kept_lines)})")
            extra_str = f" + {', '.join(extras)}" if extras else ""

            warnings = []
            if compiled['num_entries'] > 400:
                warnings.append(
                    f"⚠️ {compiled['num_entries']} entradas PAT: Revit puede ir lento (>400 recomendado)"
                )
            if upscaled:
                if precision_mode:
                    warnings.append(
                        f"Imagen ampliada x{precision_info['scale']} "
                        "para el análisis de precisión"
                    )
                else:
                    warnings.append("Imagen pequeña ampliada ×2 antes del análisis")

            timing_str = " | ".join(f"{k}:{v:.2f}s" for k, v in timings.items())
            stats = (
                f"✅ {method.upper()}{extra_str}: {len(kept_lines)} líneas → "
                f"{compiled['num_entries']} entradas PAT ({total_time:.1f}s) | "
                f"grosor≈{stroke_width:.1f}px dedup={dedup_eff:.1f}px | "
                f"tile={tile_mm:.0f}mm | {timing_str}"
            )

            # S6: análisis estructural para la UI (dominantes, período, auto)
            analysis = None
            if precision_mode:
                _an = {}
                _ri = precision_info.get("regularization") or {}
                if _ri.get("dominant_deg"):
                    _an["dominant_deg"] = _ri["dominant_deg"]
                if period_info is not None:
                    _an["period"] = {
                        "nx": period_info.get("nx"),
                        "ny": period_info.get("ny"),
                        "cell_w": period_info.get("cell_w"),
                        "cell_h": period_info.get("cell_h"),
                    }
                if precision_info.get("auto_params"):
                    _an["auto_params"] = precision_info["auto_params"]
                if _an:
                    analysis = _an

            result = {
                "pat_content": compiled['pat_content'],
                "pat_preview": preview,
                "debug_img": debug_img,
                "debug_base": debug_base,
                "stats": stats,
                "warnings": warnings,
                "tile_w": tile_w_mm,
                "tile_h": tile_h_mm,
                "num_entries": compiled['num_entries'],
                "stroke_width_px": stroke_width,
                "dedup_px": dedup_eff,
                "period_x_px": period_x,
                "period_y_px": period_y,
                "segments": len(kept_lines),
                "segments_mm": [list(s) for s in segs_norm],
            }
            if debug_types is not None:
                result["debug_types"] = debug_types
            if debug_heat is not None:
                result["debug_heat"] = debug_heat
            if analysis is not None:
                result["analysis"] = analysis

            if period_info is not None:
                period_out = dict(period_info)
                period_out["tile_w_mm"] = tile_w_mm
                period_out["tile_h_mm"] = tile_h_mm
                result["period"] = period_out

            if return_geometry:
                result["lines_px"] = list(kept_lines)
                result["source_bgr"] = img
                result["side_px"] = side
                result["px_per_unit"] = px_per_unit
                if precision_mode and "primitives" in precision_info:
                    result["primitives_px"] = precision_info["primitives"]

            return result

        except Exception as e:
            return {"error": f"Error: {str(e)}"}
