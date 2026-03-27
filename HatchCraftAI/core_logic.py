import cv2
import numpy as np
import math
import ezdxf

def render_pat_preview(pat_content, tile_count=3, preview_size=600, manual_scale=1.0):
    """Renderiza el patrón PAT como lo vería Revit"""
    img = np.ones((preview_size, preview_size, 3), dtype=np.uint8) * 255
    
    lines_data = pat_content.strip().replace('\r\n', '\n').split('\n')
    
    # Encontrar límites y segmentos
    tile_size = 1.0
    segments = []
    
    for line in lines_data:
        line = line.strip()
        if not line or line.startswith('*') or line.startswith(';'):
            continue
        if ';' in line:
            line = line.split(';')[0].strip()
        
        try:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 5:
                continue
                
            angle = float(parts[0])
            ox, oy = float(parts[1]), float(parts[2])
            dx, dy = float(parts[3]), float(parts[4])
            
            dash_pattern = []
            if len(parts) > 5:
                dash_pattern = [float(p) for p in parts[5:] if p.strip()]
            
            tile_size = max(tile_size, dx, dy)
            
            segments.append({
                'ox': ox, 'oy': oy,
                'dx': dx, 'dy': dy,
                'angle': angle,
                'dash_pattern': dash_pattern
            })
        except:
            continue
    
    if not segments:
        return img
    
    # Escala basada en tile_size y manual_scale
    pattern_size = tile_size * tile_count
    scale = (preview_size / pattern_size) * manual_scale
    
    # Dibujar cada segmento para cada tile
    for seg in segments:
        ang_rad = math.radians(seg['angle'])
        dir_x = math.cos(ang_rad)
        dir_y = math.sin(ang_rad)
        
        for tile_x in range(tile_count):
            for tile_y in range(tile_count):
                base_x = (seg['ox'] + tile_x * seg['dx']) * scale
                base_y = preview_size - (seg['oy'] + tile_y * seg['dy']) * scale
                
                if seg['dash_pattern']:
                    pos = 0
                    for dash_val in seg['dash_pattern']:
                        length = abs(dash_val) * scale
                        if dash_val > 0:
                            x1 = int(base_x + dir_x * pos)
                            y1 = int(base_y - dir_y * pos)
                            x2 = int(base_x + dir_x * (pos + length))
                            y2 = int(base_y - dir_y * (pos + length))
                            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 1, cv2.LINE_AA)
                        pos += length
                else:
                    length = tile_size * scale * 0.5
                    x1 = int(base_x)
                    y1 = int(base_y)
                    x2 = int(base_x + dir_x * length)
                    y2 = int(base_y - dir_y * length)
                    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 1, cv2.LINE_AA)
    
    # Grid
    tile_px = preview_size / tile_count
    for i in range(1, tile_count):
        pos = int(i * tile_px)
        cv2.line(img, (pos, 0), (pos, preview_size), (200, 200, 200), 1)
        cv2.line(img, (0, pos), (preview_size, pos), (200, 200, 200), 1)
    
    return img


def render_dxf_debug(lines_data, min_x, min_y, tile_size, preview_size=500):
    """Renderiza una vista de debug del DXF mostrando los segmentos detectados"""
    img = np.ones((preview_size, preview_size, 3), dtype=np.uint8) * 255
    
    if not lines_data or tile_size == 0:
        return img
    
    # Escala para que el tile quepa en el preview
    scale = preview_size / tile_size * 0.9
    offset = preview_size * 0.05
    
    # Colores para diferentes ángulos
    colors = {
        0: (255, 0, 0),    # Rojo - horizontal
        90: (0, 0, 255),   # Azul - vertical
        45: (0, 255, 0),   # Verde - diagonal
        135: (255, 0, 255) # Magenta - diagonal inversa
    }
    
    for x1, y1, x2, y2 in lines_data:
        # Normalizar al origen
        nx1 = (x1 - min_x) * scale + offset
        ny1 = preview_size - ((y1 - min_y) * scale + offset)
        nx2 = (x2 - min_x) * scale + offset
        ny2 = preview_size - ((y2 - min_y) * scale + offset)
        
        # Determinar ángulo para color
        dx = x2 - x1
        dy = y2 - y1
        ang = math.degrees(math.atan2(dy, dx))
        if ang < 0:
            ang += 360
        if ang >= 180:
            ang -= 180
        
        # Color según ángulo aproximado
        if abs(ang - 0) < 10 or abs(ang - 180) < 10:
            color = colors[0]
        elif abs(ang - 90) < 10:
            color = colors[90]
        elif abs(ang - 45) < 10:
            color = colors[45]
        elif abs(ang - 135) < 10:
            color = colors[135]
        else:
            color = (100, 100, 100)
        
        cv2.line(img, (int(nx1), int(ny1)), (int(nx2), int(ny2)), color, 2, cv2.LINE_AA)
    
    # Dibujar borde del tile
    cv2.rectangle(img, (int(offset), int(offset)), 
                  (int(offset + tile_size * scale), int(preview_size - offset - tile_size * scale)), 
                  (150, 150, 150), 1)
    
    return img


class DXFtoPatConverter:
    """Convierte archivos DXF de AutoCAD a formato PAT"""
    
    def __init__(self):
        pass
    
    def convert(self, dxf_file_path):
        """Lee un archivo DXF y genera un archivo PAT"""
        try:
            doc = ezdxf.readfile(dxf_file_path)
            msp = doc.modelspace()
            
            # Extraer todas las líneas
            lines_data = []
            min_x, min_y = float('inf'), float('inf')
            max_x, max_y = float('-inf'), float('-inf')
            
            for entity in msp:
                if entity.dxftype() == 'LINE':
                    x1, y1 = entity.dxf.start.x, entity.dxf.start.y
                    x2, y2 = entity.dxf.end.x, entity.dxf.end.y
                    
                    min_x = min(min_x, x1, x2)
                    min_y = min(min_y, y1, y2)
                    max_x = max(max_x, x1, x2)
                    max_y = max(max_y, y1, y2)
                    
                    lines_data.append((x1, y1, x2, y2))
                
                elif entity.dxftype() == 'LWPOLYLINE':
                    points = list(entity.get_points())
                    for i in range(len(points) - 1):
                        x1, y1 = points[i][0], points[i][1]
                        x2, y2 = points[i+1][0], points[i+1][1]
                        
                        min_x = min(min_x, x1, x2)
                        min_y = min(min_y, y1, y2)
                        max_x = max(max_x, x1, x2)
                        max_y = max(max_y, y1, y2)
                        
                        lines_data.append((x1, y1, x2, y2))
                    
                    if entity.closed and len(points) > 2:
                        x1, y1 = points[-1][0], points[-1][1]
                        x2, y2 = points[0][0], points[0][1]
                        lines_data.append((x1, y1, x2, y2))
            
            if not lines_data:
                return {"error": "No se encontraron líneas en el archivo DXF"}
            
            # Tamaño del tile
            width = max_x - min_x
            height = max_y - min_y
            tile_size = max(width, height)
            
            if tile_size == 0:
                return {"error": "El dibujo tiene tamaño cero"}
            
            # Generar imagen de debug
            debug_img = render_dxf_debug(lines_data, min_x, min_y, tile_size)
            
            # Generar líneas PAT - NORMALIZANDO AL ORIGEN
            pat_lines = []
            
            for x1, y1, x2, y2 in lines_data:
                # NORMALIZAR coordenadas al origen (0,0)
                nx1 = x1 - min_x
                ny1 = y1 - min_y
                nx2 = x2 - min_x
                ny2 = y2 - min_y
                
                dx = nx2 - nx1
                dy = ny2 - ny1
                length = math.sqrt(dx**2 + dy**2)
                
                if length < 0.001:
                    continue
                
                # Ángulo
                ang = math.degrees(math.atan2(dy, dx))
                if ang < 0:
                    ang += 360
                
                # Cuantizar ángulo al más cercano (manejando wrap-around)
                def angle_diff(a, b):
                    """Diferencia mínima entre dos ángulos considerando el wrap-around"""
                    diff = abs(a - b)
                    return min(diff, 360 - diff)
                
                # Ángulos cada 15° para mayor precisión en patrones orgánicos
                valid_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165,
                               180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]
                ang_q = min(valid_angles, key=lambda a: angle_diff(a, ang))
                
                # Si el ángulo cuantizado está en el rango 180-360, 
                # intercambiar los puntos para que la línea vaya en la dirección correcta
                if ang_q >= 180:
                    ang_q = ang_q - 180
                    # Intercambiar origen: usar el punto final como origen
                    nx1, ny1, nx2, ny2 = nx2, ny2, nx1, ny1
                
                # Origen normalizado (ahora es el punto correcto)
                ox = round(nx1, 6)
                oy = round(ny1, 6)
                
                # Delta depende del ángulo
                # Para líneas H/V: el tile se repite en cuadrícula regular
                # Para diagonales: ajustar para que el patrón tile correctamente
                delta_x = round(tile_size, 6)
                delta_y = round(tile_size, 6)
                
                # Dash/gap - la longitud de la línea y el espacio
                dash = round(length, 6)
                # Gap debe ser negativo y = tile_size - length (para que no se repita dentro del mismo tile)
                gap = round(-(tile_size - length), 6)
                
                # Si el gap es mayor o igual a 0, hacer continua la línea
                if gap >= 0:
                    gap = -0.001
                
                pat_line = f"{ang_q}, {ox},{oy}, {delta_x},{delta_y}, {dash},{gap}"
                pat_lines.append(pat_line)
            
            # Construir el archivo PAT
            header = [
                "*DXF_Pattern, Converted from AutoCAD DXF",
                ";%TYPE=MODEL"
            ]
            header.extend(pat_lines)
            
            pat_content = "\r\n".join(header) + "\r\n"
            pat_preview = render_pat_preview(pat_content)
            
            return {
                "pat_content": pat_content,
                "pat_preview": pat_preview,
                "debug_img": debug_img,
                "stats": f"✅ DXF: {len(lines_data)} segmentos → PAT: {len(pat_lines)} líneas (tile={tile_size:.2f})"
            }
            
        except ezdxf.DXFError as e:
            return {"error": f"Error leyendo DXF: {str(e)}"}
        except Exception as e:
            return {"error": f"Error: {str(e)}"}


class ImageToPatConverter:
    """Convierte imágenes a PAT con pipeline CV avanzado:
    - CLAHE para normalización de contraste local
    - Adaptive Threshold como alternativa a Canny
    - LSD (Line Segment Detector) sub-pixel
    - Zhang-Suen Skeletonization (scikit-image)
    - Merge de segmentos colineales
    """
    
    def __init__(self):
        pass
    
    # ── Helpers ──────────────────────────────────────────────
    
    @staticmethod
    def _merge_colinear(lines, angle_tol=5.0, dist_tol=3.0, gap_tol=10.0):
        """Une segmentos cercanos con ángulo similar para reducir fragmentación.
        angle_tol: tolerancia angular en grados
        dist_tol: distancia perpendicular máxima entre segmentos
        gap_tol: gap máximo a lo largo de la línea para unir
        """
        if not lines:
            return lines
        
        # Calcular ángulo y posición perpendicular de cada segmento
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
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'merged': False
            })
        
        merged_lines = []
        for i, a in enumerate(entries):
            if a['merged']:
                continue
            group_min = a['p_min']
            group_max = a['p_max']
            a['merged'] = True
            
            # Buscar vecinos colineales
            changed = True
            while changed:
                changed = False
                for j, b in enumerate(entries):
                    if b['merged']:
                        continue
                    # Mismo ángulo y misma línea perpendicular
                    ang_diff = abs(a['ang'] - b['ang'])
                    if ang_diff > angle_tol and (180 - ang_diff) > angle_tol:
                        continue
                    if abs(a['perp'] - b['perp']) > dist_tol:
                        continue
                    # Verificar gap a lo largo de la línea
                    if b['p_min'] > group_max + gap_tol or b['p_max'] < group_min - gap_tol:
                        continue
                    # Merge
                    group_min = min(group_min, b['p_min'])
                    group_max = max(group_max, b['p_max'])
                    b['merged'] = True
                    changed = True
            
            # Reconstruir segmento merged
            ang_rad = math.radians(a['ang'])
            cos_a, sin_a = math.cos(ang_rad), math.sin(ang_rad)
            mx1 = group_min * cos_a - a['perp'] * sin_a  # error: should use + for perp
            my1 = group_min * sin_a + a['perp'] * cos_a
            mx2 = group_max * cos_a - a['perp'] * sin_a
            my2 = group_max * sin_a + a['perp'] * cos_a
            merged_lines.append((mx1, my1, mx2, my2))
        
        return merged_lines

    @staticmethod
    def _point_to_line_dist(px, py, lx1, ly1, lx2, ly2):
        """Distancia perpendicular de un punto a una línea (infinita)."""
        dx = lx2 - lx1
        dy = ly2 - ly1
        len_sq = dx * dx + dy * dy
        if len_sq < 1e-10:
            return math.hypot(px - lx1, py - ly1)
        # Proyección del punto sobre la línea
        return abs(dy * px - dx * py + lx2 * ly1 - ly2 * lx1) / math.sqrt(len_sq)

    @staticmethod
    def _segments_overlap(x1, y1, x2, y2, ux1, uy1, ux2, uy2):
        """Verifica si las proyecciones de dos segmentos sobre su eje compartido se solapan."""
        # Usar la dirección del primer segmento como eje
        dx = x2 - x1
        dy = y2 - y1
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-10:
            return True
        ux, uy = dx / seg_len, dy / seg_len
        
        # Proyectar los 4 puntos sobre el eje
        p1 = x1 * ux + y1 * uy
        p2 = x2 * ux + y2 * uy
        q1 = ux1 * ux + uy1 * uy
        q2 = ux2 * ux + uy2 * uy
        
        a_min, a_max = min(p1, p2), max(p1, p2)
        b_min, b_max = min(q1, q2), max(q1, q2)
        
        # Solapan si los rangos se intersectan (con tolerancia del 20% de la longitud)
        tol = seg_len * 0.2
        return a_min <= b_max + tol and b_min <= a_max + tol

    @staticmethod
    def _is_duplicate(x1, y1, x2, y2, unique_lines, threshold=8.0):
        """Filtra líneas dobles usando distancia PERPENDICULAR + solapamiento.
        
        Resuelve el problema de paralelas desfasadas longitudinalmente:
        dos líneas son duplicadas si son paralelas, están perpendicularmente
        cerca, y sus proyecciones se solapan sobre el eje compartido.
        """
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if ang < 0:
            ang += 180
        if ang >= 180:
            ang -= 180
        mx = (x1 + x2) / 2.0
        my = (y1 + y2) / 2.0
        
        for (ux1, uy1, ux2, uy2) in unique_lines:
            uang = math.degrees(math.atan2(uy2 - uy1, ux2 - ux1))
            if uang < 0:
                uang += 180
            if uang >= 180:
                uang -= 180
            
            # ── 1. Ángulo similar (tolerancia 12°) ──
            ang_diff = abs(ang - uang)
            if ang_diff > 12 and (180 - ang_diff) > 12:
                continue
            
            # ── 2. Distancia perpendicular del midpoint a la otra línea ──
            perp_dist = ImageToPatConverter._point_to_line_dist(
                mx, my, ux1, uy1, ux2, uy2
            )
            
            if perp_dist < threshold:
                # ── 3. Verificar solapamiento longitudinal ──
                if ImageToPatConverter._segments_overlap(
                    x1, y1, x2, y2, ux1, uy1, ux2, uy2
                ):
                    return True
        
        return False
    
    # ── Main ─────────────────────────────────────────────────
    
    def convert(self, image_bytes, method="hough", canny_low=50, canny_high=150, blur_size=3, 
                param1=20, param2=5,
                use_clahe=False, clahe_clip=2.0,
                use_adaptive=False, adaptive_block=11, adaptive_c=2,
                use_skeleton=False,
                merge_segments=False, merge_angle_tol=5.0, merge_gap_tol=10.0,
                dedup_threshold=8.0,
                offset_x=0.0, offset_y=0.0):
        """
        Pipeline mejorado de imagen a PAT.
        
        Parámetros nuevos:
          use_clahe       – Aplica CLAHE antes del blur (mejora contraste local)
          clahe_clip      – clipLimit para CLAHE (2.0 por defecto)
          use_adaptive    – Usa Adaptive Threshold en vez de Canny
          adaptive_block  – Tamaño de bloque para adaptive threshold (impar)
          adaptive_c      – Constante C para adaptive threshold
          use_skeleton    – Aplica Zhang-Suen skeletonization post-bordes
          merge_segments  – Une segmentos colineales cercanos
          merge_angle_tol – Tolerancia angular para merge (grados)
          merge_gap_tol   – Gap máximo para merge (pixels)
          dedup_threshold – Distancia (px) para considerar dos líneas como duplicadas
          offset_x       – Desplazamiento horizontal del patrón (0.0 a 1.0)
          offset_y       – Desplazamiento vertical del patrón (0.0 a 1.0)
        """
        try:
            # ── Decodificar imagen ──
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return {"error": "Error al cargar la imagen"}
            
            # Hacer cuadrada
            h_orig, w_orig = img.shape[:2]
            side = min(h_orig, w_orig)
            start_x = (w_orig - side) // 2
            start_y = (h_orig - side) // 2
            img = img[start_y:start_y+side, start_x:start_x+side]
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # ── 1. CLAHE (Contrast Limited Adaptive Histogram Equalization) ──
            if use_clahe:
                clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
                gray = clahe.apply(gray)
            
            # ── 2. Blur ──
            if blur_size > 1:
                blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
                blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
            else:
                blurred = gray
            
            # ── 3. Binarización: Canny vs Adaptive Threshold ──
            if use_adaptive:
                adaptive_block = adaptive_block if adaptive_block % 2 == 1 else adaptive_block + 1
                if adaptive_block < 3:
                    adaptive_block = 3
                edges = cv2.adaptiveThreshold(
                    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, adaptive_block, adaptive_c
                )
            else:
                edges = cv2.Canny(blurred, canny_low, canny_high)
            
            # ── 4. Skeleton (Zhang-Suen via scikit-image) ──
            if use_skeleton:
                try:
                    from skimage.morphology import skeletonize
                    skeleton_input = (edges > 0).astype(np.uint8)
                    skeleton = skeletonize(skeleton_input).astype(np.uint8) * 255
                    if cv2.countNonZero(skeleton) > 0:
                        edges = skeleton
                    # Si el skeleton es vacío, mantener edges original (Rule 67)
                except ImportError:
                    pass  # scikit-image no disponible, continuar sin skeleton
            
            # ── 5. Detección de líneas según método ──
            debug_img = np.ones((side, side, 3), dtype=np.uint8) * 255
            # Dibujar los bordes detectados en gris como referencia
            edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            debug_edges = np.where(edges_color > 0, 
                                   np.array([220, 220, 220], dtype=np.uint8), 
                                   np.array([255, 255, 255], dtype=np.uint8))
            debug_img = debug_edges.astype(np.uint8)
            
            raw_lines = []
            
            if method == "hough":
                # Conectar bordes rotos ligeramente
                kernel = np.ones((2, 2), np.uint8)
                closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
                
                # Transformada Probabilística de Hough
                lines = cv2.HoughLinesP(closed_edges, 1, np.pi/180, threshold=20, 
                                        minLineLength=param1, maxLineGap=param2)
                if lines is not None:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        raw_lines.append((x1, y1, x2, y2))
            
            elif method == "lsd":
                # ── LSD: Line Segment Detector (sub-pixel, auto-tuning) ──
                lsd = cv2.createLineSegmentDetector(
                    cv2.LSD_REFINE_STD  # Refinamiento estándar
                )
                lines_lsd, widths, precs, nfas = lsd.detect(blurred)
                
                if lines_lsd is not None:
                    min_len = float(param1)
                    for i, line in enumerate(lines_lsd):
                        x1, y1, x2, y2 = line[0]
                        seg_len = math.hypot(x2 - x1, y2 - y1)
                        if seg_len >= min_len:
                            raw_lines.append((x1, y1, x2, y2))
            
            else:  # contour
                # Contornos para formas orgánicas
                kernel = np.ones((2, 2), np.uint8)
                dilated = cv2.dilate(edges, kernel, iterations=1)
                contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                
                min_contour_len = param1
                epsilon_factor = float(param2)
                
                for cnt in contours:
                    arc_len = cv2.arcLength(cnt, False)
                    if arc_len < min_contour_len:
                        continue
                        
                    approx = cv2.approxPolyDP(cnt, epsilon_factor * arc_len, False)
                    pts = approx[:, 0, :]
                    
                    for i in range(len(pts) - 1):
                        x1, y1 = pts[i]
                        x2, y2 = pts[i+1]
                        raw_lines.append((x1, y1, x2, y2))

            # ── 6. Merge colineal (reduce fragmentación) ──
            if merge_segments and len(raw_lines) > 1:
                raw_lines = self._merge_colinear(
                    raw_lines, 
                    angle_tol=merge_angle_tol, 
                    gap_tol=merge_gap_tol
                )

            # ── 7. Filtrado y generación PAT ──
            unique_lines = []
            pat_lines = []
            tile_size = 1.0
            
            for x1, y1, x2, y2 in raw_lines:
                if self._is_duplicate(x1, y1, x2, y2, unique_lines, threshold=dedup_threshold):
                    continue
                unique_lines.append((x1, y1, x2, y2))
                
                # Dibujar en debug con color según ángulo
                dx_raw = x2 - x1
                dy_raw = y2 - y1
                ang_raw = math.degrees(math.atan2(dy_raw, dx_raw))
                if ang_raw < 0:
                    ang_raw += 180
                
                if abs(ang_raw) < 15 or abs(ang_raw - 180) < 15:
                    line_color = (220, 50, 50)    # Rojo - horizontal
                elif abs(ang_raw - 90) < 15:
                    line_color = (50, 50, 220)    # Azul - vertical
                elif abs(ang_raw - 45) < 15:
                    line_color = (50, 180, 50)    # Verde - diagonal
                elif abs(ang_raw - 135) < 15:
                    line_color = (180, 50, 180)   # Magenta - diagonal inv
                else:
                    line_color = (80, 80, 80)     # Gris - otros
                
                cv2.line(debug_img, (int(x1), int(y1)), (int(x2), int(y2)), 
                         line_color, 1, cv2.LINE_AA)
                
                # Normalizar coordenadas de 0 a 1 e invertir Y
                nx1 = x1 / side
                ny1 = 1.0 - (y1 / side)
                nx2 = x2 / side
                ny2 = 1.0 - (y2 / side)
                
                dx = nx2 - nx1
                dy = ny2 - ny1
                length = math.sqrt(dx**2 + dy**2)
                
                if length < 0.005:
                    continue
                
                ang = math.degrees(math.atan2(dy, dx))
                if ang < 0:
                    ang += 360
                    
                if ang >= 180:
                    ang -= 180
                    nx1, ny1, nx2, ny2 = nx2, ny2, nx1, ny1
                    
                # Aplicar offset SOLO al origin (no a endpoints de geometría)
                ox = round((nx1 + offset_x) % 1.0, 5)
                oy = round((ny1 + offset_y) % 1.0, 5)
                ang_q = round(ang, 3)
                dash = round(length, 5)
                gap = round(-(tile_size - length), 5)
                
                if gap >= 0:
                    gap = -0.001
                    
                pat_line = f"{ang_q}, {ox},{oy}, {tile_size},{tile_size}, {dash},{gap}"
                pat_lines.append(pat_line)
                
            if not pat_lines:
                return {"error": "No se detectaron líneas válidas. Ajusta los parámetros."}
                
            # ── Generar archivo PAT ──
            header = [
                "*Image_Pattern, HatchCraft precision pattern",
                ";%TYPE=MODEL"
            ]
            header.extend(pat_lines)
            pat_content = "\r\n".join(header) + "\r\n"
            
            pat_preview = render_pat_preview(pat_content)
            
            # Stats detalladas
            method_names = {"hough": "HOUGH", "lsd": "LSD", "contour": "CONTOUR"}
            extras = []
            if use_clahe:
                extras.append("CLAHE")
            if use_adaptive:
                extras.append("AdaptiveThresh")
            if use_skeleton:
                extras.append("Skeleton")
            if merge_segments:
                extras.append(f"Merge({len(raw_lines)}→{len(pat_lines)})")
            extra_str = f" + {', '.join(extras)}" if extras else ""
            
            return {
                "pat_content": pat_content,
                "pat_preview": pat_preview,
                "debug_img": debug_img,
                "stats": f"✅ {method_names.get(method, method.upper())}{extra_str}: {len(pat_lines)} líneas"
            }
            
        except Exception as e:
            return {"error": f"Error: {str(e)}"}