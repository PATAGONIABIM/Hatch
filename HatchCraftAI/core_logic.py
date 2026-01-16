import cv2
import numpy as np
import math
from skimage.morphology import skeletonize
from collections import defaultdict

# Solo 4 ángulos principales
VALID_ANGLES = [0, 45, 90, 135]

def quantize_angle(ang):
    """Redondea el ángulo a 0-180 y luego al válido más cercano"""
    ang = ang % 180
    best = 0
    best_diff = 180
    for valid in VALID_ANGLES:
        diff = min(abs(ang - valid), abs(ang - valid - 180), abs(ang - valid + 180))
        if diff < best_diff:
            best_diff = diff
            best = valid
    return best

def get_line_position(x, y, angle):
    """Retorna (perpendicular_pos, parallel_pos) para un punto dado un ángulo"""
    ang_rad = math.radians(angle)
    perp = -x * math.sin(ang_rad) + y * math.cos(ang_rad)
    para = x * math.cos(ang_rad) + y * math.sin(ang_rad)
    return perp, para

class PatternGenerator:
    def __init__(self, size=100.0):
        self.size = float(size)

    def process_image(self, image_file, epsilon_factor=0.005, closing_size=2, mode="Auto-Detectar", use_skeleton=True):
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None: return {"error": "Error al cargar imagen"}

        h_orig, w_orig = img.shape[:2]
        side = min(h_orig, w_orig)
        start_x = (w_orig - side) // 2
        start_y = (h_orig - side) // 2
        img = img[start_y:start_y+side, start_x:start_x+side]
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        if (mode == "Auto-Detectar" and np.sum(binary==255) > binary.size/2) or mode == "Líneas Negras":
            binary = cv2.bitwise_not(binary)

        if closing_size > 0:
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((closing_size,closing_size), np.uint8))

        if use_skeleton:
            binary = (skeletonize(binary > 0) * 255).astype(np.uint8)

        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        vec_preview = np.ones((side, side, 3), dtype=np.uint8) * 255
        
        # Agrupar segmentos por ángulo Y posición perpendicular (líneas colineales)
        # Clave: (angulo, perp_pos_redondeada) -> lista de (para_start, para_end, x, y)
        line_families = defaultdict(list)
        PERP_TOLERANCE = 0.02  # Tolerancia para considerar líneas como colineales
        
        for cnt in contours:
            arc_len = cv2.arcLength(cnt, True)
            if arc_len < 15: continue
            
            approx = cv2.approxPolyDP(cnt, epsilon_factor * arc_len, True)
            pts = approx[:, 0, :]
            cv2.polylines(vec_preview, [pts], True, (0,0,0), 1, cv2.LINE_AA)

            for i in range(len(pts)):
                p1, p2 = pts[i], pts[(i + 1) % len(pts)]
                
                x1, y1 = p1[0] / side, (side - p1[1]) / side
                x2, y2 = p2[0] / side, (side - p2[1]) / side
                
                dx, dy = x2 - x1, y2 - y1
                L = math.sqrt(dx**2 + dy**2)
                if L < 0.02: continue
                
                ang = math.degrees(math.atan2(dy, dx))
                if ang < 0: ang += 360
                ang_q = quantize_angle(ang)
                
                # Calcular posiciones
                perp1, para1 = get_line_position(x1, y1, ang_q)
                perp2, para2 = get_line_position(x2, y2, ang_q)
                
                # Ordenar para_start < para_end
                para_start = min(para1, para2)
                para_end = max(para1, para2)
                perp_avg = (perp1 + perp2) / 2
                
                # Agrupar por posición perpendicular redondeada
                perp_key = round(perp_avg / PERP_TOLERANCE) * PERP_TOLERANCE
                key = (ang_q, round(perp_key, 4))
                
                line_families[key].append({
                    'para_start': para_start,
                    'para_end': para_end,
                    'x': x1,
                    'y': y1,
                    'length': L
                })
        
        # Generar líneas PAT
        pat_lines = []
        
        for (angle, perp_pos), segments in sorted(line_families.items()):
            if not segments:
                continue
            
            # Ordenar segmentos por posición paralela
            segments = sorted(segments, key=lambda s: s['para_start'])
            
            # Usar el primer segmento como origen
            first = segments[0]
            ox = round(first['x'], 4)
            oy = round(first['y'], 4)
            
            # Construir secuencia dash-space
            # Cada segmento es un dash, el espacio entre segmentos es un gap
            dash_space = []
            current_pos = first['para_start']
            
            for seg in segments:
                # Gap desde la posición actual hasta el inicio de este segmento
                gap = seg['para_start'] - current_pos
                if gap > 0.01 and dash_space:  # Solo si hay gap significativo
                    dash_space.append(round(-gap, 4))
                
                # Dash = longitud de este segmento
                dash = seg['para_end'] - seg['para_start']
                if dash > 0.01:
                    dash_space.append(round(dash, 4))
                    current_pos = seg['para_end']
            
            if not dash_space:
                continue
            
            # Espacio final para completar el ciclo (si no llega al final del unit cell)
            # Para que el patrón repita correctamente
            remaining = 1.0 - (current_pos % 1.0)
            if remaining > 0.01 and remaining < 0.99:
                dash_space.append(round(-remaining, 4))
            
            # Shift basado en el ángulo
            if angle in [45, 135]:
                s_x, s_y = 0.7071067812, 0.7071067812
            else:
                s_x, s_y = 1.0, 1.0
            
            # Formatear línea PAT
            dash_str = ",".join(str(v) for v in dash_space)
            line = f"{angle}, {ox},{oy}, {s_x},{s_y}, {dash_str}"
            pat_lines.append(line)
        
        # Header
        lines = [
            "*HatchCraftModel, Generated Pattern",
            ";%TYPE=MODEL"
        ]
        lines.extend(pat_lines)
        
        full_content = "\r\n".join(lines) + "\r\n"
        
        return {
            "processed_img": binary,
            "vector_img": vec_preview,
            "pat_content": full_content,
            "stats": f"Patrón generado: {len(pat_lines)} familias de líneas."
        }