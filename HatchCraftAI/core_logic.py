import cv2
import numpy as np
import math
from skimage.morphology import skeletonize
from collections import defaultdict

# Solo 4 ángulos principales para estabilidad
VALID_ANGLES = [0, 45, 90, 135]

def quantize_angle(ang):
    """Redondea el ángulo al valor válido más cercano (0-180 range)"""
    # Normalizar a 0-180 (las líneas son bidireccionales)
    ang = ang % 180
    
    best = 0
    best_diff = 180
    for valid in VALID_ANGLES:
        diff = min(abs(ang - valid), abs(ang - valid - 180), abs(ang - valid + 180))
        if diff < best_diff:
            best_diff = diff
            best = valid
    return best

def get_perpendicular_position(x, y, angle):
    """Calcula la posición perpendicular de un punto respecto a una línea en el ángulo dado"""
    ang_rad = math.radians(angle)
    # Vector perpendicular al ángulo: (-sin, cos)
    return -x * math.sin(ang_rad) + y * math.cos(ang_rad)

def get_parallel_position(x, y, angle):
    """Calcula la posición paralela (a lo largo) de un punto respecto a una línea en el ángulo dado"""
    ang_rad = math.radians(angle)
    # Vector paralelo al ángulo: (cos, sin)
    return x * math.cos(ang_rad) + y * math.sin(ang_rad)

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
        
        # Agrupar segmentos por ángulo cuantizado
        angle_groups = defaultdict(list)
        
        for cnt in contours:
            arc_len = cv2.arcLength(cnt, True)
            if arc_len < 20: continue
            
            approx = cv2.approxPolyDP(cnt, epsilon_factor * 1.5 * arc_len, True)
            pts = approx[:, 0, :]
            cv2.polylines(vec_preview, [pts], True, (0,0,0), 1, cv2.LINE_AA)

            for i in range(len(pts)):
                p1, p2 = pts[i], pts[(i + 1) % len(pts)]
                
                # Normalizar a 0-1
                x1, y1 = p1[0] / side, (side - p1[1]) / side
                x2, y2 = p2[0] / side, (side - p2[1]) / side
                
                dx, dy = x2 - x1, y2 - y1
                L = math.sqrt(dx**2 + dy**2)
                if L < 0.03: continue
                
                ang = math.degrees(math.atan2(dy, dx))
                if ang < 0: ang += 360
                
                ang_q = quantize_angle(ang)
                
                # Calcular posiciones perpendicular y paralela
                perp_pos = get_perpendicular_position(x1, y1, ang_q)
                para_pos = get_parallel_position(x1, y1, ang_q)
                
                # Guardar: (perp_pos, para_start, para_end, x1, y1, L)
                angle_groups[ang_q].append({
                    'perp': perp_pos,
                    'para_start': para_pos,
                    'para_end': para_pos + L,
                    'x': x1,
                    'y': y1,
                    'length': L
                })
        
        # Generar líneas PAT con familias agrupadas
        pat_lines = []
        
        for angle, segments in sorted(angle_groups.items()):
            if len(segments) < 2:
                continue
            
            # Ordenar por posición perpendicular
            segments = sorted(segments, key=lambda s: s['perp'])
            
            # Calcular delta-y (distancia perpendicular típica entre líneas)
            perp_positions = [s['perp'] for s in segments]
            if len(perp_positions) >= 2:
                # Usar la mediana de las diferencias para evitar outliers
                diffs = [perp_positions[i+1] - perp_positions[i] for i in range(len(perp_positions)-1)]
                diffs = [d for d in diffs if d > 0.01]  # Filtrar diferencias muy pequeñas
                if diffs:
                    delta_y = np.median(diffs)
                else:
                    delta_y = 0.1
            else:
                delta_y = 0.1
            
            # Para cada segmento, crear una línea con dash corto
            for seg in segments:
                # Origen del segmento
                ox = round(seg['x'], 4)
                oy = round(seg['y'], 4)
                
                # Shift: delta-x=0 (sin escalonado), delta-y = espaciado perpendicular
                # Pero para Revit MODEL type, usamos shift unitario
                if angle in [45, 135]:
                    s_x, s_y = 0.7071067812, 0.7071067812
                else:
                    s_x, s_y = 1.0, 1.0
                
                # Dash: longitud real del segmento (limitada)
                dash = min(0.1, seg['length'])
                
                # Space: para que no se repita inmediatamente, dejamos espacio grande
                space = -(1.0 - dash)
                
                line = f"{angle}, {ox},{oy}, {s_x},{s_y}, {round(dash, 4)},{round(space, 4)}"
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
            "stats": f"Patrón generado: {len(pat_lines)} líneas en {len(angle_groups)} familias de ángulos."
        }