import cv2
import numpy as np
import math
from skimage.morphology import skeletonize
from collections import defaultdict

# Ángulos válidos según el patrón ghiaia3 que funciona en Revit
VALID_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]

# Shift por ángulo (del patrón ghiaia3)
ANGLE_SHIFTS = {
    0: (1, 1),
    45: (0.707106781, 0.707106781),
    90: (1, 1),
    135: (0.707106781, 0.707106781),
    180: (1, 1),
    225: (0.707106781, 0.707106781),
    270: (1, 1),
    315: (0.707106781, 0.707106781),
}

def quantize_angle(ang):
    """Redondea el ángulo al valor válido más cercano"""
    best = 0
    best_diff = 360
    for valid in VALID_ANGLES:
        diff = min(abs(ang - valid), abs(ang - valid - 360), abs(ang - valid + 360))
        if diff < best_diff:
            best_diff = diff
            best = valid
    return best

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
        
        # Agrupar por ángulo cuantizado y posición
        line_groups = defaultdict(list)
        position_tolerance = 0.02
        
        for cnt in contours:
            if cv2.arcLength(cnt, True) < 5: continue
            approx = cv2.approxPolyDP(cnt, epsilon_factor * cv2.arcLength(cnt, True), True)
            pts = approx[:, 0, :]
            cv2.polylines(vec_preview, [pts], True, (0,0,0), 1, cv2.LINE_AA)

            for i in range(len(pts)):
                p1, p2 = pts[i], pts[(i + 1) % len(pts)]
                
                x1, y1 = p1[0] / side, (side - p1[1]) / side
                x2, y2 = p2[0] / side, (side - p2[1]) / side
                
                dx, dy = x2 - x1, y2 - y1
                L = math.sqrt(dx**2 + dy**2)
                if L < 0.01: continue
                
                ang = math.degrees(math.atan2(dy, dx))
                if ang < 0: ang += 360
                
                # CUANTIZAR ÁNGULO a valores válidos
                ang_q = quantize_angle(ang)
                
                # Posición perpendicular para agrupar
                ang_rad = math.radians(ang_q)
                perp_pos = x1 * math.sin(ang_rad) - y1 * math.cos(ang_rad)
                perp_key = round(perp_pos / position_tolerance) * position_tolerance
                
                # Posición paralela
                para_pos = x1 * math.cos(ang_rad) + y1 * math.sin(ang_rad)
                
                key = (ang_q, round(perp_key, 3))
                line_groups[key].append((para_pos, para_pos + L, x1, y1))
        
        # Generar .PAT
        lines = [
            "*HatchCraftModel, Generated Pattern",
            ";%TYPE=MODEL"
        ]
        
        count = 0
        for (ang, perp_pos), segments in sorted(line_groups.items()):
            segments = sorted(segments, key=lambda s: s[0])
            
            first_seg = segments[0]
            origin_x = round(first_seg[2], 4)
            origin_y = round(first_seg[3], 4)
            
            dash_space = []
            current_pos = first_seg[0]
            
            for para_start, para_end, _, _ in segments:
                gap = para_start - current_pos
                if gap > 0.01 and dash_space:
                    dash_space.append(round(-gap, 4))
                
                dash = para_end - para_start
                if dash > 0.01:
                    dash_space.append(round(dash, 4))
                    current_pos = para_end
            
            if not dash_space:
                continue
            
            # Espacio final
            remaining = 1.0 - (current_pos % 1.0)
            if 0.01 < remaining < 0.99:
                dash_space.append(round(-remaining, 4))
            
            # Obtener shift para este ángulo
            s_x, s_y = ANGLE_SHIFTS.get(ang, (1, 1))
            
            # Formatear dash-space como string
            dash_str = ",".join(str(v) for v in dash_space)
            
            line = f"{ang}, {origin_x},{origin_y}, {s_x},{s_y}, {dash_str}"
            lines.append(line)
            count += 1
        
        full_content = "\r\n".join(lines) + "\r\n"
        
        return {
            "processed_img": binary,
            "vector_img": vec_preview,
            "pat_content": full_content,
            "stats": f"Patrón generado: {count} familias de líneas. Tipo: MODELO."
        }