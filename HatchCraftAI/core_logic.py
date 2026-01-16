import cv2
import numpy as np
import math
from skimage.morphology import skeletonize
from collections import defaultdict

class PatternGenerator:
    def __init__(self, size=100.0):
        self.size = float(size)

    def process_image(self, image_file, epsilon_factor=0.005, closing_size=2, mode="Auto-Detectar", use_skeleton=True):
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None: return {"error": "Error al cargar imagen"}

        # AUTO-CROP cuadrado
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
        
        # AGRUPAR SEGMENTOS POR ÁNGULO Y POSICIÓN
        # Clave: (ángulo_redondeado, posición_perpendicular)
        line_groups = defaultdict(list)
        
        angle_tolerance = 5  # grados para redondear ángulos
        position_tolerance = 0.02  # 2% del tile
        
        for cnt in contours:
            if cv2.arcLength(cnt, True) < 5: continue
            approx = cv2.approxPolyDP(cnt, epsilon_factor * cv2.arcLength(cnt, True), True)
            pts = approx[:, 0, :]
            cv2.polylines(vec_preview, [pts], True, (0,0,0), 1, cv2.LINE_AA)

            for i in range(len(pts)):
                p1, p2 = pts[i], pts[(i + 1) % len(pts)]
                
                # Normalizar a 0-1
                x1, y1 = p1[0] / side, (side - p1[1]) / side
                x2, y2 = p2[0] / side, (side - p2[1]) / side
                
                dx, dy = x2 - x1, y2 - y1
                L = math.sqrt(dx**2 + dy**2)
                if L < 0.005: continue
                
                ang = math.degrees(math.atan2(dy, dx))
                if ang < 0: ang += 360
                
                # Redondear ángulo a múltiplos de 5°
                ang_rounded = round(ang / angle_tolerance) * angle_tolerance
                if ang_rounded >= 360: ang_rounded = 0
                
                # Calcular posición perpendicular (para agrupar líneas paralelas)
                ang_rad = math.radians(ang_rounded)
                # Proyección perpendicular del punto de origen
                perp_pos = x1 * math.sin(ang_rad) - y1 * math.cos(ang_rad)
                perp_key = round(perp_pos / position_tolerance) * position_tolerance
                
                # Proyección paralela (para ordenar segmentos)
                para_pos = x1 * math.cos(ang_rad) + y1 * math.sin(ang_rad)
                para_end = para_pos + L
                
                key = (ang_rounded, round(perp_key, 3))
                line_groups[key].append((para_pos, para_end, x1, y1))
        
        # GENERAR LÍNEAS .PAT
        lines = [
            "*HatchCraftModel, Generated Pattern",
            ";%TYPE=MODEL"
        ]
        
        count = 0
        
        for (ang, perp_pos), segments in sorted(line_groups.items()):
            # Ordenar segmentos por posición paralela
            segments = sorted(segments, key=lambda s: s[0])
            
            # Usar el primer segmento para el origen
            first_seg = segments[0]
            origin_x = round(first_seg[2], 4)
            origin_y = round(first_seg[3], 4)
            
            # Crear secuencia dash-space
            dash_space = []
            current_pos = first_seg[0]
            
            for para_start, para_end, _, _ in segments:
                gap = para_start - current_pos
                if gap > 0.01 and dash_space:
                    dash_space.append(f"-{round(gap, 4)}")
                
                dash = para_end - para_start
                if dash > 0.005:
                    dash_space.append(f"{round(dash, 4)}")
                    current_pos = para_end
            
            if not dash_space:
                continue
            
            # Espacio final
            remaining = 1.0 - current_pos % 1.0  # Normalizar al tile
            if remaining < 1.0 and remaining > 0.01:
                dash_space.append(f"-{round(remaining, 4)}")
            
            # Calcular shift basado en el ángulo
            ang_rad = math.radians(ang)
            if ang == 0 or ang == 180:
                s_x, s_y = 1, 1
            elif ang == 90 or ang == 270:
                s_x, s_y = 1, 1
            elif ang == 45 or ang == 225:
                s_x, s_y = 0.707106781, 0.707106781
            elif ang == 135 or ang == 315:
                s_x, s_y = 0.707106781, 0.707106781
            else:
                # Para otros ángulos, usar valores del ghiaia3 como referencia
                # El shift debe ser perpendicular y sumar 1 en magnitud
                cos_a = abs(math.cos(ang_rad))
                sin_a = abs(math.sin(ang_rad))
                s_x = round(cos_a, 6)
                s_y = round(sin_a, 6)
                if s_x == 0: s_x = 1
                if s_y == 0: s_y = 1
            
            dash_str = ",".join(dash_space)
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