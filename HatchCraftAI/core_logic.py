import cv2
import numpy as np
import math
from skimage.morphology import skeletonize

class PatternGenerator:
    def __init__(self, size=100.0):
        self.size = float(size)

    def process_image(self, image_file, epsilon_factor=0.005, closing_size=2, mode="Auto-Detectar", use_skeleton=True):
        # 1. Leer imagen
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None: return {"error": "Error al cargar imagen"}

        # --- AUTO-CROP: Forzamos un cuadrado perfecto ---
        h_orig, w_orig = img.shape[:2]
        side = min(h_orig, w_orig)
        start_x = (w_orig - side) // 2
        start_y = (h_orig - side) // 2
        img = img[start_y:start_y+side, start_x:start_x+side]
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Inversión de color
        if (mode == "Auto-Detectar" and np.sum(binary==255) > binary.size/2) or mode == "Líneas Negras":
            binary = cv2.bitwise_not(binary)

        # Conectividad
        if closing_size > 0:
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((closing_size,closing_size), np.uint8))

        # Esqueleto
        if use_skeleton:
            binary = (skeletonize(binary > 0) * 255).astype(np.uint8)

        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        vec_preview = np.ones((side, side, 3), dtype=np.uint8) * 255
        
        # Header - basado en el ejemplo ghiaia3 que funciona
        lines = [
            "*HatchCraftModel, Generated Pattern",
            ";%TYPE=MODEL"
        ]
        
        count = 0
        for cnt in contours:
            if cv2.arcLength(cnt, True) < 5: continue
            approx = cv2.approxPolyDP(cnt, epsilon_factor * cv2.arcLength(cnt, True), True)
            pts = approx[:, 0, :]
            
            cv2.polylines(vec_preview, [pts], True, (0,0,0), 1, cv2.LINE_AA)

            for i in range(len(pts)):
                p1, p2 = pts[i], pts[(i + 1) % len(pts)]
                
                # COORDENADAS NORMALIZADAS AL RANGO [0, 1]
                # Esto es CRÍTICO - el ejemplo ghiaia3 usa valores como .4, .1, etc.
                x1 = p1[0] / side  # Normalizar a 0-1
                y1 = (side - p1[1]) / side  # Flip Y + normalizar
                x2 = p2[0] / side
                y2 = (side - p2[1]) / side
                
                dx, dy = x2 - x1, y2 - y1
                L = math.sqrt(dx**2 + dy**2)
                if L < 0.001: continue
                
                ang = math.degrees(math.atan2(dy, dx))
                if ang < 0: ang += 360
                
                # SHIFT BASADO EN EL EJEMPLO GHIAIA3
                # Para 0° y 90°: shift = (1, 1)
                # Para 45°: shift = (0.707, 0.707)
                # Para otros ángulos: calcular perpendicular
                
                ang_rad = math.radians(ang)
                
                # Determinar shift basado en ángulos comunes
                if abs(ang) < 1 or abs(ang - 180) < 1 or abs(ang - 360) < 1:
                    # Horizontal
                    s_x = 1
                    s_y = 1
                elif abs(ang - 90) < 1 or abs(ang - 270) < 1:
                    # Vertical
                    s_x = 1
                    s_y = 1
                elif abs(ang - 45) < 5 or abs(ang - 225) < 5:
                    # Diagonal 45°
                    s_x = 0.707106781
                    s_y = 0.707106781
                elif abs(ang - 135) < 5 or abs(ang - 315) < 5:
                    # Diagonal 135°
                    s_x = 0.707106781
                    s_y = 0.707106781
                else:
                    # Otros ángulos: usar vector perpendicular
                    s_x = round(math.cos(ang_rad + math.pi/2), 6)
                    s_y = round(math.sin(ang_rad + math.pi/2), 6)
                
                # Redondear valores
                ang = round(ang, 2)
                x1 = round(x1, 4)
                y1 = round(y1, 4)
                L = round(L, 4)
                
                # Formatear sin .0 innecesario
                def fmt(val):
                    if isinstance(val, float) and val == int(val):
                        return str(int(val))
                    return str(val)
                
                # Formato: angle, x, y, shift_x, shift_y, dash, -space
                # Usamos -.95 para el space como en ghiaia3
                space = round(-0.95, 2)
                
                line = f"{fmt(ang)}, {fmt(x1)},{fmt(y1)}, {fmt(s_x)},{fmt(s_y)}, {fmt(L)},{space}"
                lines.append(line)
                count += 1
        
        # JOIN con CRLF
        full_content = "\r\n".join(lines)
        full_content += "\r\n"
        
        return {
            "processed_img": binary,
            "vector_img": vec_preview,
            "pat_content": full_content,
            "stats": f"Patrón generado: {count} segmentos. Tipo: MODELO."
        }