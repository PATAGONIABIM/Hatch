import cv2
import numpy as np
import math
from skimage.morphology import skeletonize
from collections import defaultdict

# Ángulos válidos extendidos (incluyendo los de ghiaia3)
VALID_ANGLES = [0, 26.565, 45, 63.435, 90, 116.565, 135, 153.435, 180, 206.565, 225, 243.435, 270, 296.565, 315, 333.435]

# Shift por ángulo - calculado como (cos(ang), sin(ang)) para ángulos no cardinales
def get_shift(ang):
    if ang in [0, 180]:
        return (1, 1)
    elif ang in [90, 270]:
        return (1, 1)
    elif ang in [45, 135, 225, 315]:
        return (0.707106781, 0.707106781)
    elif ang in [26.565, 206.565]:
        return (0.894427191, 0.4472135955)
    elif ang in [63.435, 243.435]:
        return (0.4472135955, 0.894427191)
    elif ang in [116.565, 296.565]:
        return (0.4472135955, 0.894427191)
    elif ang in [153.435, 333.435]:
        return (0.894427191, 0.4472135955)
    else:
        rad = math.radians(ang)
        return (abs(math.cos(rad)), abs(math.sin(rad)))

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
        
        # NUEVO ENFOQUE: Estilo ghiaia3 - cada segmento genera UNA línea con dash corto
        pat_lines = []
        
        # Dash corto fijo (estilo ghiaia3: 0.05 a 0.1)
        DASH_LENGTH = 0.07
        
        for cnt in contours:
            arc_len = cv2.arcLength(cnt, True)
            if arc_len < 20: continue  # Ignorar contornos muy pequeños
            
            # Usar epsilon más alto para menos vértices
            approx = cv2.approxPolyDP(cnt, epsilon_factor * 2 * arc_len, True)
            pts = approx[:, 0, :]
            cv2.polylines(vec_preview, [pts], True, (0,0,0), 1, cv2.LINE_AA)

            for i in range(len(pts)):
                p1, p2 = pts[i], pts[(i + 1) % len(pts)]
                
                # Normalizar a 0-1
                x1, y1 = p1[0] / side, (side - p1[1]) / side
                x2, y2 = p2[0] / side, (side - p2[1]) / side
                
                dx, dy = x2 - x1, y2 - y1
                L = math.sqrt(dx**2 + dy**2)
                if L < 0.04: continue  # Ignorar segmentos muy cortos
                
                # Calcular ángulo real
                ang = math.degrees(math.atan2(dy, dx))
                if ang < 0: ang += 360
                
                # Cuantizar al ángulo válido más cercano
                ang_q = quantize_angle(ang)
                
                # Obtener shift para este ángulo
                s_x, s_y = get_shift(ang_q)
                
                # Calcular el dash y espacio
                # Dash corto (como ghiaia3: ~0.05-0.1)
                dash = min(DASH_LENGTH, L)
                
                # Espacio = resto del unit cell (para que no se repita densamente)
                space = -(1.0 - dash)
                
                # Redondear origen
                ox = round(x1, 2)
                oy = round(y1, 2)
                
                # Formatear línea
                line = f"{ang_q}, {ox},{oy}, {s_x},{s_y}, {round(dash, 4)},{round(space, 4)}"
                pat_lines.append(line)
        
        # Construir archivo .PAT
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
            "stats": f"Patrón generado: {len(pat_lines)} líneas. Tipo: MODELO (estilo ghiaia3)."
        }