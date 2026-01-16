import cv2
import numpy as np
import math
from skimage.morphology import skeletonize

# Solo 4 ángulos principales
VALID_ANGLES = [0, 45, 90, 135]

def quantize_angle(ang):
    """Redondea el ángulo a 0-180 y al válido más cercano"""
    ang = ang % 180
    best = 0
    best_diff = 180
    for valid in VALID_ANGLES:
        diff = min(abs(ang - valid), abs(ang - valid - 180), abs(ang - valid + 180))
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
        
        # PASO 1: Suavizar para eliminar ruido y texturas
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
        
        # PASO 2: Detección de bordes con umbrales más altos (menos sensible)
        edges = cv2.Canny(blurred, 80, 200)
        
        # PASO 3: Operaciones morfológicas para limpiar
        kernel = np.ones((3, 3), np.uint8)
        # Cerrar pequeños huecos
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        # Abrir para eliminar puntos pequeños
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        
        # PASO 4: Esqueletizar para adelgazar líneas
        if use_skeleton:
            edges = (skeletonize(edges > 0) * 255).astype(np.uint8)
        
        binary = edges
        
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        vec_preview = np.ones((side, side, 3), dtype=np.uint8) * 255
        
        # Cada segmento = 1 línea PAT
        pat_lines = []
        
        # Aumentar el umbral mínimo de longitud para filtrar ruido
        MIN_CONTOUR_LEN = 25
        MIN_SEGMENT_LEN = 0.03
        
        for cnt in contours:
            arc_len = cv2.arcLength(cnt, True)
            if arc_len < MIN_CONTOUR_LEN: continue
            
            # Usar epsilon más alto para simplificar más
            approx = cv2.approxPolyDP(cnt, epsilon_factor * 1.5 * arc_len, True)
            pts = approx[:, 0, :]
            cv2.polylines(vec_preview, [pts], False, (0,0,0), 1, cv2.LINE_AA)

            for i in range(len(pts) - 1):
                p1, p2 = pts[i], pts[i + 1]
                
                x1, y1 = p1[0] / side, (side - p1[1]) / side
                x2, y2 = p2[0] / side, (side - p2[1]) / side
                
                dx, dy = x2 - x1, y2 - y1
                L = math.sqrt(dx**2 + dy**2)
                if L < MIN_SEGMENT_LEN: continue
                
                ang = math.degrees(math.atan2(dy, dx))
                if ang < 0: ang += 360
                ang_q = quantize_angle(ang)
                
                ox = round(x1, 4)
                oy = round(y1, 4)
                dash = round(L, 4)
                gap = round(-(1.0 - L), 4)
                
                if ang_q in [45, 135]:
                    s_x, s_y = 0.7071067812, 0.7071067812
                else:
                    s_x, s_y = 1.0, 1.0
                
                line = f"{ang_q}, {ox},{oy}, {s_x},{s_y}, {dash},{gap}"
                pat_lines.append(line)
        
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
            "stats": f"Patrón generado: {len(pat_lines)} líneas."
        }