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
        
        # NUEVO: Usar detección de bordes Canny en lugar de binarización + esqueletización
        # Esto detecta SOLO los bordes/líneas, sin crear líneas internas
        edges = cv2.Canny(gray, 50, 150)
        
        # Opcional: dilatar ligeramente para conectar líneas rotas
        if closing_size > 0:
            kernel = np.ones((2, 2), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Si el usuario quiere esqueletizar, adelgazar las líneas
        if use_skeleton:
            edges = (skeletonize(edges > 0) * 255).astype(np.uint8)
        
        binary = edges
        
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        vec_preview = np.ones((side, side, 3), dtype=np.uint8) * 255
        
        # Cada segmento = 1 línea PAT con su longitud real
        pat_lines = []
        
        for cnt in contours:
            arc_len = cv2.arcLength(cnt, True)
            if arc_len < 15: continue
            
            approx = cv2.approxPolyDP(cnt, epsilon_factor * arc_len, True)
            pts = approx[:, 0, :]
            cv2.polylines(vec_preview, [pts], False, (0,0,0), 1, cv2.LINE_AA)

            for i in range(len(pts) - 1):  # No cerrar el contorno
                p1, p2 = pts[i], pts[i + 1]
                
                x1, y1 = p1[0] / side, (side - p1[1]) / side
                x2, y2 = p2[0] / side, (side - p2[1]) / side
                
                dx, dy = x2 - x1, y2 - y1
                L = math.sqrt(dx**2 + dy**2)
                if L < 0.02: continue
                
                ang = math.degrees(math.atan2(dy, dx))
                if ang < 0: ang += 360
                ang_q = quantize_angle(ang)
                
                ox = round(x1, 4)
                oy = round(y1, 4)
                
                # Dash = longitud REAL del segmento
                dash = round(L, 4)
                # Gap = complemento para evitar repetición visible
                gap = round(-(1.0 - L), 4)
                
                if ang_q in [45, 135]:
                    s_x, s_y = 0.7071067812, 0.7071067812
                else:
                    s_x, s_y = 1.0, 1.0
                
                line = f"{ang_q}, {ox},{oy}, {s_x},{s_y}, {dash},{gap}"
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
            "stats": f"Patrón generado: {len(pat_lines)} líneas."
        }