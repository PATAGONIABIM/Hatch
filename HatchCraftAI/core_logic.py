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

        # --- AUTO-CROP: Forzamos un cuadrado perfecto para evitar traslapes ---
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
        
        scale = self.size / side
        vec_preview = np.ones((side, side, 3), dtype=np.uint8) * 255
        
        # --- ENCABEZADO "BULLETPROOF" PARA REVIT ---
        # El nombre NO DEBE tener espacios. No debe haber líneas vacías arriba.
        pat_content = "*HatchCraft_Modelo\n"
        pat_content += ";%TYPE=MODEL\n"
        
        count = 0
        for cnt in contours:
            if cv2.arcLength(cnt, True) < 5: continue
            approx = cv2.approxPolyDP(cnt, epsilon_factor * cv2.arcLength(cnt, True), True)
            pts = approx[:, 0, :]
            cv2.polylines(vec_preview, [pts], True, (0,0,0), 1, cv2.LINE_AA)

            for i in range(len(pts)):
                p1, p2 = pts[i], pts[(i + 1) % len(pts)]
                
                # Coordenadas relativas al cuadrado de repetición
                x1, y1 = p1[0] * scale, (side - p1[1]) * scale
                x2, y2 = p2[0] * scale, (side - p2[1]) * scale
                
                dx, dy = x2 - x1, y2 - y1
                L = math.sqrt(dx**2 + dy**2)
                if L < 0.001: continue
                
                ang = math.degrees(math.atan2(dy, dx))
                if ang < 0: ang += 360
                rad = math.radians(ang)

                # --- MATEMÁTICA DE TILING ESTRICTO ---
                # Usamos el desplazamiento perpendicular (Shift) basado en el tamaño S.
                # Para evitar traslapes, forzamos que cada segmento se dibuje una sola vez por tile.
                # Shift-x = S * sin(ang), Shift-y = S * cos(ang)
                s_x = self.size * math.sin(rad)
                s_y = self.size * math.cos(rad)
                
                # Espacio negativo muy grande para que no se repita en su propia línea
                # Esto obliga a Revit a usar solo el salto de tileado S.
                line = f"{ang:.5f}, {x1:.5f},{y1:.5f}, {s_x:.5f},{s_y:.5f}, {L:.5f}, -1000000.0\n"
                pat_content += line
                count += 1

        # Revit necesita una línea vacía al final del archivo
        pat_content += "\n"

        return {
            "processed_img": binary,
            "vector_img": vec_preview,
            "pat_content": pat_content,
            "stats": f"Patrón generado: {count} segmentos. Tipo: MODELO."
        }