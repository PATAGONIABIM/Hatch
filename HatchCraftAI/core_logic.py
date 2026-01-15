import cv2
import numpy as np
import math
from skimage.morphology import skeletonize

class PatternGenerator:
    def __init__(self, size=100.0, units="Centímetros"):
        self.size = float(size)
        self.units = units

    def process_image(self, image_file, epsilon_factor=0.008, closing_size=3, mode="Auto-Detectar", use_skeleton=True):
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None: return {"error": "Error de imagen"}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Inversión inteligente
        if (mode == "Auto-Detectar" and np.sum(binary==255) > binary.size/2) or mode == "Líneas Negras":
            binary = cv2.bitwise_not(binary)

        if closing_size > 0:
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((closing_size,closing_size), np.uint8))

        if use_skeleton:
            binary = (skeletonize(binary > 0) * 255).astype(np.uint8)

        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = gray.shape
        sx, sy = self.size / w, self.size / h
        vec_preview = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        pat_content = f"*HatchCraft_Seamless, {self.units}\n;%TYPE=MODEL\n"
        count = 0

        for cnt in contours:
            if cv2.arcLength(cnt, True) < 5: continue
            approx = cv2.approxPolyDP(cnt, epsilon_factor * cv2.arcLength(cnt, True), True)
            pts = approx[:, 0, :]
            cv2.polylines(vec_preview, [pts], True, (0,0,0), 1, cv2.LINE_AA)

            for i in range(len(pts)):
                p1, p2 = pts[i], pts[(i + 1) % len(pts)]
                x1, y1 = p1[0] * sx, (h - p1[1]) * sy
                x2, y2 = p2[0] * sx, (h - p2[1]) * sy
                
                dx, dy = x2 - x1, y2 - y1
                L = math.sqrt(dx**2 + dy**2)
                if L < 0.001: continue
                
                ang = math.degrees(math.atan2(dy, dx))
                if ang < 0: ang += 360
                rad = math.radians(ang)

                # --- LÓGICA DE TILING POR CUADRANTE ---
                # Si la línea es más horizontal, repetimos verticalmente (Vector 0, S)
                if abs(math.cos(rad)) >= abs(math.sin(rad)):
                    dx_p = self.size * math.sin(rad)
                    dy_p = self.size * math.cos(rad)
                    period = self.size / abs(math.cos(rad))
                # Si es más vertical, repetimos horizontalmente (Vector S, 0)
                else:
                    dx_p = self.size * math.cos(rad)
                    dy_p = -self.size * math.sin(rad)
                    period = self.size / abs(math.sin(rad))

                # Calculamos el espacio para que el segmento no se pierda en su propio eje
                space = -(period - L)
                
                line = f"{ang:.4f}, {x1:.5f},{y1:.5f}, {dx_p:.5f},{dy_p:.5f}, {L:.5f}, {space:.5f}\n"
                pat_content += line
                count += 1

        return {
            "processed_img": binary,
            "vector_img": vec_preview,
            "pat_content": pat_content,
            "stats": f"Muro completado: {count} segmentos vectorizados."
        }