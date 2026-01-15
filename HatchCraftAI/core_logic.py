import cv2
import numpy as np
import math
from skimage.morphology import skeletonize

class PatternGenerator:
    def __init__(self, size=100.0):
        self.size = float(size)

    def process_image(self, image_file, epsilon_factor=0.01, closing_size=2, mode="Auto-Detectar", use_skeleton=True):
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None: return {"error": "Error al leer imagen"}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Binarización
        binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Lógica de Inversión
        white_pix = np.sum(binary == 255)
        black_pix = binary.size - white_pix
        if (mode == "Auto-Detectar" and white_pix > black_pix) or (mode == "Fuerza Fondo Blanco (Líneas Negras)"):
            binary = cv2.bitwise_not(binary)

        # Unión de líneas
        if closing_size > 0:
            kernel = np.ones((closing_size, closing_size), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Esqueleto o Contorno
        if use_skeleton:
            skeleton = skeletonize(binary > 0)
            final_bin = (skeleton * 255).astype(np.uint8)
        else:
            final_bin = binary

        # Vectorización (Extraer líneas)
        contours, _ = cv2.findContours(final_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = gray.shape
        scale_x = self.size / w
        scale_y = self.size / h
        
        # Canvas de Previsualización: BLANCO puro
        vec_preview = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        pat_lines = [f"*HatchCraft_V21, Preview_BlackOnWhite", ";%TYPE=MODEL"]
        count = 0

        for cnt in contours:
            if cv2.arcLength(cnt, True) < 5: continue
            
            epsilon = epsilon_factor * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            pts = approx[:, 0, :]

            # Dibujar en la previsualización: NEGRO (0, 0, 0)
            cv2.polylines(vec_preview, [pts], True, (0, 0, 0), 1, cv2.LINE_AA)

            for i in range(len(pts)):
                p1 = pts[i]
                p2 = pts[(i + 1) % len(pts)]
                
                # Coordenadas para el .PAT
                x1, y1 = p1[0] * scale_x, (h - p1[1]) * scale_y
                x2, y2 = p2[0] * scale_x, (h - p2[1]) * scale_y
                
                dx, dy = x2 - x1, y2 - y1
                dist = math.sqrt(dx**2 + dy**2)
                if dist < 0.001: continue
                
                angle = math.degrees(math.atan2(dy, dx))
                if angle < 0: angle += 360
                
                # Definición de línea .PAT estándar
                # Ángulo, X-Origen, Y-Origen, Delta-X, Delta-Y, Trazo, Espacio
                line = f"{angle:.4f}, {x1:.5f},{y1:.5f}, {self.size:.5f},{self.size:.5f}, {dist:.5f},-2000"
                pat_lines.append(line)
                count += 1

        return {
            "processed_img": final_bin,
            "vector_img": vec_preview, # Esta es la imagen fondo blanco líneas negras
            "pat_content": "\n".join(pat_lines),
            "stats": f"Geometría: {count} segmentos vectoriales detectados."
        }