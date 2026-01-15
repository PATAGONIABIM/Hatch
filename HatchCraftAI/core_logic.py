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
        if img is None: return {"error": "Error al leer imagen"}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        
        binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        white_pix = np.sum(binary == 255)
        black_pix = binary.size - white_pix
        if (mode == "Auto-Detectar" and white_pix > black_pix) or (mode == "Fuerza Fondo Blanco (Líneas Negras)"):
            binary = cv2.bitwise_not(binary)

        if closing_size > 0:
            kernel = np.ones((closing_size, closing_size), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        if use_skeleton:
            skeleton = skeletonize(binary > 0)
            final_bin = (skeleton * 255).astype(np.uint8)
        else:
            final_bin = binary

        contours, _ = cv2.findContours(final_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = gray.shape
        scale_x = self.size / w
        scale_y = self.size / h
        
        vec_preview = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # Cabecera de Revit
        pat_lines = [f"*HatchCraft_Revit, Unidades: {self.units}", ";%TYPE=MODEL"]
        count = 0

        for cnt in contours:
            if cv2.arcLength(cnt, True) < 5: continue
            
            epsilon = epsilon_factor * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            pts = approx[:, 0, :]

            cv2.polylines(vec_preview, [pts], True, (0, 0, 0), 1, cv2.LINE_AA)

            for i in range(len(pts)):
                p1 = pts[i]
                p2 = pts[(i + 1) % len(pts)]
                
                x1, y1 = p1[0] * scale_x, (h - p1[1]) * scale_y
                x2, y2 = p2[0] * scale_x, (h - p2[1]) * scale_y
                
                dx, dy = x2 - x1, y2 - y1
                dist = math.sqrt(dx**2 + dy**2)
                if dist < 0.001: continue
                
                angle = math.degrees(math.atan2(dy, dx))
                if angle < 0: angle += 360
                
                # --- MATEMÁTICA DE REPETICIÓN RECURRENTE (FIX PARA REVIT) ---
                # Queremos que la línea se repita cada 'self.size' unidades en los ejes globales.
                # En .pat, dx_pat y dy_pat son desplazamientos en el eje local de la línea.
                angle_rad = math.radians(angle)
                
                # Desplazamiento para repetir el patrón verticalmente cada 'size' unidades
                # dx = shift a lo largo de la línea, dy = shift perpendicular
                dx_pat = self.size * math.sin(angle_rad)
                dy_pat = self.size * math.cos(angle_rad)
                
                # Usamos un espacio negativo enorme (-9999) para que la línea no se repita
                # sobre sí misma, sino que cree el segmento y "salte" a la siguiente celda.
                line = f"{angle:.4f}, {x1:.5f},{y1:.5f}, {dx_pat:.5f},{dy_pat:.5f}, {dist:.5f}, {-9999:.1f}"
                pat_lines.append(line)
                count += 1

        return {
            "processed_img": final_bin,
            "vector_img": vec_preview,
            "pat_content": "\n".join(pat_lines),
            "stats": f"Geometría: {count} segmentos. Repetición: {self.size} {self.units}."
        }