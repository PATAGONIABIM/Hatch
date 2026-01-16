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

def render_pat_preview(pat_content, tile_count=3, preview_size=600):
    """
    Renderiza el patrón PAT como lo vería Revit (con tiles repetidos)
    """
    img = np.ones((preview_size, preview_size, 3), dtype=np.uint8) * 255
    tile_size = preview_size / tile_count
    
    # Parsear las líneas del PAT
    lines = pat_content.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith('*') or line.startswith(';') or not line:
            continue
        
        try:
            parts = [p.strip() for p in line.split(',')]
            angle = float(parts[0])
            ox, oy = float(parts[1]), float(parts[2])
            dx, dy = float(parts[3]), float(parts[4])
            
            # Parsear el patrón de dash-gap
            dash_pattern = [float(p) for p in parts[5:]]
            
            # Dibujar para cada tile
            for tile_x in range(tile_count):
                for tile_y in range(tile_count):
                    # Origen en píxeles para este tile
                    base_x = tile_x * tile_size + ox * tile_size
                    base_y = preview_size - (tile_y * tile_size + oy * tile_size)
                    
                    # Dirección de la línea
                    ang_rad = math.radians(angle)
                    dir_x = math.cos(ang_rad)
                    dir_y = -math.sin(ang_rad)  # Y invertido en imagen
                    
                    # Dibujar el patrón de dashes a lo largo de la línea
                    pos = 0
                    for i, dash_val in enumerate(dash_pattern):
                        length = abs(dash_val) * tile_size
                        if dash_val > 0:  # Es un dash (dibujar)
                            x1 = int(base_x + dir_x * pos)
                            y1 = int(base_y + dir_y * pos)
                            x2 = int(base_x + dir_x * (pos + length))
                            y2 = int(base_y + dir_y * (pos + length))
                            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 1, cv2.LINE_AA)
                        pos += length
                        
        except (ValueError, IndexError):
            continue
    
    # Dibujar bordes de tiles para referencia
    for i in range(1, tile_count):
        pos = int(i * tile_size)
        cv2.line(img, (pos, 0), (pos, preview_size), (200, 200, 200), 1)
        cv2.line(img, (0, pos), (preview_size, pos), (200, 200, 200), 1)
    
    return img

class PatternGenerator:
    def __init__(self, size=100.0):
        self.size = float(size)

    def process_image(self, image_file, epsilon_factor=0.005, closing_size=2, mode="Auto-Detectar", 
                     use_skeleton=True, canny_low=30, canny_high=100, blur_size=3,
                     min_contour_len=20, min_segment_len=0.025):
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None: return {"error": "Error al cargar imagen"}

        h_orig, w_orig = img.shape[:2]
        side = min(h_orig, w_orig)
        start_x = (w_orig - side) // 2
        start_y = (h_orig - side) // 2
        img = img[start_y:start_y+side, start_x:start_x+side]
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Suavizado configurable para reducir ruido
        if blur_size > 1:
            blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1  # Debe ser impar
            blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        else:
            blurred = gray
        
        # Detección de bordes con umbrales configurables
        edges = cv2.Canny(blurred, canny_low, canny_high)
        
        # Dilatar para conectar líneas rotas
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Esqueletizar para adelgazar
        if use_skeleton:
            edges = (skeletonize(edges > 0) * 255).astype(np.uint8)
        
        binary = edges
        
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        vec_preview = np.ones((side, side, 3), dtype=np.uint8) * 255
        
        pat_lines = []
        
        for cnt in contours:
            arc_len = cv2.arcLength(cnt, True)
            if arc_len < min_contour_len: continue
            
            approx = cv2.approxPolyDP(cnt, epsilon_factor * arc_len, True)
            pts = approx[:, 0, :]
            cv2.polylines(vec_preview, [pts], False, (0,0,0), 1, cv2.LINE_AA)

            for i in range(len(pts) - 1):
                p1, p2 = pts[i], pts[i + 1]
                
                x1, y1 = p1[0] / side, (side - p1[1]) / side
                x2, y2 = p2[0] / side, (side - p2[1]) / side
                
                dx, dy = x2 - x1, y2 - y1
                L = math.sqrt(dx**2 + dy**2)
                if L < min_segment_len: continue
                
                ang = math.degrees(math.atan2(dy, dx))
                if ang < 0: ang += 360
                ang_q = quantize_angle(ang)
                
                ox = round(x1, 4)
                oy = round(y1, 4)
                dash = round(L, 4)
                gap = round(-(1.0 - L), 4)
                
                # Shift estándar del formato PAT (como en ghiaia3)
                if ang_q in [45, 135]:
                    s_x, s_y = 0.7071067812, 0.7071067812
                else:
                    s_x, s_y = 1, 1
                
                line = f"{ang_q}, {ox},{oy}, {s_x},{s_y}, {dash},{gap}"
                pat_lines.append(line)
        
        header_lines = [
            "*HatchCraftModel, Generated Pattern",
            ";%TYPE=MODEL"
        ]
        header_lines.extend(pat_lines)
        
        full_content = "\r\n".join(header_lines) + "\r\n"
        
        # Generar preview del PAT como lo vería Revit
        pat_preview = render_pat_preview(full_content)
        
        return {
            "processed_img": binary,
            "vector_img": vec_preview,
            "pat_preview": pat_preview,
            "pat_content": full_content,
            "stats": f"Patrón generado: {len(pat_lines)} líneas."
        }