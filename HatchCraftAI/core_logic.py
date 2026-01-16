import cv2
import numpy as np
import math
import base64
import json
import re
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
    """Renderiza el patrón PAT como lo vería Revit (con tiles repetidos)"""
    img = np.ones((preview_size, preview_size, 3), dtype=np.uint8) * 255
    tile_size = preview_size / tile_count
    
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
            dash_pattern = [float(p) for p in parts[5:]]
            
            for tile_x in range(tile_count):
                for tile_y in range(tile_count):
                    base_x = tile_x * tile_size + ox * tile_size
                    base_y = preview_size - (tile_y * tile_size + oy * tile_size)
                    
                    ang_rad = math.radians(angle)
                    dir_x = math.cos(ang_rad)
                    dir_y = -math.sin(ang_rad)
                    
                    pos = 0
                    for i, dash_val in enumerate(dash_pattern):
                        length = abs(dash_val) * tile_size
                        if dash_val > 0:
                            x1 = int(base_x + dir_x * pos)
                            y1 = int(base_y + dir_y * pos)
                            x2 = int(base_x + dir_x * (pos + length))
                            y2 = int(base_y + dir_y * (pos + length))
                            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 1, cv2.LINE_AA)
                        pos += length
                        
        except (ValueError, IndexError):
            continue
    
    for i in range(1, tile_count):
        pos = int(i * tile_size)
        cv2.line(img, (pos, 0), (pos, preview_size), (200, 200, 200), 1)
        cv2.line(img, (0, pos), (preview_size, pos), (200, 200, 200), 1)
    
    return img


class AIPatternGenerator:
    """Generador de patrones usando IA (Gemini Vision)"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        
    def analyze_and_generate(self, image_bytes, tile_size=100.0):
        """Analiza la imagen con Gemini y genera un patrón PAT geométrico"""
        import requests
        from io import BytesIO
        
        # Comprimir imagen para reducir tamaño (máx 800px)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "Error al cargar la imagen"}
        
        # Redimensionar si es muy grande
        max_dim = 800
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        
        # Convertir a JPEG comprimido
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        _, img_encoded = cv2.imencode('.jpg', img, encode_param)
        img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
        
        # Prompt para Gemini
        prompt = """Analyze this image which shows a texture or pattern (like bricks, tiles, herringbone, etc).

I need you to generate a Revit/AutoCAD PAT hatch pattern file that reproduces this pattern geometrically.

First, identify:
1. Pattern type (herringbone, running bond, stack bond, random, etc.)
2. Main angles used (typically 0, 45, 90, 135 degrees)
3. Approximate proportions (width/height ratio of elements)

Then generate the PAT file content. The format is:
*PatternName, Description
;%TYPE=MODEL
angle, x-origin, y-origin, delta-x, delta-y, dash-length, -gap-length, ...

Rules:
- Coordinates should be normalized 0-1 (unit cell)
- Use angles: 0, 45, 90, 135 only
- For 0° or 90°: use shift 1,1
- For 45° or 135°: use shift 0.7071067812,0.7071067812
- Keep lines minimal but representative of the pattern structure
- Maximum 20 lines for clean patterns

Return ONLY the PAT file content, nothing else. Start with *PatternName line."""

        # API call a Gemini
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.api_key}"
        
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": img_base64
                        }
                    }
                ]
            }],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 2048
            }
        }
        
        try:
            # Timeout más largo (60s) y reintentos
            for attempt in range(3):
                try:
                    response = requests.post(url, json=payload, timeout=60)
                    response.raise_for_status()
                    break
                except requests.exceptions.Timeout:
                    if attempt == 2:
                        raise
                    continue
            
            result = response.json()
            
            # Extraer el texto de la respuesta
            pat_content = result['candidates'][0]['content']['parts'][0]['text']
            
            # Limpiar el contenido (quitar markdown si existe)
            pat_content = pat_content.strip()
            if pat_content.startswith('```'):
                pat_content = re.sub(r'^```\w*\n?', '', pat_content)
                pat_content = re.sub(r'\n?```$', '', pat_content)
            
            # Asegurar que empiece con *
            if not pat_content.startswith('*'):
                # Buscar la línea que empieza con *
                lines = pat_content.split('\n')
                for i, line in enumerate(lines):
                    if line.strip().startswith('*'):
                        pat_content = '\n'.join(lines[i:])
                        break
            
            # Generar preview
            pat_preview = render_pat_preview(pat_content)
            
            return {
                "pat_content": pat_content + "\r\n",
                "pat_preview": pat_preview,
                "stats": "Patrón generado con IA (Gemini Vision)"
            }
            
        except requests.exceptions.RequestException as e:
            return {"error": f"Error de conexión: {str(e)}"}
        except (KeyError, IndexError) as e:
            return {"error": f"Error parseando respuesta: {str(e)}"}
        except Exception as e:
            return {"error": f"Error inesperado: {str(e)}"}


class PatternGenerator:
    """Generador clásico de patrones (detección de bordes)"""
    
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
        
        if blur_size > 1:
            blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
            blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        else:
            blurred = gray
        
        edges = cv2.Canny(blurred, canny_low, canny_high)
        
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
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
        pat_preview = render_pat_preview(full_content)
        
        return {
            "processed_img": binary,
            "vector_img": vec_preview,
            "pat_preview": pat_preview,
            "pat_content": full_content,
            "stats": f"Patrón generado: {len(pat_lines)} líneas."
        }