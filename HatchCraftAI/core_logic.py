import cv2
import numpy as np
import math
import base64
import requests

VALID_ANGLES = [0, 45, 90, 135]

def quantize_angle(ang):
    """Redondea el ángulo al válido más cercano"""
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
    """Renderiza el patrón PAT como lo vería Revit"""
    img = np.ones((preview_size, preview_size, 3), dtype=np.uint8) * 255
    tile_size = preview_size / tile_count
    
    lines = pat_content.strip().replace('\r\n', '\n').split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('*') or line.startswith(';'):
            continue
        if ';' in line:
            line = line.split(';')[0].strip()
        
        try:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 5:
                continue
                
            angle = float(parts[0])
            ox, oy = float(parts[1]), float(parts[2])
            dx, dy = float(parts[3]), float(parts[4])
            
            dash_pattern = []
            if len(parts) > 5:
                dash_pattern = [float(p) for p in parts[5:] if p.strip()]
            
            for tile_x in range(tile_count):
                for tile_y in range(tile_count):
                    base_x = tile_x * tile_size + ox * tile_size
                    base_y = preview_size - (tile_y * tile_size + oy * tile_size)
                    
                    ang_rad = math.radians(angle)
                    dir_x = math.cos(ang_rad)
                    dir_y = -math.sin(ang_rad)
                    
                    if not dash_pattern:
                        x1 = int(base_x)
                        y1 = int(base_y)
                        x2 = int(base_x + dir_x * tile_size)
                        y2 = int(base_y + dir_y * tile_size)
                        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 1, cv2.LINE_AA)
                    else:
                        pos = 0
                        while pos < tile_size * 1.5:
                            for dash_val in dash_pattern:
                                length = abs(dash_val) * tile_size
                                if dash_val > 0:
                                    x1 = int(base_x + dir_x * pos)
                                    y1 = int(base_y + dir_y * pos)
                                    x2 = int(base_x + dir_x * (pos + length))
                                    y2 = int(base_y + dir_y * (pos + length))
                                    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 1, cv2.LINE_AA)
                                pos += length
                                if pos > tile_size * 1.5:
                                    break
                        
        except (ValueError, IndexError):
            continue
    
    for i in range(1, tile_count):
        pos = int(i * tile_size)
        cv2.line(img, (pos, 0), (pos, preview_size), (200, 200, 200), 1)
        cv2.line(img, (0, pos), (preview_size, pos), (200, 200, 200), 1)
    
    return img


class GeminiPatternGenerator:
    """Generador de patrones usando Gemini 1.5 Pro"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.model = "gemini-1.5-pro"  # Modelo más potente para análisis + texto
    
    def generate_pattern(self, image_bytes):
        """Genera un patrón PAT desde una imagen usando Gemini 3 Pro"""
        
        # Comprimir imagen
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "Error al cargar la imagen"}
        
        # Redimensionar
        max_dim = 1024
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        
        _, img_encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
        
        # Prompt optimizado para Gemini 3 Pro con "Thinking"
        prompt = """You are an expert in Revit/AutoCAD hatch pattern files (.PAT format).

TASK: Analyze this tile/brick pattern image and generate a precise PAT file that recreates it.

STEP 1 - ANALYSIS:
Look at the image carefully and identify:
- What type of pattern is this? (bricks, tiles, herringbone, basketweave, stone, etc.)
- What are the main line directions? (horizontal at 0°, vertical at 90°, diagonal at 45° or 135°)
- How are elements arranged? (staggered, aligned, rotated groups, etc.)
- What is the approximate ratio/proportion of elements?

STEP 2 - PAT FORMAT RULES:
Each line in a PAT file defines a family of parallel lines:
angle, x-origin, y-origin, delta-x, delta-y, dash, -gap, dash, -gap...

Where:
- angle: Direction (only use 0, 45, 90, or 135)
- x-origin, y-origin: Starting point (normalized 0-1)
- delta-x: Offset along line direction for next parallel line
- delta-y: Perpendicular spacing between parallel lines
- dash: Length to draw (positive number)
- gap: Length to skip (NEGATIVE number)

STEP 3 - GENERATE THE PAT:
Create a PAT file with:
- Header: *PatternName, Description
- Type line: ;%TYPE=MODEL
- Pattern lines: One line per family of parallel lines needed

CRITICAL RULES:
1. Use ONLY angles 0, 45, 90, 135
2. All coordinates normalized to 0-1 range
3. For horizontal/vertical lines: delta-y should be the spacing between courses
4. Dash lengths should match the visible line segments
5. Generate 5-20 lines depending on pattern complexity

OUTPUT: Return ONLY the PAT file content. Start with *PatternName. No explanations."""

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": img_base64}}
                ]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 4096
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            # Debug: imprimir estructura de respuesta
            import json as json_module
            
            # Extraer texto - manejar diferentes estructuras
            pat_content = None
            
            if 'candidates' in result and len(result['candidates']) > 0:
                candidate = result['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    parts = candidate['content']['parts']
                    if len(parts) > 0 and 'text' in parts[0]:
                        pat_content = parts[0]['text'].strip()
                elif 'text' in candidate:
                    pat_content = candidate['text'].strip()
            elif 'text' in result:
                pat_content = result['text'].strip()
            
            if not pat_content:
                return {"error": f"Respuesta vacía del modelo. Estructura: {json_module.dumps(result, indent=2)[:500]}"}
            
            # Limpiar markdown si existe
            if '```' in pat_content:
                import re
                pat_content = re.sub(r'^```\w*\n?', '', pat_content)
                pat_content = re.sub(r'\n?```$', '', pat_content)
            
            # Asegurar que empiece con *
            if not pat_content.startswith('*'):
                lines = pat_content.split('\n')
                for i, line in enumerate(lines):
                    if line.strip().startswith('*'):
                        pat_content = '\n'.join(lines[i:])
                        break
            
            # Contar líneas de patrón
            num_lines = len([l for l in pat_content.split('\n') if l.strip() and not l.startswith('*') and not l.startswith(';')])
            
            # Generar preview
            pat_preview = render_pat_preview(pat_content)
            
            return {
                "pat_content": pat_content + "\r\n",
                "pat_preview": pat_preview,
                "stats": f"✅ Patrón generado con Gemini 3 Pro: {num_lines} líneas"
            }
            
        except requests.exceptions.HTTPError as e:
            error_msg = str(e)
            try:
                error_detail = response.json()
                if 'error' in error_detail:
                    error_msg = error_detail['error'].get('message', str(e))
            except:
                pass
            return {"error": f"Error API: {error_msg}"}
        except requests.exceptions.Timeout:
            return {"error": "Timeout - El modelo tardó demasiado. Intenta con una imagen más pequeña."}
        except KeyError as e:
            return {"error": f"Error parseando respuesta: {str(e)}"}
        except Exception as e:
            return {"error": f"Error: {str(e)}"}