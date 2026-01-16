import cv2
import numpy as np
import math
import ezdxf

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


class DXFtoPatConverter:
    """Convierte archivos DXF de AutoCAD a formato PAT"""
    
    def __init__(self, tile_size=1.0):
        self.tile_size = tile_size
    
    def convert(self, dxf_file):
        """Lee un archivo DXF y genera un archivo PAT"""
        try:
            # Leer el DXF
            doc = ezdxf.read(dxf_file)
            msp = doc.modelspace()
            
            # Encontrar los límites del dibujo
            min_x, min_y = float('inf'), float('inf')
            max_x, max_y = float('-inf'), float('-inf')
            
            lines_data = []
            
            # Extraer todas las líneas
            for entity in msp:
                if entity.dxftype() == 'LINE':
                    x1, y1 = entity.dxf.start.x, entity.dxf.start.y
                    x2, y2 = entity.dxf.end.x, entity.dxf.end.y
                    
                    min_x = min(min_x, x1, x2)
                    min_y = min(min_y, y1, y2)
                    max_x = max(max_x, x1, x2)
                    max_y = max(max_y, y1, y2)
                    
                    lines_data.append((x1, y1, x2, y2))
                
                elif entity.dxftype() == 'LWPOLYLINE':
                    points = list(entity.get_points())
                    for i in range(len(points) - 1):
                        x1, y1 = points[i][0], points[i][1]
                        x2, y2 = points[i+1][0], points[i+1][1]
                        
                        min_x = min(min_x, x1, x2)
                        min_y = min(min_y, y1, y2)
                        max_x = max(max_x, x1, x2)
                        max_y = max(max_y, y1, y2)
                        
                        lines_data.append((x1, y1, x2, y2))
                    
                    # Si es cerrada, conectar último con primero
                    if entity.closed and len(points) > 2:
                        x1, y1 = points[-1][0], points[-1][1]
                        x2, y2 = points[0][0], points[0][1]
                        lines_data.append((x1, y1, x2, y2))
            
            if not lines_data:
                return {"error": "No se encontraron líneas en el archivo DXF"}
            
            # Calcular el tamaño del tile
            width = max_x - min_x
            height = max_y - min_y
            tile_dim = max(width, height)
            
            if tile_dim == 0:
                return {"error": "El dibujo tiene tamaño cero"}
            
            # Generar líneas PAT
            pat_lines = []
            
            for x1, y1, x2, y2 in lines_data:
                # Normalizar coordenadas a 0-1
                nx1 = (x1 - min_x) / tile_dim
                ny1 = (y1 - min_y) / tile_dim
                nx2 = (x2 - min_x) / tile_dim
                ny2 = (y2 - min_y) / tile_dim
                
                dx = nx2 - nx1
                dy = ny2 - ny1
                length = math.sqrt(dx**2 + dy**2)
                
                if length < 0.001:
                    continue
                
                # Calcular ángulo
                ang = math.degrees(math.atan2(dy, dx))
                if ang < 0:
                    ang += 360
                ang_q = quantize_angle(ang)
                
                # Origen y parámetros
                ox = round(nx1, 4)
                oy = round(ny1, 4)
                dash = round(length, 4)
                gap = round(-(1.0 - length), 4)
                
                # Shift para evitar repetición
                if ang_q in [45, 135]:
                    s_x, s_y = 0, 0.1414
                else:
                    s_x, s_y = 0, 0.25
                
                pat_line = f"{ang_q}, {ox},{oy}, {s_x},{s_y}, {dash},{gap}"
                pat_lines.append(pat_line)
            
            # Construir el archivo PAT
            header = [
                "*DXF_Pattern, Converted from AutoCAD DXF",
                ";%TYPE=MODEL"
            ]
            header.extend(pat_lines)
            
            pat_content = "\r\n".join(header) + "\r\n"
            
            # Generar preview
            pat_preview = render_pat_preview(pat_content)
            
            return {
                "pat_content": pat_content,
                "pat_preview": pat_preview,
                "stats": f"✅ Convertido: {len(pat_lines)} líneas desde DXF"
            }
            
        except ezdxf.DXFError as e:
            return {"error": f"Error leyendo DXF: {str(e)}"}
        except Exception as e:
            return {"error": f"Error: {str(e)}"}