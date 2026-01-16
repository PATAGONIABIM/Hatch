import cv2
import numpy as np
import math
import ezdxf
from collections import defaultdict

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
            
            # Dibujar múltiples líneas paralelas
            ang_rad = math.radians(angle)
            dir_x = math.cos(ang_rad)
            dir_y = -math.sin(ang_rad)
            
            # Vector perpendicular para el offset
            perp_x = -math.sin(math.radians(angle))
            perp_y = -math.cos(math.radians(angle))
            
            # Calcular cuántas líneas paralelas dibujar
            if dy > 0.001:
                num_lines = int(tile_count / dy) + 2
            else:
                num_lines = 1
            
            for tile_x in range(tile_count):
                for tile_y in range(tile_count):
                    for line_idx in range(-1, num_lines):
                        # Posición base
                        base_x = tile_x * tile_size + ox * tile_size + line_idx * dy * tile_size * perp_x
                        base_y = preview_size - (tile_y * tile_size + oy * tile_size) + line_idx * dy * tile_size * perp_y
                        
                        if not dash_pattern:
                            # Línea continua
                            x1 = int(base_x - dir_x * tile_size)
                            y1 = int(base_y - dir_y * tile_size)
                            x2 = int(base_x + dir_x * tile_size * 2)
                            y2 = int(base_y + dir_y * tile_size * 2)
                            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 1, cv2.LINE_AA)
                        else:
                            # Patrón de dashes
                            pos = -tile_size
                            while pos < tile_size * 2:
                                for dash_val in dash_pattern:
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
    
    # Grid
    for i in range(1, tile_count):
        pos = int(i * tile_size)
        cv2.line(img, (pos, 0), (pos, preview_size), (200, 200, 200), 1)
        cv2.line(img, (0, pos), (preview_size, pos), (200, 200, 200), 1)
    
    return img


class DXFtoPatConverter:
    """Convierte archivos DXF de AutoCAD a formato PAT"""
    
    def __init__(self):
        pass
    
    def convert(self, dxf_file_path):
        """Lee un archivo DXF y genera un archivo PAT"""
        try:
            doc = ezdxf.readfile(dxf_file_path)
            msp = doc.modelspace()
            
            # Extraer todas las líneas
            lines_data = []
            min_x, min_y = float('inf'), float('inf')
            max_x, max_y = float('-inf'), float('-inf')
            
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
                    
                    if entity.closed and len(points) > 2:
                        x1, y1 = points[-1][0], points[-1][1]
                        x2, y2 = points[0][0], points[0][1]
                        lines_data.append((x1, y1, x2, y2))
            
            if not lines_data:
                return {"error": "No se encontraron líneas en el archivo DXF"}
            
            # Normalizar al tamaño del tile
            width = max_x - min_x
            height = max_y - min_y
            tile_dim = max(width, height)
            
            if tile_dim == 0:
                return {"error": "El dibujo tiene tamaño cero"}
            
            # Agrupar líneas por ángulo
            angle_groups = defaultdict(list)
            
            for x1, y1, x2, y2 in lines_data:
                # Normalizar
                nx1 = (x1 - min_x) / tile_dim
                ny1 = (y1 - min_y) / tile_dim
                nx2 = (x2 - min_x) / tile_dim
                ny2 = (y2 - min_y) / tile_dim
                
                dx = nx2 - nx1
                dy = ny2 - ny1
                length = math.sqrt(dx**2 + dy**2)
                
                if length < 0.001:
                    continue
                
                # Ángulo
                ang = math.degrees(math.atan2(dy, dx))
                if ang < 0:
                    ang += 360
                ang_q = quantize_angle(ang)
                
                # Guardar como segmento normalizado
                angle_groups[ang_q].append({
                    'x1': nx1, 'y1': ny1,
                    'x2': nx2, 'y2': ny2,
                    'length': length
                })
            
            # Generar líneas PAT - una por cada grupo de ángulo
            pat_lines = []
            
            for angle, segments in angle_groups.items():
                if not segments:
                    continue
                
                # Para cada ángulo, calcular el espaciado entre líneas paralelas
                ang_rad = math.radians(angle)
                
                # Vector de dirección y perpendicular
                dir_x = math.cos(ang_rad)
                dir_y = math.sin(ang_rad)
                perp_x = -dir_y
                perp_y = dir_x
                
                # Proyectar todos los puntos de inicio sobre el eje perpendicular
                perp_positions = []
                for seg in segments:
                    # Posición perpendicular del punto de inicio
                    perp_pos = seg['x1'] * perp_x + seg['y1'] * perp_y
                    perp_positions.append((perp_pos, seg))
                
                # Ordenar por posición perpendicular
                perp_positions.sort(key=lambda x: x[0])
                
                # Calcular el espaciado (delta-y) - la distancia entre líneas paralelas
                if len(perp_positions) > 1:
                    spacings = []
                    for i in range(1, len(perp_positions)):
                        diff = perp_positions[i][0] - perp_positions[i-1][0]
                        if diff > 0.01:  # Ignorar líneas muy cercanas
                            spacings.append(diff)
                    
                    if spacings:
                        delta_y = min(spacings)  # Usar el espaciado más pequeño
                    else:
                        delta_y = 0.5
                else:
                    delta_y = 0.5
                
                # Usar el primer segmento como origen
                first_seg = perp_positions[0][1]
                ox = round(first_seg['x1'], 4)
                oy = round(first_seg['y1'], 4)
                
                # Calcular el patrón de dash/gap
                # Por ahora, usar línea continua para simplicidad
                # delta_x = 0 (no hay offset paralelo entre líneas)
                delta_x = 0
                delta_y = round(delta_y, 4)
                
                # Línea continua (sin dash/gap = dibujar todo)
                pat_line = f"{angle}, {ox},{oy}, {delta_x},{delta_y}"
                pat_lines.append(pat_line)
            
            # Construir el archivo PAT
            header = [
                "*DXF_Pattern, Converted from AutoCAD DXF",
                ";%TYPE=MODEL"
            ]
            header.extend(pat_lines)
            
            pat_content = "\r\n".join(header) + "\r\n"
            pat_preview = render_pat_preview(pat_content)
            
            return {
                "pat_content": pat_content,
                "pat_preview": pat_preview,
                "stats": f"✅ Convertido: {len(pat_lines)} familias de líneas ({len(lines_data)} segmentos)"
            }
            
        except ezdxf.DXFError as e:
            return {"error": f"Error leyendo DXF: {str(e)}"}
        except Exception as e:
            return {"error": f"Error: {str(e)}"}