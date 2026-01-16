import cv2
import numpy as np
import math
import ezdxf

def render_pat_preview(pat_content, tile_count=3, preview_size=600, manual_scale=1.0):
    """Renderiza el patrón PAT como lo vería Revit"""
    img = np.ones((preview_size, preview_size, 3), dtype=np.uint8) * 255
    
    lines_data = pat_content.strip().replace('\r\n', '\n').split('\n')
    
    # Encontrar límites y segmentos
    tile_size = 1.0
    segments = []
    
    for line in lines_data:
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
            
            tile_size = max(tile_size, dx, dy)
            
            segments.append({
                'ox': ox, 'oy': oy,
                'dx': dx, 'dy': dy,
                'angle': angle,
                'dash_pattern': dash_pattern
            })
        except:
            continue
    
    if not segments:
        return img
    
    # Escala basada en tile_size y manual_scale
    pattern_size = tile_size * tile_count
    scale = (preview_size / pattern_size) * manual_scale
    
    # Dibujar cada segmento para cada tile
    for seg in segments:
        ang_rad = math.radians(seg['angle'])
        dir_x = math.cos(ang_rad)
        dir_y = math.sin(ang_rad)
        
        for tile_x in range(tile_count):
            for tile_y in range(tile_count):
                base_x = (seg['ox'] + tile_x * seg['dx']) * scale
                base_y = preview_size - (seg['oy'] + tile_y * seg['dy']) * scale
                
                if seg['dash_pattern']:
                    pos = 0
                    for dash_val in seg['dash_pattern']:
                        length = abs(dash_val) * scale
                        if dash_val > 0:
                            x1 = int(base_x + dir_x * pos)
                            y1 = int(base_y - dir_y * pos)
                            x2 = int(base_x + dir_x * (pos + length))
                            y2 = int(base_y - dir_y * (pos + length))
                            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 1, cv2.LINE_AA)
                        pos += length
                else:
                    length = tile_size * scale * 0.5
                    x1 = int(base_x)
                    y1 = int(base_y)
                    x2 = int(base_x + dir_x * length)
                    y2 = int(base_y - dir_y * length)
                    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 1, cv2.LINE_AA)
    
    # Grid
    tile_px = preview_size / tile_count
    for i in range(1, tile_count):
        pos = int(i * tile_px)
        cv2.line(img, (pos, 0), (pos, preview_size), (200, 200, 200), 1)
        cv2.line(img, (0, pos), (preview_size, pos), (200, 200, 200), 1)
    
    return img


def render_dxf_debug(lines_data, min_x, min_y, tile_size, preview_size=500):
    """Renderiza una vista de debug del DXF mostrando los segmentos detectados"""
    img = np.ones((preview_size, preview_size, 3), dtype=np.uint8) * 255
    
    if not lines_data or tile_size == 0:
        return img
    
    # Escala para que el tile quepa en el preview
    scale = preview_size / tile_size * 0.9
    offset = preview_size * 0.05
    
    # Colores para diferentes ángulos
    colors = {
        0: (255, 0, 0),    # Rojo - horizontal
        90: (0, 0, 255),   # Azul - vertical
        45: (0, 255, 0),   # Verde - diagonal
        135: (255, 0, 255) # Magenta - diagonal inversa
    }
    
    for x1, y1, x2, y2 in lines_data:
        # Normalizar al origen
        nx1 = (x1 - min_x) * scale + offset
        ny1 = preview_size - ((y1 - min_y) * scale + offset)
        nx2 = (x2 - min_x) * scale + offset
        ny2 = preview_size - ((y2 - min_y) * scale + offset)
        
        # Determinar ángulo para color
        dx = x2 - x1
        dy = y2 - y1
        ang = math.degrees(math.atan2(dy, dx))
        if ang < 0:
            ang += 360
        if ang >= 180:
            ang -= 180
        
        # Color según ángulo aproximado
        if abs(ang - 0) < 10 or abs(ang - 180) < 10:
            color = colors[0]
        elif abs(ang - 90) < 10:
            color = colors[90]
        elif abs(ang - 45) < 10:
            color = colors[45]
        elif abs(ang - 135) < 10:
            color = colors[135]
        else:
            color = (100, 100, 100)
        
        cv2.line(img, (int(nx1), int(ny1)), (int(nx2), int(ny2)), color, 2, cv2.LINE_AA)
    
    # Dibujar borde del tile
    cv2.rectangle(img, (int(offset), int(offset)), 
                  (int(offset + tile_size * scale), int(preview_size - offset - tile_size * scale)), 
                  (150, 150, 150), 1)
    
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
            
            # Tamaño del tile
            width = max_x - min_x
            height = max_y - min_y
            tile_size = max(width, height)
            
            if tile_size == 0:
                return {"error": "El dibujo tiene tamaño cero"}
            
            # Generar imagen de debug
            debug_img = render_dxf_debug(lines_data, min_x, min_y, tile_size)
            
            # Generar líneas PAT - NORMALIZANDO AL ORIGEN
            pat_lines = []
            
            for x1, y1, x2, y2 in lines_data:
                # NORMALIZAR coordenadas al origen (0,0)
                nx1 = x1 - min_x
                ny1 = y1 - min_y
                nx2 = x2 - min_x
                ny2 = y2 - min_y
                
                dx = nx2 - nx1
                dy = ny2 - ny1
                length = math.sqrt(dx**2 + dy**2)
                
                if length < 0.001:
                    continue
                
                # Ángulo
                ang = math.degrees(math.atan2(dy, dx))
                if ang < 0:
                    ang += 360
                
                # Cuantizar ángulo al más cercano (manejando wrap-around)
                def angle_diff(a, b):
                    """Diferencia mínima entre dos ángulos considerando el wrap-around"""
                    diff = abs(a - b)
                    return min(diff, 360 - diff)
                
                valid_angles = [0, 45, 90, 135, 180, 225, 270, 315]
                ang_q = min(valid_angles, key=lambda a: angle_diff(a, ang))
                
                # Normalizar a 0-180 para PAT
                if ang_q >= 180:
                    ang_q = ang_q - 180
                
                # Origen normalizado
                ox = round(nx1, 6)
                oy = round(ny1, 6)
                
                # Delta depende del ángulo
                # Para líneas H/V: el tile se repite en cuadrícula regular
                # Para diagonales: ajustar para que el patrón tile correctamente
                delta_x = round(tile_size, 6)
                delta_y = round(tile_size, 6)
                
                # Dash/gap - la longitud de la línea y el espacio
                dash = round(length, 6)
                # Gap debe ser negativo y = tile_size - length (para que no se repita dentro del mismo tile)
                gap = round(-(tile_size - length), 6)
                
                # Si el gap es mayor o igual a 0, hacer continua la línea
                if gap >= 0:
                    gap = -0.001
                
                pat_line = f"{ang_q}, {ox},{oy}, {delta_x},{delta_y}, {dash},{gap}"
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
                "debug_img": debug_img,
                "stats": f"✅ DXF: {len(lines_data)} segmentos → PAT: {len(pat_lines)} líneas (tile={tile_size:.2f})"
            }
            
        except ezdxf.DXFError as e:
            return {"error": f"Error leyendo DXF: {str(e)}"}
        except Exception as e:
            return {"error": f"Error: {str(e)}"}