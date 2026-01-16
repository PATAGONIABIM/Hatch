import cv2
import numpy as np
import math

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


def lines_to_pat(lines_data, canvas_size):
    """Convierte líneas dibujadas a formato PAT"""
    pat_lines = []
    
    for line in lines_data:
        x1, y1, x2, y2 = line['x1'], line['y1'], line['x2'], line['y2']
        
        # Normalizar a 0-1
        nx1 = x1 / canvas_size
        ny1 = 1.0 - (y1 / canvas_size)  # Invertir Y
        nx2 = x2 / canvas_size
        ny2 = 1.0 - (y2 / canvas_size)
        
        dx = nx2 - nx1
        dy = ny2 - ny1
        length = math.sqrt(dx**2 + dy**2)
        
        if length < 0.01:
            continue
        
        ang = math.degrees(math.atan2(dy, dx))
        if ang < 0:
            ang += 360
        ang_q = quantize_angle(ang)
        
        ox = round(nx1, 4)
        oy = round(ny1, 4)
        dash = round(length, 4)
        gap = round(-(1.0 - length), 4)
        
        # Shift para que no se repita
        if ang_q in [45, 135]:
            s_x, s_y = 0, 0.1414
        else:
            s_x, s_y = 0, 0.25
        
        pat_line = f"{ang_q}, {ox},{oy}, {s_x},{s_y}, {dash},{gap}"
        pat_lines.append(pat_line)
    
    header = [
        "*HatchCraft_Manual, Hand-drawn Pattern",
        ";%TYPE=MODEL"
    ]
    header.extend(pat_lines)
    
    return "\r\n".join(header) + "\r\n"


def extract_lines_from_canvas(canvas_result):
    """Extrae líneas del resultado del canvas"""
    lines = []
    
    if canvas_result is None or canvas_result.json_data is None:
        return lines
    
    objects = canvas_result.json_data.get("objects", [])
    
    for obj in objects:
        if obj.get("type") == "line":
            # Línea simple
            x1 = obj.get("x1", 0) + obj.get("left", 0)
            y1 = obj.get("y1", 0) + obj.get("top", 0)
            x2 = obj.get("x2", 0) + obj.get("left", 0)
            y2 = obj.get("y2", 0) + obj.get("top", 0)
            lines.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
        elif obj.get("type") == "path":
            # Path con múltiples segmentos
            path = obj.get("path", [])
            left = obj.get("left", 0)
            top = obj.get("top", 0)
            
            prev_point = None
            for cmd in path:
                if cmd[0] in ["M", "L"]:
                    point = (cmd[1] + left, cmd[2] + top)
                    if prev_point and cmd[0] == "L":
                        lines.append({
                            "x1": prev_point[0], "y1": prev_point[1],
                            "x2": point[0], "y2": point[1]
                        })
                    prev_point = point
    
    return lines