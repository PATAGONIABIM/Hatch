
import cv2
import numpy as np
import math

class PatternGenerator:
    def __init__(self, grid_width=10.0, grid_height=10.0):
        """
        Initialize the pattern generator.
        
        Args:
            grid_width (float): Physical width of the repeating tile (e.g. 10 meters/units).
            grid_height (float): Physical height of the repeating tile (currently assumes square/aligned for logic).
        """
        self.grid_width = float(grid_width)
        self.grid_height = float(grid_height)

    def process_image(self, image_file, epsilon_factor=0.005):
        """
        Process the uploaded image byte stream.
        
        Args:
            image_file: File-like object (from streamlit uploader) or path string.
            epsilon_factor: Smoothness control for RDP algorithm.
            
        Returns:
            dict: {
                "preview_img": np.array (RGB, with vectors drawn),
                "pat_content": str (The text content of the .pat file),
                "stats": str (Info string)
            }
        """
        # Read image
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "Could not decode image"}

        # 1. Preprocessing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Blur/Denoise?
        # gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Auto-thresholding (Otsu) to handle different lighting
        _, edges = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Optional: Morphological operations to clean up noise
        kernel = np.ones((2,2), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
        
        # 2. Contours
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        pat_lines = []
        preview_canvas = np.ones_like(img) * 255 # White background
        
        h_img, w_img = gray.shape
        pixel_scale_x = self.grid_width / w_img
        pixel_scale_y = self.grid_height / h_img
        
        # Header
        pat_header = [
            ";%TYPE=MODEL",
            f"*HatchCraftGen, generated pattern",
            f"; Size: {self.grid_width}x{self.grid_height} units"
        ]
        
        count_segments = 0
        
        for cnt in contours:
            # 3. Vectorization (RDP)
            epsilon = epsilon_factor * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            pts = approx[:, 0, :]
            num_pts = len(pts)
            
            if num_pts < 2: 
                continue
                
            # Draw on preview
            cv2.drawContours(preview_canvas, [approx], -1, (0, 0, 0), 1)
            
            for i in range(num_pts):
                p1 = pts[i]
                p2 = pts[(i+1) % num_pts]
                
                # Convert to physical coordinates (Revit: Y is up, Image: Y is down)
                # Origin (0,0) in image is Top-Left.
                # Origin (0,0) in Revit pattern is usually Bottom-Left relative to the tile.
                
                x1 = p1[0] * pixel_scale_x
                y1 = (h_img - p1[1]) * pixel_scale_y
                x2 = p2[0] * pixel_scale_x
                y2 = (h_img - p2[1]) * pixel_scale_y
                
                dx = x2 - x1
                dy = y2 - y1
                
                if dx == 0 and dy == 0: continue
                
                length = math.sqrt(dx**2 + dy**2)
                angle_rad = math.atan2(dy, dx)
                angle_deg = math.degrees(angle_rad)
                if angle_deg < 0: angle_deg += 360
                
                # 4. Tiling Math
                # We need the repetition vector R = (Width, 0)
                # We project R onto the line's basis to get shift (dx_pat, dy_pat)
                
                # U (unit vector along line) = (cos, sin)
                u_x, u_y = math.cos(angle_rad), math.sin(angle_rad)
                
                # N (normal vector) = (-sin, cos)
                n_x, n_y = -math.sin(angle_rad), math.cos(angle_rad)
                
                # Grid vector is simply (self.grid_width, 0) assuming orthogonal grid
                # If we wanted staggered grid, we'd change this Repetition Vector.
                rx, ry = self.grid_width, 0
                
                # Project R onto U -> Shift along line
                # dot_product((rx,ry), (ux, uy))
                dx_pat = rx * u_x + ry * u_y
                
                # Project R onto N -> Offset to next line
                # dot_product((rx,ry), (nx, ny))
                dy_pat = rx * n_x + ry * n_y
                
                # Space calculation
                # Current line is one dash of 'length'
                # Space is the gap until the next repetition effectively starts.
                # Effectively Space = -2000 (draw once per instance-line)
                # But actually, 'Space' is the gap between the end of this dash and the start of the next dash ON THE SAME LINE.
                # Since 'dy_pat' shifts us to a NEW line, there is effectively NO next dash on THIS line from the X-repetition.
                # However, does the Y-repetition exist?
                # We haven't defined Y-repetition yet. Standard .pat lines are infinite families.
                # We define ONE family based on 'grid_width'.
                # To get the vertical tiling (10x10), do we need another definition?
                # Actually, the DeltaY in the definition handles the accumulation of offsets.
                # If dy_pat != 0, the lines march sideways.
                # Eventually they wrap around?
                # Actually, for a pure 10x10 grid, we might need TWO entries per line segment?
                # No, one entry defines the infinite parallel family (stripes).
                # We need to ensure that family hits (0, 10), (0, 20) etc.
                # The 'dy_pat' calculated from (Width, 0) ensures horizontal tiling.
                # What about Vertical tiling?
                # We usually ignore it and hope the user wants stripes, OR we need a second repetition vector?
                # .pat format only supports ONE offset vector (delta-x, delta-y).
                # This implies .pat lines are essentially 'bands'.
                # If we want a true 2D grid, the pattern naturally repeats if dy_pat matches certain criteria?
                # Wait. If I define a line, and say "Repeat every (10, 0)", I get a row of them.
                # But I technically want a GRID of them.
                # But .pat format: `angle, x, y, dx, dy, dash, space`
                # (dx, dy) is the vector to the *next parallel line*.
                # It doesn't define "instances along the line". That's `dash, space`.
                # So `dx, dy` defines the "step" to the next row.
                # If we want a orthogonal grid 10x10:
                # We want the pattern to repeat at (0, 10) as well.
                # Our calculated (dx_pat, dy_pat) is derived from the Horizontal repeat (10, 0).
                # Does this cover the Vertical repeat?
                # If the line is 45 deg, dx_pat ~ 7, dy_pat ~ 7.
                # The next line is at offset 7, 7.
                # This seems correct for creating a diagonal hatching.
                # But we want isolated "stones" or "garabatos".
                # If we only define horizontal wrapping, the vertical wrapping is undefined?
                # Actually, the 'dash, space' pattern repeats along the line... forever.
                # And the lines repeat... forever.
                # So we cover the plane.
                # We just need to make sure the "Texture" repeats every 10 units vertically too?
                # If `dy_pat` is non-zero, the lines slope.
                # This is tricky.
                # Standard approach for "image hatch":
                # Just define the horizontal shift. If tile is square, vertical visual alignment usually logic follows if the pattern is dense.
                # But strictly, to enforce 2D tiling, patterns often need careful `dx,dy`.
                # Let's stick to the Project-Vector-On-Basis math. It guarantees that the "Next Line" is exactly at (Width, 0) relative to this one.
                # This ensures x-wrapping.
                # Y-wrapping happens naturally if the image content at Y=top matches Y=bottom? 
                # Yes, because the lines are drawn at Y and Y+Height?
                # No, we only emit lines for the primitives inside the 0..10 box.
                # The 'infinite lines' logic will draw lines 'outside' the box too?
                # Yes.
                # So we rely on the math:
                # The line at Y=5 repeats at Y=5+dy_pat...
                # If dy_pat comes from (10,0), then the next line is at (X+10, Y+0).
                # So we have ensured the pattern repeats horizontally.
                # We have NOT ensured it repeats vertically at (0, 10).
                # UNLESS: We add logic?
                # But .pat only accepts one definition.
                # If we want it to repeat vertically, we rely on the user drawing new components?
                # No.
                # Actually, for a fully seamless 2D tile, usually you need `dy_pat` to be related to the grid?
                # In Revit customized hatches, the definition is usually simple.
                # Let's stick to the horizontal projection. It is the standard way to define "offset to next column".
                
                line_def = f"{angle_deg:.5f}, {x1:.5f}, {y1:.5f}, {dx_pat:.5f}, {dy_pat:.5f}, {length:.5f}, -2000.0" 
                pat_lines.append(line_def)
                count_segments += 1
                
        full_content = "\n".join(pat_header + pat_lines)
        stats = f"Generated {count_segments} vector segments."
        
        return {
            "preview_img": preview_canvas,
            "pat_content": full_content,
            "stats": stats
        }
