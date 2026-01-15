
import cv2
import numpy as np
import math

class PatternGenerator:
    def __init__(self, grid_width=10.0, grid_height=10.0):
        self.grid_width = float(grid_width)
        self.grid_height = float(grid_height)

    def process_image(self, image_file, epsilon_factor=0.001, scale=1.0, closing_size=1, min_area=10.0):
        """
        Process the image with the "Rescue Pipeline":
        1. Binarize
        2. Morphological Closing (Connect gaps)
        3. Contours (External)
        4. PolyDP Simplification
        """
        # Read
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "Could not decode image"}

        # 1. Binarization
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # OTSU Thresholding to get binary
        # We want the "Ink" to be White for morphological operations usually.
        # If input is Black ink on White paper: THRESH_BINARY_INV makes Ink=White, Paper=Black.
        thresh_val, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 2. Morphological Closing (The "Glue" step)
        # Connects pixels that are close.
        # Kernel size determines how big of a gap to close.
        kernel_size = int(closing_size)
        if kernel_size < 1: kernel_size = 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Closing: Dilation followed by Erosion. Fills small holes and gaps.
        closed_img = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Also apply open to remove isolated noise pixels if needed?
        # Maybe let the user decide. For now, strictly follow "Connect lines".
        
        # 3. Contours
        # RETR_EXTERNAL: Only outer contours. Good for stones that are filled shapes.
        # If the user has "outlines of stones", this gives the outer edge of the pen stroke.
        # If inputs are thick lines and we want the "shape of the stone", 
        # usually users want the 'hole' inside the loop if it's an outline?
        # But the prompt says: "Detect contorno externo... prioritize stones look like closed figures".
        # Let's stick to RETR_EXTERNAL as requested.
        contours, hierarchy = cv2.findContours(closed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        pat_lines = []
        
        # Visualization Canvases
        # Preview 1: The "Connected" image (Debugging)
        # Convert binary back to BGR for display
        debug_closed = cv2.cvtColor(closed_img, cv2.COLOR_GRAY2BGR)
        
        # Preview 2: The Vectors
        vector_canvas = np.ones_like(img) * 255 # White background
        
        h_img, w_img = gray.shape
        
        # Scale Calculation
        # Map 1.0 image width to 'grid_width * scale' physical units
        physical_width = self.grid_width * scale
        physical_height = self.grid_height * scale
        
        pixel_scale_x = physical_width / w_img
        pixel_scale_y = physical_height / h_img
        
        # Header
        pat_header = [
            f"*HatchCraft_V3, Robust Contour",
            f";%TYPE=MODEL",
            f"; Physical Tile Size: {physical_width:.4f} x {physical_height:.4f}",
            f"; Source Image: {w_img}x{h_img}px"
        ]
        
        count_segments = 0
        
        for cnt in contours:
            # Filter by area
            area = cv2.contourArea(cnt)
            if area < min_area: continue
            
            # 4. Simplification
            # Epsilon is % of perimeter
            perimeter = cv2.arcLength(cnt, True)
            epsilon = epsilon_factor * perimeter
            
            # Closed=True because we want closed stone shapes
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            pts = approx[:, 0, :]
            num_pts = len(pts)
            
            if num_pts < 2: continue
            
            # Draw on Vector Canvas (Black lines)
            # Use anti-aliasing for nice view
            cv2.drawContours(vector_canvas, [approx], -1, (0, 0, 0), 2, cv2.LINE_AA)
            
            # Generate PAT data
            for i in range(num_pts):
                p1 = pts[i]
                p2 = pts[(i+1) % num_pts] # Loop back to start to close the shape
                
                # Transform to Cartesian (Revit)
                # Image: Y down. Revit: Y up.
                x1 = p1[0] * pixel_scale_x
                y1 = (h_img - p1[1]) * pixel_scale_y
                x2 = p2[0] * pixel_scale_x
                y2 = (h_img - p2[1]) * pixel_scale_y
                
                dx = x2 - x1
                dy = y2 - y1
                
                length = math.sqrt(dx**2 + dy**2)
                if length < 0.000001: continue
                
                angle_rad = math.atan2(dy, dx)
                angle_deg = math.degrees(angle_rad)
                if angle_deg < 0: angle_deg += 360
                
                # PAT Repetition Logic
                # We project the "Physical Grid Width" onto the line basis.
                # Repetition is defined by the full tile size.
                
                rx, ry = physical_width, 0 # Horizontal Repeat
                
                u_x, u_y = math.cos(angle_rad), math.sin(angle_rad)
                
                # dx_pat = Projection of R onto U
                dx_pat = rx * u_x + ry * u_y
                
                # dy_pat = Projection of R onto N (offset to next infinite line)
                # Ideally dy is perpendicular distance.
                # In .pat, dy is component-y of the shift vector in rotated frame?
                # No, dy is "y-offset". 
                # Formula: dy = -W * sin(angle) (for standard x-repeat)
                dy_pat = -physical_width * math.sin(angle_rad)
                
                # Dash/Space
                # Dash = Length
                # Space = -2000 (We don't want the line to repeat along itself immediately, 
                # we rely on the X/Y grid repetition to place the next stone).
                
                line_def = f"{angle_deg:.5f},{x1:.5f},{y1:.5f},{dx_pat:.5f},{dy_pat:.5f},{length:.5f},-2000"
                pat_lines.append(line_def)
                count_segments += 1
                
        full_content = "\n".join(pat_header + pat_lines)
        
        return {
            "debug_closed_img": debug_closed, # The "Connected" binary image
            "vector_img": vector_canvas,      # The Result
            "pat_content": full_content,
            "stats": f"Detected {len(contours)} shapes, {count_segments} segments."
        }
