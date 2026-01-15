
import cv2
import numpy as np
import math
from skimage.morphology import skeletonize

class PatternGenerator:
    def __init__(self, grid_width=10.0, grid_height=10.0):
        self.grid_width = float(grid_width)
        self.grid_height = float(grid_height)

    def process_image(self, image_file, epsilon_factor=0.005, scale=1.0):
        """
        Process the image with Skeletonization pipeline.
        """
        # Read
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "Could not decode image"}

        # 1. Binarization & Cleaning
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Invert if needed: We want patterns (lines) to be White (1) for skeletonize
        # Assuming typical drawing: Black lines on white paper.
        # Threshold: Binary Inverted (Black lines become White regions)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Cleanup small noise (optional)
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 2. Skeletonization (The Key Step)
        # Skeletonize requires boolean/0-1 array.
        binary_bool = binary > 0
        skeleton = skeletonize(binary_bool)
        skeleton_uint8 = (skeleton * 255).astype(np.uint8)
        
        # 3. Contours on Skeleton
        # Skeleton is 1-pixel wide. RETR_TREE or RETR_CCOMP? 
        # RETR_EXTERNAL won't find holes inside stones.
        # RETR_LIST or RETR_CCOMP is best.
        contours, _ = cv2.findContours(skeleton_uint8, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        pat_lines = []
        preview_canvas = np.ones_like(img) * 255
        
        h_img, w_img = gray.shape
        
        # Setup Scale
        # User defined Scale (Scale Factor slider)
        # If grid_width is 10, then scale affects how much of the grid the image covers?
        # Actually, scaling is usually pixel_to_unit.
        # Let's assume grid_width is the final size of the tile.
        pixel_scale_x = (self.grid_width * scale) / w_img
        pixel_scale_y = (self.grid_height * scale) / h_img
        
        # Header
        pat_header = [
            f"*HatchCraft_Clean, Generated via Skeletonize",
            f";%TYPE=MODEL",
            f"; Size: {self.grid_width}x{self.grid_height} units"
        ]
        
        count_segments = 0
        
        for cnt in contours:
            # 4. Simplification (Douglas-Peucker)
            epsilon = epsilon_factor * cv2.arcLength(cnt, False) # Closed=False for skeleton lines usually
            # Note: A skeleton loop is Closed=True?
            # Skeleton structures are often graphs. findContours traces around the 1-px line.
            # This effectively makes a very thin closed loop.
            # We treat it as closed to keep the shape, then we might want to kill one side?
            # Actually, simply approxPolyDP on the loop will likely result in a line going there and back if it's open,
            # or a loop if it's a loop.
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            pts = approx[:, 0, :]
            
            # Filter tiny specs
            if len(pts) < 2: continue
            
            # Check length to avoid noise points
            if cv2.arcLength(approx, True) < 5: continue

            # Draw
            cv2.polylines(preview_canvas, [pts], True, (0, 0, 0), 1, cv2.LINE_AA)
            
            for i in range(len(pts)):
                p1 = pts[i]
                p2 = pts[(i+1) % len(pts)]
                
                # Transform
                x1 = p1[0] * pixel_scale_x
                y1 = (h_img - p1[1]) * pixel_scale_y # Flip Y
                x2 = p2[0] * pixel_scale_x
                y2 = (h_img - p2[1]) * pixel_scale_y
                
                dx = x2 - x1
                dy = y2 - y1
                
                length = math.sqrt(dx**2 + dy**2)
                if length < 0.0001: continue # Skip zero length
                
                angle_rad = math.atan2(dy, dx)
                angle_deg = math.degrees(angle_rad)
                if angle_deg < 0: angle_deg += 360
                
                # Calcs
                u_x, u_y = math.cos(angle_rad), math.sin(angle_rad)
                # n_x, n_y = -math.sin(angle_rad), math.cos(angle_rad)
                
                # Repetition (Horizontal tile width)
                # Using the FULL grid width as repetition even if scale changes?
                # Usually Tiling is fixed to the box.
                rx, ry = self.grid_width * scale, 0
                
                dx_pat = rx * u_x + ry * u_y
                dy_pat = rx * -u_y # Project onto normal. Wait. Normal is (-sin, cos)?
                # Correct projection for .pat dy is onto the normal vector.
                # let's map normal N = (-sin, cos)
                # dy_pat = rx * (-sin) + ry * (cos)
                # rx=W, ry=0 => dy_pat = -W * sin
                
                # Special fix for vertical lines (angle=90, sin=1) -> dy = -W. 
                # .pat dy is "offset to family line".
                dy_pat = - (self.grid_width * scale) * math.sin(angle_rad)
                
                # Dash/Space
                # To draw JUST this segment:
                # Dash = Length
                # Space = - (Repetition Shift - Length) ??
                # Actually, simply use -2000. It works practically to ensure no repeats "locally".
                
                line_def = f"{angle_deg:.4f},{x1:.4f},{y1:.4f},{dx_pat:.4f},{dy_pat:.4f},{length:.4f},-2000"
                pat_lines.append(line_def)
                count_segments += 1

        full_content = "\n".join(pat_header + pat_lines)
        
        return {
            "preview_img": preview_canvas,
            "pat_content": full_content,
            "stats": f"Segments: {count_segments} (Method: Skeletonize)"
        }
