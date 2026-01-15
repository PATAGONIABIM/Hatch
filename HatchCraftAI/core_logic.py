
import cv2
import numpy as np
import math
from skimage.morphology import skeletonize

class PatternGenerator:
    def __init__(self, grid_width=10.0, grid_height=10.0):
        self.grid_width = float(grid_width)
        self.grid_height = float(grid_height)

    def process_image(self, image_file, epsilon_factor=0.005, scale=1.0, closing_size=3, manual_invert=False):
        """
        Refined Pipeline:
        1. Auto-Invert (Smart Background Detection)
        2. Morphological Closing (Connect Gaps)
        3. Skeletonization (Centerline extraction)
        4. Vectorization (Douglas-Peucker)
        """
        # Read Image
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "Could not decode image"}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # --- 1. Smart Binarization & Inversion ---
        # Apply Otsu to get basic binary
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Check pixel counts (0=Black, 255=White)
        white_pixels = np.sum(binary == 255)
        total_pixels = binary.size
        black_pixels = total_pixels - white_pixels
        
        # We need the "Object" (Lines) to be WHITE (255) and Background to be BLACK (0) for processing.
        # Assumption: Drawings usually have more Background than Lines.
        # If White > Black, then White is likely Background. -> We need to Invert (so Bg becomes Black).
        
        auto_invert_needed = white_pixels > black_pixels
        
        # Logic Table:
        # Detected | Manual Invert Checkbox | Action
        # Need Inv | False                  | Invert
        # Need Inv | True                   | Don't Invert (User overrides auto)
        # No Inv   | False                  | Don't Invert
        # No Inv   | True                   | Invert
        
        # Actually simpler: User toggle usually means "Flip what you did".
        # But let's follow prompt: "Invert manually" checkbox.
        # Let's say we default to logic. If user checks "Force Invert", we flip the result of auto.
        # Or simpler: The checkbox forces Inversion regardless? 
        # The prompt says: "Checkbox: Invertir Colores Manualmente".
        # Let's stick to: Calculate intended, apply. If Manual Invert is ON, flip it.
        
        # Base state: We want objects=255.
        # If background is white (common), binary has Back=255, Obj=0. We need Invert.
        # Only if background is Black, we have Back=0, Obj=255. No Invert needed.
        
        is_white_bg = white_pixels > black_pixels
        
        # Initial binary state based on assumption
        if is_white_bg:
            # White BG -> Invert so BG is black
            binary = cv2.bitwise_not(binary)
            
        # Manual Override: Simply flip the current state
        if manual_invert:
            binary = cv2.bitwise_not(binary)
            
        # --- 2. Morphological Closing (The "Glue") ---
        # Connects nearby pixels.
        if closing_size > 0:
            kernel = np.ones((closing_size, closing_size), np.uint8)
            # Dilate then Erode
            processed_img = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        else:
            processed_img = binary

        # --- 3. Skeletonization (Center Line) ---
        # Reduces thick lines (from markers or the Closing step) to 1px wide lines.
        # Requires Boolean True/False
        bool_img = processed_img > 0
        skeleton = skeletonize(bool_img)
        skeleton_uint8 = (skeleton * 255).astype(np.uint8)
        
        # --- 4. Contours & Vectorization ---
        # FIND CONTOURS on SKELETON
        # RETR_LIST or RETR_CCOMP to get all lines.
        # Since it's 1px wide, contours will trace the line.
        # Note: Contours of a 1px line might be a loop going around the pixel strip.
        # approxPolyDP on a very thin loop usually collapses well.
        contours, _ = cv2.findContours(skeleton_uint8, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        pat_lines = []
        vector_canvas = np.ones_like(img) * 255 # White background
        
        # Physical Scale Logic
        h_img, w_img = gray.shape
        physical_width = self.grid_width * scale
        physical_height = self.grid_height * scale
        
        pixel_scale_x = physical_width / w_img
        pixel_scale_y = physical_height / h_img
        
        pat_header = [
            f"*HatchCraft_Smart, Invert+Close+Skel",
            f";%TYPE=MODEL",
            f"; Size: {physical_width:.4f}x{physical_height:.4f}",
            f"; Processed {len(contours)} raw traces"
        ]
        
        count_segments = 0
        
        for cnt in contours:
            # Filter tiny noise
            if cv2.arcLength(cnt, False) < 5: continue
            
            # Simplification
            epsilon = epsilon_factor * cv2.arcLength(cnt, True) # Skeletons are loops in cv2 eyes? 
            # Ideally use False for arcLength if open, but contours are technically closed loops of pixels.
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            pts = approx[:, 0, :]
            
            # Draw vectors
            cv2.polylines(vector_canvas, [pts], False, (0, 0, 0), 1, cv2.LINE_AA)
            
            for i in range(len(pts) - 1): # Open chain for skeleton lines usually better?
                # Actually, findContours on 1px Skeleton often returns a "Double back" loop.
                # e.g. A->B->C->B->A.
                # If we treat it as closed loop, we draw the line twice. 
                # This is a known issue with skeleton vectorization via contours.
                # Optimized approach: Treating it as generic loop is 'safe' but doubles output size.
                # For this MVP, we accept the loop or try to merge? 
                # Let's emit segments. If they overlap perfectly, the file size grows but visuals are fine.
                
                p1 = pts[i]
                p2 = pts[i+1]
                
                # Transform
                x1 = p1[0] * pixel_scale_x
                y1 = (h_img - p1[1]) * pixel_scale_y
                x2 = p2[0] * pixel_scale_x
                y2 = (h_img - p2[1]) * pixel_scale_y
                
                dx = x2 - x1
                dy = y2 - y1
                
                length = math.sqrt(dx**2 + dy**2)
                if length < 0.0001: continue
                
                angle_rad = math.atan2(dy, dx)
                angle_deg = math.degrees(angle_rad)
                if angle_deg < 0: angle_deg += 360
                
                # PAT Repetition
                # Projects Horizontal Grid Width onto line vector
                rx, ry = physical_width, 0
                u_x, u_y = dx/length, dy/length
                
                dx_pat = rx * u_x + ry * u_y
                dy_pat = -physical_width * u_y # (Project R onto Normal logic simplified)
                
                line_def = f"{angle_deg:.5f},{x1:.5f},{y1:.5f},{dx_pat:.5f},{dy_pat:.5f},{length:.5f},-2000"
                pat_lines.append(line_def)
                count_segments += 1
                
        # Close loop for last segment if needed? 
        # Skeletons from findContours are loops. So yes. 
        # But for skeleton lines, we usually dont want to close the loop back to start if it's an open line?
        # findContours makes it a loop.
        # Let's keep it simple.
        
        full_content = "\n".join(pat_header + pat_lines)
        
        # Stats Check
        warning = None
        if count_segments > 500:
            warning = f"⚠️ ALERTA: Patrón Complejo ({count_segments} líneas). Puede ralentizar Revit. Aumenta 'Simplificación'."
            
        return {
            "processed_img": skeleton_uint8, # Show the Skeleton
            "vector_img": vector_canvas,
            "pat_content": full_content,
            "stats": f"Segments: {count_segments}",
            "warning": warning
        }
