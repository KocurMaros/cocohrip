import cv2
from matplotlib.pyplot import imshow
from checkers_game.camera.ximea_camera import XimeaCamera
import numpy as np
import copy
import sys
from checkers_game.checkers.piece import Piece
from checkers_game.constants import BLACK, ROWS, RED, SQUARE_SIZE, COLS, WHITE, GREY, BROWN

# Debug mode - set to True to show additional windows for debugging
DEBUG_MODE = True  # Enabled for debugging auto-detection

def log(msg):
    """Print with immediate flush so output appears in terminal right away"""
    print(msg, flush=True)
    sys.stdout.flush()

class BoardDetection:

    def __init__(self, ximeaCamera):
        self.ximeaCamera = ximeaCamera
        self._init()
        

    def _init(self):
        # 1. Camera Adjustment Phase
        self._camera_adjustment_window()

        # 2. Board Corner Detection (AUTO)
        log("\nAttempting automatic board detection...")
        auto_corners, auto_success_reason = self._auto_detect_corners()
        
        if auto_corners is not None:
            self.bounderies = auto_corners
            log("✓ Automatic detection successful!")
            log(f"  → {auto_success_reason}")
            
            # 2b. Verify and adjust corners if needed
            self.bounderies = self._verify_and_adjust_corners(self.bounderies)
        else:
            log("⚠ Automatic detection failed. Falling back to manual selection.")
            self.bounderies = self._get_trim_param_manual()
            
        # 3. Auto-detect piece thresholds based on 12 black + 12 white pieces
        # Then show settings window with detected values for user adjustment
        log("\nAuto-detecting piece color thresholds...")
        auto_thresholds = self._auto_detect_thresholds()
        
        if auto_thresholds is not None:
            self.empty_variance_threshold = auto_thresholds['empty']
            self.black_variance_threshold = auto_thresholds['black']
            self.white_piece_threshold = auto_thresholds['white']
            log("✓ Auto-detected thresholds:")
            log(f"  Empty < {self.empty_variance_threshold:.0f} < Black < {self.black_variance_threshold:.0f} < White")
        else:
            # Set defaults if auto-detection fails
            self.empty_variance_threshold = 600.0
            self.black_variance_threshold = 1000.0
            self.white_piece_threshold = 1000.0
            log("⚠ Using default thresholds - please adjust in settings window")
        
        # 4. Show settings window with detected values for user confirmation/adjustment
        self._settings_window()

        # 5. Final initialization
        self.numberOfEmptyFields = 40
        self.param1ForGetAllContours = 255
        self.gameBoardFieldsContours = self._get_grid_squares_contours()
        
        self.is_initialized = False
        self.selected_difficulty = 3

    def _auto_detect_corners(self):
        """
        Automatically detect the board corners using multiple detection strategies.
        Shows visual feedback during detection.
        
        Returns: (corners, reason_message) or (None, failure_reason)
        """
        log("\n" + "="*60)
        log("AUTO-DETECTING BOARD CORNERS")
        log("="*60)
        
        # Get image
        image = self.ximeaCamera.get_camera_image()
        if image is None:
            log("  ✗ FAILED: Could not get camera image")
            return None, "Camera image unavailable"
        
        log(f"  → Image size: {image.shape[1]}x{image.shape[0]}")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        min_board_area = (min(h, w) * 0.2) ** 2  # Board should be at least 20% of image
        log(f"  → Minimum board area: {min_board_area:.0f} pixels")
        
        # Show the image we're analyzing
        debug_image = image.copy()
        cv2.putText(debug_image, "Detecting board...", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Auto Detection", debug_image)
        cv2.waitKey(100)  # Brief pause to show image
        
        # Strategy 1: Find largest quadrilateral contour
        board_cnt = None
        largest_area = 0
        used_method = None
        all_quads_found = []
        all_large_contours = []
        
        methods = [
            ("Adaptive Gaussian", lambda: cv2.adaptiveThreshold(
                cv2.GaussianBlur(gray, (5, 5), 0), 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)),
            ("Otsu Inv", lambda: cv2.threshold(
                cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, 
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]),
            ("Otsu Normal", lambda: cv2.threshold(
                cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, 
                cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
            ("Canny Soft", lambda: cv2.dilate(
                cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 100), 
                np.ones((5,5), np.uint8), iterations=2)),
            ("Canny Strong", lambda: cv2.dilate(
                cv2.Canny(cv2.GaussianBlur(gray, (7, 7), 0), 50, 150), 
                np.ones((7,7), np.uint8), iterations=3)),
            ("Morph Close", lambda: cv2.morphologyEx(
                cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, 
                cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                cv2.MORPH_CLOSE, np.ones((15,15), np.uint8))),
        ]
        
        for method_name, threshold_func in methods:
            try:
                thresh = threshold_func()
                
                # Show threshold result briefly
                cv2.imshow("Threshold Debug", thresh)
                cv2.waitKey(50)
                
                # Try both external and tree contour retrieval
                for mode_name, mode in [("EXTERNAL", cv2.RETR_EXTERNAL), ("TREE", cv2.RETR_TREE)]:
                    contours, _ = cv2.findContours(thresh, mode, cv2.CHAIN_APPROX_SIMPLE)
                    
                    log(f"  → {method_name} ({mode_name}): {len(contours)} contours")
                    
                    # Draw all contours on debug image
                    contour_debug = image.copy()
                    
                    quad_count = 0
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        
                        # Track all large contours for debugging
                        if area > 10000:
                            peri = cv2.arcLength(cnt, True)
                            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                            all_large_contours.append((method_name, area, len(approx)))
                            
                            # Draw this contour
                            color = (0, 255, 0) if len(approx) == 4 else (0, 0, 255)
                            cv2.drawContours(contour_debug, [approx], -1, color, 2)
                            cv2.putText(contour_debug, f"{len(approx)}pts", 
                                       (int(cnt[0][0][0]), int(cnt[0][0][1])),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        
                        if area < min_board_area:
                            continue
                        
                        peri = cv2.arcLength(cnt, True)
                        # Try different approximation factors
                        for eps_factor in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1]:
                            approx = cv2.approxPolyDP(cnt, eps_factor * peri, True)
                            
                            if len(approx) == 4:
                                quad_count += 1
                                all_quads_found.append((method_name, mode_name, area, approx, eps_factor))
                                
                                if area > largest_area:
                                    largest_area = area
                                    board_cnt = approx
                                    used_method = f"{method_name} ({mode_name}, eps={eps_factor})"
                                break  # Found a quad for this contour
                    
                    # Show contour debug
                    cv2.putText(contour_debug, f"{method_name}: {quad_count} quads found", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.imshow("Contour Detection", contour_debug)
                    cv2.waitKey(50)
                    
                    if quad_count > 0:
                        log(f"    → Found {quad_count} quadrilaterals, largest: {largest_area:.0f}")
                        
            except Exception as e:
                log(f"  → Method '{method_name}' failed: {e}")
                continue
        
        # Log large contours found
        if len(all_large_contours) > 0:
            log(f"\n  Large contours found (area > 10000):")
            for m, a, pts in sorted(all_large_contours, key=lambda x: -x[1])[:10]:
                log(f"    - {m}: area={a:.0f}, vertices={pts}")
        
        # Strategy 2: If no quad found, try to find the checkerboard pattern
        if board_cnt is None:
            log("  → No quad found, trying checkerboard pattern...")
            for grid_size in [(7, 7), (6, 6), (5, 5)]:
                ret, corners_chess = cv2.findChessboardCorners(gray, grid_size, None)
                if ret:
                    log(f"  → Found checkerboard pattern {grid_size}!")
                    corners_chess = corners_chess.reshape(-1, 2)
                    x_min, y_min = corners_chess.min(axis=0)
                    x_max, y_max = corners_chess.max(axis=0)
                    
                    margin_x = (x_max - x_min) / grid_size[0]
                    margin_y = (y_max - y_min) / grid_size[1]
                    
                    board_cnt = np.array([
                        [[x_min - margin_x, y_min - margin_y]],
                        [[x_max + margin_x, y_min - margin_y]],
                        [[x_max + margin_x, y_max + margin_y]],
                        [[x_min - margin_x, y_max + margin_y]]
                    ], dtype=np.float32)
                    used_method = f"Checkerboard {grid_size}"
                    largest_area = (x_max - x_min + 2*margin_x) * (y_max - y_min + 2*margin_y)
                    break
        
        cv2.destroyWindow("Threshold Debug")
        cv2.destroyWindow("Contour Detection")
        cv2.destroyWindow("Auto Detection")
        
        if board_cnt is None:
            log("\n  ✗ FAILED: No quadrilateral contour found")
            log("    Diagnostic info:")
            log(f"    - Image size: {w}x{h}")
            log(f"    - Min board area threshold: {min_board_area:.0f} pixels")
            log(f"    - Total quads found: {len(all_quads_found)}")
            log(f"    - Total large contours: {len(all_large_contours)}")
            
            if len(all_quads_found) > 0:
                log(f"    - Largest quad area: {max(q[2] for q in all_quads_found):.0f}")
            
            log("\n    Possible reasons:")
            log("    - Board edges blend with background/table")
            log("    - No clear border around the board")
            log("    - Try adding a dark border around the board")
            log("\n    → Falling back to manual selection...")
            
            return None, "No board contour detected"
        
        log(f"\n  ✓ Found board using '{used_method}'")
        log(f"    Area: {largest_area:.0f} pixels")
        
        # Reshape to 4x2
        pts = board_cnt.reshape(4, 2).astype(np.float32)
        
        # Sort points: TL, TR, BR, BL
        rect = np.zeros((4, 2), dtype="float32")
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # TL
        rect[2] = pts[np.argmax(s)]  # BR

        diff = np.diff(pts, axis=1).flatten()
        rect[1] = pts[np.argmin(diff)]  # TR
        rect[3] = pts[np.argmax(diff)]  # BL
        
        # Orient the corners based on piece detection
        rect, orient_success, orient_reason = self._orient_corners(rect, image)
        
        if not orient_success:
            log(f"  ⚠ Orientation detection uncertain: {orient_reason}")
            log("    → Using geometric orientation, may need manual rotation")
        
        return rect, orient_reason

    def _orient_corners(self, corners, image):
        """
        Check all 4 rotations of the corners to find the one that matches 
        the expected board setup:
        TL: White empty square (Low Variance)
        TR: Black square with BLACK piece (Medium Variance)
        BR: White empty square (Low Variance)
        BL: Black square with WHITE piece (High Variance - white pieces are brighter)
        
        Returns: (best_corners, success, reason_message)
        """
        best_corners = corners.copy()
        best_score = float('inf')
        best_rotation = 0
        variances_log = []
        
        for rotation in range(4):
            # Warp to inspect corners
            warped = self._trim_image_perspective(image, corners)
            
            if warped is None or warped.shape[0] < 800 or warped.shape[1] < 800:
                corners = np.roll(corners, 1, axis=0)
                continue
            
            # Extract corner ROIs (from the 100x100 corner squares)
            margin = 15
            size = 70
            
            roi_tl = warped[margin:margin+size, margin:margin+size]
            roi_tr = warped[margin:margin+size, 800-margin-size:800-margin]
            roi_br = warped[800-margin-size:800-margin, 800-margin-size:800-margin]
            roi_bl = warped[800-margin-size:800-margin, margin:margin+size]
            
            v_tl = self._calculate_variance(roi_tl)
            v_tr = self._calculate_variance(roi_tr)
            v_br = self._calculate_variance(roi_br)
            v_bl = self._calculate_variance(roi_bl)
            
            variances_log.append({
                'rotation': rotation,
                'TL': v_tl, 'TR': v_tr, 'BR': v_br, 'BL': v_bl
            })
            
            # Score the orientation:
            # - TL and BR should be empty (low variance)
            # - TR should have black piece (medium variance)
            # - BL should have white piece (highest variance)
            score = 0
            
            # TL and BR should be the lowest (empty squares)
            score += v_tl + v_br
            
            # TR must be higher than TL (has piece)
            if v_tr <= v_tl * 1.5:
                score += 5000  # Penalty: TR should have a piece
            
            # BL must be higher than TL (has piece)
            if v_bl <= v_tl * 1.5:
                score += 5000  # Penalty: BL should have a piece
            
            # BL (white piece) should be higher variance than TR (black piece)
            if v_bl <= v_tr:
                score += 3000  # Penalty: white should be brighter/higher variance than black
            
            # Additional check: TR and BL must be significantly higher than empty
            if v_tr < 200 or v_bl < 200:
                score += 10000  # Both corners should have pieces
            
            if score < best_score:
                best_score = score
                best_corners = corners.copy()
                best_rotation = rotation
            
            # Rotate corners for next iteration
            corners = np.roll(corners, 1, axis=0)
        
        # Log variance analysis
        log(f"  → Corner variance analysis:")
        for v in variances_log:
            marker = " ← SELECTED" if v['rotation'] == best_rotation else ""
            log(f"    Rotation {v['rotation']}: TL={v['TL']:.0f}, TR={v['TR']:.0f}, BR={v['BR']:.0f}, BL={v['BL']:.0f}{marker}")
        
        # Determine success
        success = best_score < 5000
        if success:
            reason = f"Orientation found (rotation={best_rotation}, score={best_score:.0f})"
        else:
            reason = f"Uncertain orientation (rotation={best_rotation}, score={best_score:.0f}) - corner variances may not match expected pattern"
        
        return best_corners, success, reason

    def _verify_and_adjust_corners(self, corners):
        """
        Show the warped board with grid overlay so user can verify if corners are correct.
        If not correct, allow manual adjustment by clicking new corners.
        
        Controls:
        - SPACE/ENTER: Accept current corners
        - 'R': Reset and select new corners manually
        - 1-4: Select corner to adjust, then click new position
        """
        log("\n" + "="*60)
        log("CORNER VERIFICATION")
        log("="*60)
        log("Check if the grid aligns with the board squares.")
        log("Controls:")
        log("  SPACE/ENTER - Accept corners if grid looks good")
        log("  R - Reset and manually select new corners")
        log("  1/2/3/4 - Select corner to adjust (TL/TR/BR/BL)")
        log("  Click - Set new position for selected corner")
        log("-" * 60 + "\n")
        
        current_corners = corners.copy()
        selected_corner = None  # None, 0, 1, 2, 3 for TL, TR, BR, BL
        corner_names = ["TL", "TR", "BR", "BL"]
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal current_corners, selected_corner
            if event == cv2.EVENT_LBUTTONDOWN and selected_corner is not None:
                # Update the selected corner position in original image coordinates
                # We need to map from warped coordinates back to original
                # For now, let's work in the original image space
                pass
        
        # For corner adjustment, we'll work with original image
        def mouse_callback_original(event, x, y, flags, param):
            nonlocal current_corners, selected_corner
            if event == cv2.EVENT_LBUTTONDOWN and selected_corner is not None:
                current_corners[selected_corner] = [x, y]
                log(f"  → Corner {corner_names[selected_corner]} moved to ({x}, {y})")
                selected_corner = None
        
        cv2.namedWindow("Original with Corners")
        cv2.setMouseCallback("Original with Corners", mouse_callback_original)
        
        while True:
            # Get current camera image
            image = self.ximeaCamera.get_camera_image()
            if image is None:
                continue
            
            # Create warped view with grid overlay
            warped = self._trim_image_perspective(image, current_corners)
            
            if warped is not None and warped.shape[0] >= 800 and warped.shape[1] >= 800:
                display_warped = warped.copy()
                
                # Draw 8x8 grid
                cell_size = 100
                for i in range(9):
                    # Vertical lines
                    cv2.line(display_warped, (i * cell_size, 0), (i * cell_size, 800), (0, 255, 0), 2)
                    # Horizontal lines
                    cv2.line(display_warped, (0, i * cell_size), (800, i * cell_size), (0, 255, 0), 2)
                
                # Add labels
                cv2.putText(display_warped, "Grid should align with squares", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(display_warped, "SPACE=Accept, R=Reset, 1-4=Adjust corner", (10, 780), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                cv2.imshow("Warped Board (Grid Check)", display_warped)
            
            # Show original image with corner markers
            display_original = image.copy()
            
            # Draw corners and polygon
            pts = current_corners.astype(np.int32)
            cv2.polylines(display_original, [pts], True, (0, 255, 0), 3)
            
            for i, (corner, name) in enumerate(zip(current_corners, corner_names)):
                x, y = int(corner[0]), int(corner[1])
                color = (0, 0, 255) if selected_corner == i else (0, 255, 0)
                cv2.circle(display_original, (x, y), 10, color, -1)
                cv2.putText(display_original, f"{i+1}:{name}", (x+15, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if selected_corner is not None:
                cv2.putText(display_original, f"Click to move corner {selected_corner+1} ({corner_names[selected_corner]})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(display_original, "Press 1-4 to select corner, SPACE to accept", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Original with Corners", display_original)
            
            key = cv2.waitKey(30) & 0xFF
            
            # Accept corners
            if key == 32 or key == 13:  # SPACE or ENTER
                log("✓ Corners accepted!")
                cv2.destroyWindow("Warped Board (Grid Check)")
                cv2.destroyWindow("Original with Corners")
                return current_corners.astype(np.float32)
            
            # Reset to manual selection
            if key == ord('r') or key == ord('R'):
                log("  → Switching to manual corner selection...")
                cv2.destroyWindow("Warped Board (Grid Check)")
                cv2.destroyWindow("Original with Corners")
                return self._get_trim_param_manual()
            
            # Select corner to adjust
            if key == ord('1'):
                selected_corner = 0
                log(f"  → Selected corner 1 (TL) - click to move")
            elif key == ord('2'):
                selected_corner = 1
                log(f"  → Selected corner 2 (TR) - click to move")
            elif key == ord('3'):
                selected_corner = 2
                log(f"  → Selected corner 3 (BR) - click to move")
            elif key == ord('4'):
                selected_corner = 3
                log(f"  → Selected corner 4 (BL) - click to move")
            
            # ESC to cancel
            if key == 27:
                log("  → Cancelled, using current corners")
                cv2.destroyWindow("Warped Board (Grid Check)")
                cv2.destroyWindow("Original with Corners")
                return current_corners.astype(np.float32)

    def _camera_adjustment_window(self):
        log("\n" + "="*60)
        log("STEP 0: CAMERA ADJUSTMENT")
        log("="*60)
        log("Adjust the camera so the whole board is visible.")
        log("Press SPACE to continue when ready.")
        log("-" * 60 + "\n")

        while True:
            image = self.ximeaCamera.get_camera_image()
            display = image.copy()
            cv2.putText(display, "Adjust Camera. Press SPACE to continue", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Camera Adjustment", display)
            
            if cv2.waitKey(1) & 0xFF == 32: # SPACE
                cv2.destroyWindow("Camera Adjustment")
                break

    def _calibrate_thresholds_from_corners(self):
        """
        Calculate variance thresholds based on the 4 selected corners.
        Corner 1 (TL): White square on black side (Empty)
        Corner 2 (TR): Black square with black piece
        Corner 3 (BR): White square on white side
        Corner 4 (BL): White piece on black square
        """
        image = self.ximeaCamera.get_camera_image()
        warped = self._trim_image_perspective(image, self.bounderies)
        
        # Grid size is 800x800, so cell is 100x100
        # Get ROIs (top-left, top-right, bottom-right, bottom-left)
        # Add small margin to avoid edge artifacts
        margin = 10
        cell_size = 100
        
        # TL: warped[0:100, 0:100]
        roi_tl = warped[margin:cell_size-margin, margin:cell_size-margin]
        # TR: warped[0:100, 700:800]
        roi_tr = warped[margin:cell_size-margin, 700+margin:800-margin]
        # BR: warped[700:800, 700:800]
        roi_br = warped[700+margin:800-margin, 700+margin:800-margin]
        # BL: warped[700:800, 0:100]
        roi_bl = warped[700+margin:800-margin, margin:cell_size-margin]
        
        var_tl = self._calculate_variance(roi_tl) # Empty White Square
        var_tr = self._calculate_variance(roi_tr) # Black Piece
        var_br = self._calculate_variance(roi_br) # Empty White Square (or similar to TL)
        var_bl = self._calculate_variance(roi_bl) # White Piece
        
        print("\n" + "="*60)
        print("CALIBRATION RESULTS")
        print("="*60)
        print(f"Corner 1 (Empty/White Sq): Var = {var_tl:.2f}")
        print(f"Corner 2 (Black Piece):    Var = {var_tr:.2f}")
        print(f"Corner 3 (Empty/White Sq): Var = {var_br:.2f}")
        print(f"Corner 4 (White Piece):    Var = {var_bl:.2f}")
        
        # Set thresholds based on user logic and calibration
        # User: Empty < 10
        # User: Black Piece < 1200 (typically < 800)
        # User: White Piece is second threshold
        
        # We'll use the measured values to refine, but respect user limits
        self.empty_variance_threshold = 15.0 # Slightly lenient than 10
        if var_tl < 20:
             self.empty_variance_threshold = max(10, var_tl + 5)
        
        # Define Black Piece threshold
        # It should be > empty but covers the black piece variance
        # If var_tr is e.g. 500, we can set threshold around 800-1000
        self.black_variance_threshold = 1200 # Default upper limit
        # If monitored black piece is lower, we can tighten it, but 1200 is safe if white is higher
        
        # Check if White Piece is distinguished by higher variance
        self.white_piece_threshold = 1200 # Default
        
        if var_bl > var_tr:
            # White piece has higher variance
            # Set threshold between Black and White
            midpoint = (var_tr + var_bl) / 2
            self.white_piece_threshold = midpoint
        else:
            # Fallback if logic is inverted (unlikely for standard pieces)
            self.white_piece_threshold = 1200
            
        print(f"Set Empty Threshold: < {self.empty_variance_threshold}")
        print(f"Set Black Threshold: < {self.black_variance_threshold}")
        print(f"Set White Threshold: > {self.white_piece_threshold}")
        print("-" * 60 + "\n")

    def _calculate_variance(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean, stddev = cv2.meanStdDev(gray)
        return stddev[0][0] ** 2

    def _auto_detect_thresholds(self):
        """
        Automatically detect piece color thresholds by analyzing all 64 squares.
        
        In starting position:
        - 12 black pieces on rows 0-2 (on black squares only)
        - 12 white pieces on rows 5-7 (on black squares only)
        - 40 empty squares (all white squares + middle black squares)
        
        Black squares are where (row + col) is odd.
        We detect pieces only on black squares in the starting rows.
        
        Returns: dict with 'empty', 'black', 'white' thresholds or None if failed
        """
        image = self.ximeaCamera.get_camera_image()
        if image is None:
            print("  ✗ Could not get camera image for threshold detection")
            return None
        
        warped = self._trim_image_perspective(image, self.bounderies)
        if warped is None:
            print("  ✗ Could not warp image for threshold detection")
            return None
        
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        cell_size = 100
        padding = 10
        
        # Collect variances by expected category
        empty_variances = []
        black_piece_variances = []
        white_piece_variances = []
        all_variances = []
        
        for row in range(8):
            for col in range(8):
                is_black_square = (row + col) % 2 == 1
                
                # Extract square
                y1 = row * cell_size + padding
                y2 = (row + 1) * cell_size - padding
                x1 = col * cell_size + padding
                x2 = (col + 1) * cell_size - padding
                
                square = blur[y1:y2, x1:x2]
                if square.size == 0:
                    continue
                
                variance = np.var(square)
                all_variances.append((row, col, variance, is_black_square))
                
                # Categorize by expected starting position
                if not is_black_square:
                    # White squares are always empty
                    empty_variances.append(variance)
                elif row <= 2:
                    # Rows 0-2: black pieces on black squares
                    black_piece_variances.append(variance)
                elif row >= 5:
                    # Rows 5-7: white pieces on black squares
                    white_piece_variances.append(variance)
                else:
                    # Rows 3-4: empty black squares in middle
                    empty_variances.append(variance)
        
        # Validate we have enough data
        if len(black_piece_variances) < 10 or len(white_piece_variances) < 10:
            print(f"  ✗ Insufficient piece data: Black={len(black_piece_variances)}, White={len(white_piece_variances)}")
            return None
        
        # Calculate statistics
        empty_max = max(empty_variances) if empty_variances else 0
        empty_avg = np.mean(empty_variances) if empty_variances else 0
        
        black_min = min(black_piece_variances)
        black_max = max(black_piece_variances)
        black_avg = np.mean(black_piece_variances)
        
        white_min = min(white_piece_variances)
        white_max = max(white_piece_variances)
        white_avg = np.mean(white_piece_variances)
        
        print(f"  → Empty squares: count={len(empty_variances)}, max={empty_max:.0f}, avg={empty_avg:.0f}")
        print(f"  → Black pieces:  count={len(black_piece_variances)}, range={black_min:.0f}-{black_max:.0f}, avg={black_avg:.0f}")
        print(f"  → White pieces:  count={len(white_piece_variances)}, range={white_min:.0f}-{white_max:.0f}, avg={white_avg:.0f}")
        
        # Determine thresholds
        # Empty threshold: between empty max and black min
        if black_min > empty_max:
            empty_threshold = (empty_max + black_min) / 2
        else:
            # Overlap - use a higher value
            empty_threshold = empty_avg + (black_avg - empty_avg) * 0.3
            print(f"  ⚠ Variance overlap between empty and black pieces")
        
        # Black/White threshold: between black max and white min
        if white_min > black_max:
            black_threshold = (black_max + white_min) / 2
        else:
            # Overlap
            black_threshold = black_avg + (white_avg - black_avg) * 0.5
            print(f"  ⚠ Variance overlap between black and white pieces")
        
        return {
            'empty': empty_threshold,
            'black': black_threshold,
            'white': black_threshold  # Same as black threshold (anything above is white)
        }

    def _settings_window(self):
        """
        Show settings window with auto-detected thresholds and AI difficulty.
        User can adjust thresholds and confirm or use detected values.
        """
        print("\n" + "="*60)
        print("SETTINGS: Threshold Adjustment & AI Difficulty")
        print("="*60)
        print("Adjust thresholds if needed. Target: 40 empty, 12 black, 12 white")
        print("Press SPACE to confirm settings and continue.")
        print("-" * 60 + "\n")
        
        cv2.namedWindow('Settings')
        
        # Create trackbars with auto-detected values
        initial_empty = int(self.empty_variance_threshold)
        initial_black = int(self.black_variance_threshold)
        initial_difficulty = 3  # Default medium
        
        cv2.createTrackbar('empty_threshold', 'Settings', min(initial_empty, 5000), 5000, self._nothing)
        cv2.createTrackbar('black_threshold', 'Settings', min(initial_black, 5000), 5000, self._nothing)
        cv2.createTrackbar('AI_difficulty', 'Settings', initial_difficulty, 5, self._nothing)
        cv2.createTrackbar('show_values', 'Settings', 1, 1, self._nothing)
        
        self.selected_difficulty = initial_difficulty
        
        while True:
            image = self.ximeaCamera.get_camera_image()
            if image is None:
                continue
            
            warped = self._trim_image_perspective(image, self.bounderies)
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Get current trackbar values
            empty_threshold = cv2.getTrackbarPos('empty_threshold', 'Settings')
            black_threshold = cv2.getTrackbarPos('black_threshold', 'Settings')
            ai_difficulty = max(1, cv2.getTrackbarPos('AI_difficulty', 'Settings'))
            show_values = cv2.getTrackbarPos('show_values', 'Settings')
            
            self.selected_difficulty = ai_difficulty
            
            # Analyze board
            cell_size = 100
            padding = 10
            
            empty_count = 0
            black_count = 0
            white_count = 0
            
            result = warped.copy()
            all_variances = []
            
            for row in range(8):
                for col in range(8):
                    y1 = row * cell_size
                    y2 = (row + 1) * cell_size
                    x1 = col * cell_size
                    x2 = (col + 1) * cell_size
                    
                    square = blur[y1+padding:y2-padding, x1+padding:x2-padding]
                    if square.size == 0:
                        continue
                    
                    variance = np.var(square)
                    all_variances.append(variance)
                    
                    # Classify
                    if variance < empty_threshold:
                        empty_count += 1
                        color = (0, 255, 0)  # Green
                        label = "E"
                    elif variance < black_threshold:
                        black_count += 1
                        color = (255, 0, 255)  # Magenta
                        label = "B"
                    else:
                        white_count += 1
                        color = (0, 255, 255)  # Cyan
                        label = "W"
                    
                    cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
                    
                    # Label with background
                    label_pos = (x1+5, y1+25)
                    cv2.rectangle(result, (label_pos[0]-2, label_pos[1]-18), 
                                 (label_pos[0]+20, label_pos[1]+4), (0, 0, 0), -1)
                    cv2.putText(result, label, label_pos, 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    if show_values:
                        var_text = f"{int(variance)}"
                        var_pos = (x1+5, y2-8)
                        cv2.rectangle(result, (var_pos[0]-2, var_pos[1]-12), 
                                     (var_pos[0]+40, var_pos[1]+4), (0, 0, 0), -1)
                        cv2.putText(result, var_text, var_pos, 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Check if correct
            is_correct = (empty_count == 40 and black_count == 12 and white_count == 12)
            
            # Create info panel
            info_panel = np.zeros((500, 350, 3), dtype=np.uint8)
            info_panel[:] = (40, 40, 40)
            
            y_pos = 30
            cv2.putText(info_panel, "DETECTION RESULTS", (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_pos += 40
            
            # Counts
            for label, count, expected, clr in [("Empty", empty_count, 40, (0, 255, 0)),
                                                 ("Black", black_count, 12, (255, 0, 255)),
                                                 ("White", white_count, 12, (0, 255, 255))]:
                status_color = (0, 255, 0) if count == expected else (0, 0, 255)
                cv2.circle(info_panel, (30, y_pos - 6), 8, clr, -1)
                cv2.putText(info_panel, f"{label}: {count} / {expected}", (50, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                y_pos += 30
            
            y_pos += 20
            cv2.line(info_panel, (20, y_pos), (330, y_pos), (100, 100, 100), 1)
            y_pos += 30
            
            # Thresholds
            cv2.putText(info_panel, "THRESHOLDS", (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_pos += 35
            cv2.putText(info_panel, f"Empty:  < {empty_threshold}", (30, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_pos += 25
            cv2.putText(info_panel, f"Black:  {empty_threshold} - {black_threshold}", (30, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            y_pos += 25
            cv2.putText(info_panel, f"White:  > {black_threshold}", (30, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            y_pos += 40
            cv2.line(info_panel, (20, y_pos), (330, y_pos), (100, 100, 100), 1)
            y_pos += 30
            
            # AI Difficulty
            cv2.putText(info_panel, "AI DIFFICULTY", (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_pos += 35
            
            if ai_difficulty <= 1:
                diff_name, diff_color = "EASY", (0, 255, 0)
            elif ai_difficulty <= 3:
                diff_name, diff_color = "MEDIUM", (0, 255, 255)
            else:
                diff_name, diff_color = "HARD", (0, 0, 255)
            
            cv2.putText(info_panel, f"Level: {ai_difficulty} ({diff_name})", (30, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, diff_color, 2)
            
            y_pos += 50
            cv2.line(info_panel, (20, y_pos), (330, y_pos), (100, 100, 100), 1)
            y_pos += 30
            
            # Status
            if is_correct:
                cv2.putText(info_panel, "READY! Press SPACE", (20, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(info_panel, "Adjust thresholds...", (20, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            cv2.imshow("Settings", result)
            cv2.imshow("Settings_Info", info_panel)
            
            key = cv2.waitKey(30) & 0xFF
            
            if key == 32:  # SPACE
                self.empty_variance_threshold = empty_threshold
                self.black_variance_threshold = black_threshold
                self.selected_difficulty = ai_difficulty
                
                if ai_difficulty <= 1:
                    diff_name = "EASY"
                elif ai_difficulty <= 3:
                    diff_name = "MEDIUM"
                else:
                    diff_name = "HARD"
                
                print(f"\n✓ Settings confirmed!")
                print(f"  Thresholds: Empty<{empty_threshold}<Black<{black_threshold}<White")
                print(f"  AI Difficulty: {diff_name} (depth={ai_difficulty})")
                print(f"  Detection: Empty={empty_count}, Black={black_count}, White={white_count}")
                
                if not is_correct:
                    print(f"  ⚠ Warning: Detection not perfect, but continuing...")
                
                print("\n  → Press 'S' to start the game!")
                
                cv2.destroyWindow("Settings")
                cv2.destroyWindow("Settings_Info")
                break
            
            elif key == 27:  # ESC
                print("\n✗ Settings cancelled, using current values")
                cv2.destroyWindow("Settings")
                cv2.destroyWindow("Settings_Info")
                break

    def _piece_placement_window(self):
        print("\n" + "="*60)
        print("STEP 3: PIECE PLACEMENT CHECK")
        print("="*60)
        print("Place pieces in starting positions.")
        print("Press SPACE or ENTER to confirm and Start Game.")
        print("-" * 60 + "\n")
        
        while True:
            image = self.ximeaCamera.get_camera_image()
            if image is None:
                continue

            warped = self._trim_image_perspective(image, self.bounderies)
            display = warped.copy()
            
            # Analyze grid
            cell_size = 100
            margin = 10
            
            for row in range(8):
                for col in range(8):
                    x1 = col * cell_size
                    y1 = row * cell_size
                    x2 = x1 + cell_size
                    y2 = y1 + cell_size
                    
                    roi = warped[y1+margin:y2-margin, x1+margin:x2-margin]
                    variance = self._calculate_variance(roi)
                    
                    # Detection
                    color = (0, 255, 255) # Yellow unknown
                    label = "?"
                    
                    if variance < self.empty_variance_threshold:
                        label = "E"
                        color = (0, 255, 0) # Green Empty
                    elif variance < self.black_variance_threshold:
                        label = "B"
                        color = (0, 0, 255) # Red for Black
                    elif variance >= self.white_piece_threshold: # Assuming white is higher
                        label = "W"
                        color = (255, 255, 255) # White
                    else:
                        label = "?" 
                        
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display, f"{int(variance)}", (x1+5, y2-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    cv2.putText(display, label, (x1+40, y1+60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.imshow("Piece Placement & Detection", display)
            
            key = cv2.waitKey(10) & 0xFF
            # Accept ENTER (13) or SPACE (32)
            if key == 13 or key == 32: 
                cv2.destroyWindow("Piece Placement & Detection")
                print("✓ Board setup confirmed! Starting game...\n")
                break
                
            # Handle standard exit (ESC)
            if key == 27:
                cv2.destroyWindow("Piece Placement & Detection")
                print("⚠ Detection skipped by user.")
                break

    def _get_grid_squares_contours(self):
        """
        Generate strict 8x8 grid contours for the warped 800x800 image.
        Each square is 100x100.
        Returns: 8x8 list of [x, y, w, h]
        """
        contours = []
        cell_size = 100
        for row in range(8):
            row_cnts = []
            for col in range(8):
                x = col * cell_size
                y = row * cell_size
                # Format: [x, y, w, h]
                row_cnts.append([x, y, cell_size, cell_size])
            contours.append(row_cnts)
        return contours

    def get_board(self, cameraImage, game):
        """
        Main entry point for board detection during gameplay
        """
        # Use perspective transform with cached corners
        cameraImage = self._trim_image_perspective(cameraImage, self.bounderies)
        
        # We don't need _get_empty_fields_and_pieces anymore as calibration is done.
        # Just use the thresholds to detect board state.
        
        # Ensure gameBoardFieldsContours is set
        if not hasattr(self, 'gameBoardFieldsContours') or self.gameBoardFieldsContours is None:
             self.gameBoardFieldsContours = self._get_grid_squares_contours()

        self.set_number_of_empty_fields(game)
        
        # Pass [] as emptyFieldsContours since it's unused in _get_board_from_image
        return self._get_board_from_image(cameraImage, [])

    def _get_trim_param_manual(self):
        """
        Manual board corner selection - click 4 corners in order:
        1. White square on black side (Top-Left)
        2. Black square with black piece (Top-Right)
        3. White square on white side (Bottom-Right)
        4. White piece on black square (Bottom-Left)
        """
        corners = []
        clone = None
        
        def click_event(event, x, y, flags, params):
            nonlocal corners, clone
            
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(corners) < 4:
                    corners.append([x, y])
                    print(f"  ✓ Corner {len(corners)}/4 selected: ({x}, {y})")
                    
                    # Draw the point
                    cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)
                    
                    # Draw line if we have more than one point
                    if len(corners) > 1:
                        cv2.line(clone, tuple(corners[-2]), tuple(corners[-1]), (0, 255, 0), 2)
                    
                    # Close the polygon after 4 points
                    if len(corners) == 4:
                        cv2.line(clone, tuple(corners[-1]), tuple(corners[0]), (0, 255, 0), 2)
                        print("\n  → All 4 corners selected!")
                        print("  → Press SPACE in 'Select Board Corners' window to confirm")
                        print("  → Press 'R' to reset and reselect corners\n")
                    
                    cv2.imshow("Select Board Corners", clone)
        
        print("\n" + "="*60)
        print("STEP 1: BOARD CORNER SELECTION")
        print("="*60)
        print("Click on the 4 corners of the board in this order:")
        print("  1. White square on black side (Top-Left)")
        print("  2. Black square with black piece (Top-Right)")
        print("  3. White square on white side (Bottom-Right)")
        print("  4. White piece on black square (Bottom-Left)")
        print("\nControls:")
        print("  SPACE - Confirm selection")
        print("  R     - Reset points")
        print("  ESC   - Exit")
        print("-"*60 + "\n")
        
        while True:
            # Get fresh image
            image = self.ximeaCamera.get_camera_image()
            clone = image.copy()
            
            # Redraw existing corners
            for i, corner in enumerate(corners):
                cv2.circle(clone, tuple(corner), 5, (0, 255, 0), -1)
                cv2.putText(clone, str(i+1), (corner[0]+10, corner[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if i > 0:
                    cv2.line(clone, tuple(corners[i-1]), tuple(corners[i]), (0, 255, 0), 2)
            
            if len(corners) == 4:
                cv2.line(clone, tuple(corners[-1]), tuple(corners[0]), (0, 255, 0), 2)
            
            cv2.imshow("Select Board Corners", clone)
            cv2.setMouseCallback("Select Board Corners", click_event)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Reset selection
            if key == ord('r') or key == ord('R'):
                corners = []
                print("\n  ↻ Points reset - start selecting again\n")
            
            # Confirm selection
            if key == 32 and len(corners) == 4:
                cv2.destroyWindow("Select Board Corners")
                print("✓ Board corners saved!\n")
                print("="*60)
                print("STEP 2: DETECTING BOARD GRID")
                print("="*60 + "\n")
                return np.array(corners, dtype=np.float32)
            
            # Exit
            if key == 27:
                cv2.destroyWindow("Select Board Corners")
                print("\n✗ Board selection cancelled\n")
                return None

    def _trim_image_perspective(self, image, corners):
        """
        Apply perspective transform to get top-down view of the board
        """
        if corners is None or len(corners) != 4:
            return image
        
        # Define the output size (you can adjust this)
        board_size = 800  # 800x800 pixel output
        
        # Define destination points for perspective transform
        dst_points = np.array([
            [0, 0],
            [board_size, 0],
            [board_size, board_size],
            [0, board_size]
        ], dtype=np.float32)
        
        # Calculate perspective transform matrix
        matrix = cv2.getPerspectiveTransform(corners, dst_points)
        
        # Apply perspective transform
        warped = cv2.warpPerspective(image, matrix, (board_size, board_size))
        
        return warped

    def _get_trim_param(self):
        """
        DEPRECATED: Old automatic trimming method (keeping for reference)
        Use _get_trim_param_manual() instead
        """
        bounderies = []
        temp_bounderies = []
        while 1:
            temp_bounderies = []
            cameraImage = self.ximeaCamera.get_camera_image()
            cameraImage = self._trim_image(cameraImage, bounderies)
            cv2.imshow("original", cameraImage)

            temp_bounderies.append(self._get_bounderies(cameraImage))
            
            # Get the bounding rectangle for the largest contour
            x, y, w, h = cv2.boundingRect(temp_bounderies[-1])

            # Crop the image using the bounding rectangle
            trimmed_image = cameraImage.copy()
            trimmed_image = trimmed_image[y:y+h, x:x+w]

            cv2.imshow("trimming", trimmed_image)

            key = cv2.waitKey(1) & 0xFF
            if key == 32:
                bounderies.extend(temp_bounderies)
            
            if key == 27:
                cv2.destroyWindow("trimming")
                cv2.destroyWindow("original")
                cv2.destroyWindow("contours")
                cv2.destroyWindow("thresh")

                break

        return bounderies

    def _get_bounderies(self, image):
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)

        # Threshold the image to separate the black frame
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        cv2.imshow("thresh", thresh)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort the contours by area and keep the largest one
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        result = image.copy()
        for c in contours:
            cv2.drawContours(result, [c], -1, (0, 255, 0), 2)
        cv2.imshow("contours", result)

        return contours[0]

    def _trim_image(self, image, bounderies):
        """
        DEPRECATED: Old trimming method using contours
        Use _trim_image_perspective() instead
        """
        trimmed_image = image.copy()
        for boundery in bounderies:
            # Get the bounding rectangle for the largest contour
            x, y, w, h = cv2.boundingRect(boundery)

            # Crop the image using the bounding rectangle
            trimmed_image = trimmed_image[y:y+h, x:x+w]
            
        return trimmed_image
    

    def _get_contours(self, cameraImage):
        all_contours = self._get_contours_off_all_rectangles(cameraImage)
        return self._get_sorted_contours(all_contours)


    def _get_contours_off_all_rectangles(self, image):
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        attempts = 1
        while 1:
            if(self.param1ForGetAllContours < 1):
                attempts += 1
                if(attempts > 1):
                    break
                
            self.param1ForGetAllContours -= 1

            # Detect lines using the Hough Line Transform
            lines = cv2.HoughLines(edges, 1, np.pi / 180, self.param1ForGetAllContours)

            # Create a copy of the original image to draw lines on
            black_image = image.copy()
            black_image = np.zeros_like(black_image)

            # Draw the lines on the image
            if lines is not None:
                for rho, theta in lines[:, 0]:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * a)
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * a)

                    cv2.line(black_image, (x1, y1), (x2, y2), (255, 255, 255), 2)

            gray = cv2.cvtColor(black_image, cv2.COLOR_RGB2GRAY)

            # cv2.imshow("all_rectangles_black_image", gray)

            # Find contours
            contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Filter for rectangles
            rectangles = []
            for cnt in contours:
                # Get convex hull
                hull = cv2.convexHull(cnt)
                
                # Get approximate polygon
                epsilon = 0.02 * cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, epsilon, True)
                
                # Check if it is a rectangle
                if len(approx) == 4:
                    rectangles.append(approx)


            # Sort the contours by area and keep the largest one
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            new_image = image.copy()
            if(len(contours) >= 64 and len(contours) <= 150):
                correct_result = 0
                new_contours = []
                for position in range(1, len(contours)):

                    x,y,w,h = cv2.boundingRect(contours[position])
                
                    if(w > 20 and h > 20 and correct_result < 64):
                        new_contours.append(contours[position])
                        correct_result += 1
                        cv2.rectangle(new_image, (x, y), (x + w, y + h), (36,255,12), 1)
                # cv2.drawContours(result, [contours[position]], -1, (0, 0, 255), 2)

                # cv2.imshow('all_rectangles_new_image', new_image)
                if(correct_result == 64):
                    break

        return new_contours

    def _get_sorted_contours(self, all_contours):
        array_2d_in_2d = []
        position = 0
        group_range = 25  # Rozsah pre skupiny
        if(len(all_contours) == 64):
            for i in range(8):
                inner_array = []
                for j in range(8):
                    # Store the actual contour instead of just bounding rect
                    inner_array.append(all_contours[position])
                    position += 1
                array_2d_in_2d.append(inner_array)

        # Function to get bounding rect for sorting
        def get_bounding_rect(contour):
            x, y, w, h = cv2.boundingRect(contour)
            return [contour, x, y, w, h]

        # Function to flatten and sort the 2D array based on "y" values
        def flatten_and_sort(array):
            flat_list = [get_bounding_rect(item) for sublist in array for item in sublist]
            return sorted(flat_list, key=lambda x: x[2])  # Sort by y

        # Function to group elements by "y" value within a specified range
        def group_by_y_range(sorted_list, group_range):
            grouped = []
            while sorted_list:
                base_y = sorted_list[0][2]
                group = []

                for element in sorted_list:
                    if base_y - group_range <= element[2] <= base_y + group_range:
                        group.append(element)
                        if len(group) == 8:
                            break

                grouped.append(group)
                # Use index-based filtering instead of 'not in' comparison
                group_ids = {id(item) for item in group}
                sorted_list = [x for x in sorted_list if id(x) not in group_ids]

            return grouped

        # Function to ensure the final array is 8x8
        def ensure_8x8_array(grouped_elements):
            while len(grouped_elements) < 8:
                grouped_elements.append([[None, 0, 0, 0, 0]] * 8)
            return grouped_elements[:8]

        def order_rows_by_x(array_2d):
            ordered_array_2d = []
            for row in array_2d:
                ordered_row = sorted(row, key=lambda x: x[1])  # Sort by x
                ordered_array_2d.append(ordered_row)
            return ordered_array_2d
        
        # Function to reorder the rows based on the first element's "y" value in each row
        def reorder_rows_by_first_y(array_2d):
            # Sorting the entire 2D array based on the "y" value of the first element in each row
            reordered_array_2d = sorted(array_2d, key=lambda row: row[0][2])
            return reordered_array_2d
        
        # Main execution
        sorted_flat_list = flatten_and_sort(array_2d_in_2d)
        grouped_elements = group_by_y_range(sorted_flat_list, group_range)
        new_array_2d_in_2d = ensure_8x8_array(grouped_elements)
        # Ordering each row by "x" value
        ordered_new_array_2d_in_2d = order_rows_by_x(new_array_2d_in_2d)
        # Reordering rows based on the first element's "y" value
        reordered_new_array_2d_in_2d = reorder_rows_by_first_y(ordered_new_array_2d_in_2d)

        # Extract just [x, y, w, h] for compatibility
        final_result = []
        for row in reordered_new_array_2d_in_2d:
            final_row = []
            for element in row:
                final_row.append([element[1], element[2], element[3], element[4]])  # [x, y, w, h]
            final_result.append(final_row)

        return final_result

    
    def _nothing(self, x):
        pass

    def _get_empty_fields_and_pieces(self, cameraImage):
        """
        Unified detection: classify each square as empty, black piece, or white piece
        based on VARIANCE with 2 adjustable thresholds
        
        Variance ranges (from your data):
        - Empty white squares: ~250
        - Empty black squares: ~500
        - Black pieces: ~700-900
        - White pieces: >1100
        """
        createTrackBars = False
        image = cameraImage.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Check if trackbar exists
        try:
            cv2.getTrackbarPos('empty_variance', 'Square_Classification')
        except:
            createTrackBars = True

        cv2.namedWindow('Square_Classification')
        
        if createTrackBars:
            print("\n" + "="*60)
            print("STEP 3: SQUARE CLASSIFICATION & AI DIFFICULTY")
            print("="*60)
            print("\nAdjust the 2 variance thresholds to classify all 64 squares:")
            print("  • Empty squares: variance BELOW 'empty_variance'")
            print("  • Black pieces: variance between 'empty_variance' and 'black_variance'")
            print("  • White pieces: variance ABOVE 'black_variance'")
            print(f"\nTarget: {self.numberOfEmptyFields} empty, 12 black, 12 white")
            print("\nBased on your previous calibration:")
            print("  Empty squares: variance ~200-500")
            print("  Black pieces: variance ~700-900")
            print("  White pieces: variance >1100")
            print("\n★ AI DIFFICULTY (use 'difficulty' slider):")
            print("  1 = EASY   (fast, makes mistakes)")
            print("  3 = MEDIUM (balanced)")
            print("  5 = HARD   (slow, very challenging)")
            print("\n→ Adjust trackbars in 'Square_Classification' window")
            print("→ Press SPACE when you see correct counts")
            print("-"*60)
            
            # Default values based on your data, increased max to 5000
            cv2.createTrackbar('empty_variance', 'Square_Classification', 600, 5000, self._nothing)
            cv2.createTrackbar('black_variance', 'Square_Classification', 2000, 5000, self._nothing)
            cv2.createTrackbar('show_values', 'Square_Classification', 1, 1, self._nothing)
            # AI Difficulty trackbar: 1=Easy, 3=Medium, 5=Hard
            cv2.createTrackbar('AI_difficulty', 'Square_Classification', 3, 5, self._nothing)
            # Initialize difficulty storage
            self.selected_difficulty = 3  # Default: Medium

        # INITIALIZATION MODE - Wait loop for user to adjust and confirm
        while createTrackBars:
            # Get current image
            current_image = self.ximeaCamera.get_camera_image()
            current_image = self._trim_image_perspective(current_image, self.bounderies)
            current_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
            current_blur = cv2.GaussianBlur(current_gray, (5, 5), 0)
            
            empty_variance = cv2.getTrackbarPos('empty_variance', 'Square_Classification')
            black_variance = cv2.getTrackbarPos('black_variance', 'Square_Classification')
            show_values = cv2.getTrackbarPos('show_values', 'Square_Classification')
            
            # Get AI difficulty (ensure minimum of 1)
            ai_difficulty = max(1, cv2.getTrackbarPos('AI_difficulty', 'Square_Classification'))
            self.selected_difficulty = ai_difficulty
            
            # Get the dimensions of each square
            h, w = current_image.shape[:2]
            square_h = h // 8
            square_w = w // 8
            
            # Classify each square
            empty_fields = []
            black_pieces = []
            white_pieces = []
            result = current_image.copy()
            
            # Store all variance values for statistics
            all_variances = []
            
            # Create visualization - CLEAN board with only colored rectangles and values
            for row in range(8):
                for col in range(8):
                    # Extract the square region
                    y1 = row * square_h
                    y2 = (row + 1) * square_h
                    x1 = col * square_w
                    x2 = (col + 1) * square_w
                    
                    # Get the square with padding to avoid edges
                    padding = 10
                    y1_pad = y1 + padding
                    y2_pad = y2 - padding
                    x1_pad = x1 + padding
                    x2_pad = x2 - padding
                    
                    square = current_blur[y1_pad:y2_pad, x1_pad:x2_pad]
                    
                    if square.size == 0:
                        continue
                    
                    # Calculate variance of the square
                    variance = np.var(square)
                    all_variances.append(variance)
                    
                    # Classify based on variance thresholds
                    if variance < empty_variance:
                        # Empty square (low variance ~200-500)
                        empty_fields.append((row, col))
                        color = (0, 255, 0)  # Green
                        label = "E"
                    elif variance < black_variance:
                        # Black piece (medium variance ~700-900)
                        black_pieces.append((row, col))
                        color = (255, 0, 255)  # Magenta
                        label = "B"
                    else:
                        # White piece (high variance >1100)
                        white_pieces.append((row, col))
                        color = (0, 255, 255)  # Cyan
                        label = "W"
                    
                    # Draw rectangle
                    cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label with black background for readability
                    label_pos = (x1+5, y1+22)
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(result, (label_pos[0]-2, label_pos[1]-text_size[1]-2), 
                                 (label_pos[0]+text_size[0]+2, label_pos[1]+2), 
                                 (0, 0, 0), -1)
                    cv2.putText(result, label, label_pos, 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Show variance value with background
                    if show_values:
                        var_text = f"{int(variance)}"
                        var_pos = (x1+5, y2-7)
                        text_size_var = cv2.getTextSize(var_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                        cv2.rectangle(result, (var_pos[0]-2, var_pos[1]-text_size_var[1]-2), 
                                     (var_pos[0]+text_size_var[0]+2, var_pos[1]+2), 
                                     (0, 0, 0), -1)
                        cv2.putText(result, var_text, var_pos, 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Count results
            num_empty = len(empty_fields)
            num_black = len(black_pieces)
            num_white = len(white_pieces)
            
            # Check if correct
            is_correct = (num_empty == self.numberOfEmptyFields and 
                         num_black == 12 and num_white == 12)
            
            # Calculate statistics (handle empty list case)
            if len(all_variances) > 0:
                min_var = int(min(all_variances))
                max_var = int(max(all_variances))
                avg_var = int(np.mean(all_variances))
            else:
                min_var = max_var = avg_var = 0
            
            # Create separate INFO PANEL (400x600)
            info_panel = np.zeros((600, 400, 3), dtype=np.uint8)
            info_panel[:] = (40, 40, 40)  # Dark gray background
            
            y_pos = 40
            line_height = 35
            
            # Title
            cv2.putText(info_panel, "CLASSIFICATION INFO", (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            y_pos += 50
            
            # Separator
            cv2.line(info_panel, (20, y_pos), (380, y_pos), (100, 100, 100), 2)
            y_pos += 40
            
            # Piece counts with color indicators
            text = f"Empty: {num_empty} / {self.numberOfEmptyFields}"
            text_color = (0, 255, 0) if num_empty == self.numberOfEmptyFields else (0, 0, 255)
            cv2.circle(info_panel, (30, y_pos - 8), 10, (0, 255, 0), -1)
            cv2.putText(info_panel, text, (50, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            y_pos += line_height
            
            text = f"Black: {num_black} / 12"
            text_color = (0, 255, 0) if num_black == 12 else (0, 0, 255)
            cv2.circle(info_panel, (30, y_pos - 8), 10, (255, 0, 255), -1)
            cv2.putText(info_panel, text, (50, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            y_pos += line_height
            
            text = f"White: {num_white} / 12"
            text_color = (0, 255, 0) if num_white == 12 else (0, 0, 255)
            cv2.circle(info_panel, (30, y_pos - 8), 10, (0, 255, 255), -1)
            cv2.putText(info_panel, text, (50, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            y_pos += 50
            
            # Separator
            cv2.line(info_panel, (20, y_pos), (380, y_pos), (100, 100, 100), 2)
            y_pos += 40
            
            # Variance statistics
            cv2.putText(info_panel, "VARIANCE STATISTICS", (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_pos += 40
            
            cv2.putText(info_panel, f"Min:  {min_var}", (30, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            y_pos += 30
            
            cv2.putText(info_panel, f"Max:  {max_var}", (30, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            y_pos += 30
            
            cv2.putText(info_panel, f"Avg:  {avg_var}", (30, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            y_pos += 50
            
            # Separator
            cv2.line(info_panel, (20, y_pos), (380, y_pos), (100, 100, 100), 2)
            y_pos += 40
            
            # Thresholds
            cv2.putText(info_panel, "THRESHOLDS", (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_pos += 40
            
            cv2.putText(info_panel, f"Empty:  < {empty_variance}", (30, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            y_pos += 30
            
            cv2.putText(info_panel, f"Black:  {empty_variance} - {black_variance}", (30, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
            y_pos += 30
            
            cv2.putText(info_panel, f"White:  > {black_variance}", (30, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            y_pos += 50
            
            # Separator
            cv2.line(info_panel, (20, y_pos), (380, y_pos), (100, 100, 100), 2)
            y_pos += 40
            
            # AI Difficulty section
            cv2.putText(info_panel, "AI DIFFICULTY", (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_pos += 40
            
            # Display difficulty name based on value
            if ai_difficulty <= 1:
                diff_name = "EASY"
                diff_color = (0, 255, 0)  # Green
            elif ai_difficulty <= 3:
                diff_name = "MEDIUM"
                diff_color = (0, 255, 255)  # Yellow
            else:
                diff_name = "HARD"
                diff_color = (0, 0, 255)  # Red
            
            cv2.putText(info_panel, f"Level: {ai_difficulty} ({diff_name})", (30, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, diff_color, 2)
            y_pos += 50
            
            # Separator
            cv2.line(info_panel, (20, y_pos), (380, y_pos), (100, 100, 100), 2)
            y_pos += 40
            
            # Status message
            if is_correct:
                status_text = "PERFECT!"
                status_color = (0, 255, 0)
                instruction = "Press SPACE"
            else:
                status_text = "Adjust Thresholds"
                status_color = (0, 165, 255)
                instruction = "Use trackbars above"
            
            cv2.putText(info_panel, status_text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
            y_pos += 40
            
            cv2.putText(info_panel, instruction, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Show windows side by side
            cv2.imshow("Square_Classification", result)
            cv2.imshow("Classification_Info", info_panel)
            cv2.imshow("boardCamera", current_image)
            
            # Wait for key press
            key = cv2.waitKey(30) & 0xFF
            
            if key == 32:  # SPACE pressed
                # Get difficulty name for display
                if ai_difficulty <= 1:
                    diff_name = "EASY"
                elif ai_difficulty <= 3:
                    diff_name = "MEDIUM"
                else:
                    diff_name = "HARD"
                    
                if is_correct:
                    print(f"\n✓ Perfect classification achieved!")
                    print(f"  Empty: {num_empty}, Black: {num_black}, White: {num_white}")
                    print(f"  Variance range: {min_var} - {max_var} (avg: {avg_var})")
                    print(f"  Thresholds: Empty<{empty_variance}<Black<{black_variance}<White")
                    print(f"  ★ AI Difficulty: {diff_name} (depth={ai_difficulty})")
                    print("  → Initialization complete, press 'S' to start game!\n")
                else:
                    print(f"\n⚠ Warning: Not perfect but continuing...")
                    print(f"  Empty: {num_empty}/{self.numberOfEmptyFields}, Black: {num_black}/12, White: {num_white}/12")
                    print(f"  Variance range: {min_var} - {max_var} (avg: {avg_var})")
                    print(f"  ★ AI Difficulty: {diff_name} (depth={ai_difficulty})")
                
                # Store thresholds for runtime use
                self.empty_variance_threshold = empty_variance
                self.black_variance_threshold = black_variance
                
                # Store selected difficulty for checkers_node to read
                self.selected_difficulty = ai_difficulty
                
                # Close info window
                cv2.destroyWindow("Classification_Info")
                
                # Create empty field contours for compatibility
                contours = []
                for row, col in empty_fields:
                    y = row * square_h + square_h // 2
                    x = col * square_w + square_w // 2
                    cnt = np.array([
                        [[x-10, y-10]],
                        [[x+10, y-10]],
                        [[x+10, y+10]],
                        [[x-10, y+10]]
                    ], dtype=np.int32)
                    contours.append(cnt)
                
                print(f"  → Returning {len(contours)} empty field contours")
                return contours
            
            elif key == 27:  # ESC pressed
                print("\n✗ Classification cancelled\n")
                cv2.destroyWindow("Classification_Info")
                return []
        
        # RUNTIME MODE - If trackbars already exist, use stored thresholds quickly
        # print("  → Using cached thresholds for runtime detection")
        h, w = image.shape[:2]
        square_h = h // 8
        square_w = w // 8
        
        empty_fields = []
        
        for row in range(8):
            for col in range(8):
                y1 = row * square_h
                y2 = (row + 1) * square_h
                x1 = col * square_w
                x2 = (col + 1) * square_w
                
                padding = 10
                y1_pad = y1 + padding
                y2_pad = y2 - padding
                x1_pad = x1 + padding
                x2_pad = x2 - padding
                
                square = blur[y1_pad:y2_pad, x1_pad:x2_pad]
                
                if square.size == 0:
                    continue
                    
                variance = np.var(square)
                
                if variance < self.empty_variance_threshold:
                    empty_fields.append((row, col))
        
        # Create contours
        contours = []
        for row, col in empty_fields:
            y = row * square_h + square_h // 2
            x = col * square_w + square_w // 2
            cnt = np.array([
                [[x-10, y-10]],
                [[x+10, y-10]],
                [[x+10, y+10]],
                [[x-10, y+10]]
            ], dtype=np.int32)
            contours.append(cnt)
        
        return contours

    def _get_board_from_image(self, cameraImage, emptyFieldsContours):
        """
        Simplified: use the same variance thresholds from initialization
        """
        board = np.empty((8, 8), dtype=object)
        board.fill(0)

        new_image = cameraImage.copy()
        gray = cv2.cvtColor(cameraImage, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Get dimensions
        h, w = cameraImage.shape[:2]
        square_h = h // 8
        square_w = w // 8
        
        # Track counts
        black_count = 0
        white_count = 0
        
        position = 0
        for row in range(len(self.gameBoardFieldsContours)):
            for col in range(len(self.gameBoardFieldsContours[row])):
                gameBoardFieldsContours = self.gameBoardFieldsContours[row][col]
                if len(gameBoardFieldsContours) != 4:
                    continue
                    
                # Get square region
                x, y, w_rect, h_rect = gameBoardFieldsContours
                
                # Add padding
                padding = 10
                x_pad = x + padding
                y_pad = y + padding
                w_pad = w_rect - 2 * padding
                h_pad = h_rect - 2 * padding
                
                # Ensure valid region
                if w_pad <= 0 or h_pad <= 0:
                    board[row][col] = 0
                    position += 1
                    continue
                
                # Extract square and calculate variance
                square = blur[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
                
                if square.size == 0:
                    board[row][col] = 0
                    position += 1
                    continue
                    
                variance = np.var(square)
                
                # Classify using stored variance thresholds
                if variance < self.empty_variance_threshold:
                    # Empty
                    board[row][col] = 0
                    color = (0, 255, 255)  # Yellow
                    label = str(position)
                elif variance < self.black_variance_threshold:
                    # Black piece
                    board[row][col] = 2
                    black_count += 1
                    color = (255, 0, 255)  # Magenta
                    label = str(position)
                else:
                    # White piece
                    board[row][col] = 1
                    white_count += 1
                    color = (0, 255, 255)  # Cyan
                    label = str(position)
                
                # Draw on image with background for readability
                point_x = x + w_rect // 2
                point_y = y + h_rect // 2
                
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(new_image, 
                             (point_x - 12, point_y - text_size[1] - 2), 
                             (point_x + text_size[0] - 8, point_y + 7), 
                             (0, 0, 0), -1)
                cv2.putText(new_image, label, (point_x-10, point_y+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                position += 1
        
        # Add info overlay with background
        text = f"Black: {black_count} | White: {white_count}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(new_image, (8, 2), (12 + text_size[0], 29), (0, 0, 0), -1)
        cv2.putText(new_image, text, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("gameboard", new_image)
        
        # Print status on first run or when counts change
        if not self.is_initialized:
            print(f"\n" + "="*60)
            print("BOARD STATE")
            print("="*60)
            print(f"  Detected: Black={black_count}, White={white_count}")
            print(f"  Using thresholds: Empty<{self.empty_variance_threshold}<Black<{self.black_variance_threshold}<White")
            
            if black_count == 12 and white_count == 12:
                print(f"  ✓ Perfect! Game ready")
                print(f"  → Press 'S' in any OpenCV window to start the game")
                self.is_initialized = True
            else:
                print(f"  ⚠ Piece count mismatch - adjust thresholds or continue anyway")
            print("="*60 + "\n")
        
        return board

    def set_number_of_empty_fields(self, game):
        """
        Update the expected number of empty fields based on current game state
        """
        self.numberOfEmptyFields = 64 - game.board.black_left - game.board.white_left