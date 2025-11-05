import cv2
from matplotlib.pyplot import imshow
from checkers_game.camera.ximea_camera import XimeaCamera
import numpy as np
import copy
from checkers_game.checkers.piece import Piece
from checkers_game.constants import BLACK, ROWS, RED, SQUARE_SIZE, COLS, WHITE, GREY, BROWN

class BoardDetection:

    def __init__(self, ximeaCamera):
        self.ximeaCamera = ximeaCamera
        self._init()
        

    def _init(self):
        self.numberOfEmptyFields = 40
        self.param1ForGetAllContours = 255
        self.bounderies = self._get_trim_param_manual()

        cameraImage = self.ximeaCamera.get_camera_image()
        cameraImage = self._trim_image_perspective(cameraImage, self.bounderies)

        self.gameBoardFieldsContours = self._get_contours(cameraImage)
        
        # Initialize variance thresholds (will be set during calibration)
        self.empty_variance_threshold = 500  # Empty squares have low variance ~200-500
        self.black_variance_threshold = 1000  # Black pieces ~700-900, White pieces >1100
        
        # Add flags to track initialization state
        self.is_initialized = False
        self.trackbars_created = False

    def _get_trim_param_manual(self):
        """
        Manual board corner selection - click 4 corners in order:
        1. Top-left
        2. Top-right
        3. Bottom-right
        4. Bottom-left
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
        print("  1. Top-left corner")
        print("  2. Top-right corner")
        print("  3. Bottom-right corner")
        print("  4. Bottom-left corner")
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
            print("STEP 3: SQUARE CLASSIFICATION (EMPTY/BLACK/WHITE)")
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
            print("\n→ Adjust trackbars in 'Square_Classification' window")
            print("→ Press SPACE when you see correct counts")
            print("-"*60)
            
            # Default values based on your data, increased max to 2200
            cv2.createTrackbar('empty_variance', 'Square_Classification', 600, 2200, self._nothing)
            cv2.createTrackbar('black_variance', 'Square_Classification', 1000, 2200, self._nothing)
            cv2.createTrackbar('show_values', 'Square_Classification', 1, 1, self._nothing)

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
                if is_correct:
                    print(f"\n✓ Perfect classification achieved!")
                    print(f"  Empty: {num_empty}, Black: {num_black}, White: {num_white}")
                    print(f"  Variance range: {min_var} - {max_var} (avg: {avg_var})")
                    print(f"  Thresholds: Empty<{empty_variance}<Black<{black_variance}<White")
                    print("  → Initialization complete, press 'S' to start game!\n")
                else:
                    print(f"\n⚠ Warning: Not perfect but continuing...")
                    print(f"  Empty: {num_empty}/{self.numberOfEmptyFields}, Black: {num_black}/12, White: {num_white}/12")
                    print(f"  Variance range: {min_var} - {max_var} (avg: {avg_var})")
                
                # Store thresholds for runtime use
                self.empty_variance_threshold = empty_variance
                self.black_variance_threshold = black_variance
                
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

    def get_board(self, cameraImage, game):
        """
        Main entry point for board detection during gameplay
        """
        # Use perspective transform
        cameraImage = self._trim_image_perspective(cameraImage, self.bounderies)
        cv2.imshow("boardCamera", cameraImage)
        
        self.set_number_of_empty_fields(game)

        # Use unified detection method
        emptyFieldsContours = self._get_empty_fields_and_pieces(cameraImage)
        
        return self._get_board_from_image(cameraImage, emptyFieldsContours)
        
    def set_number_of_empty_fields(self, game):
        """
        Update the expected number of empty fields based on current game state
        """
        self.numberOfEmptyFields = 64 - game.board.black_left - game.board.white_left