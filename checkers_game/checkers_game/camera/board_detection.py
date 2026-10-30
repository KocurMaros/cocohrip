"""Interactive board detection for the checkers node (camera + OpenCV GUI).

Startup (all in OpenCV windows):
  0. camera adjustment            - SPACE to continue
  1. board corners                - cached / automatic (board_geometry) /
                                    manual, verified on a grid overlay
  2. orientation + piece model    - put the pieces in the START position; the
                                    board orientation (robot's black pieces on
                                    top) and the piece classifier are learned
                                    from it.  SPACE to accept.
At runtime get_board() returns an 8x8 grid (see board_vision for the values),
combining the classifier with change detection against the last confirmed
position (set_reference), which is what keeps black-on-dark pieces reliable.

The pure image processing lives in board_geometry.py (corners) and
board_vision.py (pieces), both testable without a camera.
"""

import json
import os
import sys
from datetime import datetime

import cv2
import numpy as np

from checkers_game.camera import board_geometry
from checkers_game.camera import board_vision as bv

# Debug mode - set to True to show additional windows for debugging
DEBUG_MODE = True

# Where the last verified board calibration (corners + orientation) is cached,
# so a fixed camera mount doesn't need detection redone every session.
CALIBRATION_DIR = os.path.expanduser("~/.cocohrip")
CALIBRATION_PATH = os.path.join(CALIBRATION_DIR, "board_calibration.json")

# classifier confidence needed to trust a square without temporal evidence
CONFIDENT = 0.90
# above this classifier confidence the temporal prior is not used at all
TRUST_CLASSIFIER = 0.97
# change z-score below which a square is considered unchanged
Z_UNCHANGED = 3.0
# change z-score above which a square has certainly changed
Z_CHANGED = 6.0


def log(msg):
    """Print with immediate flush so output appears in terminal right away"""
    print(msg, flush=True)
    sys.stdout.flush()


class BoardDetection:

    def __init__(self, ximeaCamera, interactive=True):
        self.ximeaCamera = ximeaCamera
        self.bounderies = None          # corners BL, TL, TR, BR (camera image px)
        self.rotation = 0               # 90-degree steps into game orientation
        self.dark_parity = 1
        self.classifier = bv.PieceClassifier()
        self.selected_difficulty = 3
        self.ref_norm = None            # normalised warp of the last confirmed position
        self.ref_grid = None
        self.last_probs = None
        self.last_change = None
        self.last_light = None
        self.centre_model = None        # where a centred piece shows up (bv.fit_centre_model)
        self.gameBoardFieldsContours = self._get_grid_squares_contours()
        if interactive:
            self._init()

    def _init(self):
        # 1. Camera Adjustment Phase
        self._camera_adjustment_window()

        # 2. Board corners - cached / auto / manual, verified by the user
        self.bounderies = self._resolve_board_corners()

        # 3. Orientation + piece classifier from the start position
        self._piece_calibration_window()

    # ------------------------------------------------------------------
    # public API used by checkers_node
    # ------------------------------------------------------------------
    def warp(self, image):
        """Rectified board (800x800) in game orientation."""
        return bv.warp(image, self.bounderies, self.rotation)

    def board_bbox(self, margin=0):
        """(x0, y0, x1, y1) of the board in camera-image pixels."""
        pts = np.asarray(self.bounderies, dtype=np.float32)
        x0, y0 = pts.min(axis=0) - margin
        x1, y1 = pts.max(axis=0) + margin
        return float(x0), float(y0), float(x1), float(y1)

    def cell_center_in_image(self, row, col):
        H = bv.homography(self.bounderies, self.rotation)
        p = np.array([[[col * bv.CELL + bv.CELL / 2, row * bv.CELL + bv.CELL / 2]]], np.float32)
        q = cv2.perspectiveTransform(p, np.linalg.inv(H))
        return float(q[0, 0, 0]), float(q[0, 0, 1])

    def analyse(self, image, expected=None):
        """Full analysis of one camera frame.

        Returns dict(grid, probs, change, warped, norm).  `expected` (8x8 of
        0/1/2) may be given when the position is known (e.g. right after the
        robot moved) - it is used only to resolve squares the camera cannot
        decide on its own."""
        warped = self.warp(image)
        gray = bv.to_gray(warped)
        norm, dark, light = bv.normalise(gray, self.dark_parity)
        self.last_light = light
        cells, feats = bv.cell_features(warped, norm, self.dark_parity)
        grid = np.zeros((8, 8), dtype=int)
        probs = np.zeros((8, 8, 3), np.float32)
        probs[:, :, 0] = 1.0
        # (works uncalibrated too - universal defaults from recorded data)
        p = self.classifier.predict_proba(feats)
        change = None
        if self.ref_norm is not None:
            change, _ = bv.change_scores(self.ref_norm, norm)
        classes = bv.PieceClassifier.CLASSES
        for i, (r, c) in enumerate(cells):
            probs[r, c] = p[i]
            post = self._posterior(p[i], r, c, change, expected)
            k = int(np.argmax(post))
            if post[k] >= CONFIDENT:
                val = classes[k]
            elif post[0] < 1.0 - CONFIDENT:
                val = bv.OCCUPIED            # surely a piece, colour unsure
            else:
                val = bv.UNKNOWN
            grid[r, c] = val
        self.last_probs, self.last_change = probs, change
        return {"grid": grid, "probs": probs, "change": change,
                "warped": warped, "norm": norm, "cells": cells, "feats": feats}

    def _posterior(self, p_cls, r, c, change, expected):
        """Combine the classifier with what we already know about the square.

        Prior: the last confirmed value (ref_grid) is very likely if the
        square did not change since then, unlikely if it clearly changed.
        A known expected position (after a robot move) adds a second prior.
        The classifier decides whenever it is confident; the priors decide
        when it is not (black piece vs. empty dark square)."""
        p_cls = np.asarray(p_cls, np.float32)
        if float(p_cls.max()) >= TRUST_CLASSIFIER:
            return p_cls                 # camera is sure - don't second-guess it
        prior = np.full(3, 1.0 / 3, np.float32)
        classes = bv.PieceClassifier.CLASSES
        if change is not None and self.ref_grid is not None:
            ref = int(self.ref_grid[r][c])
            z = float(change[r, c])
            if z < Z_UNCHANGED:
                stay = 0.97
            elif z < Z_CHANGED:
                stay = 0.5
            else:
                stay = 0.2           # moved (or nudged) - classifier must decide
            j = classes.index(ref) if ref in classes else 0
            prior[:] = (1.0 - stay) / 2
            prior[j] = stay
        if expected is not None:
            e = int(expected[r][c])
            j = classes.index(e) if e in classes else 0
            boost = np.full(3, 0.15, np.float32)
            boost[j] = 0.7
            prior = prior * boost
        post = np.asarray(p_cls, np.float32) * prior
        tot = float(post.sum())
        return post / tot if tot > 0 else prior / prior.sum()

    def get_board(self, cameraImage, game=None, expected=None):
        """8x8 grid for the game (0 empty, 1 white, 2 black, 3 occupied of
        unknown colour, -1 unknown).  Draws the 'gameboard' debug window."""
        res = self.analyse(cameraImage, expected)
        self._show_gameboard(res)
        return res["grid"]

    def calibrate_from_start(self, image, rotation=None):
        """Learn orientation (unless given) and the piece classifier from a
        frame showing the start position.  Returns (ok, info)."""
        info = "given orientation"
        if rotation is None:
            rotation, _, score = bv.find_start_orientation(image, self.bounderies)
            if rotation is None:
                return False, score
            info = f"auto orientation (score {score:.2f})"
        self.rotation = int(rotation)
        warped = self.warp(image)
        gray = bv.to_gray(warped)
        self.dark_parity = bv.dark_parity(gray)
        norm, _, _ = bv.normalise(gray, self.dark_parity)
        cells, feats = bv.cell_features(warped, norm, self.dark_parity)
        _, labels = bv.start_position_labels(self.dark_parity)
        self.classifier = bv.PieceClassifier().fit(feats, labels)
        self.clear_reference()
        return True, info

    def set_reference(self, image, grid, learn=True):
        """Remember a confirmed position (start, accepted move, verified robot
        move): future frames are compared against it, and the classifier is
        refreshed with its now-known labels (follows lighting drift)."""
        warped = self.warp(image)
        gray = bv.to_gray(warped)
        norm, _, _ = bv.normalise(gray, self.dark_parity)
        self.ref_norm = norm
        self.ref_grid = np.array(grid, dtype=int)
        if learn:
            cells, feats = bv.cell_features(warped, norm, self.dark_parity)
            labels = np.array([int(grid[r][c]) for r, c in cells])
            self.classifier.update(feats, labels)

    def clear_reference(self):
        self.ref_norm = None
        self.ref_grid = None

    def _bordered_gray(self, image):
        return bv.to_gray(bv.warp_bordered(image, self.bounderies, self.rotation))

    def learn_piece_centres(self, image, grid):
        """Learn where a centred piece shows up from a position of centred pieces."""
        cells = [(r, c) for r in range(8) for c in range(8) if int(grid[r][c]) in (1, 2)]
        self.centre_model = bv.fit_centre_model(self._bordered_gray(image), cells)

    def piece_offset(self, image, row, col):
        """(rows, cols) the piece on (row, col) sits off its square's centre, or (0, 0)."""
        if self.centre_model is None or image is None:
            return 0.0, 0.0
        return bv.piece_offset(self._bordered_gray(image), row, col, self.centre_model)

    # ------------------------------------------------------------------
    # display
    # ------------------------------------------------------------------
    def _overlay(self, res, title=None):
        img = bv.to_bgr(res["warped"]).copy()
        grid, probs, change = res["grid"], res["probs"], res["change"]
        colors = {0: (0, 200, 0), 1: (255, 255, 0), 2: (255, 0, 255),
                  3: (0, 165, 255), -1: (0, 0, 255)}
        labels = {0: "", 1: "W", 2: "B", 3: "?", -1: "??"}
        for r in range(8):
            for c in range(8):
                if not bv.is_dark(r, c, self.dark_parity):
                    continue
                v = int(grid[r, c])
                x, y = c * bv.CELL, r * bv.CELL
                cv2.rectangle(img, (x + 3, y + 3), (x + bv.CELL - 3, y + bv.CELL - 3),
                              colors.get(v, (0, 0, 255)), 2)
                if labels.get(v):
                    cv2.putText(img, labels[v], (x + 8, y + 26), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, colors[v], 2)
                pe, pw, pb = probs[r, c]
                cv2.putText(img, f"{pe:.1f}/{pw:.1f}/{pb:.1f}", (x + 6, y + bv.CELL - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1)
                if change is not None and change[r, c] >= Z_UNCHANGED:
                    cv2.circle(img, (x + bv.CELL - 14, y + 14), 6, (0, 0, 255), -1)
        if title:
            cv2.rectangle(img, (0, 0), (bv.SIZE, 30), (0, 0, 0), -1)
            cv2.putText(img, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return img

    def _show_gameboard(self, res):
        g = res["grid"]
        title = (f"white={int((g == 1).sum())} black={int((g == 2).sum())} "
                 f"unsure={int(((g == 3) | (g == -1)).sum())}   "
                 "(red dot = changed since last confirmed position)")
        cv2.imshow("gameboard", self._overlay(res, title))

    def board_white_level(self, image):
        """Median grey of the board's light squares (border bands)."""
        g = bv.to_gray(bv.warp(image, self.bounderies, 0))
        lv = bv.border_levels(g)
        par = np.add.outer(np.arange(8), np.arange(8)) % 2
        a, b = float(np.median(lv[par == 0])), float(np.median(lv[par == 1]))
        return max(a, b)

    def auto_expose_on_board(self):
        cam = self.ximeaCamera
        if cam is None or not hasattr(cam, "auto_expose") or self.bounderies is None:
            return
        log("Adjusting exposure on the board's white squares ...")
        cam.auto_expose(target=190.0, measure=self.board_white_level)

    # ------------------------------------------------------------------
    # step 2: orientation + piece classifier from the start position
    # ------------------------------------------------------------------
    def _piece_calibration_window(self):
        log("\n" + "=" * 60)
        log("STEP 2: PIECE CALIBRATION")
        log("=" * 60)
        log("Put ALL pieces in the START position (robot's black pieces on its side).")
        log("The board orientation and the piece colours are learned from it.")
        log("  SPACE - accept   O - rotate orientation by 90 deg   E - redo exposure   ESC - skip")
        log("  AI_difficulty trackbar: 1 easy ... 5 hard")
        log("-" * 60 + "\n")

        self.auto_expose_on_board()
        cached_rot = self._load_cached_orientation()
        win = "Piece Calibration"
        cv2.namedWindow(win)
        cv2.createTrackbar("AI_difficulty", win, 3, 5, lambda x: None)
        manual_rot = None
        frame_i = 0
        rot, par, info = (cached_rot if cached_rot is not None else 0), 1, ""
        while True:
            image = self.ximeaCamera.get_camera_image()
            if image is None:
                continue
            if manual_rot is None and frame_i % 10 == 0:
                r_found, p_found, score = bv.find_start_orientation(image, self.bounderies)
                if r_found is not None:
                    rot, par, info = r_found, p_found, f"auto orientation (score {score:.2f})"
                elif cached_rot is not None:
                    rot, info = cached_rot, f"cached orientation ({score})"
                else:
                    info = str(score)
            frame_i += 1
            self.rotation = rot
            warped = self.warp(image)
            gray = bv.to_gray(warped)
            self.dark_parity = par = bv.dark_parity(gray)
            norm, dark, light = bv.normalise(gray, par)
            cells, feats = bv.cell_features(warped, norm, par)
            _, labels = bv.start_position_labels(par)
            clf = bv.PieceClassifier().fit(feats, labels)
            self.classifier = clf
            res = self.analyse(image)
            pred = res["grid"]
            ok = all(int(pred[r, c]) == int(lab) for (r, c), lab in zip(cells, labels))
            sep = self._class_separation(feats, labels)
            title = (f"{info} | parity {par} | board black={dark:.0f} white={light:.0f} | "
                     f"gap piece/empty {sep[0]:.2f}  white/black {sep[1]:.2f}")
            img = self._overlay(res, title)
            msg = "START POSITION OK - SPACE to accept" if ok and par == 1 else \
                "not a clean start position / orientation - fix pieces or press O"
            cv2.putText(img, msg, (8, bv.SIZE - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if ok else (0, 0, 255), 2)
            cv2.imshow(win, img)
            self.selected_difficulty = max(1, cv2.getTrackbarPos("AI_difficulty", win))
            key = cv2.waitKey(30) & 0xFF
            if key in (ord('e'), ord('E')):
                self.auto_expose_on_board()
            if key in (ord('o'), ord('O')):
                manual_rot = (rot + 1) % 4
                rot = manual_rot
                info = "manual orientation"
            elif key in (32, 13):
                log(f"✓ Piece calibration accepted (rotation={rot}, parity={par}, "
                    f"gap piece/empty={sep[0]:.2f}, white/black={sep[1]:.2f})")
                if sep[0] < 0.8:
                    log("  ⚠ black pieces are hard to tell from empty dark squares under this "
                        "lighting - moves are still tracked by change detection, but "
                        "consider more light / the desk lamp")
                self._save_orientation(rot, par)
                break
            elif key == 27:
                log("⚠ Piece calibration skipped - using default piece model "
                    f"and orientation {rot}")
                self.classifier = bv.PieceClassifier()
                break
        cv2.destroyWindow(win)

    @staticmethod
    def _class_separation(feats, labels):
        """(occupancy gap, colour gap) in log units learned from labels -
        roughly: > 1.0 is comfortable, < 0.6 is marginal."""
        return bv.PieceClassifier().fit(feats, labels).separation()

    def _load_cached_orientation(self):
        try:
            with open(CALIBRATION_PATH) as f:
                data = json.load(f)
            return int(data["rotation"]) if "rotation" in data else None
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _save_orientation(self, rotation, parity):
        try:
            data = {}
            if os.path.exists(CALIBRATION_PATH):
                with open(CALIBRATION_PATH) as f:
                    data = json.load(f)
            data["rotation"] = int(rotation)
            data["dark_parity"] = int(parity)
            os.makedirs(CALIBRATION_DIR, exist_ok=True)
            with open(CALIBRATION_PATH, "w") as f:
                json.dump(data, f, indent=2)
        except (OSError, ValueError) as e:
            log(f"  ⚠ Could not save orientation ({e})")

    def _get_grid_squares_contours(self):
        """8x8 list of [x, y, w, h] cells of the 800x800 rectified board."""
        return [[[c * bv.CELL, r * bv.CELL, bv.CELL, bv.CELL] for c in range(8)]
                for r in range(8)]

    def _trim_image_perspective(self, image, corners):
        """Rectify without the game-orientation rotation (corner check view)."""
        if corners is None or len(corners) != 4:
            return image
        return bv.warp(image, corners, 0)

    def _resolve_board_corners(self):
        """
        Get the board's 4 corners (BL, TL, TR, BR - see board_geometry.py),
        preferring a cached calibration from a previous run over redoing
        detection, since the camera is rigidly mounted and the corners
        rarely change between sessions. Falls back to automatic detection,
        then manual clicking, exactly like before if there is no usable
        cache.
        """
        image = self.ximeaCamera.get_camera_image()
        image_shape = image.shape if image is not None else None

        cached = self._load_cached_calibration(image_shape)
        if cached is not None:
            log("\n" + "="*60)
            log("USING CACHED BOARD CALIBRATION")
            log("="*60)
            log("  → Grid should already line up with the squares.")
            log("  → SPACE to accept, 'A' to re-run auto-detection, or 'R' to")
            log("    select the corners manually.")
            corners = self._verify_and_adjust_corners(cached)
            self._save_calibration(corners, image_shape)
            return corners

        log("\nAttempting automatic board detection...")
        auto_corners, auto_success_reason = self._auto_detect_corners()

        if auto_corners is not None:
            log("✓ Automatic detection successful!")
            log(f"  → {auto_success_reason}")
            corners = self._verify_and_adjust_corners(auto_corners)
        else:
            log("⚠ Automatic detection failed. Falling back to manual selection.")
            corners = self._get_trim_param_manual()

        self._save_calibration(corners, image_shape)
        return corners

    def _load_cached_calibration(self, image_shape):
        """Return cached corners (np.float32, shape (4,2)) if a valid cache
        exists for the given camera image shape, else None.
        """
        if not os.path.exists(CALIBRATION_PATH):
            return None
        try:
            with open(CALIBRATION_PATH, "r") as f:
                data = json.load(f)
            corners = np.array(data["corners"], dtype=np.float32)
            if corners.shape != (4, 2):
                return None
            cached_shape = data.get("image_shape")
            if image_shape is not None and cached_shape is not None:
                if list(cached_shape[:2]) != list(image_shape[:2]):
                    log(f"  ⚠ Cached calibration was for a "
                        f"{cached_shape[1]}x{cached_shape[0]} image, current "
                        f"camera is {image_shape[1]}x{image_shape[0]} - "
                        f"ignoring cache.")
                    return None
            log(f"  → Found cached calibration from "
                f"{data.get('saved_at', 'an unknown time')} "
                f"({CALIBRATION_PATH})")
            return corners
        except (OSError, ValueError, KeyError, TypeError) as e:
            log(f"  ⚠ Could not read cached calibration ({e}), ignoring it.")
            return None

    def _save_calibration(self, corners, image_shape=None):
        """Persist verified corners so future runs can skip detection."""
        if corners is None:
            return
        try:
            os.makedirs(CALIBRATION_DIR, exist_ok=True)
            data = {
                "corners": np.asarray(corners, dtype=float).tolist(),
                "corner_order": list(board_geometry.CORNER_NAMES),
                "image_shape": list(image_shape) if image_shape is not None else None,
                "saved_at": datetime.now().isoformat(timespec="seconds"),
            }
            with open(CALIBRATION_PATH, "w") as f:
                json.dump(data, f, indent=2)
            log(f"  → Calibration saved to {CALIBRATION_PATH}")
        except OSError as e:
            log(f"  ⚠ Could not save calibration ({e})")

    def _auto_detect_corners(self):
        """
        Automatically detect the outline of the 8x8 squares: detect the
        individual squares, fit a perspective lattice to them and refine on
        the checkerboard's internal grid junctions. See board_geometry.py
        for the algorithm and test/test_board_geometry.py for its tests.

        Returns: (corners, reason_message) or (None, failure_reason)
        """
        log("\n" + "="*60)
        log("AUTO-DETECTING BOARD CORNERS")
        log("="*60)

        image = self.ximeaCamera.get_camera_image()
        if image is None:
            log("  ✗ FAILED: Could not get camera image")
            return None, "Camera image unavailable"

        log(f"  → Image size: {image.shape[1]}x{image.shape[0]}")

        corners, message = board_geometry.find_board_corners(image)

        if corners is None:
            log(f"  ✗ FAILED: {message}")
            log("    → Falling back to manual selection...")
            return None, message

        log(f"  ✓ Found board corners ({message})")
        for name, (x, y) in zip(board_geometry.CORNER_NAMES, corners):
            log(f"    {name}: ({x:.0f}, {y:.0f})")

        return corners, message

    def _verify_and_adjust_corners(self, corners):
        """
        Show the warped board with grid overlay so user can verify if corners are correct.
        If not correct, allow manual adjustment by clicking new corners.
        
        Controls:
        - SPACE/ENTER: Accept current corners
        - 'A': Re-run automatic detection on the current camera image
        - 'R': Reset and select new corners manually
        - 1-4: Select corner to adjust, then click new position
        """
        log("\n" + "="*60)
        log("CORNER VERIFICATION")
        log("="*60)
        log("Check if the grid aligns with the board squares.")
        log("Controls:")
        log("  SPACE/ENTER - Accept corners if grid looks good")
        log("  A - Re-run automatic detection")
        log("  R - Reset and manually select new corners")
        log("  1/2/3/4 - Select corner to adjust (BL/TL/TR/BR)")
        log("  Click - Set new position for selected corner")
        log("-" * 60 + "\n")

        current_corners = corners.copy()
        selected_corner = None  # None, 0, 1, 2, 3 for BL, TL, TR, BR
        corner_names = list(board_geometry.CORNER_NAMES)
        
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
                cv2.putText(display_warped, "SPACE=Accept, A=Auto, R=Manual, 1-4=Adjust", (10, 780), 
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
            
            # Re-run automatic detection (e.g. after the camera was moved or
            # when the cached calibration is stale)
            if key == ord('a') or key == ord('A'):
                auto_corners, _ = self._auto_detect_corners()
                if auto_corners is not None:
                    current_corners = auto_corners.copy()
                    selected_corner = None

            # Reset to manual selection
            if key == ord('r') or key == ord('R'):
                log("  → Switching to manual corner selection...")
                cv2.destroyWindow("Warped Board (Grid Check)")
                cv2.destroyWindow("Original with Corners")
                return self._get_trim_param_manual()
            
            # Select corner to adjust
            if key == ord('1'):
                selected_corner = 0
                log(f"  → Selected corner 1 ({corner_names[0]}) - click to move")
            elif key == ord('2'):
                selected_corner = 1
                log(f"  → Selected corner 2 ({corner_names[1]}) - click to move")
            elif key == ord('3'):
                selected_corner = 2
                log(f"  → Selected corner 3 ({corner_names[2]}) - click to move")
            elif key == ord('4'):
                selected_corner = 3
                log(f"  → Selected corner 4 ({corner_names[3]}) - click to move")
            
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
        log("Exposure is set automatically (E = redo, +/- = brighter/darker).")
        log("Press SPACE to continue when ready.")
        log("-" * 60 + "\n")

        cam = self.ximeaCamera
        if hasattr(cam, "auto_expose"):
            cam.auto_expose()
        while True:
            image = cam.get_camera_image()
            display = image.copy()
            cv2.putText(display, "Adjust Camera. Press SPACE to continue", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if hasattr(cam, "exposure_us"):
                cv2.putText(display, f"exposure {cam.exposure_us / 1000:.1f} ms  (E auto, +/-)",
                            (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Camera Adjustment", display)

            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # SPACE
                cv2.destroyWindow("Camera Adjustment")
                break
            if hasattr(cam, "auto_expose"):
                if key in (ord('e'), ord('E')):
                    cam.auto_expose()
                elif key in (ord('+'), ord('=')):
                    cam.set_exposure(cam.exposure_us * 1.25)
                elif key in (ord('-'), ord('_')):
                    cam.set_exposure(cam.exposure_us / 1.25)

    def _get_trim_param_manual(self):
        """
        Manual board corner selection - click 4 corners in order, on the
        inner edge of the frame (right where the squares start), not the
        outer edge of the board:
        1. Bottom-Left
        2. Top-Left
        3. Top-Right
        4. Bottom-Right
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
        print("Click on the 4 corners of the board, on the inner edge of the")
        print("frame (right where the squares start), in this order:")
        print("  1. Bottom-Left")
        print("  2. Top-Left")
        print("  3. Top-Right")
        print("  4. Bottom-Right")
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

