"""Finding the board in the camera image at startup: automatic detection,
the cached corners of the last run, or four clicks."""

import json
import os

import cv2
import numpy as np

from chess_game.vision.camera import display_point, raw_point, to_display

try:
    from checkers_game.camera.board_geometry import find_board_corners, order_corners
except ImportError:  # source tree without the workspace sourced
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                                    "checkers_game"))
    from checkers_game.camera.board_geometry import find_board_corners, order_corners

CACHE = os.path.expanduser(os.environ.get("CHESS_BOARD_CACHE", "~/.cocohrip/chess_board.json"))
CORNER_ORDER = ("BL", "TL", "TR", "BR")


def load_cached(shape, path=CACHE):
    """(corners, a1_corner) saved for an image of this shape, or None."""
    try:
        with open(path) as f:
            data = json.load(f)
        if tuple(data["shape"]) != tuple(shape[:2]):
            return None
        return np.asarray(data["corners"], np.float32), data.get("a1_corner", "BR")
    except (OSError, ValueError, KeyError):
        return None


def save_cached(corners, a1_corner, shape, path=CACHE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"corners": np.asarray(corners).round(2).tolist(), "a1_corner": a1_corner,
                   "shape": list(shape[:2])}, f)


def locate(work, a1_corner="BR", max_shift=25.0, path=CACHE):
    """Detect the board; fall back to the cache.  Returns (corners,
    a1_corner, how) or (None, a1_corner, reason)."""
    cached = load_cached(work.shape, path)
    corners, msg = find_board_corners(work)
    if corners is not None:
        if cached is not None and np.abs(corners - cached[0]).max() > max_shift:
            msg += f" - moved more than {max_shift:.0f} px since the last run"
        a1 = cached[1] if cached is not None else a1_corner
        save_cached(corners, a1, work.shape, path)
        return corners, a1, "detected: " + msg
    if cached is not None:
        return cached[0], cached[1], "cached corners (" + msg + ")"
    return None, a1_corner, msg


def click_corners(window, work, height=900):
    """Let the user click the outer corners of printed a1, a8, h8, h1 in the
    (display-rotated) image.  Returns (corners BL TL TR BR, a1_corner) or
    None on ESC."""
    names = ["a1", "a8", "h8", "h1"]
    clicks = []
    scale = height / float(to_display(work).shape[0])

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append(raw_point((x / scale, y / scale), work.shape))
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    while len(clicks) < 4:
        disp = cv2.resize(to_display(work), None, fx=scale, fy=scale)
        for p in clicks:
            d = np.array(display_point(p, work.shape)) * scale
            cv2.circle(disp, tuple(int(v) for v in d), 6, (0, 0, 255), -1)
        cv2.putText(disp, f"Click the outer corner of the squares at {names[len(clicks)]}"
                    "  (ESC = cancel)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow(window, disp)
        if cv2.waitKey(20) & 0xFF == 27:
            cv2.setMouseCallback(window, lambda *a: None)
            return None
    cv2.setMouseCallback(window, lambda *a: None)
    ordered = order_corners(clicks)
    a1 = np.asarray(clicks[0], np.float32)
    a1_corner = CORNER_ORDER[int(np.argmin(np.linalg.norm(ordered - a1, axis=1)))]
    return ordered, a1_corner
