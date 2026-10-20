"""Single game window ("Checkers") composed of tiles:

    +-----------------------------------------------------------------+
    | banner: whose turn / what to do / errors (colour coded)          |
    +----------------------------------+----------------+-------------+
    |                                  | CAMERA + HANDS | DETECTION   |
    |          GAME BOARD              +----------------+-------------+
    |   (authoritative game state,     | status, stability, exposure, |
    |    last move, problem squares,   | messages, move log, keys      |
    |    camera dots per square)       |                               |
    +----------------------------------+-------------------------------+

Pure OpenCV/numpy.  The window is resizable (WINDOW_NORMAL, keeps the
aspect ratio); keys pressed while it is focused reach cv2.waitKey().
"""

import cv2
import numpy as np

WINDOW = "Checkers"
W, H = 1320, 780
BANNER_H = 56
PAD = 10
BOARD = H - BANNER_H - 2 * PAD           # 704
RIGHT_X = PAD + BOARD + PAD               # 724
RIGHT_W = W - RIGHT_X - PAD               # 586
TILE = (RIGHT_W - PAD) // 2               # 288
CAPTION_H = 22

BG = (32, 32, 32)
PANEL = (48, 48, 48)
TEXT = (235, 235, 235)
DIM = (165, 165, 165)

# banner colours (BGR)
COLORS = {
    "info": (90, 90, 90),
    "your_turn": (60, 140, 40),
    "robot": (150, 90, 20),
    "error": (40, 40, 190),
    "warn": (0, 140, 220),
    "over": (0, 160, 200),
}

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _fit(img, w, h):
    """Resize keeping aspect ratio, letterboxed onto a (h, w) canvas."""
    canvas = np.full((h, w, 3), 18, np.uint8)
    if img is None:
        return canvas
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    y, x = (h - nh) // 2, (w - nw) // 2
    canvas[y:y + nh, x:x + nw] = small
    return canvas


def _text(img, txt, org, scale=0.5, color=TEXT, thick=1):
    cv2.putText(img, txt, org, FONT, scale, color, thick, cv2.LINE_AA)


def _wrap(txt, max_chars):
    words, lines, cur = txt.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > max_chars and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        lines.append(cur)
    return lines or [""]


class Dashboard:
    def __init__(self, show=True):
        self.show = show
        self.canvas = np.full((H, W, 3), BG, np.uint8)
        self._created = False

    def _ensure_window(self):
        if self._created or not self.show:
            return
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(WINDOW, W, H)
        self._created = True

    def is_open(self):
        """False once the user closed the window."""
        if not self._created:
            return True
        try:
            return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False

    def _tile(self, img, x, y, w, h, caption):
        c = self.canvas
        c[y:y + CAPTION_H, x:x + w] = PANEL
        _text(c, caption, (x + 6, y + 16), 0.48, DIM)
        c[y + CAPTION_H:y + h, x:x + w] = _fit(img, w, h - CAPTION_H)

    def render(self, board_img, camera_img, detection_img, status):
        """status: dict with
             banner      (text, kind)   kind in COLORS
             details     [str]          lines under the banner topic
             facts       [(label, value)]
             stability   0..1           how far the board is from 'settled'
             log         [str]          moves so far
             keys        str            key help line
        """
        c = self.canvas
        c[:] = BG
        text, kind = status.get("banner", ("", "info"))
        c[:BANNER_H] = COLORS.get(kind, COLORS["info"])
        _text(c, text, (PAD + 4, 38), 0.95, (255, 255, 255), 2)

        # game board (square)
        c[BANNER_H + PAD:BANNER_H + PAD + BOARD, PAD:PAD + BOARD] = _fit(board_img, BOARD, BOARD)

        # camera + detection tiles
        y0 = BANNER_H + PAD
        self._tile(camera_img, RIGHT_X, y0, TILE, TILE + CAPTION_H, "CAMERA + HAND TRACKING")
        self._tile(detection_img, RIGHT_X + TILE + PAD, y0, TILE, TILE + CAPTION_H,
                   "DETECTION (rectified)")

        # info panel
        py = y0 + TILE + CAPTION_H + PAD
        ph = H - PAD - py
        c[py:py + ph, RIGHT_X:RIGHT_X + RIGHT_W] = PANEL
        x = RIGHT_X + 10
        y = py + 22
        max_chars = 62
        for line in status.get("details", [])[:6]:
            for sub in _wrap(line, max_chars):
                _text(c, sub, (x, y), 0.5, TEXT)
                y += 19
        y += 6
        facts = status.get("facts", [])
        for i, (label, value) in enumerate(facts):
            fx = x + (i % 2) * (RIGHT_W // 2)
            _text(c, f"{label}:", (fx, y), 0.45, DIM)
            _text(c, str(value), (fx + 92, y), 0.45, TEXT)
            if i % 2 == 1 or i == len(facts) - 1:
                y += 18
        stab = float(np.clip(status.get("stability", 0.0), 0.0, 1.0))
        _text(c, "board still:", (x, y + 12), 0.45, DIM)
        bx, bw = x + 92, RIGHT_W - 120
        cv2.rectangle(c, (bx, y + 2), (bx + bw, y + 14), (90, 90, 90), 1)
        cv2.rectangle(c, (bx + 1, y + 3), (bx + 1 + int((bw - 2) * stab), y + 13),
                      (60, 180, 60) if stab >= 1.0 else (0, 165, 255), -1)
        y += 32
        log = status.get("log", [])
        _text(c, "MOVES", (x, y), 0.45, DIM)
        y += 18
        bottom = py + ph - 26
        rows = max(0, (bottom - y) // 17)
        col_w = RIGHT_W // 2
        shown = log[-2 * rows:] if rows else []
        first = len(log) - len(shown)
        for i, m in enumerate(shown):
            cx = x + (i // rows) * col_w if rows else x
            cy = y + (i % rows) * 17 if rows else y
            _text(c, f"{first + i + 1:2d}. {m}"[:34], (cx, cy), 0.42, TEXT)
        _text(c, status.get("keys", ""), (x, py + ph - 9), 0.42, DIM)

        if self.show:
            self._ensure_window()
            cv2.imshow(WINDOW, c)
        return c
