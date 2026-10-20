"""Tiny renderer of a checkers board seen by a tilted camera - used to test
the vision + rules pipeline end-to-end without hardware.

It is deliberately pessimistic about the black pieces: they are only a few
grey levels brighter than the dark squares (with a faint rim), like plain
black checkers on a black square.
"""

import cv2
import numpy as np

SIZE = 800
CELL = 100


def render_topdown(grid, parity=1, black_piece_level=38, dark_level=22,
                   light_level=205, white_piece_level=215, rng=None, centres=None, feet=None):
    """Top-down board; centres / feet = {(r, c): (dx, dy)} piece top / foot offsets in px."""
    rng = rng or np.random.RandomState(0)
    img = np.zeros((SIZE, SIZE, 3), np.uint8)
    for r in range(8):
        for c in range(8):
            v = dark_level if (r + c) % 2 == parity else light_level
            img[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL] = (v, v, v + 3)
    for r in range(8):
        for c in range(8):
            d = int(grid[r][c])
            if d == 0:
                continue
            if centres is not None:
                cx = c * CELL + CELL // 2 + centres[(r, c)][0]
                cy = r * CELL + CELL // 2 + centres[(r, c)][1]
            else:
                # tall pieces seen at an angle: top face shifted a bit
                cx = c * CELL + CELL // 2 + int(rng.randint(-6, 7))
                cy = r * CELL + CELL // 2 - 8 + int(rng.randint(-4, 5))
            wall = (150, 150, 145) if d == 1 else (black_piece_level - 8,) * 3
            if feet is None:
                cv2.circle(img, (cx, cy + 6), 31, wall, -1)                  # side wall
            else:
                # side wall: the silhouette from the piece's foot up to its top
                fx = c * CELL + CELL // 2 + feet[(r, c)][0]
                fy = r * CELL + CELL // 2 + feet[(r, c)][1]
                for t in np.linspace(0.0, 1.0, 8):
                    pt = (int(round(fx + t * (cx - fx))), int(round(fy + t * (cy - fy))))
                    cv2.circle(img, pt, 31, wall, -1)
            if d == 1:
                cv2.circle(img, (cx, cy), 30, (white_piece_level,) * 3, -1)
                for rr in (22, 14, 6):
                    cv2.circle(img, (cx, cy), rr, (white_piece_level - 35,) * 3, 1)
            else:
                b = black_piece_level
                cv2.circle(img, (cx, cy), 30, (b,) * 3, -1)
                cv2.ellipse(img, (cx, cy), (27, 27), 0, 200, 320, (b + 25,) * 3, 2)  # sheen
                cv2.circle(img, (cx, cy), 18, (b - 6,) * 3, 1)
    return img


def random_corners(rng, shape=(578, 585)):
    h, w = shape
    base = np.array([[40, h - 30], [25, 20], [w - 60, 10], [w - 30, h - 80]], np.float32)
    return base + rng.uniform(-15, 15, base.shape).astype(np.float32)


def to_camera(topdown, corners, shape=(578, 585), gain=1.0, gradient=0.0,
              noise=3.0, blur=1.0, rotate_k=0, rng=None, frame_margin=True):
    """Project the top-down board into a camera image.  corners are the image
    positions of BL, TL, TR, BR of the *rotated* board."""
    rng = rng or np.random.RandomState(0)
    h, w = shape
    td = np.rot90(topdown, rotate_k).copy()
    src = np.array([[0, SIZE], [0, 0], [SIZE, 0], [SIZE, SIZE]], np.float32)
    if frame_margin:
        # white border + dark frame around the squares like the real board
        pad = 60
        big = np.full((SIZE + 2 * pad, SIZE + 2 * pad, 3), 225, np.uint8)
        cv2.rectangle(big, (0, 0), (SIZE + 2 * pad - 1, SIZE + 2 * pad - 1), (15, 15, 15), 18)
        big[pad:pad + SIZE, pad:pad + SIZE] = td
        td = big
        src = src + pad
    H = cv2.getPerspectiveTransform(src, np.asarray(corners, np.float32))
    bg = np.full((h, w, 3), 70, np.uint8)
    img = cv2.warpPerspective(td, H, (w, h), dst=bg, borderMode=cv2.BORDER_TRANSPARENT)
    img = img.astype(np.float32) * gain
    if gradient:
        xx = np.linspace(-1, 1, w)[None, :, None]
        img = img * (1.0 + gradient * xx)
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    img += rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)
