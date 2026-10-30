"""Board diagram drawn with OpenCV (no fonts needed): printed a1 at the
bottom left - the board as the human sees it from the rank-1 side."""

import cv2
import numpy as np

LIGHT = (196, 222, 232)
DARK = (92, 120, 150)
FROM_COLOUR = (40, 40, 230)
TO_COLOUR = (40, 190, 40)
MARK_COLOUR = (0, 200, 255)


def square_rect(sq, size, margin):
    s = (size - 2 * margin) / 8.0
    col, row = sq % 8, 7 - sq // 8
    x0, y0 = margin + col * s, margin + row * s
    return int(round(x0)), int(round(y0)), int(round(x0 + s)), int(round(y0 + s))


def draw_diagram(placement, size=560, sources=(), targets=(), marks=(), title=None):
    """placement: {printed square: FEN symbol}.  sources / targets get a red /
    green frame, marks (e.g. squares the camera disagrees on) a yellow dot."""
    margin = 24
    img = np.full((size, size, 3), 50, np.uint8)
    s = (size - 2 * margin) / 8.0
    for sq in range(64):
        x0, y0, x1, y1 = square_rect(sq, size, margin)
        dark = (sq % 8 + sq // 8) % 2 == 0
        cv2.rectangle(img, (x0, y0), (x1, y1), DARK if dark else LIGHT, -1)
        sym = placement.get(sq)
        if sym:
            cx, cy, r = (x0 + x1) // 2, (y0 + y1) // 2, int(s * 0.38)
            white = sym.isupper()
            cv2.circle(img, (cx, cy), r, (225, 240, 245) if white else (30, 35, 45), -1, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), r, (20, 20, 20) if white else (200, 200, 200), 1, cv2.LINE_AA)
            letter = sym.upper()
            (tw, th), _ = cv2.getTextSize(letter, cv2.FONT_HERSHEY_DUPLEX, s / 50.0, 2)
            cv2.putText(img, letter, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_DUPLEX,
                        s / 50.0, (20, 20, 20) if white else (235, 235, 235), 2, cv2.LINE_AA)
    for squares, colour in ((sources, FROM_COLOUR), (targets, TO_COLOUR)):
        for sq in squares:
            x0, y0, x1, y1 = square_rect(sq, size, margin)
            cv2.rectangle(img, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2), colour, 4)
    for sq in marks:
        x0, y0, x1, y1 = square_rect(sq, size, margin)
        cv2.circle(img, (x1 - 10, y0 + 10), 6, MARK_COLOUR, -1, cv2.LINE_AA)
    for i in range(8):
        x0, _, x1, _ = square_rect(i, size, margin)
        cv2.putText(img, "abcdefgh"[i], ((x0 + x1) // 2 - 5, size - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (220, 220, 220), 1, cv2.LINE_AA)
        _, y0, _, y1 = square_rect(i * 8, size, margin)
        cv2.putText(img, str(i + 1), (6, (y0 + y1) // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (220, 220, 220), 1, cv2.LINE_AA)
    if title:
        cv2.putText(img, title, (margin, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1,
                    cv2.LINE_AA)
    return img
