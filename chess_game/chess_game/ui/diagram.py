"""Board diagram: printed a1 at the bottom left - the board as the human
sees it from the rank-1 side.  Pieces are drawn as chess-piece figures
(the chess glyphs of a system font, rendered with Pillow); without Pillow
or such a font they fall back to lettered discs."""

import functools
import os

import cv2
import numpy as np

FONTS = ("/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/noto/NotoSansSymbols2-Regular.ttf")
OUTLINE = {"K": "\u2654", "Q": "\u2655", "R": "\u2656", "B": "\u2657", "N": "\u2658", "P": "\u2659"}
SOLID = {k: chr(ord(v) + 6) for k, v in OUTLINE.items()}          # U+265A..265F
WHITE_FILL, WHITE_LINE = (238, 242, 246), (25, 25, 25)            # BGR
BLACK_FILL, BLACK_LINE = (32, 32, 38), (175, 175, 175)


@functools.lru_cache(maxsize=64)
def piece_sprite(symbol, size):
    """(BGR, alpha 0..1) image of a piece, size x size, or None."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None
    path = next((f for f in FONTS if os.path.exists(f)), None)
    if path is None:
        return None
    font = ImageFont.truetype(path, int(size * 0.92))
    white = symbol.isupper()
    fill, line = (WHITE_FILL, WHITE_LINE) if white else (BLACK_FILL, BLACK_LINE)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    key = symbol.upper()
    x0, y0, x1, y1 = draw.textbbox((0, 0), SOLID[key], font=font)
    org = ((size - (x1 - x0)) / 2 - x0, (size - (y1 - y0)) / 2 - y0)
    if white:   # white body, then the outline glyph's lines on top
        draw.text(org, SOLID[key], font=font, fill=fill[::-1] + (255,))
        draw.text(org, OUTLINE[key], font=font, fill=line[::-1] + (255,))
    else:       # the solid glyph has its own detail cut-outs; a light rim
        draw.text(org, SOLID[key], font=font, fill=fill[::-1] + (255,),
                  stroke_width=max(1, size // 45), stroke_fill=line[::-1] + (255,))
    rgba = np.asarray(img)
    return np.ascontiguousarray(rgba[:, :, 2::-1]), rgba[:, :, 3:4].astype(np.float32) / 255.0


def _paste(img, sprite, x0, y0):
    bgr, alpha = sprite
    h, w = alpha.shape[:2]
    roi = img[y0:y0 + h, x0:x0 + w]
    h, w = roi.shape[:2]
    roi[:] = (roi * (1 - alpha[:h, :w]) + bgr[:h, :w] * alpha[:h, :w]).astype(np.uint8)

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


def draw_diagram(placement, size=560, sources=(), targets=(), marks=(), title=None, promoted=()):
    """placement: {printed square: FEN symbol}.  sources / targets get a red /
    green frame, marks (e.g. squares the camera disagrees on) a yellow dot,
    promoted pieces (a pawn on the board) a small orange "P" badge."""
    margin = 24
    img = np.full((size, size, 3), 50, np.uint8)
    s = (size - 2 * margin) / 8.0
    for sq in range(64):
        x0, y0, x1, y1 = square_rect(sq, size, margin)
        dark = (sq % 8 + sq // 8) % 2 == 0
        cv2.rectangle(img, (x0, y0), (x1, y1), DARK if dark else LIGHT, -1)
        sym = placement.get(sq)
        sprite = piece_sprite(sym, int(x1 - x0)) if sym else None
        if sprite is not None:
            _paste(img, sprite, x0, y0)
        elif sym:
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
    for sq in promoted:
        x0, y0, x1, y1 = square_rect(sq, size, margin)
        cv2.rectangle(img, (x0 + 3, y1 - 19), (x0 + 19, y1 - 3), (0, 120, 255), -1)
        cv2.putText(img, "P", (x0 + 6, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
                    cv2.LINE_AA)
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
