#!/usr/bin/env python3
"""Draws the launcher icons (chess.png, checkers.png, stop.png, 256 px)."""
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
S = 256
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def base(colour):
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((8, 8, S - 8, S - 8), radius=48, fill=colour)
    return img, d


def chess():
    img, d = base((46, 94, 140, 255))
    font = ImageFont.truetype(FONT, 200)
    d.text((S // 2, S // 2 - 4), "\u265e", font=font, anchor="mm", fill=(250, 250, 250, 255))
    return img


def checkers():
    img, d = base((150, 52, 44, 255))
    for x, y, fill in ((92, 150, (30, 30, 30)), (164, 150, (245, 240, 230)),
                                      (128, 100, (30, 30, 30))):
        for k in range(3):                   # a stack of discs
            yy = y - k * 10
            d.ellipse((x - 46, yy - 22, x + 46, yy + 22), fill=fill,
                      outline=(120, 120, 120) if fill[0] > 128 else (90, 90, 90), width=3)
    return img


def stop():
    img, d = base((192, 57, 43, 255))
    d.rounded_rectangle((78, 78, S - 78, S - 78), radius=14, fill=(255, 255, 255, 255))
    return img


if __name__ == "__main__":
    for name, fn in (("chess", chess), ("checkers", checkers), ("stop", stop)):
        fn().save(os.path.join(HERE, f"{name}.png"))
        print(f"{name}.png")
