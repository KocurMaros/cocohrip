"""Pygame view of a rules.GameState (drawing only - no game logic here)."""

import pygame

from checkers_game.checkers.rules import is_king, owner, WHITE as R_WHITE
from checkers_game.constants import (BLACK, BLUE, BROWN, COLS, GREEN, GREY,
                                     ORANGE, RED, ROWS, SQUARE_SIZE, WHITE,
                                     get_crown)

PADDING = 15
OUTLINE = 2


class Board:
    def __init__(self, dark_parity=1):
        self.dark_parity = dark_parity

    def draw_squares(self, win):
        win.fill(BROWN)
        for row in range(ROWS):
            for col in range(COLS):
                if (row + col) % 2 != self.dark_parity:
                    pygame.draw.rect(win, GREY, (col * SQUARE_SIZE, row * SQUARE_SIZE,
                                                 SQUARE_SIZE, SQUARE_SIZE))

    @staticmethod
    def _center(row, col):
        return (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2)

    def draw_piece(self, win, row, col, value):
        radius = SQUARE_SIZE // 2 - PADDING
        x, y = self._center(row, col)
        fill = WHITE if owner(value) == R_WHITE else BLACK
        edge = BLACK if fill == WHITE else WHITE
        pygame.draw.circle(win, edge, (x, y), radius + OUTLINE)
        pygame.draw.circle(win, fill, (x, y), radius)
        if is_king(value):
            crown = get_crown()
            if crown is not None:
                win.blit(crown, (x - crown.get_width() // 2, y - crown.get_height() // 2))
            else:
                pygame.draw.circle(win, ORANGE, (x, y), radius // 3)

    def draw(self, win, state, last_move=None, highlight=(), detected=None):
        """Draw `state`.  last_move: rules.Move to show with a blue line;
        highlight: squares to outline red (e.g. pieces to huff / mismatches);
        detected: optional 8x8 grid from the camera, drawn as small dots so
        detection problems are visible at a glance."""
        self.draw_squares(win)
        if state is None:
            return
        if last_move is not None:
            pts = [self._center(r, c) for r, c in last_move.path]
            if len(pts) > 1:
                pygame.draw.lines(win, BLUE, False, pts, 4)
        for r in range(ROWS):
            for c in range(COLS):
                v = state.at(r, c)
                if v:
                    self.draw_piece(win, r, c, v)
        for (r, c) in highlight:
            pygame.draw.rect(win, RED, (c * SQUARE_SIZE, r * SQUARE_SIZE,
                                        SQUARE_SIZE, SQUARE_SIZE), 4)
        if detected is not None:
            for r in range(ROWS):
                for c in range(COLS):
                    d = int(detected[r][c])
                    if d == 0:
                        continue
                    x, y = self._center(r, c)
                    col = GREEN
                    exp = state.at(r, c)
                    if d in (3, -1):            # occupied / unknown colour
                        col = ORANGE
                    elif exp == 0 or (exp > 0) != (d == 1):
                        col = RED
                    pygame.draw.circle(win, col, (x + SQUARE_SIZE // 2 - 12,
                                                  y - SQUARE_SIZE // 2 + 12), 6)
