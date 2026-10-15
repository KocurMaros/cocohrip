"""Small value object kept for backwards compatibility (old code imported
checkers.piece.Piece).  Game logic now lives in checkers.rules."""

from dataclasses import dataclass


@dataclass
class Piece:
    row: int
    col: int
    color: tuple          # constants.WHITE / constants.BLACK (rgb)
    king: bool = False

    def move(self, row, col):
        self.row, self.col = row, col

    def make_king(self):
        self.king = True

    def equals(self, other):
        return (self.row, self.col, self.color, self.king) == \
            (other.row, other.col, other.color, other.king)
