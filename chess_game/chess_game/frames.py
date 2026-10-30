"""Square and coordinate conventions shared by the rules, vision and robot.

Three ways to name a square:

* **printed** square - the coordinates printed on the physical board
  (python-chess numbering, a1 = 0 ... h8 = 63).  The camera, the dataset
  labels and the board calibration all use printed squares.
* **logical** square - the square the rules engine sees.  The human sits on
  the printed rank-1 side.  Playing white, the pieces are set up normally
  and logical == printed.  Playing black, black is set up on printed ranks
  1-2, so logical = printed rotated by 180 degrees (logical a1 = printed h8).
* **game frame** (row, col) of robot_control (same as checkers): row 0 is
  the robot's side (printed rank 8), row 7 the human's side (printed rank
  1), col 0 the human's left (printed a-file).  robot_control turns these
  into robot XY from the two measured corner squares; off-board places
  (storage) are just fractional / out-of-range rows and cols.
"""

import chess


def printed_to_logical(printed_sq, human_white):
    return printed_sq if human_white else 63 - printed_sq


def logical_to_printed(logical_sq, human_white):
    # rotating by 180 degrees is its own inverse
    return printed_to_logical(logical_sq, human_white)


def printed_to_game(printed_sq):
    """Printed square -> (row, col) in the robot's game frame."""
    return 7 - chess.square_rank(printed_sq), chess.square_file(printed_sq)


def game_to_printed(row, col):
    return chess.square(int(col), 7 - int(row))


def logical_to_game(logical_sq, human_white):
    return printed_to_game(logical_to_printed(logical_sq, human_white))


class OffBoardLayout:
    """Storage slots for the human's pieces the robot captures, in game-frame
    squares: one column of ``storage_rows`` slots beside the human's left
    edge (col ``left_col``), then one beside the right edge (``right_col``) -
    the same places as in checkers, 16 slots for at most 15 pieces.  Each
    column fills from the robot's side (row 0)."""

    def __init__(self, left_col=-1.8, right_col=8.8, storage_rows=8):
        self.cols = (float(left_col), float(right_col))
        self.storage_rows = int(storage_rows)

    @property
    def storage_slots(self):
        return self.storage_rows * len(self.cols)

    def storage(self, k):
        if not 0 <= k < self.storage_slots:
            raise IndexError(f"storage slot {k} out of range (0..{self.storage_slots - 1})")
        return float(k % self.storage_rows), self.cols[k // self.storage_rows]
