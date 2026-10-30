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
  (storage, reserve) are just fractional / out-of-range rows and cols.
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
    """Where pieces go off the board, in game-frame squares.

    storage: captured human pieces (and the robot's promoted pawns) - a grid
    of ``storage_rows`` x ``storage_cols`` slots beside the human's left
    edge (negative cols), filled column by column from the robot's side.
    reserve: one slot per piece symbol for promotions (the human puts e.g. a
    spare queen there); beside the human's right edge.
    """

    def __init__(self, storage_first_col=-1.4, storage_col_step=-1.1, storage_rows=8,
                 storage_cols=2, reserve_col=8.4, reserve_first_row=1.0, reserve_row_step=1.1):
        self.storage_first_col = float(storage_first_col)
        self.storage_col_step = float(storage_col_step)
        self.storage_rows = int(storage_rows)
        self.storage_cols = int(storage_cols)
        self.reserve_col = float(reserve_col)
        self.reserve_first_row = float(reserve_first_row)
        self.reserve_row_step = float(reserve_row_step)

    @property
    def storage_slots(self):
        return self.storage_rows * self.storage_cols

    def storage(self, k):
        if not 0 <= k < self.storage_slots:
            raise IndexError(f"storage slot {k} out of range (0..{self.storage_slots - 1})")
        return (float(k % self.storage_rows),
                self.storage_first_col + (k // self.storage_rows) * self.storage_col_step)

    def reserve(self, symbol):
        """Reserve slot of a piece symbol: Q/q first, then R, B, N."""
        order = "QRBN"
        i = order.index(symbol.upper())
        return self.reserve_first_row + i * self.reserve_row_step, self.reserve_col
