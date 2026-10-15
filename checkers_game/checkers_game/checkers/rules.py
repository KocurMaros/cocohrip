"""English draughts / American checkers rules engine (WCDF rules).

Pure python, no pygame / ROS / OpenCV, so it can be unit-tested anywhere.

Rules implemented (World Checkers/Draughts Federation, "English draughts"):

* 8x8 board, pieces only ever stand on one colour of squares (the engine does
  not care which - diagonal moves keep the square parity).
* Men move one square diagonally *forward* onto an empty square.
  WHITE (the human) moves towards row 0, BLACK (the robot) towards row 7.
* Kings move one square diagonally in any direction (no "flying" kings).
* Capturing: jump diagonally over an adjacent opponent piece onto the empty
  square right behind it.  Men capture forward only, kings in all directions.
* Capturing is MANDATORY.  If several captures are possible the player may
  choose any of them (there is no "capture the most pieces" rule).
* A multi-jump must be completed: after a jump the same piece must keep
  jumping as long as it can.  A piece cannot be jumped twice in one move and
  jumped pieces stay on the board (as blockers) until the move is finished.
* A man that reaches the far row is crowned and the move ENDS there, even if
  as a king it could continue jumping.
* A player with no pieces or no legal move loses.
* Draw: 40 moves per side (80 plies) without a capture or a man move, or the
  same position (with the same side to move) occurring 3 times (see Game).
* BLACK moves first (WCDF rule 1.3); the node can override this.

On top of that the "huff" rule (traditional, informal): if a player ignores a
mandatory capture (plays a non-capturing move although a capture was
available, or stops a multi-jump early), the move stands but the opponent
removes ("huffs") one of the pieces that should have captured.  Use
:meth:`GameState.classify_move` / :func:`huff_candidates` for that.

Squares are ``(row, col)`` tuples, row 0 is the top of the pygame window
(BLACK's home row), row 7 is WHITE's home row.
"""

from dataclasses import dataclass, field

ROWS = COLS = 8

EMPTY = 0
WHITE = 1        # human, men move to row 0
BLACK = -1       # robot / AI, men move to row 7

MAN = 1
KING = 2

DRAW = "draw"

# 80 plies = 40 moves per side without a capture or a man moving
QUIET_PLY_LIMIT = 80


def color_name(color):
    return "WHITE" if color == WHITE else "BLACK"


def owner(value):
    if value == 0:
        return 0
    return WHITE if value > 0 else BLACK


def is_king(value):
    return abs(value) == KING


def square_name(sq):
    """Human readable square name: 'r5c2 (#42)'."""
    r, c = sq
    return f"r{r}c{c}(#{r * 8 + c})"


def _idx(r, c):
    return r * COLS + c


def _on_board(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS


def _forward(color):
    return -1 if color == WHITE else 1


def _crown_row(color):
    return 0 if color == WHITE else ROWS - 1


@dataclass(frozen=True)
class Move:
    """A complete (or, for huff analysis, partial) move.

    path     -- squares visited, path[0] = start, path[-1] = end
    captured -- squares of the jumped pieces, in jump order
    promotes -- the moving man is crowned at the end of the move
    """
    path: tuple
    captured: tuple = ()
    promotes: bool = False

    @property
    def start(self):
        return self.path[0]

    @property
    def end(self):
        return self.path[-1]

    @property
    def is_capture(self):
        return bool(self.captured)

    def __str__(self):
        sep = "x" if self.captured else "-"
        s = sep.join(f"({r},{c})" for r, c in self.path)
        return s + (" =K" if self.promotes else "")


@dataclass
class MoveClass:
    """Result of :meth:`GameState.classify_move`."""
    kind: str                     # 'legal', 'huff', 'illegal'
    move: Move = None
    reason: str = ""
    huff_squares: list = field(default_factory=list)   # squares AFTER the move


class GameState:
    """Immutable-ish position: 64 cells + side to move + quiet ply counter.

    Cell values: 0 empty, +1 white man, +2 white king, -1 black man,
    -2 black king.
    """

    __slots__ = ("cells", "turn", "quiet_plies")

    def __init__(self, cells, turn=BLACK, quiet_plies=0):
        cells = tuple(int(v) for v in cells)
        if len(cells) != ROWS * COLS:
            raise ValueError("need 64 cells")
        self.cells = cells
        self.turn = turn
        self.quiet_plies = quiet_plies

    # ------------------------------------------------------------------ setup
    @classmethod
    def initial(cls, dark_parity=1, turn=BLACK):
        """Standard start: 12 black men on rows 0-2, 12 white men on rows 5-7,
        on squares with (row + col) % 2 == dark_parity."""
        cells = [0] * 64
        for r in range(ROWS):
            for c in range(COLS):
                if (r + c) % 2 != dark_parity:
                    continue
                if r <= 2:
                    cells[_idx(r, c)] = BLACK * MAN
                elif r >= 5:
                    cells[_idx(r, c)] = WHITE * MAN
        return cls(cells, turn)

    @classmethod
    def from_grid(cls, grid, turn=BLACK, kings=()):
        """Build from a detection grid (0 empty, 1 white, 2 black)."""
        cells = [0] * 64
        for r in range(ROWS):
            for c in range(COLS):
                v = int(grid[r][c])
                if v == 1:
                    cells[_idx(r, c)] = WHITE * MAN
                elif v == 2:
                    cells[_idx(r, c)] = BLACK * MAN
        for (r, c) in kings:
            if cells[_idx(r, c)]:
                cells[_idx(r, c)] = owner(cells[_idx(r, c)]) * KING
        return cls(cells, turn)

    def to_grid(self):
        """Detection-style grid: 0 empty, 1 white, 2 black."""
        g = [[0] * COLS for _ in range(ROWS)]
        for r in range(ROWS):
            for c in range(COLS):
                v = self.cells[_idx(r, c)]
                if v > 0:
                    g[r][c] = 1
                elif v < 0:
                    g[r][c] = 2
        return g

    def copy_with(self, cells=None, turn=None, quiet_plies=None):
        return GameState(self.cells if cells is None else cells,
                         self.turn if turn is None else turn,
                         self.quiet_plies if quiet_plies is None else quiet_plies)

    # -------------------------------------------------------------- queries
    def at(self, r, c):
        return self.cells[_idx(r, c)]

    def pieces(self, color):
        """[(row, col, value)] of all pieces of `color`."""
        out = []
        for i, v in enumerate(self.cells):
            if v and owner(v) == color:
                out.append((i // COLS, i % COLS, v))
        return out

    def count(self, color):
        return sum(1 for v in self.cells if v and owner(v) == color)

    def key(self):
        return (self.cells, self.turn)

    def __eq__(self, other):
        return isinstance(other, GameState) and self.key() == other.key()

    def __hash__(self):
        return hash(self.key())

    def __repr__(self):
        return f"GameState(turn={color_name(self.turn)}, quiet={self.quiet_plies})\n" + self.ascii()

    def ascii(self):
        sym = {0: ".", 1: "w", 2: "W", -1: "b", -2: "B"}
        lines = ["  " + " ".join(str(c) for c in range(COLS))]
        for r in range(ROWS):
            lines.append(f"{r} " + " ".join(sym[self.at(r, c)] for c in range(COLS)))
        return "\n".join(lines)

    # ------------------------------------------------------- move generation
    @staticmethod
    def _dirs(value):
        color = owner(value)
        if is_king(value):
            return ((-1, -1), (-1, 1), (1, -1), (1, 1))
        f = _forward(color)
        return ((f, -1), (f, 1))

    def simple_moves_from(self, r, c):
        v = self.at(r, c)
        color = owner(v)
        moves = []
        for dr, dc in self._dirs(v):
            nr, nc = r + dr, c + dc
            if _on_board(nr, nc) and self.at(nr, nc) == EMPTY:
                promotes = (not is_king(v)) and nr == _crown_row(color)
                moves.append(Move(((r, c), (nr, nc)), (), promotes))
        return moves

    def _jump_tree(self, r, c, include_partial):
        """All capture sequences starting at (r, c).

        Returns complete sequences, and additionally every strict prefix of a
        sequence if include_partial is True.  Jumped pieces stay on the board
        while jumping (can't be jumped twice, can't be landed on); the start
        square counts as empty (a king may end on its own start square)."""
        v = self.at(r, c)
        color = owner(v)
        king = is_king(v)
        start = (r, c)
        results = []

        def occupied(sq):
            if sq == start:
                return 0
            return self.cells[_idx(*sq)]

        def dfs(pos, path, captured):
            extended = False
            for dr, dc in self._dirs(v):
                mr, mc = pos[0] + dr, pos[1] + dc
                lr, lc = pos[0] + 2 * dr, pos[1] + 2 * dc
                if not _on_board(lr, lc):
                    continue
                mid = (mr, mc)
                if mid in captured:
                    continue
                mv = occupied(mid)
                if mv == 0 or owner(mv) == color:
                    continue
                if occupied((lr, lc)) != 0:
                    continue
                extended = True
                npath = path + ((lr, lc),)
                ncap = captured + (mid,)
                if not king and lr == _crown_row(color):
                    # crowning ends the move
                    results.append(Move(npath, ncap, True))
                    continue
                if include_partial:
                    results.append(Move(npath, ncap, False))
                dfs((lr, lc), npath, ncap)
            if not extended and captured and not include_partial:
                results.append(Move(path, captured, False))

        dfs(start, (start,), ())
        if include_partial:
            # dedupe (a complete non-crowning sequence was appended as prefix)
            seen, out = set(), []
            for m in results:
                if m not in seen:
                    seen.add(m)
                    out.append(m)
            return out
        return results

    def captures_from(self, r, c):
        return self._jump_tree(r, c, include_partial=False)

    def all_captures(self, color=None):
        color = self.turn if color is None else color
        out = []
        for r, c, _ in self.pieces(color):
            out.extend(self.captures_from(r, c))
        return out

    def legal_moves(self, color=None):
        color = self.turn if color is None else color
        caps = self.all_captures(color)
        if caps:
            return caps
        out = []
        for r, c, _ in self.pieces(color):
            out.extend(self.simple_moves_from(r, c))
        return out

    def capturing_squares(self, color=None):
        """Squares of `color` pieces that have at least one capture."""
        color = self.turn if color is None else color
        return [(r, c) for r, c, _ in self.pieces(color) if self.captures_from(r, c)]

    def huffable_moves(self, color=None):
        """Moves that break ONLY the capture obligation (huff-able):
        simple moves while a capture was available, and incomplete jumps."""
        color = self.turn if color is None else color
        if not self.all_captures(color):
            return []
        out = []
        for r, c, _ in self.pieces(color):
            out.extend(self.simple_moves_from(r, c))
            complete = set(self.captures_from(r, c))
            for m in self._jump_tree(r, c, include_partial=True):
                if m not in complete:
                    out.append(m)
        return out

    # ------------------------------------------------------------- applying
    def apply(self, move, check=False):
        """Return the position after `move` (turn passes to the opponent)."""
        if check and move not in self.legal_moves():
            raise ValueError(f"illegal move {move}")
        cells = list(self.cells)
        sr, sc = move.start
        er, ec = move.end
        v = cells[_idx(sr, sc)]
        if v == 0:
            raise ValueError(f"no piece on {move.start}")
        cells[_idx(sr, sc)] = 0
        for (cr, cc) in move.captured:
            cells[_idx(cr, cc)] = 0
        color = owner(v)
        if not is_king(v) and er == _crown_row(color):
            v = color * KING
        cells[_idx(er, ec)] = v
        man_moved = not is_king(self.cells[_idx(sr, sc)])
        quiet = 0 if (move.captured or man_moved) else self.quiet_plies + 1
        return GameState(cells, -self.turn, quiet)

    def remove(self, squares):
        """Remove pieces (huff); side to move unchanged."""
        cells = list(self.cells)
        for (r, c) in squares:
            cells[_idx(r, c)] = 0
        return GameState(cells, self.turn, self.quiet_plies)

    # ------------------------------------------------------------- outcome
    def winner(self):
        """WHITE / BLACK if the game is decided, DRAW, or None."""
        if self.count(self.turn) == 0 or not self.legal_moves():
            return -self.turn
        if self.quiet_plies >= QUIET_PLY_LIMIT:
            return DRAW
        return None

    # ------------------------------------------------------- huff analysis
    def classify_move(self, move):
        """Is `move` (by the side to move) legal, huff-able or illegal?"""
        if move in self.legal_moves():
            return MoveClass("legal", move)
        if move in self.huffable_moves():
            return MoveClass("huff", move, _huff_reason(self, move),
                             huff_candidates(self, move))
        return MoveClass("illegal", move, "not a legal checkers move")


def _huff_reason(state, move):
    if move.is_capture:
        return (f"multi-jump stopped early at {square_name(move.end)} - "
                "the jumping piece must keep capturing")
    caps = state.all_captures()
    ex = caps[0]
    return (f"capture was mandatory (e.g. {square_name(ex.start)} x "
            f"{square_name(ex.captured[0])}) but a non-capturing move was played")


def huff_candidates(state, move):
    """Squares (in the position AFTER `move`) of the pieces that may be huffed.

    * incomplete multi-jump -> the piece that stopped (at move.end)
    * non-capturing move    -> every piece that had a capture available; if
                               the moved piece was one of them it is listed at
                               its new square.
    """
    if move.is_capture:
        return [move.end]
    out = []
    for sq in state.capturing_squares():
        out.append(move.end if sq == move.start else sq)
    return out
