"""Game controller: keeps the authoritative rules.GameState and turns camera
detections into moves.

The camera only tells us, per square, empty / white / black (0 / 1 / 2).
The human's move is found by applying every legal (and every huff-able)
move to the current position and picking the one whose resulting board
matches the detection.  Matching is done primarily on *occupancy*, which is
much more reliable than the white/black colour decision (a black piece on a
dark square is hard to see, but a moved piece always empties one square and
fills another); colour is only a secondary check.

No pygame / ROS here (drawing lives in checkers.board), so it is testable.
"""

from collections import Counter
from dataclasses import dataclass, field

from checkers_game.checkers.rules import (BLACK, DRAW, WHITE, GameState, Move,
                                          color_name, is_king, owner,
                                          square_name)
from checkers_game.minimax.algorithm import choose_huff, choose_move

HUMAN = WHITE
ROBOT = BLACK

# repetition draw (same position, same side to move)
REPETITION_LIMIT = 3


# detection grid values (see camera.board_vision)
D_EMPTY, D_WHITE, D_BLACK, D_OCCUPIED, D_UNKNOWN = 0, 1, 2, 3, -1


def _occ(grid, r, c):
    """1 occupied, 0 empty, None unknown."""
    v = int(grid[r][c])
    if v == D_UNKNOWN:
        return None
    return 1 if v != D_EMPTY else 0


@dataclass
class Inference:
    """Outcome of matching a detected board against the possible moves."""
    kind: str                      # 'no_change' | 'legal' | 'huff' | 'invalid' | 'not_your_turn'
    move: Move = None
    huff_squares: list = field(default_factory=list)
    reason: str = ""
    noise_squares: list = field(default_factory=list)   # tolerated detection errors
    closest: Move = None           # for 'invalid': most similar possible move
    closest_diff: list = field(default_factory=list)
    changed: list = field(default_factory=list)          # squares that differ from position

    @property
    def accepted(self):
        return self.kind in ("legal", "huff")


@dataclass
class RobotPlan:
    move: Move                      # None if the robot has no move (it lost)
    huffed: tuple = None            # square of the human piece removed, or None
    piece_value: int = 0            # value of the moving piece BEFORE the move
    removed: list = field(default_factory=list)   # [(row, col, value)] to take off the board


class Game:
    def __init__(self, human_first=False, max_noise_squares=1, color_tolerance=2):
        self.state = None
        self.first = HUMAN if human_first else ROBOT
        self.history = Counter()
        self.log = []                 # list of strings, one per ply
        self.pending_huff = []        # candidate squares (after the human move)
        self.max_noise_squares = max_noise_squares
        self.color_tolerance = color_tolerance
        self.last_robot_plan = None

    # ----------------------------------------------------------------- setup
    @property
    def started(self):
        return self.state is not None

    @property
    def turn(self):
        return self.state.turn if self.state else None

    def start_standard(self, dark_parity=1):
        self.state = GameState.initial(dark_parity, turn=self.first)
        self._reset_history()
        return self.state

    def start_from_grid(self, grid, turn=None, kings=()):
        """Start (or resume) from a detected position.  Men standing on
        their crowning row are kings (they can only have got there that
        way)."""
        kings = set(kings)
        for c in range(8):
            if int(grid[0][c]) == D_WHITE:
                kings.add((0, c))
            if int(grid[7][c]) == D_BLACK:
                kings.add((7, c))
        self.state = GameState.from_grid(grid, turn or self.first, kings)
        self._reset_history()
        return self.state

    @staticmethod
    def check_standard_start(grid):
        """(ok, message, dark_parity) - is `grid` the standard start position
        (12 black men on rows 0-2, 12 white men on rows 5-7)?"""
        counts = Counter(int(v) for row in grid for v in row)

        def same(d, e):
            # 'occupied, colour unknown' is fine - the start position is fixed
            return d == e or (d == D_OCCUPIED and e != D_EMPTY)

        for parity in (1, 0):
            ref = GameState.initial(parity).to_grid()
            if all(same(int(grid[r][c]), ref[r][c]) for r in range(8) for c in range(8)):
                return True, "standard start position", parity
        return (False, f"not the start position (detected white={counts.get(1, 0)}, "
                f"black={counts.get(2, 0)}; need 12 + 12 on rows 5-7 / 0-2)", None)

    def _reset_history(self):
        self.history = Counter([self.state.key()])
        self.log = []
        self.pending_huff = []
        self.last_robot_plan = None

    # --------------------------------------------------------------- outcome
    def winner(self):
        """WHITE / BLACK / DRAW / None."""
        if not self.state:
            return None
        w = self.state.winner()
        if w is not None:
            return w
        if self.history[self.state.key()] >= REPETITION_LIMIT:
            return DRAW
        return None

    def winner_text(self):
        w = self.winner()
        if w is None:
            return None
        if w == DRAW:
            return "DRAW"
        return f"{color_name(w)} ({'you' if w == HUMAN else 'robot'})"

    def expected_grid(self):
        return self.state.to_grid()

    def mismatches(self, grid, use_color=True):
        """Squares where `grid` disagrees with the current position."""
        exp = self.expected_grid()
        out = []
        for r in range(8):
            for c in range(8):
                d, e = int(grid[r][c]), exp[r][c]
                if d == D_UNKNOWN:
                    continue
                if (d != 0) != (e != 0):
                    out.append((r, c))
                elif use_color and d in (D_WHITE, D_BLACK) and d != e:
                    out.append((r, c))
        return out

    # ------------------------------------------------------ human move input
    def _compare(self, after, grid, touched=()):
        """Squares where `grid` contradicts the position `after`.
        An unknown reading is ignored - except on a square the move touches
        (start, landing squares, captured pieces): those must be seen."""
        occ_diff, col_diff = [], []
        for r in range(8):
            for c in range(8):
                v = after.at(r, c)
                d = int(grid[r][c])
                if d == D_UNKNOWN:
                    if (r, c) in touched:
                        occ_diff.append((r, c))
                    continue
                if (v != 0) != (d != 0):
                    occ_diff.append((r, c))
                elif v != 0 and d in (D_WHITE, D_BLACK) and d != (1 if v > 0 else 2):
                    col_diff.append((r, c))
        return occ_diff, col_diff

    def infer_human_move(self, grid):
        """Find the human move that explains `grid` (8x8 of 0/1/2)."""
        s = self.state
        if s is None:
            return Inference("invalid", reason="game not started")
        if s.turn != HUMAN:
            return Inference("not_your_turn", reason="it is the robot's turn")
        return self.infer_move(grid)

    def infer_move(self, grid):
        """Find the move of the side to move that explains `grid` (used for
        the human; also for replaying recordings where a person moved both
        colours)."""
        s = self.state

        changed = [(r, c) for r in range(8) for c in range(8)
                   if _occ(grid, r, c) is not None
                   and (s.at(r, c) != 0) != (_occ(grid, r, c) == 1)]
        if not changed:
            return Inference("no_change")

        cands = [(m, "legal") for m in s.legal_moves()]
        cands += [(m, "huff") for m in s.huffable_moves()]
        scored = []
        seen = set()
        after_of = {}
        for m, kind in cands:
            after = s.apply(m)
            after_of[id(m)] = after
            if after.key() in seen:      # e.g. a king taking the same pieces in another order
                continue
            seen.add(after.key())
            touched = set(m.path) | set(m.captured)
            occ_diff, col_diff = self._compare(after, grid, touched)
            scored.append((len(occ_diff), len(col_diff), 0 if kind == "legal" else 1,
                           m, kind, occ_diff, col_diff, touched))
        scored.sort(key=lambda t: (t[0], t[1], t[2]))

        best = scored[0] if scored else None
        if best is not None:
            n_occ, n_col, _, m, kind, occ_diff, col_diff, touched = best
            second = scored[1][0] if len(scored) > 1 else 99
            # every square the move touches must agree; tolerate a little
            # detection noise elsewhere when the match is unambiguous
            ambiguous = (len(scored) > 1 and scored[1][0] == n_occ and scored[1][1] == n_col)
            # tolerated noise: a piece the camera missed (reads empty where
            # the position has a piece) away from the move - never a piece
            # that appeared somewhere (that could be the real, illegal move)
            # (and only an OPPONENT piece - the black pieces are the hard
            # ones to see; a "missing" piece of the mover is a real change)
            missed_only = all(int(grid[r][c]) == D_EMPTY
                              and owner(after_of[id(m)].at(r, c)) == -s.turn
                              for r, c in occ_diff)
            noise_ok = not ambiguous and (n_occ == 0 or
                        (n_occ <= self.max_noise_squares
                         and missed_only
                         and not (set(occ_diff) & touched)
                         and second >= n_occ + 2))
            # the moved piece itself must not read as the other colour
            colour_ok = n_col <= self.color_tolerance and not (set(col_diff) & touched)
            if noise_ok and colour_ok:
                inf = Inference(kind, m, noise_squares=occ_diff + col_diff, changed=changed)
                if kind == "huff":
                    cls = s.classify_move(m)
                    inf.huff_squares = cls.huff_squares
                    inf.reason = cls.reason
                return inf

        inf = Inference("invalid", changed=changed)
        if best is not None:
            inf.closest, inf.closest_diff = best[3], best[5]
        inf.reason = self._explain_invalid(grid, changed)
        if best is not None and best[0] == 0 and len(scored) > 1 and scored[1][0] == 0:
            inf.reason = ("ambiguous - the camera can't see some squares well enough to "
                          f"tell {best[3]} from {scored[1][3]}")
        return inf

    def _explain_invalid(self, grid, changed):
        s = self.state
        gone = [sq for sq in changed if s.at(*sq) != 0]
        new = [sq for sq in changed if s.at(*sq) == 0]
        parts = []
        robot_touched = [sq for sq in gone if owner(s.at(*sq)) == ROBOT]
        if len(gone) == 1 and len(new) == 1 and owner(s.at(*gone[0])) == HUMAN:
            parts.append(f"{square_name(gone[0])} -> {square_name(new[0])} is not a legal move")
            caps = s.all_captures()
            if caps:
                parts.append("(a capture is available - captures are mandatory)")
        elif robot_touched and not any(sq in [c for m in s.legal_moves() + s.huffable_moves()
                                              for c in m.captured] for sq in robot_touched):
            parts.append("a robot (black) piece seems to have moved or disappeared: "
                         + ", ".join(square_name(q) for q in robot_touched))
        else:
            if gone:
                parts.append("pieces gone from " + ", ".join(square_name(q) for q in gone))
            if new:
                parts.append("new pieces on " + ", ".join(square_name(q) for q in new))
        return "; ".join(parts) or "board does not match any legal move"

    def apply_human(self, inference):
        """Apply an accepted Inference. Returns the new state."""
        assert inference.accepted
        m = inference.move
        mover = color_name(self.state.turn)
        self.state = self.state.apply(m)
        self.history[self.state.key()] += 1
        self.log.append(f"{mover} {m}" + (" (huff!)" if inference.kind == "huff" else ""))
        self.pending_huff = list(inference.huff_squares) if inference.kind == "huff" else []
        return self.state

    # -------------------------------------------------------------- robot
    def plan_robot_turn(self, level=3, time_limit=4.0, rng=None):
        """Decide the huff (if any) and the robot's move.  Does not apply it."""
        s = self.state
        assert s.turn == ROBOT, "not the robot's turn"
        huffed = None
        removed = []
        if self.pending_huff:
            cands = [sq for sq in self.pending_huff if owner(s.at(*sq)) == HUMAN]
            if cands:
                huffed = choose_huff(s, cands, level=level, rng=rng)
                removed.append((huffed[0], huffed[1], s.at(*huffed)))
                s = s.remove([huffed])
        move = None
        value = 0
        if s.winner() is None:
            move, _, _ = choose_move(s, level=level, time_limit=time_limit, rng=rng)
            if move is not None:
                value = s.at(*move.start)
                removed += [(r, c, s.at(r, c)) for (r, c) in move.captured]
        return RobotPlan(move, huffed, value, removed)

    def apply_robot(self, plan):
        s = self.state
        if plan.huffed is not None:
            s = s.remove([plan.huffed])
            self.log.append(f"BLACK huffs {square_name(plan.huffed)}")
        if plan.move is not None:
            s = s.apply(plan.move)
            self.log.append(f"BLACK {plan.move}")
        elif s.turn == ROBOT and s.winner() is None:
            raise ValueError("robot plan without a move")
        self.state = s
        self.pending_huff = []
        self.history[self.state.key()] += 1
        self.last_robot_plan = plan
        return self.state

    # ------------------------------------------------------------ helpers
    def pieces_for_msg(self):
        """[(row, col, color_str, king)] for the Board message."""
        out = []
        for color in (HUMAN, ROBOT):
            for r, c, v in self.state.pieces(color):
                out.append((r, c, "white" if color == WHITE else "red", is_king(v)))
        return out

    def hint_moves(self):
        """Readable list of the human's legal moves."""
        if not self.state or self.state.turn != HUMAN:
            return []
        return [str(m) for m in self.state.legal_moves()]

    @staticmethod
    def other(color):
        return BLACK if color == WHITE else WHITE


__all__ = ["Game", "Inference", "RobotPlan", "HUMAN", "ROBOT", "DRAW", "WHITE", "BLACK"]
