"""Alpha-beta search for the robot (BLACK) on top of checkers.rules.

Only ever returns legal moves (mandatory capture, completed multi-jumps,
crowning ends the move - all enforced by the rules engine).

Difficulty "level" 1..5 (the AI_difficulty trackbar) maps to search depth in
plies; a time budget caps the search (iterative deepening keeps the best
move of the last fully searched depth).
"""

import random
import time

from checkers_game.checkers.rules import (BLACK, DRAW, KING, WHITE, GameState,
                                          is_king, owner)

LEVEL_DEPTH = {1: 1, 2: 2, 3: 4, 4: 6, 5: 8}
DEFAULT_TIME_LIMIT = 4.0          # seconds

WIN_SCORE = 100000
MAN_VALUE = 100
KING_VALUE = 160

# small positional terms (from the point of view of the piece owner)
ADVANCE_BONUS = 4      # per row a man has advanced
BACK_ROW_BONUS = 12    # men still guarding their own back row
CENTER_BONUS = 6       # pieces on the central 4x2 squares
EDGE_KING_PENALTY = 8  # kings stuck on an edge column

_CENTER = {(3, 2), (3, 3), (3, 4), (3, 5), (4, 2), (4, 3), (4, 4), (4, 5)}


def depth_for_level(level):
    level = max(1, min(5, int(level)))
    return LEVEL_DEPTH[level]


def evaluate(state, color):
    """Static evaluation from `color`'s point of view."""
    score = 0
    mat = {WHITE: 0, BLACK: 0}
    men = {WHITE: 0, BLACK: 0}
    for i, v in enumerate(state.cells):
        if not v:
            continue
        r, c = divmod(i, 8)
        o = owner(v)
        s = 0
        if is_king(v):
            s += KING_VALUE
            if c in (0, 7):
                s -= EDGE_KING_PENALTY
        else:
            s += MAN_VALUE
            men[o] += 1
            advanced = (7 - r) if o == WHITE else r
            s += ADVANCE_BONUS * advanced
            home = 7 if o == WHITE else 0
            if r == home:
                s += BACK_ROW_BONUS
        if (r, c) in _CENTER:
            s += CENTER_BONUS
        mat[o] += MAN_VALUE if not is_king(v) else KING_VALUE
        score += s if o == color else -s
    # when ahead, trading down is good
    total = mat[WHITE] + mat[BLACK]
    if total:
        diff = mat[color] - mat[-color]
        score += int(diff * 2400 / (total + 400) / 10)
    return score


def _ordered(moves):
    # longer captures and promotions first: better alpha-beta cut-offs
    return sorted(moves, key=lambda m: (len(m.captured), m.promotes), reverse=True)


class _Timeout(Exception):
    pass


class Searcher:
    def __init__(self, time_limit=DEFAULT_TIME_LIMIT, rng=None):
        self.time_limit = time_limit
        self.rng = rng or random.Random()
        self.nodes = 0
        self.deadline = None

    def _negamax(self, state, depth, alpha, beta, ply):
        self.nodes += 1
        if self.deadline and (self.nodes & 1023) == 0 and time.time() > self.deadline:
            raise _Timeout()
        moves = state.legal_moves()
        if not moves:
            return -WIN_SCORE + ply          # side to move has lost
        if state.quiet_plies >= 80:
            return 0
        # quiescence: keep searching forced capture sequences past the horizon
        if depth <= 0 and not moves[0].is_capture:
            return evaluate(state, state.turn)
        if depth <= -6:
            return evaluate(state, state.turn)
        best = -WIN_SCORE * 2
        for m in _ordered(moves):
            val = -self._negamax(state.apply(m), depth - 1, -beta, -alpha, ply + 1)
            if val > best:
                best = val
            if val > alpha:
                alpha = val
            if alpha >= beta:
                break
        return best

    def search(self, state, max_depth, need_score=False):
        """Return (best_move, score, depth_reached)."""
        moves = state.legal_moves()
        if not moves:
            return None, -WIN_SCORE, 0
        if len(moves) == 1 and not need_score:
            return moves[0], 0, 0
        self.deadline = time.time() + self.time_limit if self.time_limit else None
        best_move, best_score, reached = None, None, 0
        ordered = _ordered(moves)
        for depth in range(1, max_depth + 1):
            try:
                scored = []
                alpha = -WIN_SCORE * 2
                for m in ordered:
                    val = -self._negamax(state.apply(m), depth - 1, -WIN_SCORE * 2, -alpha + 1, 1)
                    scored.append((val, m))
                    alpha = max(alpha, val)
            except _Timeout:
                break
            top = max(v for v, _ in scored)
            ties = [m for v, m in scored if v == top]
            best_move, best_score, reached = self.rng.choice(ties), top, depth
            # search the best move first next iteration
            ordered = [best_move] + [m for m in ordered if m != best_move]
            if abs(top) >= WIN_SCORE - 100:
                break
        if best_move is None:          # timed out even at depth 1
            best_move = ordered[0]
            best_score = evaluate(state, state.turn)
        return best_move, best_score, reached


def choose_move(state, level=3, time_limit=DEFAULT_TIME_LIMIT, rng=None):
    """Best legal move for the side to move in `state`."""
    return Searcher(time_limit, rng).search(state, depth_for_level(level))


def choose_huff(state, candidates, level=3, time_limit=1.5, rng=None):
    """Pick which of the opponent's pieces to huff (remove) - the one that
    leaves the side to move best off.  Returns a square."""
    if len(candidates) <= 1:
        return candidates[0] if candidates else None
    # the time budget is shared by all candidates
    s = Searcher(time_limit / len(candidates) if time_limit else None, rng)
    depth = max(1, depth_for_level(level) - 2)
    best_sq, best_val = None, None
    for sq in candidates:
        after = state.remove([sq])
        _, val, _ = s.search(after, depth, need_score=True)
        # prefer removing kings when the search can't tell the difference
        val += 5 if is_king(state.at(*sq)) else 0
        if best_val is None or val > best_val:
            best_sq, best_val = sq, val
    return best_sq


# --------------------------------------------------------------------------
# backwards compatible wrapper (old code called minimax(board, depth, ...))
# --------------------------------------------------------------------------
def minimax(state, depth, max_player=None, game=None):  # noqa: ARG001
    """Deprecated: kept so old imports don't break. Returns (score, move)."""
    if not isinstance(state, GameState):
        raise TypeError("minimax now expects a rules.GameState")
    move, score, _ = Searcher(DEFAULT_TIME_LIMIT).search(state, max(1, depth))
    return score, move


__all__ = ["choose_move", "choose_huff", "evaluate", "depth_for_level", "Searcher",
           "LEVEL_DEPTH", "KING", "DRAW"]
