"""Chess game flow on top of python-chess: what did the human play (from
the camera), what does the robot have to pick and place, is the board set
up right.  No ROS, no camera - see test/test_game.py.

Camera observations
-------------------
The vision gives, for every *printed* square (python-chess numbering, see
frames.py), a probability vector over either

* 13 classes: CLASSES = empty, P N B R Q K (white), p n b r q k (black), or
* 3 classes:  empty, white piece, black piece.

There are no spare pieces: a promoted pawn stays a pawn on the board and is
a queen (or the piece chosen) only in the game - the dashboard marks it.
The camera is therefore matched against the *appearance* of the board
(promoted pieces look like pawns, python-chess tracks them in
``board.promoted``).

Move inference does not trust any single square.  Every legal move (and "no
move") gives a predicted board; each prediction is scored by the
log-likelihood of the whole observation and the best one wins if it beats
the runner-up clearly.  A colour-only (3-class) detector is enough to follow
a game - every legal move changes the colour occupancy differently.  The
promotion piece cannot be seen; it is ``ChessGame.promotion_piece`` (queen
unless the human picks another one).
"""

import math
from dataclasses import dataclass, field

import chess
import numpy as np

from chess_game.frames import logical_to_printed, printed_to_logical

CLASSES = ".PNBRQKpnbrqk"          # index 0 = empty square
COLOURS = ".WB"                    # 3-class detector: empty / white / black
EPS = 1e-4                         # probability floor for the log-likelihood
CONFIDENT_WRONG = 0.10             # p(expected) below this = square contradicts


def class_index(piece):
    return 0 if piece is None else CLASSES.index(piece.symbol())


def to_colour_labels(labels13):
    """13-class labels -> 3-class labels (0 empty, 1 white, 2 black)."""
    labels13 = np.asarray(labels13)
    return np.where(labels13 == 0, 0, np.where(labels13 <= 6, 1, 2))


def collapse_to_colour(probs13):
    """(64, 13) class probabilities -> (64, 3) empty / white / black."""
    p = np.asarray(probs13, dtype=float)
    return np.stack([p[:, 0], p[:, 1:7].sum(axis=1), p[:, 7:13].sum(axis=1)], axis=1)


def labels_for(board, human_white, appearance=True):
    """13-class label of every printed square for a python-chess board.
    appearance: promoted pieces are labelled as the pawns they physically
    are (what the camera sees)."""
    out = np.zeros(64, dtype=int)
    for logical_sq, piece in board.piece_map().items():
        if appearance and board.promoted & chess.BB_SQUARES[logical_sq]:
            piece = chess.Piece(chess.PAWN, piece.color)
        out[logical_to_printed(logical_sq, human_white)] = class_index(piece)
    return out


def observable(labels13, n_classes):
    return labels13 if n_classes == 13 else to_colour_labels(labels13)


def log_likelihood(obs, labels):
    """Sum over squares of log p(observed class = label)."""
    p = obs[np.arange(64), labels]
    return float(np.log(np.maximum(p, EPS)).sum())


def contradicting(obs, labels, threshold=CONFIDENT_WRONG):
    """Printed squares where the observation clearly disagrees with labels."""
    p = obs[np.arange(64), labels]
    return [int(s) for s in np.flatnonzero(p < threshold)]


def _check_obs(obs):
    obs = np.asarray(obs, dtype=float)
    if obs.shape not in ((64, 3), (64, 13)):
        raise ValueError(f"observation must be (64, 3) or (64, 13), got {obs.shape}")
    return obs


def obs_from_labels(labels, n_classes=13, confidence=0.98):
    """Synthetic observation (tests, replays): `confidence` on the label."""
    labels = observable(np.asarray(labels), n_classes)
    obs = np.full((64, n_classes), (1.0 - confidence) / (n_classes - 1))
    obs[np.arange(64), labels] = confidence
    return obs


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------

@dataclass
class Inference:
    kind: str                      # 'no_change' | 'move' | 'invalid' | 'unsure'
    move: chess.Move = None
    reason: str = ""
    margin: float = 0.0            # log-likelihood gap to the runner-up
    changed: list = field(default_factory=list)       # printed squares unlike the game
    noise: list = field(default_factory=list)         # ignored contradicting squares
    closest: chess.Move = None     # best legal move when the board matches none
    closest_diff: list = field(default_factory=list)  # printed squares it disagrees on

    @property
    def accepted(self):
        return self.kind == "move"


@dataclass
class Place:
    """Where the robot picks or places a piece."""
    kind: str                      # 'square' | 'storage'
    square: int = None             # logical square (kind 'square')
    slot: int = None               # storage slot (kind 'storage')

    def __str__(self):
        if self.kind == "square":
            return chess.square_name(self.square)
        return f"storage#{self.slot}"


@dataclass
class Step:
    """One pick-and-place of the robot."""
    symbol: str                    # FEN symbol of the carried piece
    pick: Place
    place: Place

    def __str__(self):
        return f"{self.symbol}: {self.pick} -> {self.place}"


@dataclass
class SetupCheck:
    ok: bool
    human_white: bool
    message: str
    wrong: list = field(default_factory=list)      # printed squares


# --------------------------------------------------------------------------
# game
# --------------------------------------------------------------------------

class ChessGame:
    """Rules state + camera interpretation for one game.

    promotion_piece: what a human pawn promotes to (the camera cannot see
    it - the pawn stays on the board); the robot picks its own.
    max_noise_squares: confidently contradicting squares (outside the
    move's own squares) that are ignored as detection noise.  0 = a single
    lifted / misplaced piece makes the board "invalid" until it is fixed.
    """

    def __init__(self, human_white=True, max_noise_squares=0, min_margin=math.log(50)):
        self.human_white = human_white
        self.max_noise_squares = max_noise_squares
        self.min_margin = min_margin
        self.promotion_piece = chess.QUEEN
        self.board = None
        self.storage_used = 0
        self.log = []

    # ------------------------------------------------------------ state
    @property
    def started(self):
        return self.board is not None

    @property
    def human_color(self):
        return chess.WHITE if self.human_white else chess.BLACK

    @property
    def human_to_move(self):
        return self.started and self.board.turn == self.human_color

    def start(self, fen=chess.STARTING_FEN, human_white=None):
        if human_white is not None:
            self.human_white = human_white
        self.board = chess.Board(fen)
        self.storage_used = 0
        self.log = []

    def start_from_observation(self, obs, human_to_move=True):
        """Start from whatever is on the board (needs the 13-class detector).
        Castling rights are kept where king and rook stand on their home
        squares; returns (ok, message)."""
        obs = _check_obs(obs)
        if obs.shape[1] != 13:
            return False, "starting from a position needs the piece classifier"
        board = chess.Board(None)
        for printed_sq in range(64):
            c = int(np.argmax(obs[printed_sq]))
            if c:
                board.set_piece_at(printed_to_logical(printed_sq, self.human_white),
                                   chess.Piece.from_symbol(CLASSES[c]))
        board.turn = self.human_color if human_to_move else not self.human_color
        board.set_castling_fen("KQkq")
        board.castling_rights = board.clean_castling_rights()
        status = board.status()
        if status != chess.STATUS_VALID:
            return False, f"not a legal position ({_status_text(status)})"
        self.board = board
        self.storage_used = 0
        self.log = []
        return True, board.fen()

    def expected_labels(self, board=None):
        return labels_for(board if board is not None else self.board, self.human_white)

    def push(self, move, steps=()):
        """Play a move (either side); `steps` = the robot's plan, to count
        the storage slots it filled."""
        dots = "." if self.board.turn == chess.WHITE else "..."
        self.log.append(f"{self.board.fullmove_number}{dots} {self.board.san(move)}")
        self.board.push(move)
        self.storage_used += sum(1 for s in steps if s.place.kind == "storage")

    def outcome(self):
        return self.board.outcome(claim_draw=True) if self.started else None

    # ------------------------------------------------------------ setup
    @staticmethod
    def check_setup(obs, fen=chess.STARTING_FEN):
        """Is the start position on the board, and which colour is on the
        human's side?  Decided on colour occupancy (works with either
        detector); with the piece classifier wrong piece types are reported
        too ("knight and bishop swapped")."""
        obs = _check_obs(obs)
        board = chess.Board(fen)
        n = obs.shape[1]
        colour_obs = collapse_to_colour(obs) if n == 13 else obs
        best = None
        for human_white in (True, False):
            labels = labels_for(board, human_white)
            score = log_likelihood(colour_obs, to_colour_labels(labels))
            if best is None or score > best[0]:
                best = (score, human_white, labels)
        _, human_white, labels = best
        wrong = contradicting(colour_obs, to_colour_labels(labels))
        side = "white" if human_white else "black"
        if wrong:
            return SetupCheck(False, human_white,
                              f"not the start position ({side} on your side?) - check "
                              + ", ".join(chess.square_name(s) for s in wrong), wrong)
        if n == 13:
            wrong = contradicting(obs, labels)
            if wrong:
                seen = {s: CLASSES[int(np.argmax(obs[s]))] for s in wrong}
                text = ", ".join(f"{chess.square_name(s)}: {CLASSES[labels[s]]} expected, "
                                 f"looks like {seen[s]}" for s in wrong)
                return SetupCheck(False, human_white, "wrong pieces - " + text, wrong)
        return SetupCheck(True, human_white, f"start position, you play {side}")

    # ------------------------------------------------------------ human move
    def infer_human_move(self, obs):
        """Match the camera observation against every legal human move."""
        obs = _check_obs(obs)
        n = obs.shape[1]
        if not self.human_to_move:
            raise RuntimeError("not the human's turn")
        current = self.expected_labels()

        # candidates grouped by what the camera can tell apart (a colour-only
        # detector cannot see which piece a pawn promoted to)
        groups = {}
        for move in [None] + list(self.board.legal_moves):
            labels = current if move is None else self._labels_after(move)
            obs_labels = observable(labels, n)
            score = log_likelihood(obs, obs_labels)
            key = obs_labels.tobytes()
            prev = groups.get(key)
            if prev is None or score > prev[0] + 1e-9 or (
                    abs(score - prev[0]) <= 1e-9 and self._prefer(move, prev[1])):
                groups[key] = (score, move, obs_labels)
        ranked = sorted(groups.values(), key=lambda g: -g[0])
        best_score, best_move, best_labels = ranked[0]
        margin = best_score - ranked[1][0] if len(ranked) > 1 else math.inf
        bad = contradicting(obs, best_labels)
        cur_obs = observable(current, n)
        changed = contradicting(obs, cur_obs)

        if best_move is None:
            if len(bad) <= self.max_noise_squares:
                return Inference("no_change", margin=margin, noise=bad)
            return self._invalid(obs, ranked, changed, "the board matches no legal move")

        move_squares = set(np.flatnonzero(best_labels != cur_obs).tolist())
        if len(bad) > self.max_noise_squares or move_squares & set(bad):
            return self._invalid(obs, ranked, changed, "the board matches no legal move")
        if margin < self.min_margin:
            return Inference("unsure", best_move, margin=margin, changed=changed,
                             reason=f"camera unsure between {self.board.san(best_move)} and "
                                    f"{self._describe(ranked[1][1])}")
        return Inference("move", best_move, margin=margin, changed=changed, noise=bad,
                         reason=self.board.san(best_move))

    def _invalid(self, obs, ranked, changed, reason):
        closest = next((g for g in ranked if g[1] is not None), None)
        inf = Inference("invalid", reason=reason, changed=changed)
        if closest is not None:
            inf.closest = closest[1]
            inf.closest_diff = contradicting(obs, closest[2])
        return inf

    def _prefer(self, move, other):
        """Tie-break inside a group the camera cannot tell apart (the
        promotion piece): the human's chosen promotion piece."""
        want = self.promotion_piece
        return (move is not None and move.promotion == want
                and (other is None or other.promotion != want))

    def _describe(self, move):
        return "no move" if move is None else self.board.san(move)

    def _labels_after(self, move):
        self.board.push(move)
        try:
            return self.expected_labels()
        finally:
            self.board.pop()

    def mismatches(self, obs, board=None):
        """Printed squares where the camera contradicts the game position
        (colour occupancy only - robust check after the robot's move)."""
        obs = _check_obs(obs)
        colour_obs = collapse_to_colour(obs) if obs.shape[1] == 13 else obs
        return contradicting(colour_obs, to_colour_labels(self.expected_labels(board)))

    # ------------------------------------------------------------ robot move
    def plan_robot_move(self, move):
        """Pick-and-place steps for a robot move, captured pieces first.

        capture       captured piece -> storage, then the move
        en passant    the pawn beside the target -> storage, then the move
        castling      king, then rook
        promotion     the pawn moves; it is the new piece only in the game
        """
        b = self.board
        piece = b.piece_at(move.from_square)
        if piece is None or piece.color == self.human_color:
            raise ValueError(f"{move} is not a robot move")
        slot = self.storage_used
        steps = []

        def square(s):
            return Place("square", square=s)

        def storage():
            nonlocal slot
            slot += 1
            return Place("storage", slot=slot - 1)

        if b.is_en_passant(move):
            cap = chess.square(chess.square_file(move.to_square),
                               chess.square_rank(move.from_square))
            steps.append(Step(self.physical_symbol(cap), square(cap), storage()))
        elif b.is_capture(move):
            steps.append(Step(self.physical_symbol(move.to_square), square(move.to_square),
                              storage()))

        if b.is_castling(move):
            rank = chess.square_rank(move.from_square)
            kingside = chess.square_file(move.to_square) > chess.square_file(move.from_square)
            rook_from = chess.square(7 if kingside else 0, rank)
            rook_to = chess.square(5 if kingside else 3, rank)
            steps.append(Step(piece.symbol(), square(move.from_square), square(move.to_square)))
            steps.append(Step(self.physical_symbol(rook_from), square(rook_from), square(rook_to)))
        else:
            steps.append(Step(self.physical_symbol(move.from_square), square(move.from_square),
                              square(move.to_square)))
        return steps

    def physical_symbol(self, logical_sq):
        """FEN symbol of the piece that physically stands on a square (a
        promoted piece is a pawn) - the robot grips by it."""
        piece = self.board.piece_at(logical_sq)
        if self.board.promoted & chess.BB_SQUARES[logical_sq]:
            piece = chess.Piece(chess.PAWN, piece.color)
        return piece.symbol()




def _status_text(status):
    names = [name for name in dir(chess) if name.startswith("STATUS_")
             and name != "STATUS_VALID" and getattr(chess, name) & status]
    return ", ".join(n[7:].lower().replace("_", " ") for n in names) or str(status)
