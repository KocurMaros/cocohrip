"""Guided dataset recording: what should the human change next, did they
do it, and how is a photo labelled.  No camera / GUI here (see
tools/capture_dataset.py and test/test_dataset.py).

The planner walks through positions one small physical change at a time
(1-2 pieces), so between photos the human only moves a piece or two:

* ``game``    random but legal games (realistic positions; captures,
              castling, en passant; promotions are avoided),
* ``scatter`` one piece to a random square, taken off, or put back -
              preferring piece/square combinations seen least so far, so
              every piece type gets photographed on every square,
* ``reset``   empty board, start position with white or black on the
              human's side.

All squares are *printed* squares (the coordinates printed on the board),
which is also how the labels are stored: the FEN piece placement of the
physical board.
"""

import json
import random
from collections import Counter
from dataclasses import dataclass, field

import chess
import numpy as np

from chess_game.game import labels_for

NAMES = {"p": "pawn", "n": "knight", "b": "bishop", "r": "rook", "q": "queen", "k": "king"}
FULL_SET = Counter("PPPPPPPPNNBBRRQKppppppppnnbbrrqk")


def piece_name(symbol):
    return ("WHITE " if symbol.isupper() else "BLACK ") + NAMES[symbol.lower()]


def placement_of(board, human_white=True):
    """Printed-square placement {square: symbol} of a logical board."""
    labels = labels_for(board, human_white)
    return {int(s): ".PNBRQKpnbrqk"[labels[s]] for s in np.flatnonzero(labels)}


def placement_fen(placement):
    b = chess.Board(None)
    for sq, sym in placement.items():
        b.set_piece_at(sq, chess.Piece.from_symbol(sym))
    return b.board_fen()


def placement_from_fen(fen):
    return {sq: p.symbol() for sq, p in chess.Board(fen + " w - - 0 1").piece_map().items()}


@dataclass
class Instruction:
    text: str
    before: dict
    after: dict
    kind: str = "move"                          # 'reset' | 'game' | 'scatter'
    lines: list = field(default_factory=list)   # extra lines for the GUI
    move: chess.Move = None                     # kind 'game': the logical move

    @property
    def changed(self):
        """Printed squares whose content changes."""
        keys = set(self.before) | set(self.after)
        return sorted(s for s in keys if self.before.get(s) != self.after.get(s))

    @property
    def sources(self):
        return [s for s in self.changed if s in self.before]

    @property
    def targets(self):
        return [s for s in self.changed if s in self.after]


class DatasetPlanner:
    """Endless sequence of Instructions.  `coverage` counts how often each
    (symbol, square) has been photographed; feed it with record()."""

    def __init__(self, placement=None, seed=None, game_plies=(20, 60), scatter_steps=(15, 30)):
        self.rng = random.Random(seed)
        self.placement = dict(placement or {})
        self.coverage = Counter()
        self.game_plies = game_plies
        self.scatter_steps = scatter_steps
        self._queue = []
        self._board = None
        self._human_white = True
        self._plies_left = 0
        self._phase = None
        self._scatter_left = 0
        self._resets = 0

    # ------------------------------------------------------------ public
    def record(self, placement):
        for sq, sym in placement.items():
            self.coverage[(sym, sq)] += 1

    def next(self):
        """The next instruction (not applied until accept())."""
        if not self._queue:
            self._queue.append(self._plan())
        return self._queue[0]

    def accept(self):
        ins = self._queue.pop(0)
        self.placement = dict(ins.after)
        if ins.kind == "game":
            self._board.push(ins.move)
        return ins

    def skip(self):
        """Drop the current instruction (e.g. a piece is missing)."""
        self._queue.clear()
        self._phase = "scatter"
        self._scatter_left = self.rng.randint(*self.scatter_steps)

    # ------------------------------------------------------------ planning
    def _plan(self):
        if self._phase is None:
            return self._reset("empty")
        if self._phase == "game":
            ins = self._game_step() if self._plies_left > 0 else None
            if ins is not None:
                return ins
            self._phase = "scatter"
            self._scatter_left = self.rng.randint(*self.scatter_steps)
        if self._phase == "scatter" and self._scatter_left > 0:
            self._scatter_left -= 1
            return self._scatter_step()
        self._resets += 1
        return self._reset("start")

    def _reset(self, what):
        if what == "empty":
            self._phase = "reset-empty"
            after = {}
            text = "Clear the board completely (photo of the empty board)"
        else:
            self._human_white = self._resets % 2 == 1
            self._board = chess.Board()
            after = placement_of(self._board, self._human_white)
            side = "WHITE" if self._human_white else "BLACK"
            text = f"Set up the START position - {side} on your side (ranks 1-2)"
            self._phase = "game"
            self._plies_left = self.rng.randint(*self.game_plies)
        return Instruction(text, dict(self.placement), after, kind="reset")

    def _game_step(self):
        board = self._board
        if board.is_game_over():
            return None
        moves = [m for m in board.legal_moves if not m.promotion]
        if not moves:
            return None
        caps = [m for m in moves if board.is_capture(m)]
        castles = [m for m in moves if board.is_castling(m)]
        r = self.rng.random()
        if castles and r < 0.5:
            move = self.rng.choice(castles)
        elif caps and r < 0.6:
            move = self.rng.choice(caps)
        else:
            move = self.rng.choice(moves)
        self._plies_left -= 1
        after_board = board.copy(stack=False)
        after_board.push(move)
        after = placement_of(after_board, self._human_white)
        return Instruction(self._move_text(board, move), dict(self.placement), after, kind="game",
                           lines=[f"(game move {board.san(move)})"], move=move)

    def _move_text(self, board, move):
        hw = self._human_white
        sq = (lambda s: chess.square_name(s if hw else 63 - s))     # logical -> printed
        piece = board.piece_at(move.from_square).symbol()
        if board.is_castling(move):
            kingside = chess.square_file(move.to_square) > chess.square_file(move.from_square)
            rank = chess.square_rank(move.from_square)
            rf, rt = chess.square(7 if kingside else 0, rank), chess.square(5 if kingside else 3, rank)
            return (f"Castle: {piece_name(piece)} {sq(move.from_square)} -> {sq(move.to_square)}, "
                    f"ROOK {sq(rf)} -> {sq(rt)}")
        text = f"Move {piece_name(piece)} {sq(move.from_square)} -> {sq(move.to_square)}"
        if board.is_en_passant(move):
            cap = chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
            text += f"  and take the pawn on {sq(cap)} off (en passant)"
        elif board.is_capture(move):
            victim = board.piece_at(move.to_square).symbol()
            text = f"Take the {piece_name(victim)} off {sq(move.to_square)}, then " + text[0].lower() + text[1:]
        return text

    def _scatter_step(self):
        before = dict(self.placement)
        on_board = list(before.items())
        box = FULL_SET - Counter(before.values())
        empty = [s for s in range(64) if s not in before]
        n = len(on_board)
        r = self.rng.random()
        if box and (n < 8 or (r < 0.2 and n < 28)):
            sym = self._least_seen_symbol(box)
            to = self._least_seen_square(sym, empty)
            after = dict(before)
            after[to] = sym
            return Instruction(f"Put a {piece_name(sym)} on {chess.square_name(to)} (from the box)",
                               before, after, kind="scatter")
        if (n > 26 and r < 0.35) or (n > 8 and r < 0.12):
            sq, sym = self.rng.choice([(s, p) for s, p in on_board if p.lower() != "k"]
                                      or on_board)
            after = dict(before)
            del after[sq]
            return Instruction(f"Take the {piece_name(sym)} off {chess.square_name(sq)} (into the box)",
                               before, after, kind="scatter")
        sq, sym = self.rng.choice(on_board)
        to = self._least_seen_square(sym, empty)
        after = dict(before)
        del after[sq]
        after[to] = sym
        return Instruction(f"Move the {piece_name(sym)} {chess.square_name(sq)} -> "
                           f"{chess.square_name(to)}", before, after, kind="scatter")

    def _least_seen_square(self, sym, squares):
        best = min(self.coverage[(sym, s)] for s in squares)
        return self.rng.choice([s for s in squares if self.coverage[(sym, s)] == best])

    def _least_seen_symbol(self, box):
        syms = [s for s, c in box.items() if c > 0]
        totals = {s: sum(v for (p, _), v in self.coverage.items() if p == s) for s in syms}
        best = min(totals.values())
        return self.rng.choice([s for s in syms if totals[s] == best])


# --------------------------------------------------------------------------
# change detection between two photos of the board
# --------------------------------------------------------------------------

class ChangeDetector:
    """Which squares look different between a reference photo and now.

    Only the central part of each square is compared (where a piece's base
    stands), in a brightness-normalised Lab image plus gradient energy, so a
    cream piece on a cream square still registers.  Tall pieces also change
    the squares their projected column passes over - `allowed()` adds those
    for the expected squares."""

    def __init__(self, board_model, image_shape, shrink=0.35, threshold=14.0):
        import cv2
        self.cv2 = cv2
        self.model = board_model
        self.threshold = threshold
        h, w = image_shape[:2]
        self.masks = []                 # flat pixel indices of each square's centre
        for sq in range(64):
            m = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(m, board_model.square_polygon(sq, shrink=shrink).astype(np.int32), 1)
            self.masks.append(np.flatnonzero(m))
        board_mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(board_mask, board_model.corners.astype(np.int32), 1)
        self.board_mask = board_mask.astype(bool)

    def _features(self, image):
        cv2 = self.cv2
        img = cv2.GaussianBlur(image[:, :, :3], (5, 5), 0)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        white = np.percentile(lab[:, :, 0][self.board_mask], 90)
        lab[:, :, 0] *= 200.0 / max(white, 1.0)
        gray = lab[:, :, 0]
        grad = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1))
        return lab, grad

    def scores(self, ref, cur):
        lab0, g0 = self._features(ref)
        lab1, g1 = self._features(cur)
        d = (np.abs(lab1 - lab0).sum(axis=2) + 0.25 * np.abs(g1 - g0)).ravel()
        return np.array([float(d[m].mean()) if len(m) else 0.0 for m in self.masks])

    def changed(self, ref, cur):
        s = self.scores(ref, cur)
        return [int(q) for q in np.flatnonzero(s > self.threshold)], s

    def allowed(self, squares, height=1.8):
        out = set()
        for sq in squares:
            out.update(self.model.column_squares(sq, height))
        return out


def check_change(instruction, changed, allowed):
    """(ok, message) - did the board change the way the instruction says?"""
    expected = set(instruction.changed)
    missing = expected - set(changed)
    extra = set(changed) - allowed - expected
    if missing:
        return False, "waiting for " + ", ".join(chess.square_name(s) for s in sorted(missing))
    if extra:
        return False, "unexpected change on " + ", ".join(chess.square_name(s) for s in sorted(extra))
    return True, "board matches the instruction"


# --------------------------------------------------------------------------
# labels file
# --------------------------------------------------------------------------

def label_record(image_file, placement, corners, exposure_us, lighting, step, instruction,
                 image_size, a1_corner, **extra):
    rec = {
        "image": image_file,
        "placement": placement_fen(placement),
        "corners": np.asarray(corners, dtype=float).round(2).tolist(),   # BL TL TR BR, px
        "image_size": list(image_size),
        "a1_corner": a1_corner,
        "exposure_us": int(exposure_us),
        "lighting": lighting,
        "step": int(step),
        "instruction": instruction,
    }
    rec.update(extra)
    return rec


def read_labels(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
