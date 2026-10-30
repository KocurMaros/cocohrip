"""Swappable chess engines behind one interface.

    engine = create_engine("stockfish")    # or "leela", "maia", "lc0:...", "builtin", "random"
    move = engine.choose_move(board, level=3, time_limit=2.0)
    engine.close()

* ``stockfish`` - any UCI binary on PATH (or ``stockfish:/path/to/binary``);
  the level (1..5) sets Stockfish's "Skill Level".
* ``leela`` - Leela Chess Zero (lc0, a neural-network engine) with the net
  ``~/chess_engines/leela.pb.gz``; the level sets the search size (1 node =
  the network's instinct ... 4000 nodes).
* ``maia`` - lc0 with the Maia nets ``~/chess_engines/maia-<rating>.pb.gz``:
  plays like a human of that rating; level 1..5 = Maia 1100 .. 1900
  (searched with 1 node, as Maia is meant to be).
* ``lc0`` - lc0 with any net (``lc0:weights=/path/net.pb.gz,nodes=100``).
* ``builtin`` - small alpha-beta search in pure Python, no binary needed
  (fallback, tests, easy levels).
* ``random`` - random legal move (tests).

* ``uci:/path/to/engine`` - any other UCI engine (all of these get the
  position and answer a move in the same UCI format).

A PyTorch policy network would be one more Engine subclass with the same
choose_move().
"""

import os
import random
import shutil
import time

import chess
import chess.engine

LEVELS = (1, 2, 3, 4, 5)


class Engine:
    name = "engine"

    def choose_move(self, board, level=3, time_limit=2.0):
        raise NotImplementedError

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class RandomEngine(Engine):
    name = "random"

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def choose_move(self, board, level=3, time_limit=2.0):
        return self.rng.choice(list(board.legal_moves))


class UciEngine(Engine):
    """Any UCI engine (Stockfish, lc0, ...) run as a subprocess.

    level_options: level -> dict of UCI options set before each search
    level_limits:  level -> dict of chess.engine.Limit arguments (depth,
                   nodes, time); the time is capped by `time_limit`.
    """

    def __init__(self, command, options=None, level_options=None, level_limits=None,
                 name="uci"):
        self.name = name
        self.engine = chess.engine.SimpleEngine.popen_uci(command)
        if options:
            self.engine.configure(options)
        self.level_options = level_options or {}
        self.level_limits = level_limits or {}
        self._configured = None

    def choose_move(self, board, level=3, time_limit=2.0):
        opts = self.level_options.get(level)
        if opts and opts != self._configured:
            self.engine.configure(opts)
            self._configured = opts
        limit = dict(self.level_limits.get(level, {}))
        limit["time"] = min(limit.get("time", time_limit), time_limit)
        result = self.engine.play(board, chess.engine.Limit(**limit))
        return result.move

    def close(self):
        try:
            self.engine.quit()
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError, OSError):
            pass


def stockfish(command=None):
    command = command or shutil.which("stockfish") or "/usr/games/stockfish"
    skill = {1: 0, 2: 4, 3: 8, 4: 14, 5: 20}
    depth = {1: 1, 2: 3, 3: 6, 4: 10}
    return UciEngine(command, options={"Threads": 2, "Hash": 64},
                     level_options={lv: {"Skill Level": s} for lv, s in skill.items()},
                     level_limits={lv: {"depth": d} for lv, d in depth.items()},
                     name="stockfish")


NET_DIR = os.path.expanduser("~/chess_engines")
MAIA_RATINGS = {1: 1100, 2: 1300, 3: 1500, 4: 1700, 5: 1900}
LEELA_NODES = {1: 1, 2: 16, 3: 128, 4: 800, 5: 4000}


def _lc0_command(command):
    return command or shutil.which("lc0") or os.path.join(NET_DIR, "lc0")


def lc0(command=None, weights=None, nodes=1):
    """lc0 with one net, searching `nodes` nodes at every level."""
    options = {"WeightsFile": os.path.expanduser(weights)} if weights else None
    limits = {lv: {"nodes": int(nodes)} for lv in LEVELS}
    return UciEngine(_lc0_command(command), options=options, level_limits=limits, name="lc0")


def leela(command=None, weights=None):
    weights = os.path.expanduser(weights or os.path.join(NET_DIR, "leela.pb.gz"))
    return UciEngine(_lc0_command(command), options={"WeightsFile": weights},
                     level_limits={lv: {"nodes": n} for lv, n in LEELA_NODES.items()}, name="leela")


def maia(command=None, net_dir=None):
    """Human-like play: the level picks the Maia net (rating 1100..1900)."""
    net_dir = os.path.expanduser(net_dir or NET_DIR)
    nets = {lv: os.path.join(net_dir, f"maia-{r}.pb.gz") for lv, r in MAIA_RATINGS.items()}
    missing = [p for p in nets.values() if not os.path.exists(p)]
    if missing:
        raise OSError(f"Maia nets missing: {', '.join(missing)}")
    return UciEngine(_lc0_command(command), options={"WeightsFile": nets[3]},
                     level_options={lv: {"WeightsFile": p} for lv, p in nets.items()},
                     level_limits={lv: {"nodes": 1} for lv in LEVELS}, name="maia")


# --------------------------------------------------------------------------
# built-in alpha-beta (pure Python)
# --------------------------------------------------------------------------

VALUES = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500,
          chess.QUEEN: 900, chess.KING: 0}

# piece-square tables (Michniewski's "simplified evaluation function"),
# from white's point of view, a8 first - index with square_mirror for white
_PST = {
    chess.PAWN: [0, 0, 0, 0, 0, 0, 0, 0,
                 50, 50, 50, 50, 50, 50, 50, 50,
                 10, 10, 20, 30, 30, 20, 10, 10,
                 5, 5, 10, 25, 25, 10, 5, 5,
                 0, 0, 0, 20, 20, 0, 0, 0,
                 5, -5, -10, 0, 0, -10, -5, 5,
                 5, 10, 10, -20, -20, 10, 10, 5,
                 0, 0, 0, 0, 0, 0, 0, 0],
    chess.KNIGHT: [-50, -40, -30, -30, -30, -30, -40, -50,
                   -40, -20, 0, 0, 0, 0, -20, -40,
                   -30, 0, 10, 15, 15, 10, 0, -30,
                   -30, 5, 15, 20, 20, 15, 5, -30,
                   -30, 0, 15, 20, 20, 15, 0, -30,
                   -30, 5, 10, 15, 15, 10, 5, -30,
                   -40, -20, 0, 5, 5, 0, -20, -40,
                   -50, -40, -30, -30, -30, -30, -40, -50],
    chess.BISHOP: [-20, -10, -10, -10, -10, -10, -10, -20,
                   -10, 0, 0, 0, 0, 0, 0, -10,
                   -10, 0, 5, 10, 10, 5, 0, -10,
                   -10, 5, 5, 10, 10, 5, 5, -10,
                   -10, 0, 10, 10, 10, 10, 0, -10,
                   -10, 10, 10, 10, 10, 10, 10, -10,
                   -10, 5, 0, 0, 0, 0, 5, -10,
                   -20, -10, -10, -10, -10, -10, -10, -20],
    chess.ROOK: [0, 0, 0, 0, 0, 0, 0, 0,
                 5, 10, 10, 10, 10, 10, 10, 5,
                 -5, 0, 0, 0, 0, 0, 0, -5,
                 -5, 0, 0, 0, 0, 0, 0, -5,
                 -5, 0, 0, 0, 0, 0, 0, -5,
                 -5, 0, 0, 0, 0, 0, 0, -5,
                 -5, 0, 0, 0, 0, 0, 0, -5,
                 0, 0, 0, 5, 5, 0, 0, 0],
    chess.QUEEN: [-20, -10, -10, -5, -5, -10, -10, -20,
                  -10, 0, 0, 0, 0, 0, 0, -10,
                  -10, 0, 5, 5, 5, 5, 0, -10,
                  -5, 0, 5, 5, 5, 5, 0, -5,
                  0, 0, 5, 5, 5, 5, 0, -5,
                  -10, 5, 5, 5, 5, 5, 0, -10,
                  -10, 0, 5, 0, 0, 0, 0, -10,
                  -20, -10, -10, -5, -5, -10, -10, -20],
    chess.KING: [-30, -40, -40, -50, -50, -40, -40, -30,
                 -30, -40, -40, -50, -50, -40, -40, -30,
                 -30, -40, -40, -50, -50, -40, -40, -30,
                 -30, -40, -40, -50, -50, -40, -40, -30,
                 -20, -30, -30, -40, -40, -30, -30, -20,
                 -10, -20, -20, -20, -20, -20, -20, -10,
                 20, 20, 0, 0, 0, 0, 20, 20,
                 20, 30, 10, 0, 0, 10, 30, 20],
}
MATE = 100000


def evaluate(board):
    """Static evaluation in centipawns from the side to move's view."""
    score = 0
    for sq, piece in board.piece_map().items():
        idx = chess.square_mirror(sq) if piece.color == chess.WHITE else sq
        v = VALUES[piece.piece_type] + _PST[piece.piece_type][idx]
        score += v if piece.color == chess.WHITE else -v
    return score if board.turn == chess.WHITE else -score


class _Timeout(Exception):
    pass


class BuiltinEngine(Engine):
    """Iterative-deepening negamax with alpha-beta and a capture-only
    quiescence search.  Level = depth (1..5, time-limited); level 1 also
    picks randomly among near-equal moves so it stays beatable."""
    name = "builtin"

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def choose_move(self, board, level=3, time_limit=2.0):
        board = board.copy(stack=False)
        moves = list(board.legal_moves)
        if len(moves) == 1:
            return moves[0]
        self._deadline = time.monotonic() + max(0.05, time_limit)
        best = moves[0]
        for depth in range(1, max(1, int(level)) + 1):
            try:
                scored = self._root(board, depth)
            except _Timeout:
                break
            scored.sort(key=lambda t: -t[0])
            best = scored[0][1]
            if level <= 1:
                near = [m for s, m in scored if s >= scored[0][0] - 60]
                best = self.rng.choice(near)
            if scored[0][0] >= MATE - 100:
                break
        return best

    def _root(self, board, depth):
        out = []
        alpha = -MATE - 1
        for move in self._ordered(board):
            board.push(move)
            score = -self._negamax(board, depth - 1, -MATE - 1, -alpha)
            board.pop()
            out.append((score, move))
            alpha = max(alpha, score)
        return out

    def _negamax(self, board, depth, alpha, beta):
        if time.monotonic() > self._deadline:
            raise _Timeout
        if board.is_checkmate():
            return -MATE + board.ply()
        if board.is_stalemate() or board.is_insufficient_material() or board.is_repetition(2):
            return 0
        if depth <= 0:
            return self._quiesce(board, alpha, beta, 4)
        for move in self._ordered(board):
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha)
            board.pop()
            if score >= beta:
                return beta
            alpha = max(alpha, score)
        return alpha

    def _quiesce(self, board, alpha, beta, depth):
        stand = evaluate(board)
        if stand >= beta or depth == 0:
            return stand
        alpha = max(alpha, stand)
        for move in self._ordered(board, captures_only=True):
            board.push(move)
            score = -self._quiesce(board, -beta, -alpha, depth - 1)
            board.pop()
            if score >= beta:
                return beta
            alpha = max(alpha, score)
        return alpha

    @staticmethod
    def _ordered(board, captures_only=False):
        """Captures first, most valuable victim / least valuable attacker."""
        def key(move):
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                v = VALUES[victim.piece_type] if victim else VALUES[chess.PAWN]
                return -(10 * v - VALUES[board.piece_at(move.from_square).piece_type] // 10)
            return 0 if move.promotion else 1
        moves = board.generate_legal_captures() if captures_only else board.legal_moves
        return sorted(moves, key=key)


# --------------------------------------------------------------------------
# factory
# --------------------------------------------------------------------------

def create_engine(spec="stockfish", fallback="builtin"):
    """"name" or "name:binary" or "name:key=value,key=value".

    stockfish[:/path]            leela[:weights=/w.pb.gz]      maia[:net_dir=/dir]
    lc0[:weights=/w.pb.gz,command=/path/lc0,nodes=1]          builtin    random
    uci:/path/to/any_uci_engine
    If a UCI engine cannot be started, `fallback` is used (None = raise)."""
    name, _, arg = spec.partition(":")
    kwargs = {}
    if "=" in arg:
        kwargs = dict(kv.split("=", 1) for kv in arg.split(","))
    elif arg:
        kwargs["command"] = arg
    try:
        if name == "stockfish":
            return stockfish(kwargs.get("command"))
        if name == "lc0":
            return lc0(kwargs.get("command"), kwargs.get("weights"), int(kwargs.get("nodes", 1)))
        if name == "leela":
            return leela(kwargs.get("command"), kwargs.get("weights"))
        if name == "maia":
            return maia(kwargs.get("command"), kwargs.get("net_dir"))
        if name == "uci":                     # any other UCI engine binary
            return UciEngine(os.path.expanduser(kwargs["command"]),
                             name=os.path.basename(kwargs["command"]))
        if name == "builtin":
            return BuiltinEngine()
        if name == "random":
            return RandomEngine()
        raise ValueError(f"unknown engine '{name}'")
    except (OSError, chess.engine.EngineError) as e:
        if fallback is None or fallback == name:
            raise
        print(f"[engine] cannot start '{spec}' ({e}) - using '{fallback}'")
        return create_engine(fallback, fallback=None)
