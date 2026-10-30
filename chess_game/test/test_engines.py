import shutil

import chess
import pytest

from chess_game.engines import BuiltinEngine, RandomEngine, create_engine, evaluate


def test_evaluate_is_symmetric():
    assert evaluate(chess.Board()) == 0
    board = chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
    assert evaluate(board) > 800
    board.turn = chess.BLACK
    assert evaluate(board) < -800


def test_builtin_finds_mate_in_one_and_wins_material():
    eng = BuiltinEngine(seed=1)
    # back-rank mate
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    assert eng.choose_move(board, level=2, time_limit=5).uci() == "a1a8"
    # hanging queen
    board = chess.Board("4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1")
    assert eng.choose_move(board, level=2, time_limit=5).uci() == "d2d5"


def test_builtin_respects_time_limit_and_returns_legal_move():
    import time
    board = chess.Board()
    t = time.monotonic()
    move = BuiltinEngine().choose_move(board, level=5, time_limit=0.3)
    assert move in board.legal_moves
    assert time.monotonic() - t < 2.0


def test_random_and_factory_fallback():
    board = chess.Board()
    assert RandomEngine(seed=0).choose_move(board) in board.legal_moves
    eng = create_engine("stockfish:/nonexistent/stockfish")
    assert isinstance(eng, BuiltinEngine)
    assert isinstance(create_engine("maia:net_dir=/nonexistent"), BuiltinEngine)
    assert isinstance(create_engine("leela:command=/nonexistent/lc0"), BuiltinEngine)
    with pytest.raises(ValueError):
        create_engine("nonsense")


@pytest.mark.skipif(shutil.which("stockfish") is None and
                    not shutil.os.path.exists("/usr/games/stockfish"),
                    reason="stockfish not installed")
def test_stockfish_plays_every_level():
    board = chess.Board()
    with create_engine("stockfish", fallback=None) as eng:
        for level in (1, 3, 5):
            assert eng.choose_move(board, level=level, time_limit=0.2) in board.legal_moves
