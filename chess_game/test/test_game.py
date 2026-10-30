import chess
import numpy as np
import pytest

from chess_game.frames import (OffBoardLayout, game_to_printed, logical_to_game,
                               printed_to_game, printed_to_logical)
from chess_game.game import (ChessGame, collapse_to_colour, labels_for, obs_from_labels)


def obs_after(game, *ucis, n=13, confidence=0.97):
    board = game.board.copy()
    for u in ucis:
        board.push_uci(u)
    return obs_from_labels(labels_for(board, game.human_white), n, confidence)


def started(fen=chess.STARTING_FEN, human_white=True, **kw):
    g = ChessGame(**kw)
    g.start(fen, human_white=human_white)
    return g


# ------------------------------------------------------------------ frames
def test_frames_round_trip_and_robot_corners():
    # printed a1 is the human's left corner, h8 the robot's right corner
    assert printed_to_game(chess.A1) == (7, 0)
    assert printed_to_game(chess.H8) == (0, 7)
    for sq in range(64):
        assert game_to_printed(*printed_to_game(sq)) == sq
    # playing black the logical board is rotated: logical a1 sits on printed h8
    assert printed_to_logical(chess.H8, human_white=False) == chess.A1
    assert logical_to_game(chess.E1, human_white=False) == printed_to_game(chess.D8)


def test_offboard_layout():
    lay = OffBoardLayout()
    assert lay.storage(0) == (0.0, -1.4)
    assert lay.storage(8) == (0.0, -2.5)
    with pytest.raises(IndexError):
        lay.storage(lay.storage_slots)
    assert lay.reserve("q")[1] == lay.reserve("N")[1] == 8.4


# ------------------------------------------------------------------ setup
@pytest.mark.parametrize("human_white", [True, False])
@pytest.mark.parametrize("n", [3, 13])
def test_check_setup_detects_side(human_white, n):
    board = chess.Board()
    obs = obs_from_labels(labels_for(board, human_white), n)
    res = ChessGame.check_setup(obs)
    assert res.ok and res.human_white == human_white


def test_check_setup_reports_swapped_knight_and_bishop():
    board = chess.Board()
    labels = labels_for(board, True)
    labels[chess.B1], labels[chess.C1] = labels[chess.C1], labels[chess.B1]
    res = ChessGame.check_setup(obs_from_labels(labels, 13))
    assert not res.ok and set(res.wrong) == {chess.B1, chess.C1}
    # a colour-only detector cannot see it
    assert ChessGame.check_setup(obs_from_labels(labels, 3)).ok


def test_start_from_observation_keeps_castling_rights():
    fen = "r3k2r/pppq1ppp/2n1bn2/3pp3/3PP3/2N1BN2/PPPQ1PPP/R3K2R w KQkq - 0 8"
    g = ChessGame()
    ok, _ = g.start_from_observation(obs_from_labels(labels_for(chess.Board(fen), True), 13))
    assert ok and g.board.board_fen() == chess.Board(fen).board_fen()
    assert g.board.has_kingside_castling_rights(chess.WHITE)
    assert g.board.has_queenside_castling_rights(chess.BLACK)


def test_start_from_observation_rejects_illegal_position():
    labels = labels_for(chess.Board(), True)
    labels[chess.E8] = 0                       # no black king
    ok, msg = ChessGame().start_from_observation(obs_from_labels(labels, 13))
    assert not ok and "king" in msg


# ------------------------------------------------------------------ human moves
@pytest.mark.parametrize("n", [3, 13])
@pytest.mark.parametrize("human_white", [True, False])
def test_infer_simple_move(n, human_white):
    g = started(human_white=human_white)
    if not human_white:
        g.board.push_uci("e2e4")              # the robot (white) moved first
    uci = "e2e4" if human_white else "e7e5"
    inf = g.infer_human_move(obs_after(g, uci, n=n))
    assert inf.kind == "move" and inf.move.uci() == uci


def test_no_change_and_noise_square():
    g = started()
    obs = obs_after(g)
    assert g.infer_human_move(obs).kind == "no_change"
    obs[chess.D5] = [0.01, 0.9] + [0.09 / 11] * 11     # one noisy square
    assert g.infer_human_move(obs).kind == "invalid"
    g.max_noise_squares = 1
    inf = g.infer_human_move(obs)
    assert inf.kind == "no_change" and inf.noise == [chess.D5]


def test_castling_en_passant_capture_are_recognised_colour_only():
    g = started("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
    assert g.infer_human_move(obs_after(g, "e1g1", n=3)).move.uci() == "e1g1"
    assert g.infer_human_move(obs_after(g, "e1c1", n=3)).move.uci() == "e1c1"
    g = started("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    assert g.infer_human_move(obs_after(g, "e5d6", n=3)).move.uci() == "e5d6"
    g = started("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1")
    assert g.infer_human_move(obs_after(g, "e4d5", n=3)).move.uci() == "e4d5"


def test_promotion_piece_from_classifier_or_queen_by_default():
    g = started("8/4P1k1/8/8/8/8/8/4K3 w - - 0 1")
    assert g.infer_human_move(obs_after(g, "e7e8n", n=13)).move.uci() == "e7e8n"
    assert g.infer_human_move(obs_after(g, "e7e8n", n=3)).move.uci() == "e7e8q"


def test_illegal_move_is_rejected_with_closest_move():
    g = started()
    labels = labels_for(g.board, True)
    labels[chess.E4], labels[chess.E2] = labels[chess.E2], 0
    labels[chess.E5], labels[chess.E4] = labels[chess.E4], 0     # pawn e2 -> e5
    inf = g.infer_human_move(obs_from_labels(labels, 13))
    assert inf.kind == "invalid"
    assert set(inf.changed) == {chess.E2, chess.E5}
    assert inf.closest is not None and inf.closest.from_square == chess.E2


def test_forgotten_capture_removal_is_invalid():
    # the human lifted the e4 pawn and has not put it down yet
    g = started("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1")
    labels = labels_for(g.board, True)
    labels[chess.E4] = 0                       # pawn lifted, nothing else changed
    inf = g.infer_human_move(obs_from_labels(labels, 13))
    assert inf.kind == "invalid"
    assert inf.closest.uci() in ("e4e5", "e4d5")


def test_unsure_when_classifier_cannot_decide():
    g = started()
    obs = obs_after(g, "g1f3", n=13)
    # f3 and h3 both half occupied -> knight to f3 or h3?
    obs[chess.F3] = [0.5, 0, 0.5] + [0] * 10
    obs[chess.H3] = [0.5, 0, 0.5] + [0] * 10
    assert g.infer_human_move(obs).kind == "unsure"


def test_mismatches_after_robot_move():
    g = started()
    g.push(chess.Move.from_uci("e2e4"))
    g.push(chess.Move.from_uci("e7e5"))
    obs = obs_after(g)
    assert g.mismatches(obs) == []
    obs[chess.E5] = obs[chess.E6].copy()       # robot left the pawn off e5
    assert g.mismatches(obs) == [chess.E5]
    assert g.mismatches(collapse_to_colour(obs)) == [chess.E5]


# ------------------------------------------------------------------ robot plans
def plan(fen, uci, human_white=False, **kw):
    g = started(fen, human_white=human_white, **kw)
    return [str(s) for s in g.plan_robot_move(chess.Move.from_uci(uci))]


def test_plan_quiet_move_and_capture():
    assert plan(chess.STARTING_FEN, "g1f3") == ["N: g1 -> f3"]
    assert plan("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5") == [
        "p: d5 -> storage#0", "P: e4 -> d5"]


def test_plan_castling_both_sides():
    fen = "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"
    assert plan(fen, "e8g8", human_white=True) == ["k: e8 -> g8", "r: h8 -> f8"]
    assert plan(fen, "e8c8", human_white=True) == ["k: e8 -> c8", "r: a8 -> d8"]


def test_plan_en_passant_removes_the_passed_pawn():
    assert plan("4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1", "d4e3", human_white=True) == [
        "P: e4 -> storage#0", "p: d4 -> e3"]


def test_plan_promotion_reserve_and_human():
    fen = "3r2k1/4P3/8/8/8/8/8/4K3 w - - 0 1"
    assert plan(fen, "e7d8q") == [
        "r: d8 -> storage#0", "P: e7 -> storage#1", "Q: reserve[Q] -> d8"]
    assert plan(fen, "e7e8q", promotion="human") == ["P: e7 -> e8"]


def test_storage_slots_count_up_over_the_game():
    g = started("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", human_white=False)
    move = chess.Move.from_uci("e4d5")
    steps = g.plan_robot_move(move)
    g.push(move, steps)
    assert g.storage_used == 1
    g.board.push_uci("e8e7")
    move = chess.Move.from_uci("e1e2")
    g.push(move, g.plan_robot_move(move))
    assert g.storage_used == 1
    assert g.log == ["1. exd5", "2. Ke2"]


def test_robot_move_of_human_piece_is_refused():
    g = started()
    with pytest.raises(ValueError):
        g.plan_robot_move(chess.Move.from_uci("e2e4"))


def test_labels_rotate_for_black():
    labels = labels_for(chess.Board(), human_white=False)
    assert labels[chess.H8] == labels_for(chess.Board(), True)[chess.A1]
    assert np.count_nonzero(labels) == 32
