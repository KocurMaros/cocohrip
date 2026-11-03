import os
from collections import Counter

import chess
import cv2
import numpy as np
import pytest

from chess_game.dataset import (FULL_SET, ChangeDetector, DatasetPlanner, check_change,
                                label_record, placement_fen, placement_from_fen)
from chess_game.vision.board_model import BoardModel, load_intrinsics

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "fixtures", "start_swapped_80ms_half.jpg")
CALIB = os.path.join(HERE, "..", "chess_game", "vision", "ximea_calibration_600.npz")
# corners of the 8x8 squares in the fixture (find_board_corners, BL TL TR BR)
FIXTURE_CORNERS = [[160.7, 916.6], [153.7, 21.4], [955.0, 73.0], [1085.6, 860.9]]


def run_planner(planner, steps):
    seen = []
    for _ in range(steps):
        ins = planner.next()
        assert ins.before == planner.placement
        planner.accept()
        planner.record(ins.after)
        seen.append(ins)
    return seen


def test_planner_steps_are_small_and_consistent():
    planner = DatasetPlanner(seed=3)
    seen = run_planner(planner, 400)
    assert seen[0].kind == "reset" and seen[0].after == {}
    kinds = Counter(i.kind for i in seen)
    assert kinds["game"] > 50 and kinds["scatter"] > 50 and kinds["reset"] >= 3
    for ins in seen:
        if ins.kind != "reset":
            # one move = at most 4 squares (castling), usually 2
            assert 1 <= len(ins.changed) <= 4, ins.text
        # never more pieces than a chess set has
        assert not Counter(ins.after.values()) - FULL_SET
    # both orientations of the start position are used
    starts = [i.after for i in seen if i.kind == "reset" and i.after]
    assert any(p.get(chess.A1) == "R" for p in starts)
    assert any(p.get(chess.A1) == "r" for p in starts)


def test_planner_covers_many_piece_square_pairs():
    planner = DatasetPlanner(seed=5)
    run_planner(planner, 600)
    assert len(planner.coverage) > 450          # of 768 (12 pieces x 64 squares)


def test_game_instruction_text_uses_printed_squares_when_black_is_near():
    planner = DatasetPlanner(seed=1)
    run_planner(planner, 1)                     # empty board
    planner._resets = 1                         # next start: black on the human side
    ins = planner.next()
    assert ins.after[chess.A1] == "r" and ins.after[chess.D1] == "k"
    planner.accept()
    ins = planner.next()
    if ins.kind == "game":
        board = chess.Board()
        move = ins.move
        printed = chess.square_name(63 - move.from_square)
        assert printed in ins.text and board.piece_at(move.from_square) is not None


def test_placement_fen_round_trip():
    placement = {chess.E4: "P", chess.H8: "k", chess.A1: "Q"}
    assert placement_from_fen(placement_fen(placement)) == placement


def test_label_record_is_json_ready():
    rec = label_record("x.jpg", {chess.E1: "K"}, np.zeros((4, 2)), 40000.0, "room", 3, "t",
                       (2464, 2056), "BR", kind="scatter")
    assert rec["placement"] == "8/8/8/8/8/8/8/4K3" and rec["kind"] == "scatter"


# ------------------------------------------------------------------ real frame
@pytest.fixture(scope="module")
def model():
    img = cv2.imread(FIXTURE)
    K, dist = load_intrinsics(CALIB, (img.shape[1], img.shape[0]))
    return img, BoardModel(FIXTURE_CORNERS, K, dist, a1_corner="BR")


def test_camera_pose_is_above_the_a8_corner(model):
    _, m = model
    x, y, z = m.camera_position
    assert x < 0 and y > 8 and 12 < z < 18          # ~86 cm above the board
    assert m.reprojection_error < 6


def test_square_centres_land_on_the_pieces(model):
    img, m = model
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # white king e1 is bright, black king e8 is dark, e4 is an empty dark square
    def patch(sq):
        x, y = m.square_centre(sq).astype(int)
        return gray[y - 4:y + 5, x - 4:x + 5].mean()
    assert patch(chess.E1) > 120
    assert patch(chess.E8) < 70


def test_column_crops_are_upright_and_tall_pieces_cover_more_squares(model):
    img, m = model
    crop = m.column_crop(img, chess.E1)
    assert crop.shape == (128, 72, 3)
    # pieces lean away from the camera (above a8): a tall one on d4 hides e3
    assert m.column_squares(chess.D4, height=3) == [chess.E3, chess.D4]
    assert m.column_squares(chess.H1, height=3) == [chess.H1]      # leans off the board


def test_change_detector_sees_a_moved_piece(model):
    img, m = model
    det = ChangeDetector(m, img.shape)
    assert det.changed(img, img)[0] == []
    # "move" the white pawn e2 to e4: paste e2's centre onto e4, e2 gets d2-like... use
    # the empty e3 square's appearance for e2
    cur = img.copy()
    for src, dst in ((chess.E2, chess.E4), (chess.E3, chess.E2)):
        ps = m.square_polygon(src).astype(np.float32)
        pd = m.square_polygon(dst).astype(np.float32)
        H = cv2.getPerspectiveTransform(ps, pd)
        warped = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, m.square_polygon(dst, shrink=0.1).astype(np.int32), 1)
        cur[mask > 0] = warped[mask > 0]
    changed, _ = det.changed(img, cur)
    assert set(changed) == {chess.E2, chess.E4}


def test_check_change_messages():
    from chess_game.dataset import Instruction
    ins = Instruction("t", {chess.E2: "P"}, {chess.E4: "P"})
    assert check_change(ins, [chess.E2, chess.E4], {chess.E2, chess.E4})[0]
    ok, msg = check_change(ins, [chess.E2], set())
    assert not ok and "e4" in msg
    ok, msg = check_change(ins, [chess.E2, chess.E4, chess.A7], {chess.E2, chess.E4})
    assert not ok and "a7" in msg
