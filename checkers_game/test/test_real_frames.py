"""Detection tests on REAL camera frames from the robot cell (recorded
2026-10-15 with tools/capture_dataset.py; plain black pieces, wooden pieces).

  start_L1_80ms   start position, room light + lamp, old fixed 80 ms exposure
                  (white squares saturated)
  start_L4_40ms   start position, lights off, 40 ms (dark squares ~10 grey)
  midgame_*       mid-game position incl. 2 kings (stacked), labels.json
  empty_L4_80ms   empty board
  video_00XX      play recording: 0 start, 11 after WHITE (5,2)-(4,3),
                  21 after BLACK (2,5)-(3,4), 33 after WHITE (5,4)-(4,5)

    cd checkers_game && python3 -m pytest test/test_real_frames.py -q
"""
import json
import os

import cv2
import numpy as np
import pytest

from checkers_game.camera import board_geometry as bg
from checkers_game.camera import board_vision as bv
from checkers_game.camera.board_detection import BoardDetection
from checkers_game.checkers.game import Game
from checkers_game.checkers.rules import WHITE, GameState, Move

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "real")


def img(name):
    im = cv2.imread(os.path.join(FIX, name))
    assert im is not None, name
    return im


def midgame_truth():
    lab = json.load(open(os.path.join(FIX, "labels.json")))["midgame"]
    g = np.zeros((8, 8), int)
    for r, c in lab["white"]:
        g[r, c] = 1
    for r, c in lab["black"]:
        g[r, c] = 2
    return g


def detector_for(name):
    im = img(name)
    corners, msg = bg.find_board_corners(im)
    assert corners is not None, msg
    det = BoardDetection(None, interactive=False)
    det.bounderies = corners
    ok, info = det.calibrate_from_start(im)
    assert ok, info
    return det


@pytest.mark.parametrize("name", ["start_L1_80ms.jpg", "start_L4_40ms.jpg", "midgame_L4_80ms.jpg",
                                  "midgame_L1_40ms.jpg", "empty_L4_80ms.jpg", "video_0000.jpg"])
def test_board_corners_found(name):
    corners, msg = bg.find_board_corners(img(name))
    assert corners is not None, msg


def test_start_orientation_is_found_and_empty_board_is_not_a_start():
    for name in ("start_L1_80ms.jpg", "start_L4_40ms.jpg", "video_0000.jpg"):
        im = img(name)
        c, _ = bg.find_board_corners(im)
        rot, par, _ = bv.find_start_orientation(im, c)
        assert rot == 1 and par == 1, name
    im = img("empty_L4_80ms.jpg")
    c, _ = bg.find_board_corners(im)
    rot, _, _ = bv.find_start_orientation(im, c)
    assert rot is None


@pytest.mark.parametrize("calib", ["start_L1_80ms.jpg", "start_L4_40ms.jpg"])
@pytest.mark.parametrize("test", ["start_L1_80ms.jpg", "start_L4_40ms.jpg",
                                  "midgame_L4_80ms.jpg", "midgame_L1_40ms.jpg"])
def test_pieces_across_lighting(calib, test):
    """Calibrate under one lighting, classify under another: never a wrong
    label (an 'unknown' is allowed - the temporal logic / rules handle it)."""
    det = detector_for(calib)
    truth = (np.array(GameState.initial(1).to_grid()) if test.startswith("start")
             else midgame_truth())
    grid = det.analyse(img(test))["grid"]
    wrong, unknown = [], []
    for r in range(8):
        for c in range(8):
            if not bv.is_dark(r, c, det.dark_parity):
                continue
            g, t = int(grid[r, c]), int(truth[r, c])
            if g in (bv.UNKNOWN, bv.OCCUPIED):
                if g == bv.UNKNOWN or t == 0:
                    unknown.append((r, c))
                continue
            if g != t:
                wrong.append((r, c, t, g))
    assert wrong == [], wrong
    assert len(unknown) <= 1, unknown


def test_start_check_passes_on_real_start_frames():
    det = detector_for("start_L1_80ms.jpg")
    for name in ("start_L1_80ms.jpg", "start_L4_40ms.jpg"):
        ok, msg, _ = Game.check_standard_start(det.analyse(img(name))["grid"])
        assert ok, f"{name}: {msg}"


def test_moves_from_real_video():
    det = detector_for("video_0000.jpg")
    g = Game(human_first=True)                  # in the recording white moved first
    g.start_standard(dark_parity=det.dark_parity)
    det.set_reference(img("video_0000.jpg"), g.expected_grid())
    expected = [("video_0011.jpg", Move(((5, 2), (4, 3)))),
                ("video_0021.jpg", Move(((2, 5), (3, 4))))]
    for name, mv in expected:
        inf = g.infer_move(det.analyse(img(name))["grid"])
        assert inf.accepted and inf.move == mv, (name, inf.kind, inf.reason)
        g.apply_human(inf)
        det.set_reference(img(name), g.expected_grid())
    # frame 33: white (5,4)-(4,5) although (4,3)x(3,4) was mandatory -> huff
    inf = g.infer_move(det.analyse(img("video_0033.jpg"))["grid"])
    assert inf.kind == "huff" and inf.move == Move(((5, 4), (4, 5)))
    assert inf.huff_squares == [(4, 3)]
    assert g.turn == WHITE
