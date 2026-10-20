"""End-to-end (synthetic) test of the vision + rules pipeline:
rendered camera frames -> board corners -> orientation -> piece grid ->
human-move inference.  No camera / ROS needed.

    cd checkers_game && python3 -m pytest test/test_board_vision.py -q
"""
import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

import synthetic_board as sb  # noqa: E402
from checkers_game.camera import board_geometry as bg  # noqa: E402
from checkers_game.camera import board_vision as bv  # noqa: E402
from checkers_game.camera.board_detection import BoardDetection  # noqa: E402
from checkers_game.checkers.game import ROBOT, Game  # noqa: E402
from checkers_game.checkers.rules import GameState  # noqa: E402


def make_detector(image, corners=None):
    det = BoardDetection(None, interactive=False)
    if corners is None:
        corners, msg = bg.find_board_corners(image)
        assert corners is not None, msg
    det.bounderies = corners
    ok, info = det.calibrate_from_start(image)
    assert ok, info
    return det


@pytest.mark.parametrize("rotate_k", [0, 1, 2, 3])
def test_orientation_and_start_classification(rotate_k):
    rng = np.random.RandomState(rotate_k)
    start = GameState.initial(1).to_grid()
    corners = sb.random_corners(rng)
    img = sb.to_camera(sb.render_topdown(start, rng=rng), corners, rotate_k=rotate_k, rng=rng)
    det = make_detector(img)
    assert det.dark_parity == 1
    grid = det.get_board(img) if False else det.analyse(img)["grid"]
    ok, msg, _ = Game.check_standard_start(grid)
    assert ok, f"{msg}\n{grid}"


def _play(det, game, rng, lighting, plies=30):
    """Robot = engine; human = random legal move; every human move goes
    through the camera pipeline.  Returns number of human moves checked."""
    checked = 0
    corners = det._corners_truth
    for _ in range(plies):
        if game.winner() is not None:
            break
        if game.turn == ROBOT:
            plan = game.plan_robot_turn(level=1, time_limit=0.2, rng=rng)
            game.apply_robot(plan)
            frame = sb.to_camera(sb.render_topdown(game.expected_grid(), rng=np.random.RandomState(rng.randint(0, 10**6))),
                                 corners, rotate_k=det._k, rng=np.random.RandomState(rng.randint(0, 10**6)),
                                 **lighting(rng))
            res = det.analyse(frame, expected=game.expected_grid())
            assert game.mismatches(res["grid"], use_color=False) == []
            det.set_reference(frame, game.expected_grid())
            continue
        m = rng.choice(game.state.legal_moves())
        truth = game.state.apply(m)
        frame = sb.to_camera(sb.render_topdown(truth.to_grid(), rng=np.random.RandomState(rng.randint(0, 10**6))),
                             corners, rotate_k=det._k, rng=np.random.RandomState(rng.randint(0, 10**6)),
                             **lighting(rng))
        grid = det.analyse(frame)["grid"]
        inf = game.infer_human_move(grid)
        assert inf.accepted, f"{inf.kind}: {inf.reason}\n{np.array(grid)}\nmove {m}"
        assert game.state.apply(inf.move) == truth
        game.apply_human(inf)
        det.set_reference(frame, game.expected_grid())
        checked += 1
    return checked


def _setup(seed, k):
    rng = np.random.RandomState(seed)
    corners = sb.random_corners(rng)
    start = GameState.initial(1).to_grid()
    img = sb.to_camera(sb.render_topdown(start, rng=rng), corners, rotate_k=k, rng=rng)
    det = make_detector(img)
    det._corners_truth, det._k = corners, k
    game = Game(human_first=False)
    game.start_standard()
    det.set_reference(img, game.expected_grid())
    return det, game


def test_game_constant_lighting():
    det, game = _setup(1, 2)
    n = _play(det, game, random.Random(1), lambda r: dict(gain=1.0))
    assert n >= 10


def test_game_changing_lighting():
    """lamp on/off, side light, noise between every move"""
    det, game = _setup(2, 1)

    def lighting(r):
        return dict(gain=r.uniform(0.55, 1.35), gradient=r.uniform(-0.3, 0.3),
                    noise=r.uniform(2, 6), blur=r.uniform(0.6, 1.4))
    n = _play(det, game, random.Random(2), lighting)
    assert n >= 10


def test_black_piece_nearly_invisible_still_tracked():
    """black pieces only 6 grey levels above the dark squares"""
    rng = np.random.RandomState(5)
    corners = sb.random_corners(rng)
    start = GameState.initial(1).to_grid()
    td = sb.render_topdown(start, black_piece_level=28, rng=rng)
    img = sb.to_camera(td, corners, rotate_k=0, rng=rng)
    det = make_detector(img)
    game = Game(human_first=True)
    game.start_standard()
    det.set_reference(img, game.expected_grid())
    # human captures need the removed black piece to be noticed
    r = random.Random(3)
    for _ in range(20):
        if game.winner() is not None:
            break
        if game.turn == ROBOT:
            game.apply_robot(game.plan_robot_turn(level=1, time_limit=0.2, rng=r))
            f = sb.to_camera(sb.render_topdown(game.expected_grid(), black_piece_level=28,
                                               rng=np.random.RandomState(r.randint(0, 10**6))),
                             corners, rng=np.random.RandomState(r.randint(0, 10**6)))
            det.set_reference(f, game.expected_grid())
            continue
        m = r.choice(game.state.legal_moves())
        truth = game.state.apply(m)
        f = sb.to_camera(sb.render_topdown(truth.to_grid(), black_piece_level=28,
                                           rng=np.random.RandomState(r.randint(0, 10**6))),
                         corners, rng=np.random.RandomState(r.randint(0, 10**6)))
        inf = game.infer_human_move(det.analyse(f)["grid"])
        assert inf.accepted and game.state.apply(inf.move) == truth, inf.reason
        game.apply_human(inf)
        det.set_reference(f, game.expected_grid())


def test_change_scores_ignore_global_lighting():
    rng = np.random.RandomState(9)
    start = GameState.initial(1).to_grid()
    td = sb.render_topdown(start, rng=rng)
    a = td
    b = np.clip(td.astype(np.float32) * 0.6, 0, 255).astype(np.uint8)
    na, _, _ = bv.normalise(bv.to_gray(a), 1)
    nb, _, _ = bv.normalise(bv.to_gray(b), 1)
    z, raw = bv.change_scores(na, nb)
    assert z.max() < 3.0


# board corners in the real camera image: the far edge is ~28 % shorter
REAL_CORNERS = np.array([[119, 541], [50, 84], [497, 41], [507, 536]], np.float32)


@pytest.mark.parametrize("rotate_k, corners", [(0, None), (1, None), (2, None), (3, None),
                                               (0, REAL_CORNERS)])
def test_piece_offset_measures_pushed_piece(rotate_k, corners):
    """A pushed piece is measured to a few px (1 px ~ 0.6 mm) despite parallax."""
    rng = np.random.RandomState(rotate_k)
    start = GameState.initial(1).to_grid()
    if corners is None:
        corners = sb.random_corners(rng)
    parallax = {(r, c): (c - 4, round(1.5 * r) - 12) for r in range(8) for c in range(8)}

    def shot(centres):
        return sb.to_camera(sb.render_topdown(start, centres=centres), corners,
                            rotate_k=rotate_k, rng=rng)

    det = make_detector(shot(parallax), corners)
    det.learn_piece_centres(shot(parallax), start)
    for (r, c), (sx, sy) in {(5, 0): (12, 10), (2, 3): (-18, 5), (6, 5): (0, -20)}.items():
        pushed = dict(parallax)
        pushed[(r, c)] = (parallax[(r, c)][0] + sx, parallax[(r, c)][1] + sy)
        off_row, off_col = det.piece_offset(shot(pushed), r, c)
        assert abs(off_col * bv.CELL - sx) <= 4 and abs(off_row * bv.CELL - sy) <= 4
