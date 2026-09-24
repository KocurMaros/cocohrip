"""Tests for checkers_game.camera.board_geometry (OpenCV + numpy only - no
ROS2, camera or Ximea SDK needed):

    cd checkers_game && python3 -m pytest test/test_board_geometry.py

Ground truth is the hand-verified outline of the 8x8 squares in
fixtures/board_sample.jpg (board_augment.REFERENCE_CORNERS). The synthetic
samples move the "camera" (rotation, scale, shift, perspective tilt) and
change the lighting, transforming the ground truth along with the image.

Accuracy is measured in board cells: a detection is correct when every
corner is within 0.15 of a square's width of the truth. Landing on the
board's outer frame instead of the squares is ~0.3-1 cells off, so it
fails this check.
"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

import board_augment  # noqa: E402
from checkers_game.camera import board_geometry as bg  # noqa: E402

TOLERANCE_CELLS = 0.15


@pytest.fixture(scope="module")
def sample_image():
    image = cv2.imread(board_augment.FIXTURE_PATH)
    assert image is not None, f"could not load {board_augment.FIXTURE_PATH}"
    return image


def corner_error_cells(found, truth):
    truth = bg.order_corners(truth)
    cell = np.linalg.norm(truth[1] - truth[2]) / 8.0
    return float(np.linalg.norm(found - truth, axis=1).max() / cell)


def run_batch(image, count, lighting):
    correct = wrong = refused = 0
    for _, im, truth in board_augment.samples(image, count, lighting):
        found, _ = bg.find_board_corners(im)
        if found is None:
            refused += 1
        elif corner_error_cells(found, truth) < TOLERANCE_CELLS:
            correct += 1
        else:
            wrong += 1
    return correct, wrong, refused


def test_reference_photo_finds_squares_not_outer_frame(sample_image):
    found, msg = bg.find_board_corners(sample_image)
    assert found is not None, msg
    assert corner_error_cells(found, board_augment.REFERENCE_CORNERS) < 0.05


def test_corner_order_is_bl_tl_tr_br(sample_image):
    found, _ = bg.find_board_corners(sample_image)
    bl, tl, tr, br = found
    assert tl[0] < tr[0] and bl[0] < br[0]
    assert tl[1] < bl[1] and tr[1] < br[1]


def test_order_corners_is_idempotent_under_shuffle():
    ref = bg.order_corners(board_augment.REFERENCE_CORNERS)
    np.testing.assert_allclose(bg.order_corners(ref[[2, 0, 3, 1]]), ref, atol=1e-3)


def test_accepts_bgra_and_gray_input(sample_image):
    """The Ximea camera is configured for XI_RGB32, i.e. 4 channels."""
    for im in (cv2.cvtColor(sample_image, cv2.COLOR_BGR2BGRA),
               cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)):
        found, msg = bg.find_board_corners(im)
        assert found is not None, msg
        assert corner_error_cells(found, board_augment.REFERENCE_CORNERS) < 0.05


def test_high_resolution_input(sample_image):
    big = cv2.resize(sample_image, None, fx=2.5, fy=2.5)
    found, msg = bg.find_board_corners(big)
    assert found is not None, msg
    assert corner_error_cells(found, board_augment.REFERENCE_CORNERS * 2.5) < 0.05


def test_refuses_garbage_instead_of_guessing():
    rng = np.random.RandomState(0)
    found, msg = bg.find_board_corners(rng.randint(0, 255, (480, 480, 3), dtype=np.uint8))
    assert found is None and msg
    found, _ = bg.find_board_corners(np.zeros((0, 0, 3), dtype=np.uint8))
    assert found is None


def test_camera_moves_with_everyday_lighting(sample_image):
    correct, wrong, refused = run_batch(sample_image, 60, "mild")
    assert wrong == 0, f"{wrong}/60 confidently wrong"
    assert correct >= 58, f"only {correct}/60 correct ({refused} refused)"


def test_camera_moves_with_harsh_lighting(sample_image):
    """Includes near-black and heavily shadowed frames, so not everything is
    expected to pass - but wrong answers must stay rare."""
    correct, wrong, refused = run_batch(sample_image, 80, "harsh")
    assert correct >= 0.85 * 80, f"only {correct}/80 correct ({wrong} wrong, {refused} refused)"
    assert wrong <= 0.08 * 80, f"{wrong}/80 confidently wrong"
