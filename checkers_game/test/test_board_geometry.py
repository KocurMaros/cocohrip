"""Regression / robustness tests for checkers_game.camera.board_geometry.

These tests only need OpenCV + numpy - no ROS2, no camera, no Ximea SDK - so
they can run on a plain dev machine with `pytest checkers_game/test`
(or `python3 -m pytest ...`), which is exactly the situation this file was
written under (no robot/camera attached).

The single available real photo (fixtures/board_sample.jpg) is used as a
"gold" reference: we trust the detector's own output on the unmodified
photo (it has been visually verified - see the PR/commit description) as
ground truth, then synthetically perturb that photo (brightness, noise,
small rotations) and check the detector still finds corners close to where
they should be after applying the same known transform to the reference.

Known limitation (deliberately not asserted here): under a poorly-lit or
strongly rotated view the detector can lose precision or safely refuse
(return None) rather than guess. That's why the interactive verify/adjust
step in BoardDetection is not being removed - this test suite documents the
regime where full automation is trustworthy, not a claim that it always is.
"""

import os

import cv2
import numpy as np
import pytest

from checkers_game.camera import board_geometry as bg

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "board_sample.jpg")


@pytest.fixture(scope="module")
def sample_image():
    image = cv2.imread(FIXTURE_PATH)
    assert image is not None, f"could not load fixture image at {FIXTURE_PATH}"
    return image


@pytest.fixture(scope="module")
def reference_corners(sample_image):
    corners, msg = bg.find_board_corners(sample_image)
    assert corners is not None, f"reference detection must succeed, got: {msg}"
    return corners


def _board_scale(corners):
    """A rough board-size reference (TL-TR edge length) used to express
    corner error as a percentage instead of raw pixels, so tolerances are
    resolution independent.
    """
    return float(np.linalg.norm(corners[1] - corners[2]))


def _mean_corner_error_pct(detected, ground_truth, scale):
    err = np.linalg.norm(detected - ground_truth, axis=1)
    return float(err.mean() / scale * 100.0)


# --------------------------------------------------------------------------
# Basic contract
# --------------------------------------------------------------------------

def test_returns_4x2_array_in_bl_tl_tr_br_order(reference_corners):
    assert reference_corners.shape == (4, 2)
    bl, tl, tr, br = reference_corners
    # TL must be left of TR and above BR/BL; BL must be left of BR.
    assert tl[0] < tr[0]
    assert tl[1] < bl[1]
    assert tl[1] < br[1]
    assert bl[0] < br[0]


def test_reference_quad_is_plausible(sample_image, reference_corners):
    h, w = sample_image.shape[:2]
    xs = reference_corners[:, 0]
    ys = reference_corners[:, 1]
    # Board should occupy a large, sane fraction of the frame (this photo's
    # board fills most of it) and stay within image bounds with only a
    # small allowance for the detector overshooting by a few pixels.
    assert -5 <= xs.min() and xs.max() <= w + 5
    assert -5 <= ys.min() and ys.max() <= h + 5
    assert (xs.max() - xs.min()) > 0.5 * w
    assert (ys.max() - ys.min()) > 0.5 * h


def test_order_corners_is_idempotent_under_shuffle(reference_corners):
    shuffled = reference_corners[[2, 0, 3, 1]]
    reordered = bg.order_corners(shuffled)
    np.testing.assert_allclose(reordered, reference_corners, atol=1e-3)


def test_never_raises_on_garbage_input():
    noise_image = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
    corners, msg = bg.find_board_corners(noise_image)
    assert corners is None
    assert isinstance(msg, str) and msg


def test_empty_image_is_handled():
    corners, msg = bg.find_board_corners(np.zeros((0, 0, 3), dtype=np.uint8))
    assert corners is None


# --------------------------------------------------------------------------
# Robustness to lighting (the realistic day-to-day variation this camera
# will actually see)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("beta", [20, 40])
def test_robust_to_brightening(sample_image, reference_corners, beta):
    scale = _board_scale(reference_corners)
    brightened = cv2.convertScaleAbs(sample_image, alpha=1.0, beta=beta)
    detected, msg = bg.find_board_corners(brightened)
    assert detected is not None, msg
    err_pct = _mean_corner_error_pct(detected, reference_corners, scale)
    assert err_pct < 2.0, f"brightness+{beta}: corner error {err_pct:.1f}% too high"


@pytest.mark.parametrize("beta", [-20, -30])
def test_robust_to_darkening(sample_image, reference_corners, beta):
    scale = _board_scale(reference_corners)
    darkened = cv2.convertScaleAbs(sample_image, alpha=1.0, beta=beta)
    detected, msg = bg.find_board_corners(darkened)
    assert detected is not None, msg
    err_pct = _mean_corner_error_pct(detected, reference_corners, scale)
    assert err_pct < 10.0, f"brightness{beta}: corner error {err_pct:.1f}% too high"


@pytest.mark.parametrize("sigma", [8, 15])
def test_robust_to_sensor_noise(sample_image, reference_corners, sigma):
    rng = np.random.RandomState(42)
    scale = _board_scale(reference_corners)
    noisy = sample_image.astype(np.float32) + rng.normal(0, sigma, sample_image.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    detected, msg = bg.find_board_corners(noisy)
    assert detected is not None, msg
    err_pct = _mean_corner_error_pct(detected, reference_corners, scale)
    assert err_pct < 12.0, f"noise sigma={sigma}: corner error {err_pct:.1f}% too high"


# --------------------------------------------------------------------------
# Robustness to small in-plane rotation (mounting/setup variation)
# --------------------------------------------------------------------------

def _rotate(image, angle_deg):
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]
    rotated = cv2.warpAffine(image, matrix, (new_w, new_h),
                              borderMode=cv2.BORDER_REFLECT101)
    return rotated, matrix


def _transform_points(points, matrix):
    ones = np.ones((points.shape[0], 1))
    homogeneous = np.hstack([points, ones])
    return (matrix @ homogeneous.T).T


@pytest.mark.parametrize("angle", [-5, 5, 10])
def test_small_rotation_either_succeeds_accurately_or_safely_refuses(
        sample_image, reference_corners, angle):
    """A modest camera-mount rotation should never produce a confidently
    wrong answer: it's acceptable to refuse (None), but if it *does* return
    corners, they must actually be close to correct.
    """
    scale = _board_scale(reference_corners)
    rotated, matrix = _rotate(sample_image, angle)
    ground_truth = bg.order_corners(_transform_points(reference_corners, matrix))

    detected, msg = bg.find_board_corners(rotated)
    if detected is None:
        pytest.skip(f"angle={angle}: safely refused ({msg}) - acceptable")

    err_pct = _mean_corner_error_pct(detected, ground_truth, scale)
    assert err_pct < 15.0, f"angle={angle}: corner error {err_pct:.1f}% too high"
