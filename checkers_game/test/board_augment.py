"""Synthetic variations of the reference board photo, for testing
board_geometry.find_board_corners without a camera.

Each sample applies a random (seeded, reproducible) camera move - rotation,
scale, translation and perspective tilt - plus lighting changes, and
returns the ground-truth board corners transformed the same way.

Run directly to dump a gallery of samples with the ground truth (green) and
the detection (red) drawn on them:

    python3 test/board_augment.py --out /tmp/board_samples --count 40
"""

import argparse
import os
import sys

import cv2
import numpy as np

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "board_sample.jpg")

# Hand-verified outline of the 8x8 squares in fixtures/board_sample.jpg
# (inner edge of the frame, NOT the board's outer edge), BL, TL, TR, BR.
REFERENCE_CORNERS = np.array([
    [42.77, 555.03],
    [19.35, 22.55],
    [428.04, 6.67],
    [507.41, 479.61],
])


def random_camera_move(rng, w, h):
    angle = rng.uniform(-25, 25)
    scale = rng.uniform(0.75, 1.15)
    tx, ty = rng.uniform(-0.08, 0.08, 2) * np.array([w, h])
    rot = np.vstack([cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale), [0, 0, 1]])
    shift = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jitter = rng.uniform(-0.06, 0.06, (4, 2)) * np.array([w, h])
    tilt = cv2.getPerspectiveTransform(src, (src + jitter).astype(np.float32))
    return shift @ tilt @ rot


def mild_lighting(rng, img):
    """Everyday variation: exposure, contrast, white balance, sensor noise."""
    out = cv2.convertScaleAbs(img, alpha=rng.uniform(0.7, 1.3), beta=rng.uniform(-35, 35))
    out = np.clip(out.astype(np.float32) * rng.uniform(0.85, 1.15, 3), 0, 255)
    out = out + rng.normal(0, rng.uniform(0, 8), out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def harsh_lighting(rng, img):
    """Stress test: strong gamma, colour cast, uneven light, shadows, blur,
    heavy noise and JPEG artefacts - deliberately worse than the lab."""
    im = img.astype(np.float32) / 255.0
    h, w = im.shape[:2]
    im = im ** np.exp(rng.uniform(np.log(0.45), np.log(2.2)))
    im = im * rng.uniform(0.7, 1.3, 3)[None, None, :]
    im = (im - 0.5) * rng.uniform(0.55, 1.5) + 0.5 + rng.uniform(-0.2, 0.2)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    kind = rng.randint(3)
    if kind == 0:
        a = rng.uniform(0, 2 * np.pi)
        ramp = np.cos(a) * xx / w + np.sin(a) * yy / h
        ramp = (ramp - ramp.min()) / (np.ptp(ramp) + 1e-6)
        im *= (rng.uniform(0.45, 0.9) + ramp * rng.uniform(0.2, 0.8))[..., None]
    elif kind == 1:
        cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(w, h)
        im *= (1.0 - rng.uniform(0.3, 0.8) * r ** 2 * 2)[..., None].clip(0.2, 1)
    if rng.rand() < 0.5:
        mask = np.zeros((h, w), np.float32)
        pts = (rng.uniform(0, 1, (rng.randint(3, 6), 2)) * [w, h]).astype(np.int32)
        cv2.fillPoly(mask, [cv2.convexHull(pts)], 1.0)
        mask = cv2.GaussianBlur(mask, (0, 0), rng.uniform(3, 25))
        im *= (1 - mask * rng.uniform(0.2, 0.55))[..., None]
    out = (np.clip(im, 0, 1) * 255).astype(np.uint8)
    k = rng.choice([0, 0, 3, 5, 7])
    if k:
        out = cv2.GaussianBlur(out, (k, k), 0)
    out = np.clip(out.astype(np.float32) + rng.normal(0, rng.uniform(0, 18), out.shape), 0, 255)
    quality = int(rng.uniform(35, 95))
    encoded = cv2.imencode('.jpg', out.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def make_sample(img, seed, lighting="mild", corners=REFERENCE_CORNERS):
    """Returns (image, ground_truth_corners) or None if this seed would move
    the board (partly) out of the frame."""
    rng = np.random.RandomState(seed)
    h, w = img.shape[:2]
    H = random_camera_move(rng, w, h)
    border = [cv2.BORDER_REFLECT101, cv2.BORDER_REPLICATE, cv2.BORDER_CONSTANT][rng.randint(3)]
    fill = tuple(int(v) for v in rng.randint(0, 120, 3))
    out = cv2.warpPerspective(img, H, (w, h), borderMode=border, borderValue=fill)
    gt = cv2.perspectiveTransform(np.asarray(corners, np.float64).reshape(-1, 1, 2), H).reshape(-1, 2)
    if (gt < 4).any() or (gt[:, 0] > w - 4).any() or (gt[:, 1] > h - 4).any():
        return None
    if lighting == "mild":
        out = mild_lighting(rng, out)
    elif lighting == "harsh":
        out = harsh_lighting(rng, out)
    return out, gt


def samples(img, count, lighting="mild", first_seed=0):
    """Yield `count` (seed, image, ground_truth) samples."""
    seed, n = first_seed, 0
    while n < count:
        s = make_sample(img, seed, lighting)
        if s is not None:
            n += 1
            yield (seed,) + s
        seed += 1


def main():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from checkers_game.camera.board_geometry import find_board_corners, order_corners

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--lighting", choices=["none", "mild", "harsh"], default="mild")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    img = cv2.imread(FIXTURE_PATH)
    for seed, im, gt in samples(img, args.count, args.lighting):
        found, msg = find_board_corners(im)
        vis = im.copy()
        cv2.polylines(vis, [order_corners(gt).astype(np.int32)], True, (0, 255, 0), 2)
        if found is not None:
            cv2.polylines(vis, [found.astype(np.int32)], True, (0, 0, 255), 1)
        cv2.imwrite(os.path.join(args.out, f"sample_{seed:04d}.jpg"), vis)
        print(f"seed {seed}: {msg}")


if __name__ == "__main__":
    main()
