"""Pure, hardware-free board-corner detection.

This module contains only image-processing logic (OpenCV + numpy) and has no
dependency on the camera SDK or any GUI code, so it can be imported and unit
tested on a machine that has neither a camera nor the Ximea SDK installed.

Corner convention
------------------
Corners are always returned as a (4, 2) float32 array in the fixed order::

    index 0: BL (bottom-left)
    index 1: TL (top-left)
    index 2: TR (top-right)
    index 3: BR (bottom-right)

This assumes the camera is rigidly mounted and looks down at the board with
roughly the orientation shown in the reference photo (board not rotated more
than ~45 degrees from upright) - see README/proposal discussion. Because the
mount is fixed, corner identity can be assigned purely from geometry (top-most
/ bottom-most / left-most / right-most) instead of from piece placement, which
only worked for the exact starting position.
"""

import cv2
import numpy as np

CORNER_NAMES = ("BL", "TL", "TR", "BR")


def order_corners(points):
    """Sort 4 arbitrary points into the fixed BL, TL, TR, BR convention.

    Uses the standard sum/difference trick: TL has the smallest x+y, BR the
    largest x+y, TR has the smallest y-x, BL has the largest y-x.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = pts[:, 1] - pts[:, 0]  # y - x

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([bl, tl, tr, br], dtype=np.float32)


def warp_to_square(image, corners_bl_tl_tr_br, size=800, margin=0):
    """Perspective-warp the board defined by corners (BL,TL,TR,BR order) to
    a `size` x `size` square, with an optional pixel margin of extra canvas
    around it. Returns (warped_image, homography_matrix).
    """
    canvas = size + 2 * margin
    dst = np.array([
        [margin, margin + size],          # BL
        [margin, margin],                 # TL
        [margin + size, margin],          # TR
        [margin + size, margin + size],   # BR
    ], dtype=np.float32)
    H = cv2.getPerspectiveTransform(
        np.asarray(corners_bl_tl_tr_br, dtype=np.float32), dst)
    # BORDER_REPLICATE avoids a hard black/valid-data seam when the margin
    # samples outside the source image; a constant-fill seam is a very
    # strong artificial edge that can fool the periodicity search below.
    warped = cv2.warpPerspective(image, H, (canvas, canvas),
                                  borderMode=cv2.BORDER_REPLICATE)
    return warped, H


def _angle_dist(a, b):
    d = abs(a - b) % 180
    return min(d, 180 - d)


def _hough_segments(gray, min_len_frac=0.30, canny_lo=40, canny_hi=120):
    h, w = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, canny_lo, canny_hi)
    min_len = int(min_len_frac * min(h, w))
    raw = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=50,
                           minLineLength=min_len, maxLineGap=12)
    if raw is None:
        return []
    segs = []
    for line in raw[:, 0]:
        x1, y1, x2, y2 = [float(v) for v in line]
        length = float(np.hypot(x2 - x1, y2 - y1))
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180)
        segs.append(dict(x1=x1, y1=y1, x2=x2, y2=y2, length=length,
                          angle=angle, mx=(x1 + x2) / 2, my=(y1 + y2) / 2))
    return segs


def _dominant_direction_clusters(segs, angle_tolerance=12):
    """Split segments into the two roughly-perpendicular dominant directions
    of the board's row/column grid. Returns (clusterA, clusterB, angleA,
    angleB) or (None, None, None, None) if no clear structure is found.
    """
    if len(segs) < 8:
        return None, None, None, None

    hist = np.zeros(180)
    for s in segs:
        hist[int(s['angle']) % 180] += s['length']
    angle_a = int(np.argmax(hist))

    angle_b, best_score = None, -1.0
    for cand in range(180):
        if _angle_dist(cand, angle_a) > 60 and hist[cand] > best_score:
            best_score = hist[cand]
            angle_b = cand

    if angle_b is None:
        return None, None, None, None

    cluster_a = [s for s in segs if _angle_dist(s['angle'], angle_a) < angle_tolerance]
    cluster_b = [s for s in segs if _angle_dist(s['angle'], angle_b) < angle_tolerance]

    if len(cluster_a) < 2 or len(cluster_b) < 2:
        return None, None, None, None

    return cluster_a, cluster_b, angle_a, angle_b


def _shared_rho(seg, direction_deg):
    nrad = np.deg2rad(direction_deg + 90)
    return seg['mx'] * np.cos(nrad) + seg['my'] * np.sin(nrad)


def _seg_as_infinite_line(seg, length=2000):
    dx, dy = seg['x2'] - seg['x1'], seg['y2'] - seg['y1']
    n = np.hypot(dx, dy)
    if n < 1e-6:
        dx, dy = 1.0, 0.0
    else:
        dx, dy = dx / n, dy / n
    return ((seg['mx'] - dx * length, seg['my'] - dy * length),
            (seg['mx'] + dx * length, seg['my'] + dy * length))


def _line_intersect(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-9:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return (px, py)


def _poly_area(pts):
    pts = np.array(pts)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _coarse_quad(cluster_a, cluster_b, angle_a, angle_b, img_shape,
                  min_len_frac=0.3, topk=4):
    """Find a plausible (not necessarily precise) quadrilateral bounding the
    board, robust to a handful of stray long lines (clothing, table edge,
    etc.) by scoring candidate quads instead of blindly taking extremes.
    """
    h, w = img_shape[:2]

    def candidates(cluster, direction_deg):
        rhos = np.array([_shared_rho(s, direction_deg) for s in cluster])
        lens = np.array([s['length'] for s in cluster])
        idxs = np.where(lens > min_len_frac * min(h, w))[0]
        if len(idxs) == 0:
            idxs = np.arange(len(cluster))
        lo = idxs[np.argsort(rhos[idxs])][:topk]
        hi = idxs[np.argsort(-rhos[idxs])][:topk]
        return [cluster[i] for i in lo], [cluster[i] for i in hi]

    lo_a, hi_a = candidates(cluster_a, angle_a)
    lo_b, hi_b = candidates(cluster_b, angle_b)

    img_area = h * w
    best = None
    for sa in lo_a:
        for sb in hi_a:
            for tb in lo_b:
                for bb in hi_b:
                    line_a0 = _seg_as_infinite_line(sa)
                    line_a1 = _seg_as_infinite_line(sb)
                    line_b0 = _seg_as_infinite_line(tb)
                    line_b1 = _seg_as_infinite_line(bb)
                    c00 = _line_intersect(*line_a0, *line_b0)
                    c01 = _line_intersect(*line_a0, *line_b1)
                    c10 = _line_intersect(*line_a1, *line_b0)
                    c11 = _line_intersect(*line_a1, *line_b1)
                    if None in (c00, c01, c10, c11):
                        continue
                    quad = [c00, c01, c11, c10]
                    area = _poly_area(quad)
                    if area < 0.35 * img_area or area > 1.3 * img_area:
                        continue
                    xs = [p[0] for p in quad]
                    ys = [p[1] for p in quad]
                    if (min(xs) < -0.3 * w or max(xs) > 1.3 * w or
                            min(ys) < -0.3 * h or max(ys) > 1.3 * h):
                        continue
                    if best is None or area > best[0]:
                        best = (area, quad)

    if best is None:
        return None
    return np.array(best[1], dtype=np.float32)


def _find_profile_peaks(profile):
    thresh = profile.mean() + 0.5 * profile.std()
    peaks = []
    i, n = 0, len(profile)
    while i < n:
        if profile[i] > thresh:
            j = i
            while j < n and profile[j] > thresh:
                j += 1
            seg = profile[i:j]
            peak_idx = i + int(np.argmax(seg))
            peaks.append((peak_idx, float(profile[peak_idx])))
            i = j
        else:
            i += 1
    return peaks


def _refine_edge_pair(profile, size, n_lines=9,
                       denom_range=(6.0, 10.0), denom_steps=81,
                       min_matches=7, tolerance_frac=0.22, nominal_denom=8.0,
                       prior_sigma=1.5):
    """Given a 1D gradient-magnitude profile of an (approximately) rectified
    grid, fit a periodic pattern of `n_lines` evenly spaced boundaries and
    return the (first, last) matched position - i.e. the outermost pair that
    is consistent with the expected 8-cell spacing. This is what lets us
    reject the frame's *outer* edge (which does not fit the periodicity of
    the 8x8 grid) in favor of the true inner edge.

    `warp_to_square` maps the coarse quad exactly onto `size`, so a
    *perfectly* accurate coarse quad implies a cell spacing of exactly
    size/8. A real coarse quad is imprecise, so we still search a wide
    range of spacings (denom_range), but candidates near size/8 get a soft
    preference (gaussian prior on denom) - this avoids the fit locking onto
    an off-by-one-cell alias pattern when the correct, in-range candidate
    exists but scores only slightly lower on raw peak strength.
    """
    peaks = _find_profile_peaks(profile)
    if len(peaks) < 5:
        return None
    positions = np.array([p[0] for p in peaks])
    weights = np.array([p[1] for p in peaks])

    best = None
    for o in positions:
        for denom in np.linspace(denom_range[0], denom_range[1], denom_steps):
            spacing = size / denom
            score, matched = 0.0, []
            for k in range(n_lines):
                target = o + k * spacing
                d = np.abs(positions - target)
                idx = np.argmin(d)
                if d[idx] < spacing * tolerance_frac:
                    score += weights[idx]
                    matched.append(positions[idx])
            if len(matched) < min_matches:
                continue
            prior = np.exp(-0.5 * ((denom - nominal_denom) / prior_sigma) ** 2)
            weighted_score = score * prior
            if best is None or weighted_score > best[0]:
                best = (weighted_score, spacing, o, matched)

    if best is None:
        return None
    matched = sorted(best[3])
    return matched[0], matched[-1]


def _is_plausible_square(corners_bl_tl_tr_br, max_side_ratio=1.35,
                          max_diagonal_ratio=1.25):
    """Sanity-check that a candidate quad could plausibly be a (perspective-
    projected) square board, so a badly-fit result is rejected instead of
    silently returned. A real camera view of a square board may show
    unequal side lengths (perspective foreshortening) and unequal diagonals,
    but not by an extreme amount for the shallow viewing angles this camera
    is mounted at.
    """
    bl, tl, tr, br = corners_bl_tl_tr_br

    def dist(a, b):
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    sides = [dist(bl, tl), dist(tl, tr), dist(tr, br), dist(br, bl)]
    if min(sides) < 1e-3:
        return False
    if max(sides) / min(sides) > max_side_ratio:
        return False

    diagonals = [dist(bl, tr), dist(tl, br)]
    if min(diagonals) < 1e-3:
        return False
    if max(diagonals) / min(diagonals) > max_diagonal_ratio:
        return False

    return True


def find_board_corners(image, rectify_size=800, rectify_margin_frac=0.18,
                        debug=None):
    """Locate the 4 corners of the playing area (the edges closest to the
    squares, i.e. the *inner* edge of the board's frame/border) in `image`.

    Two-stage approach:
      1. Coarse localization via dominant-line Hough clustering - finds
         *some* quadrilateral that reasonably bounds the board.
      2. Perspective-rectify using that coarse quad (with generous margin),
         then locate the true inner grid boundary via 1D gradient projection
         profiles, fitting the known 8x8 periodicity. This works regardless
         of whether the coarse quad landed on the frame's inner or outer
         edge, because periodicity - not raw edge strength - decides.

    Returns (corners, message):
      corners: (4, 2) float32 array in BL, TL, TR, BR order, or None
      message: human-readable success/failure reason (for logging)

    `debug` (optional dict) receives intermediate artifacts (coarse_quad,
    warped, homography) for visualization/testing, when provided.
    """
    if image is None or image.size == 0:
        return None, "empty image"

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    segs = _hough_segments(gray)
    if len(segs) < 8:
        return None, f"too few line segments detected ({len(segs)})"

    cluster_a, cluster_b, angle_a, angle_b = _dominant_direction_clusters(segs)
    if cluster_a is None:
        return None, "could not find two dominant perpendicular directions"

    coarse_quad = _coarse_quad(cluster_a, cluster_b, angle_a, angle_b, image.shape)
    if coarse_quad is None:
        return None, "no plausible coarse quadrilateral found"

    if debug is not None:
        debug['coarse_quad'] = coarse_quad.copy()

    coarse_ordered = order_corners(coarse_quad)
    warped, H = warp_to_square(image, coarse_ordered, size=rectify_size,
                                margin=int(rectify_size * rectify_margin_frac))

    if debug is not None:
        debug['warped'] = warped
        debug['homography'] = H

    wgray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(wgray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(wgray, cv2.CV_32F, 0, 1, ksize=3)
    col_profile = np.abs(sx).sum(axis=0)
    row_profile = np.abs(sy).sum(axis=1)

    left_right = _refine_edge_pair(col_profile, rectify_size)
    top_bottom = _refine_edge_pair(row_profile, rectify_size)
    if left_right is None or top_bottom is None:
        return None, "profile refinement did not find a periodic 8x8 pattern"

    left, right = left_right
    top, bottom = top_bottom

    if debug is not None:
        debug['refined_edges'] = dict(left=left, right=right, top=top, bottom=bottom)

    # warped-space corners in TL,TR,BR,BL order (matching warp_to_square's
    # canvas layout), then map back and re-order canonically.
    warped_corners = np.array([
        [left, top], [right, top], [right, bottom], [left, bottom],
    ], dtype=np.float32)

    H_inv = np.linalg.inv(H)
    homo = np.hstack([warped_corners, np.ones((4, 1), dtype=np.float32)])
    mapped = (H_inv @ homo.T).T
    mapped = mapped[:, :2] / mapped[:, 2:3]

    corners = order_corners(mapped)

    if debug is not None:
        debug['final_corners'] = corners.copy()

    if not _is_plausible_square(corners):
        return None, "refined corners are not a plausible square (rejected instead of guessing)"

    return corners, "ok"
