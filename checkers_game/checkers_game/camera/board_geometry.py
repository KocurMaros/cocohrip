"""Pure, hardware-free board-corner detection.

Only OpenCV + numpy - no camera SDK, no GUI - so it can be tested on a
machine with no camera attached (see test/test_board_geometry.py).

Algorithm (finds the outline of the 8x8 squares, not the board's outer edge):
  1. Detect square candidates: adaptive/Otsu thresholds in both polarities,
     slightly eroded so diagonally-touching squares separate, keep convex
     quadrilaterals.
  2. Fit a perspective lattice (homography image -> cell coordinates) to
     those candidates with RANSAC-style seeding and region growing. Junk
     candidates (pieces, letters, background) simply don't fit the lattice.
  3. Refine on the checkerboard's X-junctions (where 4 squares meet) with
     sub-pixel accuracy. X-junctions only exist *inside* the board - where
     the squares meet the frame it's a T-junction - so they pin down the
     board extent without ever latching onto the frame's outer lines.
  4. Pick the 8x8 cell window whose 7x7 internal points match the found
     junctions, backed by a per-cell light/dark alternation check, and
     extrapolate the 4 outer corners from a homography fit to all
     junctions (so partially occluded / unevenly lit edges still work).

If the evidence is too weak (too few junctions, junctions covering only
part of the board) it returns None instead of guessing, so the caller can
fall back to manual selection.

Corners are returned as a (4, 2) float32 array in the fixed order
BL, TL, TR, BR (index 0..3), in the input image's pixel coordinates.
"""

import cv2
import numpy as np

CORNER_NAMES = ("BL", "TL", "TR", "BR")

_WORK_MAX_DIM = 640


def order_corners(points):
    """Sort 4 points into BL, TL, TR, BR (sum/difference trick)."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = pts[:, 1] - pts[:, 0]
    return np.array([pts[np.argmax(diff)], pts[np.argmin(s)],
                     pts[np.argmin(diff)], pts[np.argmax(s)]], dtype=np.float32)


def _to_gray(image):
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _map(H, pts):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def _order_quad(p):
    c = p.mean(axis=0)
    return p[np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))]


def _quad_area(q):
    return abs(cv2.contourArea(q.astype(np.float32)))


# --------------------------------------------------------------------------
# 1. square candidates
# --------------------------------------------------------------------------

def _find_square_candidates(gray):
    h, w = gray.shape
    md = min(h, w)
    min_a, max_a = (md / 45.0) ** 2, (md / 4.0) ** 2
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    erode_it = max(1, int(round(md / 300.0)))
    kernel = np.ones((3, 3), np.uint8)

    binaries = []
    for frac in (0.08, 0.15, 0.25):
        block = int(md * frac) | 1
        for mode in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
            binaries.append(cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, mode, block, 0))
    for mode in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        binaries.append(cv2.threshold(blur, 0, 255, mode + cv2.THRESH_OTSU)[1])

    quads = []
    for binary in binaries:
        binary = cv2.erode(binary, kernel, iterations=erode_it)
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            a = cv2.contourArea(c)
            if a < min_a or a > max_a:
                continue
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0 or a / hull_area < 0.85:
                continue
            approx = cv2.approxPolyDP(hull, 0.06 * cv2.arcLength(hull, True), True)
            if len(approx) != 4:
                continue
            p = _order_quad(approx.reshape(4, 2).astype(np.float64))
            sides = np.linalg.norm(p - np.roll(p, -1, axis=0), axis=1)
            if sides.min() <= 0 or sides.max() / sides.min() > 1.8:
                continue
            if _quad_area(p) / hull_area < 0.85:
                continue
            quads.append(p)

    if not quads:
        return np.zeros((0, 4, 2))
    quads = np.array(quads)
    # the different thresholds find the same squares many times over
    centers = quads.mean(axis=1)
    sizes = np.sqrt([_quad_area(q) for q in quads])
    keep = []
    for i in np.argsort(-sizes):
        if all(np.linalg.norm(centers[i] - centers[k]) > 0.3 * sizes[i] for k in keep):
            keep.append(i)
    return quads[keep]


# --------------------------------------------------------------------------
# 2. lattice fit
# --------------------------------------------------------------------------

def _cell_fit(H, quads, tol):
    """Which quads sit exactly on one lattice cell under H (image->lattice)."""
    n = len(quads)
    lat = _map(H, quads.reshape(-1, 2)).reshape(n, 4, 2)
    rounded = np.round(lat)
    err = np.abs(lat - rounded).max(axis=(1, 2))
    cell = np.floor(lat.mean(axis=1)).astype(int)
    ok = err < tol
    for k in np.where(ok)[0]:
        expected = {(cell[k, 0] + dx, cell[k, 1] + dy) for dx in (0, 1) for dy in (0, 1)}
        got = {(int(x), int(y)) for x, y in rounded[k]}
        ok[k] = got == expected
    return ok, cell, rounded


def _fit_lattice(quads, max_seeds=40):
    if len(quads) < 4:
        return None
    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    sizes = np.sqrt([_quad_area(q) for q in quads])
    seeds = np.argsort(np.abs(sizes - np.median(sizes)))[:max_seeds]

    scored = []
    for s in seeds:
        H = cv2.getPerspectiveTransform(quads[s].astype(np.float32), unit)
        ok, cell, _ = _cell_fit(H, quads, 0.25)
        scored.append(((ok & (np.abs(cell).max(axis=1) <= 2)).sum(), H))
    scored.sort(key=lambda t: -t[0])

    best = None
    for _, H in scored[:6]:
        # grow outwards from the seed: a single cell extrapolates poorly
        radius = 2
        for _ in range(8):
            ok, cell, rounded = _cell_fit(H, quads, 0.25)
            use = ok & (np.abs(cell).max(axis=1) <= radius) if radius < 99 else ok
            if use.sum() < 2:
                break
            H2, _ = cv2.findHomography(quads[use].reshape(-1, 2), rounded[use].reshape(-1, 2), 0)
            if H2 is None:
                break
            H = H2
            radius = radius + 2 if radius < 10 else 99
        ok, cell, _ = _cell_fit(H, quads, 0.2)
        if best is None or ok.sum() > best[0]:
            best = (int(ok.sum()), H, ok, cell)
    if best is None or best[0] < 6:
        return None
    return best


# --------------------------------------------------------------------------
# 3. X-junctions
# --------------------------------------------------------------------------

def _refine_junctions(gray, H, cells, cell_px):
    """Sub-pixel lattice points that look like checkerboard X-junctions.
    Returns (lattice_coords (N,2), image_coords (N,2))."""
    Hinv = np.linalg.inv(H)
    imin, jmin = cells.min(axis=0) - 1
    imax, jmax = cells.max(axis=0) + 2
    lat = np.array([[i, j] for i in range(imin, imax + 1)
                    for j in range(jmin, jmax + 1)], dtype=np.float64)

    # sample the 4 diagonal quadrants around each point: an X-junction has
    # diagonal pairs equal and the two pairs different; a T/L-junction or
    # a plain edge doesn't.
    offs = np.array([[-0.25, -0.25], [0.25, 0.25], [0.25, -0.25], [-0.25, 0.25]])
    r = max(1, int(cell_px * 0.08))
    blur = cv2.blur(gray, (2 * r + 1, 2 * r + 1)).astype(np.float64)
    h, w = gray.shape
    q = _map(Hinv, (lat[:, None, :] + offs[None]).reshape(-1, 2))
    xi = np.clip(np.round(q[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(q[:, 1]).astype(int), 0, h - 1)
    v = blur[yi, xi].reshape(-1, 4)
    contrast = np.abs(0.5 * (v[:, 0] + v[:, 1]) - 0.5 * (v[:, 2] + v[:, 3]))
    noise = np.abs(v[:, 0] - v[:, 1]) + np.abs(v[:, 2] - v[:, 3])
    structural = contrast > 2.0 * noise
    ref = np.percentile(contrast[structural], 75) if structural.sum() >= 4 else contrast.max()
    is_x = structural & (contrast > max(6.0, 0.25 * ref))

    lat = lat[is_x]
    guess = _map(Hinv, lat).astype(np.float32) if len(lat) else np.zeros((0, 2), np.float32)
    inside = ((guess[:, 0] > 3) & (guess[:, 0] < w - 4) &
              (guess[:, 1] > 3) & (guess[:, 1] < h - 4))
    lat, guess = lat[inside], guess[inside]
    if len(lat) == 0:
        return lat, np.zeros((0, 2))
    win = max(3, int(cell_px * 0.2))
    refined = cv2.cornerSubPix(
        gray, guess.reshape(-1, 1, 2).copy(), (win, win), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)).reshape(-1, 2)
    good = np.linalg.norm(refined - guess, axis=1) < 0.2 * cell_px
    return lat[good], refined[good].astype(np.float64)


# --------------------------------------------------------------------------
# 4. 8x8 window
# --------------------------------------------------------------------------

def _cell_checker_evidence(gray, H, i_range, j_range, cell_px):
    """parity * (cell brightness - mean of its 4 neighbours) per cell.
    Board cells alternate with their neighbours; frame, label strip and
    background don't. Samples near the cell border so a piece in the middle
    of a square doesn't dominate."""
    Hinv = np.linalg.inv(H)
    i0, i1 = i_range
    j0, j1 = j_range
    ring = np.array([[u, v] for u in (0.15, 0.5, 0.85) for v in (0.15, 0.5, 0.85)
                     if not (u == 0.5 and v == 0.5)])
    r = max(1, int(cell_px * 0.06))
    blur = cv2.blur(gray, (2 * r + 1, 2 * r + 1)).astype(np.float64)
    h, w = gray.shape
    ni, nj = i1 - i0, j1 - j0
    V = np.full((ni, nj), np.nan)
    for a in range(ni):
        for b in range(nj):
            q = _map(Hinv, ring + [i0 + a, j0 + b])
            if (q < 0).any() or (q[:, 0] >= w).any() or (q[:, 1] >= h).any():
                continue
            V[a, b] = np.median(blur[q[:, 1].astype(int), q[:, 0].astype(int)])
    evidence = {}
    for a in range(ni):
        for b in range(nj):
            if np.isnan(V[a, b]):
                continue
            nb = [V[x, y] for x, y in ((a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1))
                  if 0 <= x < ni and 0 <= y < nj and not np.isnan(V[x, y])]
            if len(nb) < 2:
                continue
            parity = 1 if (i0 + a + j0 + b) % 2 == 0 else -1
            evidence[(i0 + a, j0 + b)] = parity * (V[a, b] - np.mean(nb))
    return evidence


def _normalised_evidence(gray, H, lat_j, cells, cell_px):
    lo = np.minimum(lat_j.min(axis=0), cells.min(axis=0)).astype(int) - 3
    hi = np.maximum(lat_j.max(axis=0), cells.max(axis=0)).astype(int) + 3
    raw = _cell_checker_evidence(gray, H, (lo[0], hi[0]), (lo[1], hi[1]), cell_px)
    # orient the sign with cells fully surrounded by junctions (certainly on
    # the board) and scale so a clear board cell is worth ~1 point
    J = {(int(a), int(b)) for a, b in lat_j}
    core = [v for (a, b), v in raw.items()
            if {(a, b), (a + 1, b), (a, b + 1), (a + 1, b + 1)} <= J]
    if len(core) < 4:
        return None
    sign = 1.0 if np.median(core) >= 0 else -1.0
    scale = np.median(np.abs(core)) + 1e-6
    return {k: float(np.clip(sign * v / scale, -1.5, 1.5)) for k, v in raw.items()}


def _choose_window(junction_lat, cells, evidence):
    J = [(int(a), int(b)) for a, b in junction_lat]
    C = [(int(a), int(b)) for a, b in cells]
    all_i = [p[0] for p in J + C]
    all_j = [p[1] for p in J + C]
    best = None
    for i0 in range(min(all_i) - 8, max(all_i) + 1):
        for j0 in range(min(all_j) - 8, max(all_j) + 1):
            j_in = sum(1 for a, b in J if i0 < a < i0 + 8 and j0 < b < j0 + 8)
            c_in = sum(1 for a, b in C if i0 <= a < i0 + 8 and j0 <= b < j0 + 8)
            score = 2 * j_in - 3 * (len(J) - j_in) + c_in - 1.5 * (len(C) - c_in)
            if evidence:
                e_in = e_out = 0.0
                for (a, b), v in evidence.items():
                    if i0 <= a < i0 + 8 and j0 <= b < j0 + 8:
                        e_in += v
                    else:
                        e_out += v
                score += e_in - max(0.0, e_out)
            if best is None or score > best[0]:
                best = (score, i0, j0, j_in, c_in)
    return best


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def find_board_corners(image, debug=None):
    """Locate the outline of the 8x8 squares in `image` (BGR, BGRA or gray).

    Returns (corners, message): corners is a (4, 2) float32 array in
    BL, TL, TR, BR order, or None if the board couldn't be found reliably;
    message is a human-readable reason for logging. `debug`, if a dict, is
    filled with intermediate results (in working-resolution coordinates).
    """
    if image is None or image.size == 0:
        return None, "empty image"
    gray = _to_gray(image)
    scale = min(1.0, _WORK_MAX_DIM / float(max(gray.shape)))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    quads = _find_square_candidates(gray)
    if debug is not None:
        debug['quads'] = quads
    if len(quads) < 6:
        return None, f"too few square candidates ({len(quads)})"

    fit = _fit_lattice(quads)
    if fit is None:
        return None, "could not fit a square lattice"
    _, H, ok, cell = fit
    cells = cell[ok]
    cell_px = float(np.median(np.sqrt([_quad_area(q) for q in quads[ok]])))

    lat_j, img_j = _refine_junctions(gray, H, cells, cell_px)
    if len(lat_j) >= 8:
        H2, _ = cv2.findHomography(img_j, lat_j, cv2.RANSAC, 0.08)
        if H2 is not None:
            H = H2
            lat_j, img_j = _refine_junctions(gray, H, cells, cell_px)
    if len(lat_j) < 4:
        return None, f"only {len(lat_j)} grid junctions found"

    evidence = _normalised_evidence(gray, H, lat_j, cells, cell_px)
    _, i0, j0, j_in, c_in = _choose_window(lat_j, cells, evidence)
    if debug is not None:
        debug.update(H=H, cells=cells, lat_j=lat_j, img_j=img_j, window=(i0, j0))
    if j_in < 20:
        return None, f"only {j_in} of 49 internal grid junctions found"

    sel = np.array([i0 < a < i0 + 8 and j0 < b < j0 + 8 for a, b in lat_j])
    lat_in, img_in = lat_j[sel], img_j[sel]
    span_i = len(set(lat_in[:, 0].astype(int)))
    span_j = len(set(lat_in[:, 1].astype(int)))
    if span_i < 6 or span_j < 6:
        return None, (f"grid junctions only cover {span_i}x{span_j} of the 7x7 "
                      "inner rows/cols - refusing to extrapolate")

    Hb, _ = cv2.findHomography(img_in, lat_in, cv2.RANSAC, 0.08)
    if Hb is None:
        Hb = H
    board_lat = [[i0, j0], [i0 + 8, j0], [i0 + 8, j0 + 8], [i0, j0 + 8]]
    corners = _map(np.linalg.inv(Hb), board_lat) / scale
    return order_corners(corners), f"ok ({j_in}/49 grid junctions, {c_in} squares)"
