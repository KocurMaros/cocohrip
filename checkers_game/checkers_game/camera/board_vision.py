"""Piece detection on a rectified (top-down) board image.  Pure OpenCV/numpy -
no camera, GUI or ROS - so it can be developed and tested on recorded frames
(see tools/eval_detection.py and test/test_board_vision.py).

Pipeline
--------
1. ``warp``: perspective-rectify the board with the 4 corners from
   board_geometry (BL, TL, TR, BR) into a 800x800 image and rotate it by
   ``rotation`` x 90 deg into the *game* orientation: robot's (black) pieces
   on rows 0-2, human's (white) pieces on rows 5-7, top-left square light.
2. ``normalise``: a light-invariant grey image.  Every cell's border band is
   bare board (pieces never reach the cell edges), so the median of the
   light-square borders and of the dark-square borders give the local white
   and black level of the board under the *current* lighting; intensities are
   mapped so board-black = 0 and board-white = 1.  A lamp switched on/off or
   a shadow moving over the board therefore barely changes the features.
3. ``cell_features``: per playable (dark) square, contrast of the centre
   disk against the square's own border (mean / spread / bright and dark
   percentiles / edge energy / colour).  The disk position is searched within
   the cell because tall pieces seen at an angle are not centred.
4. ``PieceClassifier``: a 3-class (empty / black / white) Gaussian model
   fitted on the start position (which labels itself: 12 + 12 + 8 dark
   squares) and refreshed on every confirmed position, so it follows slow
   lighting changes.  Light squares are always empty in checkers.
5. ``change_scores``: per-square change against the last confirmed frame.
   A black piece on a dark square is hard to classify on its own, but taking
   it away or putting it down always changes the square; combined with the
   rules engine this is what makes the move detection reliable.

Grid convention: 0 empty, 1 white, 2 black, 3 occupied (colour unknown),
-1 unknown.
"""

import numpy as np
import cv2

SIZE = 800
CELL = SIZE // 8
EMPTY, WHITE_P, BLACK_P, OCCUPIED, UNKNOWN = 0, 1, 2, 3, -1

# centre-disk radius and search window (pixels in the 800x800 warp)
DISK_R = 30
SEARCH = 12
SEARCH_STEP = 6
BORDER = 9            # width of the bare-board band at the cell edges


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def rotate_corners(corners, rotation):
    """Corners (BL, TL, TR, BR) re-labelled so the warp is rotated by
    `rotation` * 90 degrees counter-clockwise."""
    return np.roll(np.asarray(corners, dtype=np.float32).reshape(4, 2), -int(rotation) % 4, axis=0)


def homography(corners, rotation=0, size=SIZE):
    c = rotate_corners(corners, rotation)
    dst = np.array([[0, size], [0, 0], [size, 0], [size, size]], dtype=np.float32)
    return cv2.getPerspectiveTransform(c, dst)


def warp(image, corners, rotation=0, size=SIZE):
    H = homography(corners, rotation, size)
    return cv2.warpPerspective(image, H, (size, size), flags=cv2.INTER_AREA)


def to_gray(img):
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def to_bgr(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def cell_slice(r, c):
    return slice(r * CELL, (r + 1) * CELL), slice(c * CELL, (c + 1) * CELL)


_band_mask = np.zeros((CELL, CELL), np.uint8)
_band_mask[:BORDER, :] = 1
_band_mask[-BORDER:, :] = 1
_band_mask[:, :BORDER] = 1
_band_mask[:, -BORDER:] = 1
_BAND = _band_mask.astype(bool)

_yy, _xx = np.mgrid[0:CELL, 0:CELL]


def _disk(cx, cy, r=DISK_R):
    return (_xx - cx) ** 2 + (_yy - cy) ** 2 <= r * r


_DISKS = {}
for _dy in range(-SEARCH, SEARCH + 1, SEARCH_STEP):
    for _dx in range(-SEARCH, SEARCH + 1, SEARCH_STEP):
        _DISKS[(_dx, _dy)] = _disk(CELL // 2 + _dx, CELL // 2 + _dy)


# --------------------------------------------------------------------------
# lighting normalisation
# --------------------------------------------------------------------------
def border_levels(gray):
    """8x8 median grey level of each cell's border band."""
    out = np.zeros((8, 8), np.float32)
    for r in range(8):
        for c in range(8):
            ys, xs = cell_slice(r, c)
            out[r, c] = np.median(gray[ys, xs][_BAND])
    return out


def dark_parity(gray):
    """Parity ((row+col) % 2) of the dark squares in a rectified grey image."""
    lv = border_levels(gray)
    par = (np.add.outer(np.arange(8), np.arange(8)) % 2)
    return 0 if lv[par == 0].mean() < lv[par == 1].mean() else 1


def board_levels(gray, parity):
    """(black_level, white_level) of the bare board under current lighting,
    plus the 8x8 border level map (smoothly varying illumination)."""
    lv = border_levels(gray)
    par = (np.add.outer(np.arange(8), np.arange(8)) % 2)
    dark = float(np.median(lv[par == parity]))
    light = float(np.median(lv[par != parity]))
    return dark, light, lv


def normalise(gray, parity):
    """Map board-black -> 0, board-white -> 1 (float32)."""
    dark, light, _ = board_levels(gray, parity)
    span = max(light - dark, 8.0)
    return (gray.astype(np.float32) - dark) / span, dark, light


def is_dark(r, c, parity):
    return (r + c) % 2 == parity


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------
FEATURE_NAMES = ("ring", "r_mean", "r_std", "r_p10")

# circular-rim matched filter: candidate piece outlines (centre offset, radius)
# inside a 100x100 cell.  The circle must stay >= 8 px from the cell edge so
# the light/dark square boundary can never look like a piece.
_N_PTS = 64
_ANG = np.linspace(0, 2 * np.pi, _N_PTS, endpoint=False)
_CANDS = [(dx, dy, R) for dx in range(-15, 16, 3) for dy in range(-18, 13, 3)
          for R in (22, 25, 28, 31) if abs(dx) + R <= 40 and abs(dy) + R <= 40]
_PX = np.clip(np.round(np.array([CELL // 2 + dx + R * np.cos(_ANG) for dx, dy, R in _CANDS])).astype(int), 0, CELL - 1)
_PY = np.clip(np.round(np.array([CELL // 2 + dy + R * np.sin(_ANG) for dx, dy, R in _CANDS])).astype(int), 0, CELL - 1)
_UX = np.tile(np.cos(_ANG), (len(_CANDS), 1)).astype(np.float32)
_UY = np.tile(np.sin(_ANG), (len(_CANDS), 1)).astype(np.float32)
_CAND_DISKS = [_disk(CELL // 2 + dx, CELL // 2 + dy, R - 4) for dx, dy, R in _CANDS]


def cell_features(bgr, norm=None, parity=1, cells=None):
    """Feature vectors for `cells` (default: all dark squares).

    Everything is computed on the REFLECTANCE RATIO image of the square:
    grey / median grey of the square's own bare border band.  Illumination is
    locally multiplicative, so a lamp switched on/off or another exposure
    hardly changes these values.

    ring   - circular-rim matched filter: the best mean |radial gradient|
             along a circle (piece outline) in the cell, divided by the noise
             of the bare dark squares in this frame.  A plain black piece on
             a black square is barely darker than the square, but its round
             rim and rings always show up here.  Measured on recorded frames
             (2 lightings x 3 exposures): black pieces 0.56-2.0, empty dark
             squares 0.09-0.35.
    r_mean - mean reflectance of the piece disk relative to the square:
             wooden pieces 2.0-8.0, black pieces 0.56-0.82.
    r_std, r_p10 - texture / darkest part of the disk (diagnostics).

    Returns (cells, F) with F of shape (len(cells), len(FEATURE_NAMES)).
    `norm` is unused (kept for API compatibility)."""
    if cells is None:
        cells = [(r, c) for r in range(8) for c in range(8) if is_dark(r, c, parity)]
    gray = to_gray(bgr).astype(np.float32)
    ratios, noise = [], []
    for (r, c) in cells:
        ys, xs = cell_slice(r, c)
        g = gray[ys, xs]
        b = max(float(np.median(g[_BAND])), 2.0)
        q = cv2.GaussianBlur(g / b, (5, 5), 1.2)
        ratios.append(q)
        noise.append(float(q[4:12, 10:90].std()))
    sigma = float(np.median(noise)) + 1e-4
    feats = np.zeros((len(cells), len(FEATURE_NAMES)), np.float32)
    for i, q in enumerate(ratios):
        gx = cv2.Sobel(q, cv2.CV_32F, 1, 0) / 8.0
        gy = cv2.Sobel(q, cv2.CV_32F, 0, 1) / 8.0
        rad = np.abs(gx[_PY, _PX] * _UX + gy[_PY, _PX] * _UY).mean(axis=1)
        k = int(np.argmax(rad))
        v = q[_CAND_DISKS[k]]
        feats[i] = (float(rad[k]) / sigma, float(v.mean()), float(v.std()),
                    float(np.percentile(v, 10)))
    return cells, feats


# --------------------------------------------------------------------------
# piece position (centres the robot's gripper on the piece)
# --------------------------------------------------------------------------
CENTRE_MARGIN = 30    # px searched beyond the cell edge: a pushed piece sticks out
CENTRE_BORDER = 2 * CENTRE_MARGIN   # surroundings kept around the board (warp_bordered)
_RADII = (22, 25, 28, 31)
_GRID_BAND = 5        # px either side of a square edge ignored by the rim search
_STEP = 3             # px between candidate circle centres (then refined to 1 px)
_K_MAX = 0.04         # largest lean per px of distance: 8 mm pieces, camera 0.2 m up
_RIM_CAP = 1.0        # log-gradient cap: score how much of a circle has an edge


def _rim_scores(gx, gy, cands, cap=np.inf):
    """Mean |radial gradient| (each point at most `cap`) along each circle (cx, cy, R)."""
    c = np.asarray(cands, np.float32)
    px = np.clip(np.round(c[:, :1] + c[:, 2:] * np.cos(_ANG)).astype(int), 0, gx.shape[1] - 1)
    py = np.clip(np.round(c[:, 1:2] + c[:, 2:] * np.sin(_ANG)).astype(int), 0, gx.shape[0] - 1)
    radial = np.abs(gx[py, px] * np.cos(_ANG) + gy[py, px] * np.sin(_ANG))
    return np.minimum(radial, cap).mean(axis=1)


def warp_bordered(image, corners, rotation=0):
    """warp() plus CENTRE_BORDER px of the board's surroundings on every side."""
    # a piece on an edge square, pushed or seen at an angle, can stick out of the board
    b = CENTRE_BORDER
    H = np.array([[1.0, 0, b], [0, 1.0, b], [0, 0, 1.0]]) @ homography(corners, rotation)
    return cv2.warpPerspective(image, H, (SIZE + 2 * b, SIZE + 2 * b), flags=cv2.INTER_AREA)


def _rim_gradients(gray, r, c):
    """(gx, gy, half): log gradients around square (r, c), its centre at (half, half)."""
    # gray: warp_bordered() image; the grid lines are blanked
    m, b = CENTRE_MARGIN, CENTRE_BORDER
    half = CELL // 2 + 2 * m
    x0 = c * CELL + CELL // 2 - half                # window origin, board px
    y0 = r * CELL + CELL // 2 - half
    win = gray[y0 + b:y0 + b + 2 * half, x0 + b:x0 + b + 2 * half]
    # log: an edge counts by its contrast RATIO, so a black piece's rim on a dark
    # square is not drowned by its own outline against a light square
    win = cv2.GaussianBlur(np.log1p(np.asarray(win, np.float32)), (5, 5), 1.2)
    gx = cv2.Sobel(win, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(win, cv2.CV_32F, 0, 1)
    # the light/dark square edges are far stronger than a black piece's rim
    for axis, origin in ((0, y0), (1, x0)):
        d = (np.arange(2 * half) + origin) % CELL
        near = np.minimum(d, CELL - d) < _GRID_BAND
        if axis == 0:
            gx[near, :] = gy[near, :] = 0
        else:
            gx[:, near] = gy[:, near] = 0
    return gx, gy, half


def lean_at(model, r, c):
    """(dx, dy) px a piece top on (r, c) shows up off its foot: k * position + t."""
    k, tx, ty = model
    return np.array([k * (c * CELL + CELL / 2) + tx, k * (r * CELL + CELL / 2) + ty])


def piece_centre(gray, r, c, lean=(0.0, 0.0)):
    """(dx, dy) px from the centre of square (r, c) to the centre of the piece's foot."""
    # gray: warp_bordered() image.  A piece is a cylinder: the rim filter of
    # cell_features scores its foot circle plus its top circle (foot + `lean`)
    # together, so whichever stands out against the background pins it - the
    # top on a dark square, the foot when the top reaches a light one
    gx, gy, half = _rim_gradients(gray, r, c)
    lean = np.clip(np.asarray(lean, np.float32), -CENTRE_MARGIN, CENTRE_MARGIN)

    def best(cands):
        cands = np.asarray(cands, np.float32)
        score = _rim_scores(gx, gy, cands, _RIM_CAP)
        if lean.any():
            tops = cands.copy()
            tops[:, :2] += lean
            score = score + _rim_scores(gx, gy, tops, _RIM_CAP)
        return cands[int(np.argmax(score))]

    span = range(-CENTRE_MARGIN, CENTRE_MARGIN + 1, _STEP)
    bx, by, br = best([(half + dx, half + dy, R) for dx in span for dy in span for R in _RADII])
    bx, by, _ = best([(bx + dx, by + dy, R) for dx in range(-2, 3) for dy in range(-2, 3)
                      for R in (br - 1, br, br + 1)])
    return np.array([bx - half, by - half], np.float32)


def fit_centre_model(gray, cells):
    """Lean model (k, tx, ty) for lean_at() from pieces standing centred, or None."""
    # a piece top seen at an angle shifts (parallax) away from the point below
    # the camera, in proportion to the distance: k * position + t for ANY camera
    # pose.  The model is the one whose predicted tops are the best circles over
    # all pieces together (each piece normalised to its own best circle), so a
    # top hard to see on its own - white on a light square - can't pull it to the foot
    if len(cells) < 6:
        return None
    m = CENTRE_MARGIN
    span = np.arange(-m, m + 1, _STEP)
    maps = []
    for r, c in cells:
        gx, gy, half = _rim_gradients(gray, r, c)
        cands = [(half + dx, half + dy, R) for dy in span for dx in span for R in _RADII]
        s = _rim_scores(gx, gy, cands, _RIM_CAP).reshape(len(span), len(span), len(_RADII))
        maps.append(s.max(axis=2) / max(float(s.max()), 1e-6))
    maps = np.array(maps)                           # (piece, dy, dx): a top there
    pos = np.array([(c * CELL + CELL / 2, r * CELL + CELL / 2) for r, c in cells])
    shifts = np.arange(-2 * m, 2 * m + 1, _STEP)
    t = np.stack(np.meshgrid(shifts, shifts), -1).reshape(-1, 2)
    piece = np.arange(len(cells))
    best, best_score = (0.0, 0.0, 0.0), -1.0
    for k in np.arange(0.0, _K_MAX + 1e-9, 0.001):
        lean = k * pos[None] + t[:, None]           # (t, piece, 2)
        ix = np.rint((lean[..., 0] + m) / _STEP).astype(int)
        iy = np.rint((lean[..., 1] + m) / _STEP).astype(int)
        inside = (ix >= 0) & (ix < len(span)) & (iy >= 0) & (iy < len(span))
        score = np.where(inside, maps[piece, iy.clip(0, len(span) - 1),
                                      ix.clip(0, len(span) - 1)], 0.0).sum(axis=1)
        j = int(np.argmax(score))
        if score[j] > best_score:
            best, best_score = (k, *t[j]), float(score[j])
    return np.array(best, np.float64)


def piece_offset(gray, r, c, model):
    """(rows, cols) the piece on (r, c) sits off its square's centre (gray: warp_bordered)."""
    dx, dy = piece_centre(gray, r, c, lean=lean_at(model, r, c))
    return float(dy) / CELL, float(dx) / CELL


# --------------------------------------------------------------------------
# classifier
# --------------------------------------------------------------------------
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class PieceClassifier:
    """Two-stage classifier for dark squares: occupied? then white or black?

    occupancy: log(ring) against a threshold that sits `OCC_K` of the way
               from this frame's empty-square baseline (mean of the lowest
               ring scores - there are always >= 8 empty dark squares) to the
               occupied level learned at calibration.  The baseline follows
               the noise level of the current frame.
    colour:    log(r_mean) against the midpoint of the learned white and
               black levels.
    Works without calibration (universal defaults from recorded data);
    calibration on the start position and update() with every confirmed
    position adapt it to the current board / camera / light."""

    CLASSES = (EMPTY, WHITE_P, BLACK_P)
    OCC_K = 0.4
    COL_K = 0.35
    N_BASELINE = 6

    def __init__(self, adapt_rate=0.2):
        self.adapt_rate = adapt_rate
        self.empty_lvl = np.log(0.15)     # log ring of empty dark squares
        self.occ_lvl = np.log(1.0)        # log ring of pieces
        self.white_lvl = np.log(3.5)      # log r_mean of wooden pieces
        self.black_lvl = np.log(0.70)     # log r_mean of black pieces
        self.fitted = False

    @staticmethod
    def _o(feats):
        return np.log(np.maximum(np.asarray(feats, np.float32)[:, 0], 1e-4))

    @staticmethod
    def _c(feats):
        return np.log(np.maximum(np.asarray(feats, np.float32)[:, 1], 1e-3))

    def _levels(self, feats, labels):
        o, c = self._o(feats), self._c(feats)
        labels = np.asarray(labels)
        out = {}
        if (labels == EMPTY).any():
            out["empty_lvl"] = float(np.median(o[labels == EMPTY]))
        # the occupied level is taken from the BLACK pieces when there are
        # any: they are the hard case (wooden pieces score far higher)
        if (labels == BLACK_P).any():
            out["occ_lvl"] = float(np.median(o[labels == BLACK_P]))
        elif (labels != EMPTY).any():
            out["occ_lvl"] = float(np.median(o[labels != EMPTY]))
        if (labels == WHITE_P).any():
            out["white_lvl"] = float(np.median(c[labels == WHITE_P]))
        if (labels == BLACK_P).any():
            out["black_lvl"] = float(np.median(c[labels == BLACK_P]))
        return out

    def fit(self, feats, labels):
        for k, v in self._levels(feats, labels).items():
            setattr(self, k, v)
        self.fitted = True
        return self

    def update(self, feats, labels):
        a = self.adapt_rate
        for k, v in self._levels(feats, labels).items():
            setattr(self, k, (1 - a) * getattr(self, k) + a * v)
        return self

    def separation(self):
        """(occupancy gap, colour gap) in log units."""
        return self.occ_lvl - self.empty_lvl, self.white_lvl - self.black_lvl

    def predict_proba(self, feats):
        o, c = self._o(feats), self._c(feats)
        gap = max(self.occ_lvl - self.empty_lvl, 0.5)
        base = self.empty_lvl
        if len(o) >= 16:
            # this frame's empty-square level: there are always >= 8 empty
            # dark squares; allowed to drift a little from the calibration
            frame = float(np.median(np.sort(o)[:8]))
            base = float(np.clip(frame, self.empty_lvl - 0.3 * gap, self.empty_lvl + 0.3 * gap))
        t_occ = base + self.OCC_K * gap
        p_occ = _sigmoid((o - t_occ) / (gap / 20.0))
        cgap = max(self.white_lvl - self.black_lvl, 0.3)
        # black pieces' relative brightness is stable across lighting
        # (0.56-0.82), the wooden pieces' is not (2.0 under strong light,
        # 8.0 in dim light) - so the threshold sits closer to black
        t_col = self.black_lvl + self.COL_K * cgap
        p_white = _sigmoid((c - t_col) / 0.06)
        return np.stack([1.0 - p_occ, p_occ * p_white, p_occ * (1.0 - p_white)], axis=1)


def start_position_labels(parity):
    """Labels for the dark squares of the standard start position in game
    orientation (black on rows 0-2)."""
    cells, labels = [], []
    for r in range(8):
        for c in range(8):
            if not is_dark(r, c, parity):
                continue
            cells.append((r, c))
            labels.append(BLACK_P if r <= 2 else WHITE_P if r >= 5 else EMPTY)
    return cells, np.array(labels)


# --------------------------------------------------------------------------
# orientation from the start position
# --------------------------------------------------------------------------
def occupancy_score(feats):
    """Unsupervised 'something is on this square' score (log ring)."""
    return np.log(np.maximum(feats[:, 0], 1e-4))


def find_start_orientation(image, corners):
    """Try the 4 rotations; return (rotation, parity, score) for the one that
    looks like the start position in game orientation (dark square parity 1,
    white pieces on rows 5-7, black on rows 0-2), or (None, None, reason)."""
    results = []
    for rot in range(4):
        w = warp(image, corners, rot)
        g = to_gray(w)
        par = dark_parity(g)
        norm, _, _ = normalise(g, par)
        cells, f = cell_features(w, norm, par)
        occ = occupancy_score(f)
        rows = np.array([r for r, _ in cells])
        bright = np.log(np.maximum(f[:, 1], 1e-3))
        top = rows <= 2
        bottom = rows >= 5
        mid = (rows >= 3) & (rows <= 4)
        # occupancy pattern: top+bottom occupied, middle empty
        occ_sep = min(occ[top].mean(), occ[bottom].mean()) - occ[mid].mean()
        # white pieces (bright) at the bottom, black (not bright) at the top
        col_sep = bright[bottom].mean() - bright[top].mean()
        score = occ_sep + col_sep + (0.25 if par == 1 else -1.0)
        results.append((score, rot, par, occ_sep, col_sep))
    results.sort(reverse=True)
    best = results[0]
    if best[3] <= 0.8 or best[4] <= 0.5:
        return None, None, ("could not recognise the start position "
                            f"(occupancy sep {best[3]:.2f}, colour sep {best[4]:.2f})")
    return best[1], best[2], best[0]


# --------------------------------------------------------------------------
# change detection against a reference frame
# --------------------------------------------------------------------------
def change_scores(ref_norm, cur_norm):
    """8x8 robust z-scores of how much each square's centre changed.

    Both inputs are normalise()d images, so a global lighting change is
    already compensated; the median over all squares removes what is left
    (a move changes at most a few squares)."""
    diff = np.abs(cv2.GaussianBlur(cur_norm, (5, 5), 0) - cv2.GaussianBlur(ref_norm, (5, 5), 0))
    raw = np.zeros((8, 8), np.float32)
    m = CELL // 2
    inner = slice(m - DISK_R - SEARCH, m + DISK_R + SEARCH)
    for r in range(8):
        for c in range(8):
            ys, xs = cell_slice(r, c)
            raw[r, c] = float(diff[ys, xs][inner, inner].mean())
    med = float(np.median(raw))
    mad = float(np.median(np.abs(raw - med))) * 1.4826 + 0.01
    return (raw - med) / mad, raw
