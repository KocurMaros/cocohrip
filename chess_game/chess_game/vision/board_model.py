"""3-D model of the board under the camera: pose, square outlines and
perspective-normalised piece crops.  OpenCV + numpy only (testable without
a camera).

The camera looks at the board obliquely (from above the a8 corner), so a
piece is seen from the top near a8 and from the side near h1, and tall
pieces lean over the squares behind them.  With the camera pose
(solvePnP on the four board corners) every square's vertical axis can be
projected into the image; `column_crop` cuts a fixed-size crop around it,
rotated so the piece always stands upright with its base at the same spot.
A classifier then sees every piece the same way up, wherever it stands.

Board frame: x along the files (a -> h), y along the ranks (1 -> 8), z up,
one unit = one square, origin at the outer corner of printed a1.
"""

import cv2
import numpy as np

CORNER_NAMES = ("BL", "TL", "TR", "BR")      # order of find_board_corners()
SQUARE_M = 0.0566                             # measured with the robot (H8 - A1)

# Calibration of the Ximea lens: done on the 600x600 frames the checkers
# node used (the full 2464x2056 sensor squeezed to 600x600, NOT rotated).
CALIB_SIZE = (600, 600)


def scaled_intrinsics(mtx, size, calib_size=CALIB_SIZE):
    """Camera matrix for frames of `size` (w, h) taken from the same sensor
    as the calibration frames (distortion coefficients are unchanged)."""
    K = np.array(mtx, dtype=np.float64).copy()
    K[0] *= size[0] / float(calib_size[0])
    K[1] *= size[1] / float(calib_size[1])
    return K


def load_intrinsics(npz_path, size):
    with np.load(npz_path) as X:
        return scaled_intrinsics(X["mtx"], size), np.array(X["dist"], dtype=np.float64)


def corner_board_points(a1_corner="BR"):
    """Board-frame (x, y) of the image corners BL, TL, TR, BR, given which of
    them is the outer corner of printed a1.  The board is seen from above,
    so going BL -> TL -> TR -> BR (clockwise on screen) visits the board
    corners clockwise seen from above too: a1, a8, h8, h1."""
    ring = [(0, 0), (0, 8), (8, 8), (8, 0)]          # a1, a8, h8, h1
    start = CORNER_NAMES.index(a1_corner)
    return [ring[(i - start) % 4] for i in range(4)]


class BoardModel:
    def __init__(self, corners, K, dist, a1_corner="BR", square_m=SQUARE_M):
        """corners: BL, TL, TR, BR of the 8x8 squares in image pixels."""
        self.corners = np.asarray(corners, dtype=np.float64).reshape(4, 2)
        self.K = np.asarray(K, dtype=np.float64)
        self.dist = np.asarray(dist, dtype=np.float64).reshape(-1)
        self.a1_corner = a1_corner
        self.square_m = float(square_m)
        obj = np.array([(x, y, 0.0) for x, y in corner_board_points(a1_corner)])
        self.rvec, self.tvec = self._solve_pose(obj * self.square_m)
        R, _ = cv2.Rodrigues(self.rvec)
        self.camera_position = (-R.T @ self.tvec).ravel() / self.square_m   # squares
        rep = self.project(obj)
        self.reprojection_error = float(np.linalg.norm(rep - self.corners, axis=1).max())

    def _solve_pose(self, obj):
        img = self.corners.reshape(-1, 1, 2)
        n, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, img, self.K, self.dist,
                                                 flags=cv2.SOLVEPNP_IPPE)
        best = None
        for rvec, tvec in zip(rvecs[:n], tvecs[:n]):
            R, _ = cv2.Rodrigues(rvec)
            cam_z = float((-R.T @ tvec).ravel()[2])
            rep, _ = cv2.projectPoints(obj, rvec, tvec, self.K, self.dist)
            err = float(np.linalg.norm(rep.reshape(-1, 2) - self.corners, axis=1).sum())
            if cam_z > 0 and (best is None or err < best[0]):     # camera above the board
                best = (err, rvec, tvec)
        if best is None:
            raise ValueError("no camera pose above the board fits the corners")
        return best[1], best[2]

    # ------------------------------------------------------------ projection
    def project(self, pts_squares):
        """Board-frame points (in squares, z up) -> image pixels (N, 2)."""
        pts = np.asarray(pts_squares, dtype=np.float64).reshape(-1, 3) * self.square_m
        img, _ = cv2.projectPoints(pts, self.rvec, self.tvec, self.K, self.dist)
        return img.reshape(-1, 2)

    @staticmethod
    def square_xy(sq):
        """Centre of a printed square (python-chess index) in the board frame."""
        return (sq % 8) + 0.5, (sq // 8) + 0.5

    def square_polygon(self, sq, shrink=0.0):
        x, y = self.square_xy(sq)
        h = 0.5 - shrink / 2.0
        return self.project([(x - h, y - h, 0), (x - h, y + h, 0),
                             (x + h, y + h, 0), (x + h, y - h, 0)])

    def square_centre(self, sq, height=0.0):
        x, y = self.square_xy(sq)
        return self.project([(x, y, height)])[0]

    def image_to_board(self, pts):
        """Image pixels -> board-frame (x, y) on the board plane."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        und = cv2.undistortPoints(pts, self.K, self.dist).reshape(-1, 2)
        R, _ = cv2.Rodrigues(self.rvec)
        t = self.tvec.ravel()
        # ray: X = s * R^T d - R^T t ; intersect with z = 0
        out = []
        for u, v in und:
            d = R.T @ np.array([u, v, 1.0])
            o = -R.T @ t
            s = -o[2] / d[2]
            out.append((o + s * d)[:2] / self.square_m)
        return np.array(out)

    # ------------------------------------------------------------ crops
    def column_crop(self, image, sq, height=1.8, size=(72, 128), below=0.65, width=1.3,
                    span=2.3):
        """Upright crop around the vertical axis of a square.

        The square centre ends up `below` squares above the bottom edge, the
        direction "up the piece" points up, the crop is `width` x `span`
        squares (local scale) and is resized to `size` (w, h).  `height` is
        the height of the column (in squares) used to find "up"; near the
        point straight under the camera the column projects to almost
        nothing and the board's rank direction is used instead."""
        x, y = self.square_xy(sq)
        base, top, ahead = self.project([(x, y, 0), (x, y, height), (x, y + 1, 0)])
        s = float(np.sqrt(abs(cv2.contourArea(self.square_polygon(sq).astype(np.float32)))))
        up = top - base
        if np.linalg.norm(up) < 0.15 * s:
            up = ahead - base
        up /= np.linalg.norm(up)
        right = np.array([-up[1], up[0]])
        bottom = base - below * s * up
        src = np.float32([bottom, bottom - right * (width * s / 2), bottom + up * (span * s)])
        w, h = size
        dst = np.float32([(w / 2, h), (0, h), (w / 2, 0)])
        M = cv2.getAffineTransform(src, dst)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_AREA,
                              borderMode=cv2.BORDER_REPLICATE)

    def all_crops(self, image, **kw):
        return [self.column_crop(image, sq, **kw) for sq in range(64)]

    def column_squares(self, sq, height=1.8):
        """Printed squares the projected column of `sq` passes over in the
        image (a tall piece on `sq` hides / changes them too)."""
        x, y = self.square_xy(sq)
        pts = self.project([(x, y, height * t) for t in np.linspace(0, 1, 7)])
        board_xy = self.image_to_board(pts)
        out = {sq}
        for bx, by in board_xy:
            if 0 <= bx < 8 and 0 <= by < 8:
                out.add(int(by) * 8 + int(bx))
        return sorted(out)


def contact_sheet(crops, labels=None, cols=8):
    """Tile 64 crops into one image (a8 at the top left, like a diagram)."""
    h, w = crops[0].shape[:2]
    rows = (len(crops) + cols - 1) // cols
    sheet = np.full((rows * h, cols * w, 3), 40, np.uint8)
    for i, crop in enumerate(crops):
        sq = i
        r, c = 7 - sq // 8, sq % 8
        tile = crop if crop.ndim == 3 else cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = tile[:, :, :3]
        if labels is not None:
            cv2.putText(sheet, str(labels[i]), (c * w + 3, r * h + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    return sheet
