"""Ximea camera for chess: full-resolution frames in the sensor's own
orientation (no resize, no undistortion, no rotation - BoardModel projects
with the lens distortion instead, and the GUI rotates only for display).

The checkers camera (checkers_game.camera.ximea_camera) squeezes frames to
600x600, which is too coarse to tell a bishop from a pawn.
"""

import threading
import time

import cv2
import numpy as np

EXPOSURE_MIN_US = 3000
EXPOSURE_MAX_US = 250000
GAIN_MAX_DB = 12.0            # used only when the exposure is at its maximum (dark room)

# Image orientation for people: the human sits on the printed rank-1 side,
# which is the right edge of the raw image -> rotate so it is at the bottom
# (a standard diagram from white's side: a1 bottom left).
DISPLAY_ROTATION = cv2.ROTATE_90_CLOCKWISE


def to_display(image):
    return cv2.rotate(image, DISPLAY_ROTATION)


def display_point(pt, raw_shape):
    """Raw-image pixel -> display pixel (for DISPLAY_ROTATION)."""
    h = raw_shape[0]
    x, y = pt
    return h - 1 - y, x


def raw_point(pt, raw_shape):
    """Display pixel -> raw-image pixel (inverse of display_point)."""
    h = raw_shape[0]
    x, y = pt
    return y, h - 1 - x


def white_level(image, mask=None):
    """90th percentile of the grey level (the board's white squares)."""
    gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    vals = gray[mask] if mask is not None else gray[gray.shape[0] // 8:-gray.shape[0] // 8,
                                                    gray.shape[1] // 8:-gray.shape[1] // 8]
    return float(np.percentile(vals, 90))


class ChessCamera:
    def __init__(self, exposure_us=40000, serial=None):
        from ximea import xiapi
        self._xiapi = xiapi
        self._lock = threading.Lock()
        self.cam = xiapi.Camera()
        if serial:
            self.cam.open_device_by_SN(serial)
        else:
            self.cam.open_device()
        sn = self.cam.get_device_sn()
        self.serial = sn.decode() if isinstance(sn, bytes) else str(sn)
        self.cam.set_param("imgdataformat", "XI_RGB24")
        self.cam.set_param("auto_wb", 1)
        self.exposure_us = 0
        self.gain_db = 0.0
        self.set_gain(0.0)
        self.set_exposure(exposure_us)
        self.img = xiapi.Image()
        self.cam.start_acquisition()
        self._first_frame()

    def _first_frame(self):
        # after a reboot the first acquisition sometimes never delivers a
        # frame - restarting the acquisition once fixes it
        for attempt in range(3):
            try:
                self.grab(timeout_ms=3000)
                return
            except self._xiapi.Xi_error as e:
                print(f"[camera] no frame ({e}) - restarting acquisition")
                self.cam.stop_acquisition()
                time.sleep(0.5)
                self.cam.start_acquisition()
        raise RuntimeError("Ximea camera delivers no frames - unplug/replug it, or run "
                           "/opt/XIMEA/bin/xiSample once")

    def grab(self, timeout_ms=5000):
        """Newest full-resolution BGR frame (raw orientation)."""
        with self._lock:
            self.cam.get_image(self.img, timeout_ms)
            return self.img.get_image_data_numpy().copy()

    def grab_fresh(self, flush=2):
        """A frame exposed with the current settings (drops `flush` frames
        that may still use the old exposure)."""
        for _ in range(flush):
            self.grab()
        return self.grab()

    def set_exposure(self, exposure_us):
        exposure_us = int(np.clip(exposure_us, EXPOSURE_MIN_US, EXPOSURE_MAX_US))
        with self._lock:
            self.cam.set_exposure(exposure_us)
            try:
                self.exposure_us = int(self.cam.get_exposure())
            except Exception:  # noqa: BLE001 - keep the requested value
                self.exposure_us = exposure_us
        return self.exposure_us

    def set_gain(self, gain_db):
        gain_db = float(np.clip(gain_db, 0.0, GAIN_MAX_DB))
        with self._lock:
            try:
                self.cam.set_gain(gain_db)
                self.gain_db = float(self.cam.get_gain())
            except Exception:  # noqa: BLE001 - camera without gain control
                self.gain_db = 0.0
        return self.gain_db

    def adjust(self, ratio):
        """Brighten / darken by `ratio`: exposure first, gain only beyond the
        longest exposure (and taken back first when darkening)."""
        if ratio < 1.0 and self.gain_db > 0.0:
            new_gain = self.gain_db + 20.0 * np.log10(ratio)
            self.set_gain(max(new_gain, 0.0))
            if new_gain >= 0.0:
                return
            ratio = 10 ** (new_gain / 20.0)
        old = self.exposure_us
        new = self.set_exposure(old * ratio)
        if ratio > 1.0 and new >= EXPOSURE_MAX_US:
            rest = ratio * old / max(new, 1)
            self.set_gain(self.gain_db + 20.0 * np.log10(max(rest, 1.0)))

    def auto_expose(self, measure=white_level, target=190.0, tol=0.08, max_iter=10):
        """Exposure (then gain) until measure(frame) is near `target`."""
        level = 0.0
        for _ in range(max_iter):
            level = float(measure(self.grab_fresh()))
            if abs(level - target) <= tol * target:
                break
            ratio = float(np.clip(target / max(level, 1.0), 0.35, 3.0))
            if level >= 250:
                ratio = min(ratio, 0.6)
            before = (self.exposure_us, self.gain_db)
            self.adjust(ratio)
            if (self.exposure_us, self.gain_db) == before:
                break                                   # at a limit
        return self.exposure_us, level

    def close(self):
        try:
            self.cam.stop_acquisition()
            self.cam.close_device()
        except Exception:  # noqa: BLE001
            pass


class ImageFolderCamera:
    """Stand-in camera replaying JPEGs (testing the tools without hardware)."""

    def __init__(self, paths, exposure_us=40000):
        self.paths = list(paths)
        self.i = 0
        self.exposure_us = exposure_us
        self.gain_db = 0.0
        self.serial = "replay"

    def grab(self, timeout_ms=0):
        img = cv2.imread(self.paths[self.i % len(self.paths)])
        return img

    def grab_fresh(self, flush=0):
        return self.grab()

    def next_image(self):
        self.i += 1

    def set_exposure(self, exposure_us):
        self.exposure_us = int(np.clip(exposure_us, EXPOSURE_MIN_US, EXPOSURE_MAX_US))
        return self.exposure_us

    def adjust(self, ratio):
        self.set_exposure(self.exposure_us * ratio)

    def auto_expose(self, measure=white_level, **kw):
        return self.exposure_us, measure(self.grab())

    def close(self):
        pass
