import threading
import time

import cv2
import numpy as np
from ximea import xiapi

from checkers_game.constants import resource_path

# Exposure limits for the software auto-exposure (microseconds).  Recorded
# data: 80 ms (the old fixed value) saturates the white squares under the
# lamp + daylight, 20-40 ms is far too dark with the room lights off.
EXPOSURE_MIN_US = 3000
EXPOSURE_MAX_US = 200000
DEFAULT_EXPOSURE_US = 40000


class XimeaCamera:
    def __init__(self, exposure_us=DEFAULT_EXPOSURE_US):
        self._lock = threading.Lock()      # capture thread + exposure changes
        self.exposure_us = int(exposure_us)
        self.cam = self.get_camera()
        self.img = xiapi.Image()
        self._calib = None   # remap tables, computed once per image size
        self._last_ae = 0.0

    def get_camera(self):
        # create instance for first connected camera
        camera = xiapi.Camera()

        # start communication
        # to open specific device, use:
        # cam.open_device_by_SN('41305651')
        # (open by serial number)
        print('Opening first camera...')
        camera.open_device()

        # settings
        camera.set_exposure(self.exposure_us)
        camera.set_param('imgdataformat', "XI_RGB32")
        camera.set_param("auto_wb", 1)
        print('Exposure was set to %i us' % camera.get_exposure())

        # start data acquisition
        print('Starting data acquisition...')
        camera.start_acquisition()

        return camera

    def get_camera_image(self):
        with self._lock:
            self.cam.get_image(self.img)
            image = self.img.get_image_data_numpy(invert_rgb_order=False)

        image = cv2.resize(image, (600, 600))

        # return image #for calibration without undistortion
        # return self.undistort(image)
        return cv2.rotate(self.undistort(image), cv2.ROTATE_180)

    # ------------------------------------------------------------------
    # exposure
    # ------------------------------------------------------------------
    def set_exposure(self, exposure_us):
        exposure_us = int(np.clip(exposure_us, EXPOSURE_MIN_US, EXPOSURE_MAX_US))
        with self._lock:
            self.cam.set_exposure(exposure_us)
            try:
                self.exposure_us = int(self.cam.get_exposure())
            except Exception:  # noqa: BLE001
                self.exposure_us = exposure_us
        return self.exposure_us

    @staticmethod
    def image_level(image):
        """Brightness of the bright part of the scene (90th percentile of
        the central area) - the board's white squares under the camera."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        return float(np.percentile(gray[h // 8:-h // 8, w // 8:-w // 8], 90))

    def auto_expose(self, target=190.0, measure=None, max_iter=10, tol=0.08, verbose=True):
        """Software auto-exposure: scale the exposure until measure(image)
        (default: image_level) is within `tol` of `target`.  Returns the
        exposure in microseconds."""
        measure = measure or self.image_level
        for _ in range(max_iter):
            for _flush in range(2):           # frames still exposed with the old value
                self.get_camera_image()
            level = float(measure(self.get_camera_image()))
            if abs(level - target) <= tol * target:
                break
            # a saturated image under-reports how much too bright it is
            ratio = target / max(level, 1.0)
            if level >= 250:
                ratio = min(ratio, 0.6)
            new = self.exposure_us * float(np.clip(ratio, 0.35, 3.0))
            if int(np.clip(new, EXPOSURE_MIN_US, EXPOSURE_MAX_US)) == self.exposure_us:
                break                           # at a limit
            self.set_exposure(new)
        if verbose:
            print(f"Auto-exposure: {self.exposure_us} us (level {level:.0f}, target {target:.0f})")
        self._last_ae = time.time()
        return self.exposure_us

    def track_exposure(self, level, target=190.0, low=120.0, high=238.0, min_interval=2.0):
        """Gentle continuous correction (call with the board's white level of
        the current frame): only acts when the level leaves [low, high], e.g.
        when the lamp is switched on/off, and at most every min_interval s.
        Returns True if the exposure was changed."""
        if low <= level <= high or time.time() - self._last_ae < min_interval:
            return False
        ratio = float(np.clip(target / max(level, 1.0), 0.5, 2.0))
        if level >= 250:
            ratio = min(ratio, 0.7)
        old = self.exposure_us
        self.set_exposure(old * ratio)
        self._last_ae = time.time()
        if self.exposure_us != old:
            print(f"[camera] lighting changed (board white level {level:.0f}) -> "
                  f"exposure {old} -> {self.exposure_us} us")
            return True
        return False

    def undistort(self, image):
        """Undistort with the stored calibration.  The remap tables are
        computed once per image size (was: file load + full undistort setup
        on every frame)."""
        h, w = image.shape[:2]
        if self._calib is None or self._calib[0] != (w, h):
            path = resource_path('camera', 'calibration_parameters_ncr_3.npz')
            with np.load(path) as X:
                mtx, dist = X['mtx'], X['dist']
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
            map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx,
                                                     (w, h), cv2.CV_16SC2)
            self._calib = ((w, h), map1, map2, roi)
        _, map1, map2, roi = self._calib
        dst = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
        x, y, rw, rh = roi
        return dst[y:y + rh, x:x + rw]
