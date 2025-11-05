import numpy as np
from ximea import xiapi
# import camera.ximea_camera as ximea_camera
import cv2
from ctypes import *

class XimeaCamera:
    def __init__(self):
        self.cam = self.get_camera()
        self.img = xiapi.Image()


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
        camera.set_exposure(80000)
        camera.set_param('imgdataformat', "XI_RGB32")
        camera.set_param("auto_wb", 1)
        print('Exposure was set to %i us' % camera.get_exposure())

        # start data acquisition
        print('Starting data acquisition...')
        camera.start_acquisition()

        return camera
    
    def get_camera_image(self):
        self.cam.get_image(self.img)
        image = self.img.get_image_data_numpy(invert_rgb_order=False)
        
        image = cv2.resize(image, (600, 600))
        
        # return image #for calibration without undistortion
        # return self.undistort(image)
        return cv2.rotate(self.undistort(image), cv2.ROTATE_180)
    
    def undistort(self, image):
        with np.load('src/checkers_game/checkers_game/camera/calibration_parameters_ncr_3.npz') as X:
            mtx, dist = X['mtx'], X['dist']

        # Assuming that 'img' is your distorted image
        h, w = image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

        # Undistort
        dst = cv2.undistort(image, mtx, dist, None, newcameramtx)
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
        return dst