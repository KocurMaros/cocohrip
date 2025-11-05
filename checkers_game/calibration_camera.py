import cv2
import numpy as np
import time
from checkers_game.camera.ximea_camera import XimeaCamera

# chessboard pattern (number of inner corners per a chessboard row and column)
CHECKERBOARD = (8, 6)   # change to the pattern you use (inner corners)
SQUARE_SIZE_MM = 22.0   # physical square size in mm (adjust to your board)
OUTPUT_FILE = 'calibration_parameters_ncr_2.npz'

def collect_and_calibrate(required_images=20):
    cam = XimeaCamera()
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = (np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
                   * SQUARE_SIZE_MM)

    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane

    print("Live camera. Press SPACE to capture when the checkerboard is clearly visible.")
    print("Press 'q' to quit and run calibration (if enough images collected), or ESC to abort.")

    while True:
        frame = cam.get_camera_image()
        vis = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            cv2.drawChessboardCorners(vis, CHECKERBOARD, corners, ret)

        cv2.putText(vis, f"Collected: {len(objpoints)}/{required_images}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow('Calibration', vis)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("Aborted by user.")
            cv2.destroyAllWindows()
            return
        if key == ord('q'):
            break
        if key == 32:  # SPACE -> capture
            if ret:
                # refine corner positions
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                            (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001))
                objpoints.append(objp)
                imgpoints.append(corners2)
                print(f"Captured image {len(objpoints)}")
            else:
                print("Chessboard not found in the frame. Adjust and try again.")

        # optional auto-stop when enough images collected
        if len(objpoints) >= required_images:
            print("Required number of images collected.")
            break

    cv2.destroyAllWindows()

    if len(objpoints) < 3:
        print("Not enough views for calibration (need at least 3).")
        return

    # Calibrate camera
    print("Calibrating...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # compute reprojection error
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    mean_error = total_error / len(objpoints)

    print(f"Calibration done. RMS: {ret}")
    print(f"Mean reprojection error: {mean_error:.6f}")

    # Save parameters
    np.savez(OUTPUT_FILE, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    print(f"Saved calibration to {OUTPUT_FILE}")

if __name__ == "__main__":
    collect_and_calibrate(required_images=20)
