import glob
import logging
import os
import pickle
from operator import mul

import cv2
import numpy as np

DATA_FILE_NAME = os.path.join(os.path.abspath(os.path.dirname(__file__)), "calibration.p")


def calibrate_camera(data_folder):
    """
    Calibrates camera using chessboard images.

    :param data_folder: folder with chessboard images
    :return: ret, mtx, dist, rvecs, tvecs, see cv2.calibrateCamera for details
    """

    objpoints = []
    imgpoints = []
    chessboard_size = (9, 6)

    objp = np.zeros((mul(*chessboard_size), 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    shape = None

    files = glob.glob(os.path.join(data_folder, "*.jpg"))
    for file_name in files:
        img = cv2.imread(file_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if shape is None:
            shape = gray.shape[::-1]
        else:
            if shape != gray.shape[::-1]:
                logging.warning("Unexpected shape %s of %s, expected %s", str(gray.shape[::-1]), file_name, str(shape))
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)

    return cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)


def get_calibration_params(data_folder):
    """
    Computes camera calibration parameters. Caches computed values on disk and reloads on subsequent invocations.

    :param data_folder: folder with chessboard images
    :return: ret, mtx, dist, rvecs, tvecs, see cv2.calibrateCamera for details
    """

    try:
        with open(DATA_FILE_NAME, "rb") as f:
            ret = pickle.load(f)
            mtx = pickle.load(f)
            dist = pickle.load(f)
            rvecs = pickle.load(f)
            tvecs = pickle.load(f)
    except Exception:
        ret, mtx, dist, rvecs, tvecs = calibrate_camera(data_folder)
        with open(DATA_FILE_NAME, "wb") as f:
            pickle.dump(ret, f)
            pickle.dump(mtx, f)
            pickle.dump(dist, f)
            pickle.dump(rvecs, f)
            pickle.dump(tvecs, f)
    return ret, mtx, dist, rvecs, tvecs
