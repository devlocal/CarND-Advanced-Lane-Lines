import cv2
import numpy as np

from calibration import get_calibration_params


class PerspectiveTransformer(object):
    CALIBRATION_DATA_FOLDER = '../camera_cal'

    def __init__(self, image_width, image_height):
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = get_calibration_params(self.CALIBRATION_DATA_FOLDER)

        self._M = None  # transformation matrix
        self._Minv = None  # inverse transformation matrix

        self.width = image_width
        self.height = image_height

        self._compute_transformation_matrix()

    def _compute_transformation_matrix(self):
        # Manually located points of lane lines
        src = np.float32((
            (586, 455),
            (693, 455),
            (189, 718),
            (1112, 718)
        ))

        wp = 0.35
        dst = np.float32((
            (self.width * wp, self.height * 0.1),
            (self.width * (1 - wp), self.height * 0.1),
            (self.width * wp, self.height),
            (self.width * (1 - wp), self.height))
        )

        self._M = cv2.getPerspectiveTransform(src, dst)
        self._M_inv = cv2.getPerspectiveTransform(dst, src)

        return self._M

    def unwarp(self, image):
        """
        Unwraps camera image to a birds-eye view image

        :param image: camera image
        :return: birds-eye view image
        """
        assert image.shape[:2] == (self.height, self.width)

        undist = cv2.undistort(image, self.mtx, self.dist)
        return cv2.warpPerspective(undist, self._M, (self.width, self.height), flags=cv2.INTER_LINEAR)

    def warp(self, image):
        """
        Warps birds-eye view image back to perspective image

        :param image: birds=eye view image
        :return: perspective image
        """
        assert image.shape[:2] == (self.height, self.width)

        return cv2.warpPerspective(image, self._M_inv, (self.width, self.height), flags=cv2.INTER_LINEAR)
