import cv2
import numpy as np


class PipelineHls(object):
    DEFAULT_S_THRESH = (110, 255)  # (170, 255)
    DEFAULT_SX_THRESH = (50, 100)  # (20, 100)

    def __init__(self, s_thresh=DEFAULT_S_THRESH, sx_thresh=DEFAULT_SX_THRESH):
        self.s_thresh = s_thresh
        self.sx_thresh = sx_thresh

    def apply(self, img):
        # Convert to HLS color space and separate the V channel
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= self.s_thresh[0]) & (s_channel <= self.s_thresh[1])] = 255

        # Sobel x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
        abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= self.sx_thresh[0]) & (scaled_sobel <= self.sx_thresh[1])] = 255

        # Stack channels
        color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)).astype(np.uint8)
        final = cv2.cvtColor(color_binary, cv2.COLOR_BGR2GRAY)
        visualisation = np.uint8(color_binary)

        return visualisation, final
