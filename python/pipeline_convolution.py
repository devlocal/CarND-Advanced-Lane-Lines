import cv2
import numpy as np
from scipy import signal


class PipelineConvolution(object):
    SOBEL_KERNEL_SIZE = 25
    LANE_FILTER_WINDOW = np.array(
        [[-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
        dtype='float'
    )
    SHADOW_FILTER_WINDOW = np.array([[
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    ]], dtype='float')

    def _get_lane_signal(self, channel):
        """Detects lanes"""

        sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=self.SOBEL_KERNEL_SIZE)
        lane_signal = signal.convolve2d(sobelx, self.LANE_FILTER_WINDOW, 'same')
        lane_signal = lane_signal / lane_signal.max()
        return lane_signal

    def _get_shade_signal(self, channel):
        """Detects transition between shade and lit areas"""

        shade_signal = signal.convolve2d(channel, self.SHADOW_FILTER_WINDOW, 'same')
        shade_signal = np.absolute(shade_signal)
        shade_signal = shade_signal * 2.0 / shade_signal.max()
        return shade_signal

    @staticmethod
    def _filter_lane_signal(lane_signal, shade_signal):
        """Filters lane signal by zeroing values in shade transition areas"""
        out_signal = lane_signal - shade_signal

        # Scale values to max=255 by looking at center area only
        h, w = out_signal.shape
        xmin, xmax = int(w * 0.3), int(w * 0.7)
        ymin, ymax = int(h * 0.3), int(h * 0.7)
        center_area = out_signal[xmin:xmax, ymin:ymax]

        return np.clip(out_signal * 255.0 / center_area.max(), 0, 255).astype(np.uint8)

    @staticmethod
    def _threshold(channel, low, high):
        """Applies threshold filter"""

        out = np.zeros_like(channel)
        out[(channel >= low) & (channel <= high)] = 255
        return out.astype(np.uint8)

    def _one_ch_sobel_filter(self, channel):
        """Applies Sobel and convolution to get a filtered image of lane lines"""

        lane_signal = self._get_lane_signal(channel)
        shade_signal = self._get_shade_signal(channel)
        out_signal = self._filter_lane_signal(lane_signal, shade_signal)

        return out_signal

    def apply(self, image):
        """Applies pipeline transformations to a single frame"""

        # Convert image to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float)

        # Extract channels
        hsv_s = hsv[:, :, 1]
        hsv_v = hsv[:, :, 2]
        lab_l = lab[:, :, 0]

        # Apply channel transformations
        ch_hsv_s = self._one_ch_sobel_filter(hsv_s)
        ch_hsv_v = self._threshold(hsv_v, 220, 255)
        ch_lab_l = self._one_ch_sobel_filter(lab_l)

        # Consolidate channel data
        out_channel = ch_hsv_s + ch_hsv_v + ch_lab_l
        out_channel = out_channel * 255.0 / out_channel.max()

        # Compose visualisation image
        visualisation = np.dstack((ch_hsv_s, ch_hsv_v, ch_lab_l)).astype(np.uint8)

        return visualisation, out_channel
