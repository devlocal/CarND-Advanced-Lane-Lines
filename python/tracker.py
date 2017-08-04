import cv2
import numpy as np

from curve import LaneCurve
from distortion import PerspectiveTransformer
from visualization import LaneVisualization, SearchWindowVisualization, MapThumbnailVisualization, \
    FrameNumberVisualization, VisualizationImageLayer, TurnRadiusVisualization, PositionVisualization
from window import SlidingWindow


class LanePixelsComputer(object):
    """Computes lane pixels from centroids"""

    def __init__(self, image_height):
        self._image_height = image_height

    def compute_from_centroids(self, centroids):
        """
        Computes lane pixels for a single lane line.

        :param centroids: lane centroids, go from bottom to top.
        """

        pixels = np.zeros((self._image_height,))
        n_centroids = len(centroids)
        assert self._image_height % n_centroids == 0
        window_height = int(self._image_height / n_centroids)

        for level in range(n_centroids):
            y_min = int(self._image_height - (level + 1) * window_height)
            y_max = int(self._image_height - level * window_height)

            pixels[y_min:y_max] = centroids[level]

        return pixels


class LaneTracker(object):
    """Cross-frame lane lines tracker"""

    def __init__(self, pipeline, draw_contour=False, draw_windows=False, draw_frame_number=False,
                 visualize_perspective=True):
        """
        :param pipeline: frame pre-processing pipeline
        :param draw_contour: True to visualise contours, False to draw original frame
        :param draw_windows: True to draw search windows
        :param draw_frame_number: True to draw frame number on each frame
        :param visualize_perspective: True to warp output image to a perspective view,
          False to draw a birds-eye view image
        """
        self.pipeline = pipeline
        self.draw_contour = draw_contour
        self.draw_windows = draw_windows
        self.draw_frame_number = draw_frame_number
        self.visualize_perspective = visualize_perspective
        self._frame_number = 0

        self._initialized = False

    def _initialize(self, frame):
        self.image_height, self.image_width = frame.shape[:2]
        self.transformer = PerspectiveTransformer(self.image_width, self.image_height)

        self.line_curve_left = LaneCurve(self.image_width, self.image_height)
        self.line_curve_right = LaneCurve(self.image_width, self.image_height)

        self.prev_centroids = None

        self.sw = SlidingWindow(self.image_width, self.image_height)
        self.swv = SearchWindowVisualization(self.image_width, self.image_height)
        self.lv = LaneVisualization(self.image_width, self.image_height)
        self.mtv = MapThumbnailVisualization(self.image_width, self.image_height)
        self.trv = TurnRadiusVisualization()
        self.pv = PositionVisualization()
        self.fnv = FrameNumberVisualization()
        self.lane_pixels_computer = LanePixelsComputer(self.image_height)

    def _visualize(self, out_frame, color_contour, left_centroids, right_centroids,
                   left_confidence, right_confidence):
        warp_trans = self.transformer if self.visualize_perspective else None

        if self.draw_windows:
            out_frame = self.swv.draw(out_frame, cv2.cvtColor(color_contour, cv2.COLOR_RGB2GRAY),
                                      left_centroids, right_centroids,
                                      left_confidence, right_confidence, warp_trans)

        out_frame = self.lv.draw(out_frame, self.line_curve_left, self.line_curve_right, warp_trans)

        visualization_layer = VisualizationImageLayer(self.image_width, self.image_height)
        image_buffer = visualization_layer.get_image_buffer()

        self.mtv.draw(image_buffer, self.line_curve_left, self.line_curve_right)
        self.trv.draw(image_buffer, self.line_curve_left, self.line_curve_right)
        self.pv.draw(image_buffer, self.line_curve_left, self.line_curve_right)
        if self.draw_frame_number:
            self.fnv.draw(image_buffer, self._frame_number)

        out_frame = visualization_layer.apply(out_frame)

        return out_frame

    def _apply_single_frame_sliding_window(self, lanes_channel):
        """
        Finds lane line (x, y) coordinates on a pre-processed image and adds them to left and right LaneCurve instances.

        :param lanes_channel: pre-processed single channel image
        """

        left_centroids, right_centroids, left_confidence, right_confidence = \
            self.sw.find_window_centroids(lanes_channel, self.prev_centroids)
        self.prev_centroids = (left_centroids, right_centroids)

        if not (left_centroids and right_centroids):
            raise RuntimeError("Cannot find one or more lane lines")

        l_pixels = self.lane_pixels_computer.compute_from_centroids(left_centroids)
        r_pixels = self.lane_pixels_computer.compute_from_centroids(right_centroids)

        self.line_curve_left.accept(l_pixels)
        self.line_curve_right.accept(r_pixels)

        return left_centroids, right_centroids, left_confidence, right_confidence

    def process_frame(self, frame):
        """
        Processes single video frame.

        :param frame: input video frame
        :return: output video frame
        """

        if not self._initialized:
            self._initialize(frame)
            self._initialized = True
        else:
            assert frame.shape[:2] == (self.image_height, self.image_width)

        undist = self.transformer.unwarp(frame)
        visualisation, lanes_channel = self.pipeline.apply(undist)

        left_centroids, right_centroids, left_confidence, right_confidence = \
            self._apply_single_frame_sliding_window(lanes_channel)

        if self.draw_contour:
            out_frame = visualisation
        else:
            out_frame = frame if self.visualize_perspective else undist

        out_frame = self._visualize(out_frame, visualisation, left_centroids, right_centroids,
                                    left_confidence, right_confidence)
        self._frame_number += 1

        return out_frame
