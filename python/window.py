import numpy as np
from scipy import signal


class SlidingWindow(object):
    # window settings
    WIDTH = 50
    HEIGHT = 80  # Break image into 9 vertical layers since image height is 720
    MARGIN = 50  # How much to slide left and right for searching

    CONV_MODE = 'same'

    LOW_CONFIDENCE_THRESHOLD = 15000

    def __init__(self, image_width, image_height):
        assert image_height % self.HEIGHT == 0, "Image height is not divisible by window height"
        assert image_width % 2 == 0, "The algorithm has not been tested with odd image width"

        self.image_width = image_width
        self.image_height = image_height

        self.n_layers = int(image_height / self.HEIGHT)
        # Window template that will be used for convolutions
        self.window = signal.gaussian(self.WIDTH, self.WIDTH / 2)
        self.half_width = int(self.image_width / 2)

        n_fade_points = int(self.image_width * 3 / 8)
        self.left_weights = np.ones(shape=(self.half_width,), dtype=np.float)
        self.left_weights[:n_fade_points] = np.linspace(0, 1, n_fade_points)
        self.right_weights = np.ones(shape=(self.half_width,), dtype=np.float)
        self.right_weights[self.half_width - n_fade_points:] = np.linspace(1, 0, n_fade_points)

    def _get_image_slice(self, image, layer):
        slice_top = int(self.image_height - (layer + 1) * self.HEIGHT)
        slice_bottom = int(self.image_height - layer * self.HEIGHT)
        return np.sum(image[slice_top:slice_bottom, :], axis=0)

    def _find_centroid(self, conv_signal, prev_centroid):
        """Finds the best centroid by using past center as a reference"""

        min_index = int(max(prev_centroid - self.MARGIN, 0))
        max_index = int(min(prev_centroid + self.MARGIN, self.image_width))
        return np.argmax(conv_signal[min_index:max_index]) + min_index

    def _find_initial_centroids(self, image):
        """
        Finds the two starting positions for the left and right lane by using np.sum to get the vertical
        image slice and then np.convolve the vertical image slice with the window template.
        """

        # Sum quarter bottom of image to get slice
        l_sum = np.sum(image[int(3 * self.image_height / 4):, :self.half_width], axis=0)
        l_convolution = np.convolve(self.window, l_sum, self.CONV_MODE) * self.left_weights
        l_idx = np.argmax(l_convolution)
        l_confidence = l_convolution[l_idx]

        r_sum = np.sum(image[int(3 * self.image_height / 4):, self.half_width:], axis=0)
        r_convolution = np.convolve(self.window, r_sum, self.CONV_MODE) * self.right_weights
        r_idx = np.argmax(r_convolution)
        r_confidence = r_convolution[r_idx]

        return l_idx, r_idx + self.half_width, l_confidence, r_confidence

    def find_window_centroids(self, image, prev_centroids):
        """
        Finds lane lines centroids using sliding window algorithm.

        :param image: a single channel image with lane lines
        :param prev_centroids: centroid coordinates found on a previous step, a tuple (left_centroids, right_centroids)
        :return: a tuple of (left_centroids, right_centroids, left_confidence, right_confidence)
        """

        assert self.image_height, self.image_width == image.shape

        # Store the (left,right) window centroid positions per layer
        left_centroids = []
        right_centroids = []
        left_confidence = []
        right_confidence = []

        if prev_centroids:
            start_layer = 0
            l_center = prev_centroids[0][0]
            r_center = prev_centroids[1][0]
        else:
            l_center, r_center, l_confidence, r_confidence = self._find_initial_centroids(image)

            # Add what we found for the first layer
            left_centroids.append(l_center)
            right_centroids.append(r_center)

            left_confidence.append(l_confidence)
            right_confidence.append(r_confidence)

            start_layer = 1

        # Go through each layer looking for max pixel locations
        for layer in range(start_layer, self.n_layers):
            # convolve the window into the vertical slice of the image
            conv_signal = np.convolve(self.window, self._get_image_slice(image, layer), self.CONV_MODE)

            # Find the best left and right centroids
            l_new_center = self._find_centroid(conv_signal, l_center)
            r_new_center = self._find_centroid(conv_signal, r_center)

            l_confidence = conv_signal[l_center]
            r_confidence = conv_signal[r_center]

            if l_confidence < self.LOW_CONFIDENCE_THRESHOLD:
                try:
                    # Use l_center from previous frame
                    l_center = prev_centroids[0][layer]
                except TypeError:
                    # Use l_center from previous layer
                    pass
            else:
                l_center = l_new_center

            if r_confidence < self.LOW_CONFIDENCE_THRESHOLD:
                try:
                    # Use r_center from previous frame
                    r_center = prev_centroids[1][layer]
                except TypeError:
                    # Use r_center from previous layer
                    pass
            else:
                r_center = r_new_center

            # Add what we found for that layer
            left_centroids.append(l_center)
            right_centroids.append(r_center)

            left_confidence.append(l_confidence)
            right_confidence.append(r_confidence)

        assert len(left_centroids) == len(right_centroids) == self.n_layers
        return left_centroids, right_centroids, left_confidence, right_confidence
