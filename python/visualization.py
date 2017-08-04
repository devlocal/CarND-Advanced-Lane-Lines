import cv2
import numpy as np


class VisualizationImageLayer(object):
    """Semi-transparent visualization layer"""

    TRANSPARENCY = 0.5

    def __init__(self, image_width, image_height):
        self._buffer = np.zeros((image_height, image_width), dtype=np.uint8)

    def get_image_buffer(self):
        return self._buffer

    def apply(self, image):
        """Combines layer content with the target image"""
        layer = np.dstack((self._buffer, self._buffer, self._buffer))
        return cv2.addWeighted(image, 1, layer, self.TRANSPARENCY, 0)


class LaneVisualization(object):
    """Highlights lane"""

    def __init__(self, image_width, image_height):
        self.image_width = image_width
        self.image_height = image_height
        self.y = np.linspace(0, image_height - 1, num=image_height)
        self.y_draw = self.y[::-1]

    def _get_lane_mask(self, line_curve_l, line_curve_r):
        """
        Creates an image with lane area filled green.

        :return: image
        """

        # Create an image to draw the lines on
        warp_zero = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        left_x = line_curve_l.get_x(self.y)
        right_x = line_curve_r.get_x(self.y)

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_x, self.y_draw]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_x, self.y_draw])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        return color_warp

    def draw(self, image, line_curve_l, line_curve_r, transformer=None):
        """Highlights detected lane area on a single image"""

        lane_mask = self._get_lane_mask(line_curve_l, line_curve_r)

        if transformer is not None:
            # Warp back to original image space
            lane_mask = transformer.warp(lane_mask)

        # Combine the result with the original image
        return cv2.addWeighted(image, 1, lane_mask, 0.3, 0)


class SearchWindowVisualization(object):
    """Draws search windows"""

    WIDTH = 50
    HEIGHT = 80
    MARGIN = 50  # How much to slide left and right for searching

    FONT_FACE = cv2.FONT_HERSHEY_PLAIN
    FONT_SCALE = 1
    THICKNESS = 1
    TEXT_COLOR = (255, 255, 255)

    def __init__(self, image_width, image_height):
        self.image_width = image_width
        self.image_height = image_height

    def _create_mask(self, center, level, width=None):
        """
        Creates a blank image of the same dimensions as img_ref and fills mask area with 1
        """
        output = np.zeros((self.image_height, self.image_width))

        if width:
            x_range = (
                max(0, int(center - width / 2)),
                min(int(center + width / 2), self.image_width)
            )
        else:
            x_range = (0, self.image_width)

        y_range = (
            int(self.image_height - (level + 1) * self.HEIGHT),
            int(self.image_height - level * self.HEIGHT)
        )

        output[y_range[0]:y_range[1], x_range[0]:x_range[1]] = 1
        return output, x_range, y_range

    def _draw_centroids(self, image1, image2, grayscale_contour, centroids, confidence):
        # Go through each level and draw the windows
        for level in range(0, len(centroids)):
            # Window_mask is a function to draw window areas
            margin_mask = self._create_mask(centroids[level], level, None if level == 0 else self.MARGIN * 2)[0]
            window_mask, x_range, y_range = self._create_mask(centroids[level], level, self.WIDTH)

            # Add graphic points from window mask here to total pixels found
            if level > 0:
                image1[margin_mask == 1] = (0, 0, 127)
            image1[window_mask == 1] = (255, 0, 0)
            image1[(grayscale_contour > 127) & (margin_mask == 1)] = (0, 0, 255)

            cv2.putText(image2, str(confidence[level]), (x_range[1], y_range[1]),
                        self.FONT_FACE, self.FONT_SCALE, self.TEXT_COLOR, self.THICKNESS)

    def draw(self, image, grayscale_contour, l_centroids, r_centroids, l_confidence, r_confidence,
             transformer=None):
        # Create an image to draw the lines on
        z1 = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        visualization_layer1 = np.dstack((z1, z1, z1))

        z2 = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        visualization_layer2 = np.dstack((z2, z2, z2))

        self._draw_centroids(visualization_layer1, visualization_layer2, grayscale_contour, l_centroids, l_confidence)
        self._draw_centroids(visualization_layer1, visualization_layer2, grayscale_contour, r_centroids, r_confidence)

        if transformer is not None:
            # Warp back to original image space
            visualization_layer1 = transformer.warp(visualization_layer1)
            visualization_layer2 = transformer.warp(visualization_layer2)

        out_image = cv2.addWeighted(image, 1, visualization_layer1, 0.3, 0)
        out_image = cv2.addWeighted(out_image, 1, visualization_layer2, 0.7, 0)
        return out_image


class MapThumbnailVisualization(object):
    """Visualises lane lines by drawing thumbnail map image"""

    LEFT = 20
    TOP = 20
    WIDTH = 100
    HEIGHT = 60

    LEFT_LANE_OFFSET = WIDTH * 3 / 8
    RIGHT_LANE_OFFSET = WIDTH * 5 / 8

    Y_DRAW = np.linspace(HEIGHT - 1, 0, num=HEIGHT) + TOP

    BG_COLOR = 31
    FG_COLOR = 255

    def __init__(self, image_width, image_height):
        self.image_width = image_width
        self.image_height = image_height
        self.scale_factor = image_height / self.HEIGHT
        self.y = np.linspace(0, image_height - 1, num=self.HEIGHT)

    def draw(self, image, line_curve_l, line_curve_r):
        """Draws thumbnail map image with left and right lane lines"""

        image[self.TOP:self.TOP + self.HEIGHT, self.LEFT:self.LEFT + self.WIDTH] = self.BG_COLOR

        # Compute x-coordinates for left and right lane lines for a thumbnail image
        left_x = line_curve_l.get_x(self.y, offset=0) / self.scale_factor + self.LEFT_LANE_OFFSET + self.LEFT
        right_x = line_curve_r.get_x(self.y, offset=0) / self.scale_factor + self.RIGHT_LANE_OFFSET + self.LEFT

        # Get (x, y) coordinate pairs to draw lines
        pts_left = np.transpose(np.vstack([left_x, self.Y_DRAW]))
        pts_right = np.transpose(np.vstack([right_x, self.Y_DRAW]))

        # Draw lines, apply anti-aliasing
        for x, y in np.concatenate((pts_left, pts_right)):
            y = int(y)
            x1 = int(x)
            x2 = x1 + 1
            v2 = x % x1
            v1 = 1 - v2
            if self.LEFT <= x1 <= self.LEFT + self.WIDTH:
                image[y, x1] = (self.FG_COLOR * v1).astype(np.uint8)
            if self.LEFT <= x2 <= self.LEFT + self.WIDTH:
                image[y, x2] = (self.FG_COLOR * v2).astype(np.uint8)


class TurnRadiusVisualization(object):
    """Visualises turn radius by printing its numerical value"""

    TOP = 31
    LEFT = 130

    FONT_FACE = cv2.FONT_HERSHEY_PLAIN
    FONT_SCALE = 1
    THICKNESS = 1
    COLOR = 255

    def draw(self, image, line_curve_l, line_curve_r):
        """Computes and prints averaged turn radius"""

        r = int((line_curve_l.get_radius() + line_curve_r.get_radius()) / 2)

        text = "Turn radius, meters: {:4}".format(r)
        cv2.putText(image, text, (self.LEFT, self.TOP), self.FONT_FACE, self.FONT_SCALE, self.COLOR, self.THICKNESS)


class PositionVisualization(object):
    """Visualises position in lane by printing offset and indicating direction with arrows"""

    TOP1 = 51
    LEFT1 = 130

    TOP2 = 71
    LEFT2 = 160

    FONT_FACE = cv2.FONT_HERSHEY_PLAIN
    FONT_SCALE = 1
    THICKNESS = 1
    COLOR = 255
    DK_COLOR = 35

    def draw(self, image, line_curve_l, line_curve_r):
        """Prints position in lane"""

        cv2.putText(image, "Position in lane, meters:", (self.LEFT1, self.TOP1), self.FONT_FACE, self.FONT_SCALE,
                    self.COLOR, self.THICKNESS)

        offset = line_curve_l.get_position(line_curve_r)
        l_color, r_color = self.COLOR, self.DK_COLOR
        if offset > 0:
            l_color, r_color = r_color, l_color

        cv2.putText(image, "<-", (self.LEFT2, self.TOP2), self.FONT_FACE, self.FONT_SCALE, l_color, self.THICKNESS)
        cv2.putText(image, "{:.2f}".format(abs(offset)), (self.LEFT2 + 27, self.TOP2), self.FONT_FACE, self.FONT_SCALE,
                    self.COLOR, self.THICKNESS)
        cv2.putText(image, "->", (self.LEFT2 + 67, self.TOP2), self.FONT_FACE, self.FONT_SCALE, r_color, self.THICKNESS)


class FrameNumberVisualization(object):
    """Visualises frame number by printing it"""

    TOP = 40
    LEFT = 800

    FONT_FACE = cv2.FONT_HERSHEY_PLAIN
    FONT_SCALE = 1
    THICKNESS = 1
    COLOR = 255

    def draw(self, image, frame_number):
        """Computes and prints averaged turn radius"""

        text = "Frame {:4}".format(frame_number)
        cv2.putText(image, text, (self.LEFT, self.TOP), self.FONT_FACE, self.FONT_SCALE, self.COLOR, self.THICKNESS)
