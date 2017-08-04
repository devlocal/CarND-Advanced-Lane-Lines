import numpy as np


class AveragingFilter(object):
    """
    Collects values and computes collected values average.
    """

    SIZE = 5

    def __init__(self):
        self._values = []

    def add(self, value):
        """
        Adds new value to the filter and removes the oldest one.
        If the filter is empty, fill filter buffer with the value.

        :param value: new value to add
        """
        if self._values:
            self._values = self._values[1:] + [value]
        else:
            self._values = [value] * self.SIZE

    def get(self):
        """Returns average value of all the values stored in the filter"""
        return np.average(self._values)


class LaneCurve(object):
    """Measures lane curvature, averaging value over multiple frames"""

    X_MPP = 3.7 / 700  # meters per pixel in x dimension
    Y_MPP = 30 / 720  # meters per pixel in y dimension

    def __init__(self, image_width, image_height):
        """
        :param image_width: image width
        :param image_height: image height
        """
        self.image_height = image_height
        self.image_center = image_width / 2.0

        self.y = np.linspace(0, self.image_height - 1, num=self.image_height)

        self.a = AveragingFilter()
        self.b = AveragingFilter()
        self.c = AveragingFilter()
        self.ma = AveragingFilter()
        self.mb = AveragingFilter()

    def accept(self, curve_points):
        """
        Accepts new curve points. Computes curvature radius.

        :param curve_points: an array of curve points, ordered from bottom to top
        :return:
        """
        cp = curve_points[::-1]  # Reverse to match top-to-bottom

        a, b, c = np.polyfit(self.y, cp, 2)
        ma, mb, _ = np.polyfit(self.y * self.Y_MPP, cp * self.X_MPP, 2)

        self.a.add(a)
        self.b.add(b)
        self.c.add(c)
        self.ma.add(ma)
        self.mb.add(mb)

    def get_x(self, y, offset=None):
        """
        Computes curve x values over a range of y coordinates. Can compute curve with a specified offset
        or with offset observed during value accumulation.

        :param y: a vector of values to compute x coordinates for
        :param offset: optional x-asis offset of the curve, if not provided then observed value if used
        :return: a vector of x coordinates of the curve
        """
        if offset is None:
            offset = self.c.get()
        return self.a.get() * y ** 2 + self.b.get() * y + offset

    def get_radius(self):
        """
        Computes curvature radius

        :return: curvature radius in meters
        """

        # Use image bottom as y-value where we want radius of curvature
        y_eval = (self.image_height - 1) * self.Y_MPP
        return ((1 + (2 * self.ma.get() * y_eval + self.mb.get()) ** 2) ** 1.5) / np.absolute(2 * self.ma.get())

    def get_position(self, right_curve):
        """
        Computes position of the vehicle in lane.
        A positive value indicates shift to the right, a negative values indicates shift to the left.

        :return: displacement from center, meters
        """

        pos_px = self.image_center - (self.c.get() + right_curve.c.get()) / 2.0
        pos_m = pos_px * self.X_MPP

        return pos_m
