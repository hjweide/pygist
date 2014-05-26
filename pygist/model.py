from __future__ import absolute_import, division, print_function


class Model:
    def __init__(self, classifiers, center, scale):
        self.classifiers = classifiers
        self.center = center
        self.scale = scale
