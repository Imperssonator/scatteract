from __future__ import division
from builtins import map
from builtins import object
from past.utils import old_div
class Rect(object):
    def __init__(self, cx, cy, width, height, confidence):
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        self.confidence = confidence
        self.true_confidence = confidence
    def overlaps(self, other):
        if abs(self.cx - other.cx) > old_div((self.width + other.width), 1.5):
            return False
        elif abs(self.cy - other.cy) > old_div((self.height + other.height), 2.0):
            return False
        else:
            return True
    def distance(self, other):
        return sum(map(abs, [self.cx - other.cx, self.cy - other.cy,
                       self.width - other.width, self.height - other.height]))
    def intersection(self, other):
        left = max(self.cx - old_div(self.width,2.), other.cx - old_div(other.width,2.))
        right = min(self.cx + old_div(self.width,2.), other.cx + old_div(other.width,2.))
        width = max(right - left, 0)
        top = max(self.cy - old_div(self.height,2.), other.cy - old_div(other.height,2.))
        bottom = min(self.cy + old_div(self.height,2.), other.cy + old_div(other.height,2.))
        height = max(bottom - top, 0)
        return width * height
    def area(self):
        return self.height * self.width
    def union(self, other):
        return self.area() + other.area() - self.intersection(other)
    def iou(self, other):
        return old_div(self.intersection(other), self.union(other))
    def __eq__(self, other):
        return (self.cx == other.cx and 
            self.cy == other.cy and
            self.width == other.width and
            self.height == other.height and
            self.confidence == other.confidence)
