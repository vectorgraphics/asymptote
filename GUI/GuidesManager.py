#!/usr/bin/env python3
import PyQt5.QtGui as QtGui
import PyQt5.QtCore as QtCore
import numpy as numpy

class Guide:
    def __init__(self, pen=None):
        if pen is None:
            pen = QtGui.QPen()
        assert isinstance(pen, QtGui.QPen)
        self.pen = pen

    def drawShape(self, pen):
        assert isinstance(pen, QtGui.QPainter)
        pen.save()
        pen.setPen(self.pen)

class LineGuide(Guide):
    def __init__(self, origin, direction, pen=None):
        super().__init__(pen)
        self.origin = origin
        self.direction = direction

    def drawShape(self, pen):
        super().drawShape(pen)
        p1 = self.origin + (9999 * QtCore.QPointF(numpy.cos(self.direction), numpy.sin(self.direction)))
        p2 = self.origin - (9999 * QtCore.QPointF(numpy.cos(self.direction), numpy.sin(self.direction)))
        pen.drawLine(QtCore.QLineF(p1, p2))
        pen.restore()

class ArcGuide(Guide):
    @classmethod
    def radTo16Deg(cls, radians):
        return int(round(numpy.rad2deg(radians) * 16))

    def __init__(self, center=None, radius=1, startAng=0, endAng=(2*numpy.pi), pen=None):
        if center is None:
            center = QtCore.QPointF(0, 0)
        super().__init__(pen)
        self.center = center
        self.radius = int(radius)
        self.startAng = startAng
        self.endAng = endAng

    def drawShape(self, pen):
        super().drawShape(pen)
        assert isinstance(pen, QtGui.QPainter)
        x, y = int(round(self.center.x())), int(round(self.center.y()))
        pen.drawArc(x - self.radius, y - self.radius, 2 * self.radius, 2 * self.radius, ArcGuide.radTo16Deg(self.startAng),
                    -ArcGuide.radTo16Deg(self.endAng - self.startAng))
        pen.restore()
