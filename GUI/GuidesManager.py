#!/usr/bin/env python3
import PyQt5.QtWidgets as Qw
import PyQt5.QtGui as Qg
import PyQt5.QtCore as Qc
import numpy as np

class Guide:
    def __init__(self, pen=None):
        if pen is None:
            pen = Qg.QPen()
        assert isinstance(pen, Qg.QPen)
        self.pen = pen

    def drawShape(self, pen):
        assert isinstance(pen, Qg.QPainter)
        pen.save()
        pen.setPen(self.pen)

class LineGuide(Guide):
    def __init__(self, origin, direction, pen=None):
        super().__init__(pen)
        self.origin = origin
        self.direction = direction

    def drawShape(self, pen):
        super().drawShape(pen)
        p1 = self.origin + (9999 * Qc.QPointF(np.cos(self.direction), np.sin(self.direction)))
        p2 = self.origin - (9999 * Qc.QPointF(np.cos(self.direction), np.sin(self.direction)))
        pen.drawLine(Qc.QLineF(p1, p2))
        pen.restore()

class ArcGuide(Guide):
    @classmethod
    def radTo16Deg(cls, radians):
        return int(round(np.rad2deg(radians) * 16))

    def __init__(self, center=None, radius=1, startAng=0, endAng=(2*np.pi), pen=None):
        if center is None:
            center = Qc.QPointF(0, 0)
        super().__init__(pen)
        self.center = center
        self.radius = int(radius)
        self.startAng = startAng
        self.endAng = endAng

    def drawShape(self, pen):
        super().drawShape(pen)
        assert isinstance(pen, Qg.QPainter)
        x, y = int(round(self.center.x())), int(round(self.center.y()))
        pen.drawArc(x - self.radius, y - self.radius, 2 * self.radius, 2 * self.radius, ArcGuide.radTo16Deg(self.startAng),
                    -ArcGuide.radTo16Deg(self.endAng - self.startAng))
        pen.restore()
